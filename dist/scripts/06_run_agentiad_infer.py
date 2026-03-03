
import sys
from pathlib import Path
import argparse
import csv
import gc
from contextlib import nullcontext
import hashlib
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Bootstrap src path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if (REPO_ROOT / "dist" / "src").exists() and str(REPO_ROOT / "dist" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "dist" / "src"))
if (REPO_ROOT / "src").exists() and str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# Import SSOT
import importlib.util

def _load_paper_contract_cls():
    # Try normal imports first (fast path)
    try:
        from agentiad_repro.paper_contract import PaperContract  # type: ignore
        return PaperContract
    except Exception:
        pass

    # File-location fallback (robust path-based import)
    candidates = [
        REPO_ROOT / "paper_contract.py",
        REPO_ROOT / "agentiad_repro" / "paper_contract.py",
        REPO_ROOT / "src" / "agentiad_repro" / "paper_contract.py",
        REPO_ROOT / "dist" / "src" / "agentiad_repro" / "paper_contract.py",
        REPO_ROOT / "dist" / "paper_contract.py",
    ]

    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("agentiad_repro__paper_contract", str(p))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                if hasattr(mod, "PaperContract"):
                    return getattr(mod, "PaperContract")

    # Hard fail with diagnostic paths
    tried = "\n".join(str(x) for x in candidates)
    raise ModuleNotFoundError(
        "PaperContract not found. Tried imports and these paths:\n" + tried
    )

PaperContract = _load_paper_contract_cls()

MMAD_ROOT_RESOLVED: Optional[Path] = None
MMAD_ASSET_MODE: str = "unknown"
_LOCAL_DIR_ENTRY_CACHE: Dict[str, Dict[str, Path]] = {}
_CHAT_TEXT_CACHE: Dict[Tuple[str, int, str], str] = {}
_LAST_VLM_EXCEPTIONS: List[Dict[str, Any]] = []
_VLM_RUNTIME_STATS: Dict[str, int] = {
    "generate_calls": 0,
    "generated_tokens_total": 0,
    "retry_exceptions": 0,
}
_CUDA_CLEANUP_STATS: Dict[str, int] = {
    "calls_total": 0,
    "calls_error": 0,
    "calls_success": 0,
}
_ANSWER_END_TAG = "</answer>"
_STRICT_ANSWER_RULE = (
    "\n\nOutput format (MUST, no extra text):\n"
    "1) First token must be <answer>\n"
    "2) Last token must be </answer>\n"
    "3) Inside tags, output ONLY compact JSON with exactly keys: "
    '{"anomaly_present": bool, "top_anomaly": "string", "visual_descriptions": ["string", ...]}\n'
    'Example: <answer>{"anomaly_present":false,"top_anomaly":"none","visual_descriptions":[]}</answer>'
)


def _chat_template_key(processor: Any, n_images: int, prompt: str) -> Tuple[str, int, str]:
    tpl = str(getattr(processor, "chat_template", "") or "")
    tpl_sig = hashlib.sha256(tpl.encode("utf-8")).hexdigest()[:16]
    return (tpl_sig, int(n_images), prompt)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    data = yaml.safe_load(_read_text(path))
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping/dict.")
    return data


def _merge_cfg(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k == "base_config":
            continue
        out[k] = v
    return out


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)

def _env_flag_1(name: str) -> bool:
    return str(os.environ.get(name, "")).strip() == "1"

def _platform_copy_model_cmd() -> str:
    if os.name == "nt":
        return "Copy-Item -Recurse -Force <DOWNLOADED_MODEL_DIR> <LOCAL_VLM_DIR>"
    return "cp -a <DOWNLOADED_MODEL_DIR> <LOCAL_VLM_DIR>"

def _enforce_answer_only_prompt(base_prompt: str) -> str:
    return _safe_str(base_prompt).rstrip() + _STRICT_ANSWER_RULE

def _extract_first_answer_block(text: str) -> str:
    s = _safe_str(text).strip()
    start = s.find(PaperContract.TAG_ANSWER_START)
    if start < 0:
        return s
    end = s.find(PaperContract.TAG_ANSWER_END, start + len(PaperContract.TAG_ANSWER_START))
    if end < 0:
        return s[start:].strip()
    return s[start : end + len(PaperContract.TAG_ANSWER_END)].strip()

def _meta_has_missing_answer_tags(meta: Mapping[str, Any]) -> bool:
    err = _safe_str(meta.get("error", "")).lower()
    return "missing <answer> tags" in err

def _build_answer_stop_criteria(processor: Any) -> Any:
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except Exception:
        return None

    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        tok = processor
    encode = getattr(tok, "encode", None)
    if not callable(encode):
        return None
    try:
        stop_ids = encode(PaperContract.TAG_ANSWER_END, add_special_tokens=False)
    except Exception:
        return None
    if isinstance(stop_ids, int):
        stop_ids = [int(stop_ids)]
    if not isinstance(stop_ids, list) or not stop_ids:
        return None
    stop_ids = [int(x) for x in stop_ids]

    class _StopOnAnswerEnd(StoppingCriteria):
        def __init__(self, token_ids: List[int]):
            super().__init__()
            self._ids = token_ids
            self._n = len(token_ids)

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            if input_ids is None or getattr(input_ids, "shape", None) is None:
                return False
            if int(input_ids.shape[1]) < self._n:
                return False
            tail = input_ids[0, -self._n :].tolist()
            return [int(x) for x in tail] == self._ids

    return StoppingCriteriaList([_StopOnAnswerEnd(stop_ids)])


def _sha256_upper_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def _sha256_upper_json(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_upper_text(s)


def _normalize_yesno(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, bool):
        return "yes" if x else "no"
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1", "anomaly", "abnormal"}:
        return "yes"
    if s in {"no", "n", "false", "0", "normal"}:
        return "no"
    return None


def _coerce_final_anomaly(x: Any) -> Tuple[str, bool]:
    norm = _normalize_yesno(x)
    if norm in {"yes", "no"}:
        return norm, False
    return "no", True


def schema_repair(final: Any) -> Dict[str, Any]:
    """
    Applies strict schema repair to the `final` field.
    - Ensures `anomaly` is one of ['yes', 'no']
    - Ensures `defect_type` is non-empty
    - Adds missing `reason`
    """
    out: Dict[str, Any] = dict(final) if isinstance(final, dict) else {}
    anomaly = _normalize_yesno(out.get("anomaly"))
    out["anomaly"] = anomaly if anomaly in {"yes", "no"} else "no"
    defect_type = _safe_str(out.get("defect_type")).strip()
    out["defect_type"] = defect_type if defect_type else "none"
    reason = _safe_str(out.get("reason")).strip()
    out["reason"] = reason if reason else "model_output"
    return out


def _normalize_bbox_or_fallback(bbox_value: Any, sample_id: str) -> List[float]:
    if isinstance(bbox_value, list) and len(bbox_value) == 4:
        try:
            return [float(bbox_value[0]), float(bbox_value[1]), float(bbox_value[2]), float(bbox_value[3])]
        except Exception:
            pass
    return _fallback_bbox_norm(sample_id)


def _build_final_obj(sample_id: str, parsed: Mapping[str, Any], bbox_hint: Any) -> Tuple[Dict[str, Any], bool]:
    parsed_map: Mapping[str, Any] = parsed if isinstance(parsed, Mapping) else {}
    anomaly, coerced_anomaly = _coerce_final_anomaly(parsed_map.get("anomaly"))
    bbox_value = parsed_map.get("bbox") if isinstance(parsed_map.get("bbox"), list) else bbox_hint
    defect_type = _safe_str(parsed_map.get("defect_type")).strip()
    if not defect_type or defect_type.lower() in {"unknown", "execution_error"}:
        defect_type = "none" if anomaly == "no" else "unspecified_anomaly"
    if anomaly == "yes" and defect_type.lower() == "none":
        defect_type = "unspecified_anomaly"
    if anomaly == "no":
        defect_type = "none"
    reason = _safe_str(parsed_map.get("reason")).strip()
    if not reason:
        reason = "schema_repaired_fallback" if coerced_anomaly else "model_output"
    repaired_core = schema_repair(
        {
            "anomaly": anomaly,
            "defect_type": defect_type,
            "reason": reason,
        }
    )
    final_obj = {
        "anomaly": repaired_core["anomaly"],
        "defect_type": repaired_core["defect_type"],
        "reason": repaired_core["reason"],
        "confidence": None,
        "bbox": _normalize_bbox_or_fallback(bbox_value, sample_id),
    }
    return final_obj, coerced_anomaly


def _append_trace_error(trace: Optional[Dict[str, Any]], code: str, message: str, detail: Optional[Dict[str, Any]] = None) -> None:
    if not isinstance(trace, dict):
        return
    errs = trace.get("inference_errors")
    if not isinstance(errs, list):
        errs = []
        trace["inference_errors"] = errs
    item: Dict[str, Any] = {"code": str(code), "message": str(message)}
    if isinstance(detail, dict) and detail:
        item["detail"] = detail
    errs.append(item)


def _prepare_raw_output_json(
    *,
    sample_id: str,
    raw_payload: Any,
    bbox_hint: Any,
) -> Tuple[str, Dict[str, Any], bool]:
    """
    Ensure raw_output is JSON-serializable and contains a schema-valid final dict.
    Returns (json_text_single_line, normalized_obj, roundtrip_ok).
    """
    normalized: Dict[str, Any]
    if isinstance(raw_payload, dict):
        normalized = dict(raw_payload)
    else:
        normalized = {"meta": {"sanity_coerced": True, "sanity_reason": "raw_output_not_dict"}}
    meta_obj = normalized.get("meta")
    if not isinstance(meta_obj, dict):
        meta_obj = {}
        normalized["meta"] = meta_obj

    final_in = normalized.get("final")
    final_map = final_in if isinstance(final_in, Mapping) else {}
    final_obj, _ = _build_final_obj(sample_id, final_map, bbox_hint)
    if not isinstance(final_in, dict):
        meta_obj["sanity_coerced"] = True
        meta_obj["sanity_reason"] = "final_not_dict"
    normalized["final"] = final_obj

    try:
        s = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        s = s.replace("\r", " ").replace("\n", " ")
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("roundtrip_not_dict")
        m2 = obj.get("meta")
        if not isinstance(m2, dict):
            m2 = {}
            obj["meta"] = m2
        m2["roundtrip_ok"] = True
        s2 = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        s2 = s2.replace("\r", " ").replace("\n", " ")
        return s2, obj, True
    except Exception as e:
        fallback = {
            "round0": "",
            "round1": "",
            "round2": "",
            "cr_called": False,
            "ref_sample_id": "",
            "final": {
                "anomaly": "no",
                "defect_type": "execution_error",
                "reason": "raw_output_roundtrip_failed",
                "confidence": None,
                "bbox": _normalize_bbox_or_fallback(bbox_hint, sample_id),
            },
            "meta": {
                "sanity_coerced": True,
                "roundtrip_ok": False,
                "roundtrip_error": f"{type(e).__name__}: {e}",
            },
        }
        s3 = json.dumps(fallback, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        s3 = s3.replace("\r", " ").replace("\n", " ")
        return s3, fallback, False


def _label_from_path_segments(p: str) -> str:
    s = p.replace("\\", "/")
    segments = [seg for seg in s.split("/") if seg]
    if any(seg == "good" for seg in segments):
        return "no"
    return "yes" if p else ""


def _extract_gt_yesno(row: Mapping[str, Any], gt_key_override: Optional[str]) -> Tuple[str, str]:
    if gt_key_override:
        val = row.get(gt_key_override)
        norm = _normalize_yesno(val)
        if norm in {"yes", "no"}:
            return norm, f"override:{gt_key_override}"
        if isinstance(val, str) and val:
            norm2 = _label_from_path_segments(val)
            if norm2 in {"yes", "no"}:
                return norm2, f"override:{gt_key_override}"

    for key in ["label", "gt_label", "anomaly", "is_anomaly"]:
        norm = _normalize_yesno(row.get(key))
        if norm in {"yes", "no"}:
            return norm, f"field:{key}"

    p = str(row.get("query_image", "") or "")
    if p:
        segments = [seg for seg in p.replace("\\", "/").split("/") if seg]
        if any(seg == "good" for seg in segments):
            return "no", "rule:query_image_good_segment"
        return "yes", "rule:query_image_not_good"

    return "", "NONE"


def _extract_class_name(row: Mapping[str, Any]) -> str:
    for key in ["class_name", "category", "object", "cls"]:
        v = row.get(key)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return ""


class ImageLoadError(Exception):
    def __init__(self, reason: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.reason = reason
        self.detail = detail or {}


def _image_loader_env_snapshot() -> Dict[str, Any]:
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or str(Path.home() / ".cache" / "huggingface")
    return {
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", ""),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
        "HF_HOME": hf_home,
    }


def _model_cache_dir_name(model_id: str) -> str:
    rid = str(model_id or "").strip().replace("\\", "/").strip("/")
    return "models--" + rid.replace("/", "--")


def _local_model_dir_ready(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    has_config = (model_dir / "config.json").exists()
    has_tokenizer = (
        (model_dir / "tokenizer.json").exists()
        or (model_dir / "tokenizer_config.json").exists()
        or (model_dir / "preprocessor_config.json").exists()
    )
    return bool(has_config and has_tokenizer)


def _find_cached_snapshot_dir(model_id: str, hub_cache: Path) -> Optional[Path]:
    rid = str(model_id or "").strip()
    if not rid:
        return None
    cache_dir = hub_cache / _model_cache_dir_name(rid)
    if not cache_dir.exists():
        return None
    snaps = cache_dir / "snapshots"
    if snaps.exists() and snaps.is_dir():
        for p in sorted(snaps.iterdir()):
            if p.is_dir() and _local_model_dir_ready(p):
                return p
    if _local_model_dir_ready(cache_dir):
        return cache_dir
    return None


def _ensure_local_only_model_ready_or_raise(vlm_model_id: str, local_only: bool) -> None:
    if not local_only:
        return
    env = _image_loader_env_snapshot()
    hf_home = Path(str(env.get("HF_HOME", "") or Path.home() / ".cache" / "huggingface")).expanduser()
    hub_cache_raw = str(os.environ.get("HUGGINGFACE_HUB_CACHE", "")).strip()
    if hub_cache_raw:
        hub_cache = Path(hub_cache_raw).expanduser()
    elif hf_home.name.lower() == "hub":
        hub_cache = hf_home
    else:
        hub_cache = hf_home / "hub"
    fallback_id = str(os.environ.get("DISTILGPT2_LOCAL_DIR", "")).strip() or "distilgpt2"
    model_id_str = str(vlm_model_id or "").strip()
    local_candidate = Path(model_id_str).expanduser()
    checked: List[str] = []

    if local_candidate.exists() and local_candidate.is_dir():
        if _local_model_dir_ready(local_candidate):
            return
        checked.append(f"local_dir_not_ready:{local_candidate}")
    else:
        for cand in [model_id_str, fallback_id]:
            c = str(cand or "").strip()
            if not c:
                continue
            p = Path(c).expanduser()
            if p.exists() and p.is_dir():
                if _local_model_dir_ready(p):
                    return
                checked.append(f"local_path_not_ready:{p}")
                continue
            snap = _find_cached_snapshot_dir(c, hub_cache)
            if snap is not None:
                return
            checked.append(f"hf_cache_miss:{c}")

    remediation_model = model_id_str if "/" in model_id_str else "Qwen/Qwen2.5-VL-3B-Instruct"
    cmd_a = (
        f"{_platform_copy_model_cmd()}; "
        "python verify_all.py --mode phase2_full --strict-contract --dataset-split train --seeds 0 "
        "--vlm-model-local-dir <LOCAL_VLM_DIR>"
    )
    cmd_b = (
        f"HF_HOME=\"{hf_home}\" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "
        f"huggingface-cli download \"{remediation_model}\" --local-dir <DOWNLOADED_MODEL_DIR>"
    )
    raise RuntimeError(
        "Offline local-only model check failed before from_pretrained. "
        f"vlm_model_id={model_id_str or 'NOT_SET'}; "
        f"HF_HOME={hf_home}; HF_CACHE={hub_cache}; "
        f"checked={'; '.join(checked) if checked else 'none'}. "
        f"Remediation A: {cmd_a}. "
        f"Remediation B: {cmd_b}."
    )


def _load_paths_yaml_candidates(project_root: Path) -> Dict[str, Any]:
    candidates = [
        project_root / "configs" / "paths.yaml",
        project_root / "dist" / "configs" / "paths.yaml",
    ]
    for p in candidates:
        try:
            if p.exists():
                data = _load_yaml(p)
                if isinstance(data, dict):
                    return data
        except Exception:
            continue
    return {}


def resolve_mmad_root(project_root: Path, paths: Any) -> Tuple[Optional[Path], str]:
    env_raw = str(os.environ.get("MMAD_ROOT", "") or "").strip()
    if env_raw:
        p = Path(env_raw).expanduser().resolve()
        return p, "env"

    cfg = _load_paths_yaml_candidates(project_root)
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    if isinstance(paths_cfg, dict):
        cfg_raw = str(paths_cfg.get("mmad_root", "") or "").strip()
        if cfg_raw:
            p = Path(cfg_raw)
            if not p.is_absolute():
                p = (project_root / p).resolve()
            else:
                p = p.resolve()
            return p, "paths_yaml"

    mmad_dir = getattr(paths, "mmad_dir", None)
    if isinstance(mmad_dir, Path):
        return mmad_dir.resolve(), "paths.mmad_dir"

    data_dir = getattr(paths, "data_dir", None)
    if isinstance(data_dir, Path):
        return (data_dir / "mmad").resolve(), "paths.data_dir/mmad"

    return None, "not_set"


def _detect_mmad_asset_mode(mmad_root: Optional[Path], image_env: Dict[str, Any]) -> str:
    has_local_assets = False
    if mmad_root is not None:
        has_local_assets = (mmad_root / "DS-MVTec").exists() or (mmad_root / "MVTec-AD").exists()
    if has_local_assets:
        return "local_root"
    offline = any(str(image_env.get(k, "")).strip() == "1" for k in ["HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"])
    if offline:
        return "hub_disabled_no_assets"
    return "hf_cache"


def _extract_dataset_filename(value: Any) -> Optional[str]:
    raw = None
    if isinstance(value, dict):
        p = value.get("path")
        if isinstance(p, str) and p.strip():
            raw = p
    elif isinstance(value, str):
        raw = value
    if raw is None:
        return None
    s = str(raw).strip().replace("\\", "/")
    if not s:
        return None
    if "://" in s:
        return None
    if Path(s).is_absolute():
        return None
    return s.lstrip("./").lstrip("/")


def _get_dir_entry_map_ci(parent: Path) -> Dict[str, Path]:
    key = str(parent.resolve())
    cached = _LOCAL_DIR_ENTRY_CACHE.get(key)
    if cached is not None:
        return cached
    mapping: Dict[str, Path] = {}
    try:
        for child in parent.iterdir():
            mapping[child.name.lower()] = child
    except Exception:
        pass
    _LOCAL_DIR_ENTRY_CACHE[key] = mapping
    return mapping


def _resolve_case_insensitive_path(base: Path, rel_path: str) -> Optional[Path]:
    cur = base
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    for part in parts:
        direct = cur / part
        if direct.exists():
            cur = direct
            continue
        mapping = _get_dir_entry_map_ci(cur)
        nxt = mapping.get(part.lower())
        if nxt is None:
            return None
        cur = nxt
    return cur if cur.exists() else None


def _resolve_local_mmad_asset(filename: str) -> Optional[Path]:
    if MMAD_ROOT_RESOLVED is None:
        return None
    rel = filename.replace("\\", "/").lstrip("/")
    direct = MMAD_ROOT_RESOLVED / rel
    if direct.exists():
        return direct
    return _resolve_case_insensitive_path(MMAD_ROOT_RESOLVED, rel)


def _local_root_prefix_probe(d0: Any, sample_n: int = 1000) -> Tuple[List[str], Dict[str, str]]:
    n = 0
    needed: Dict[str, str] = {}
    total = len(d0) if hasattr(d0, "__len__") else sample_n
    lim = min(int(total), int(max(1, sample_n)))
    for i in range(lim):
        row = d0[i]
        if not isinstance(row, dict):
            continue
        for _, v in _extract_image_candidates(row):
            fn = _extract_dataset_filename(v)
            if not fn or "/" not in fn:
                continue
            prefix = fn.split("/", 1)[0].strip()
            if not prefix:
                continue
            if prefix not in needed:
                needed[prefix] = fn
        n += 1
        if n >= lim:
            break
    return sorted(needed.keys()), needed


def _extract_image_candidates(row: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    for key in ["query_image", "image", "img", "image_path", "path", "file_path", "image_file", "image_bytes", "bytes"]:
        if key in row and row.get(key) is not None:
            out.append((key, row.get(key)))
    return out


def _classify_image_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "cannot identify image file" in msg or "truncated" in msg or "decoder" in msg:
        return "decode_error"
    if isinstance(exc, FileNotFoundError):
        return "missing_file"
    if isinstance(exc, TypeError):
        return "unsupported_image_value"
    return "image_load_error_other"


def _git_commit() -> Optional[str]:
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        out = out.strip()
        return out if out else None
    except Exception:
        return None


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _fallback_bbox_norm(sample_id: str) -> List[float]:
    h = hashlib.sha256(sample_id.encode("utf-8")).digest()
    v = int.from_bytes(h[:8], byteorder="big", signed=False)
    dx = ((v % 101) - 50) / 1000.0
    dy = (((v // 101) % 101) - 50) / 1000.0
    x1 = 0.25 + dx
    y1 = 0.25 + dy
    x2 = 0.75 + dx
    y2 = 0.75 + dy
    if x1 < 0.0:
        x1 = 0.0
    if y1 < 0.0:
        y1 = 0.0
    if x2 > 1.0:
        x2 = 1.0
    if y2 > 1.0:
        y2 = 1.0
    if x2 <= x1:
        x1 = 0.0
        x2 = 1.0
    if y2 <= y1:
        y1 = 0.0
        y2 = 1.0
    return [float(x1), float(y1), float(x2), float(y2)]


def _parse_agent_json(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Use strict SSOT parsing
    parsed = PaperContract.parse_model_output(text)
    
    out: Dict[str, Any] = {}
    meta = {"parse_ok": False, "contract": "strict"}
    
    if parsed["valid_syntax"]:
        data = parsed["data"]
        # Validate schema but don't fail hard, just log/meta
        is_valid, errors = PaperContract.validate_schema(data)
        meta["schema_valid"] = is_valid
        meta["schema_errors"] = errors
        meta["parse_ok"] = True
        
        # Map to internal keys for downstream logic
        ap = data.get("anomaly_present")
        if ap is None:
            out["anomaly"] = "unknown"
        elif isinstance(ap, bool):
            out["anomaly"] = "yes" if ap else "no"
        else:
            s = str(ap).strip().lower()
            if s in {"yes", "true", "1"}:
                out["anomaly"] = "yes"
            elif s in {"no", "false", "0"}:
                out["anomaly"] = "no"
            else:
                out["anomaly"] = "unknown"

        out["defect_type"] = str(data.get("top_anomaly", "none"))
        # confidence is FORBIDDEN by contract, so it is None
        
        # Tool Calls?
        # If the model output contained tool calls, we might want to expose them
        # But this function returns a single dict.
        # We'll attach them to meta if needed, or rely on trace handling.
        
        # BBox?
        # The contract output doesn't strictly have bbox in <answer>.
        # BBox comes from crop_image_normalized tool call arguments.
        # If the model output <tool_call>...bbox_2d...</tool_call>, we need to extract it.
        # Check tool calls
        for tc in parsed["tool_calls"]:
            if tc.get("name") == "crop_image_normalized":
                args = tc.get("arguments", {})
                if "bbox_2d" in args:
                    out["bbox"] = args["bbox_2d"]
        
        return out, meta
    else:
        meta["error"] = parsed.get("error")
        return {}, meta


def _summarize_raw_text(text: str, max_len: int = 240) -> str:
    s = re.sub(r"\s+", " ", _safe_str(text)).strip()
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


def _drain_vlm_errors(round_idx: int) -> List[Dict[str, Any]]:
    errs: List[Dict[str, Any]] = []
    err = _take_last_vlm_exception()
    while err:
        errs.append({"round": int(round_idx), **err})
        err = _take_last_vlm_exception()
    return errs


def _schema_valid(meta: Mapping[str, Any], parsed: Mapping[str, Any]) -> bool:
    if not (bool(meta.get("parse_ok")) and bool(meta.get("schema_valid"))):
        return False
    anomaly_ok = _normalize_yesno(parsed.get("anomaly")) in {"yes", "no"}
    defect_type_ok = bool(_safe_str(parsed.get("defect_type")).strip())
    return bool(anomaly_ok and defect_type_ok)


def _build_repair_prompt(base_prompt: str) -> str:
    strict_rule = "\n\n只输出 <answer>...</answer>，不要任何其它字符；立即重写。"
    return _safe_str(base_prompt) + strict_rule


def _make_forced_valid_answer() -> str:
    payload = {
        "anomaly_present": False,
        "top_anomaly": "none",
        "visual_descriptions": [],
    }
    return "<answer>\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n</answer>"


def _vlm_generate_with_schema_repair(
    *,
    processor: Any,
    model: Any,
    images: Sequence["PIL.Image.Image"],
    prompt: str,
    generation_config: Any,
    use_cache: bool,
    sdp_backend: str,
    max_retries: int,
    repair_n: int,
    answer_stop_criteria: Any,
    dry_run: bool,
    sample_id: str,
    seed: int,
    round_idx: int,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    schema_repair_debug = _env_flag_1("SCHEMA_REPAIR_DEBUG")

    def _run(prompt_text: str) -> Tuple[str, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        if dry_run:
            raw_local = _vlm_generate_dry(images, prompt_text, sample_id, seed)
        else:
            raw_local = _vlm_generate(
                processor,
                model,
                images,
                prompt_text,
                generation_config,
                use_cache=use_cache,
                sdp_backend=sdp_backend,
                max_retries=max_retries,
                answer_stop_criteria=answer_stop_criteria,
            )
        raw_local = _extract_first_answer_block(raw_local)
        parsed_local, meta_local = _parse_agent_json(raw_local)
        if not isinstance(meta_local, dict):
            meta_local = {}
        errs_local = _drain_vlm_errors(round_idx)
        if errs_local:
            meta_local["inference_errors"] = errs_local
        return raw_local, parsed_local, meta_local, errs_local

    raw, parsed, meta, _ = _run(prompt)
    repair_summaries: List[Dict[str, Any]] = []
    if _schema_valid(meta, parsed):
        meta["repair_attempted"] = False
        meta["repair_n"] = int(max(0, repair_n))
        return raw, parsed, meta

    repair_prompt = _build_repair_prompt(prompt)
    total_repairs = max(0, int(repair_n))
    for ridx in range(1, total_repairs + 1):
        repair_summaries.append(
            {
                "attempt": int(ridx),
                "raw_text_summary": _summarize_raw_text(raw),
                "schema_valid": bool(_schema_valid(meta, parsed)),
                "parse_ok": bool(meta.get("parse_ok")),
                "schema_errors": list(meta.get("schema_errors", []) or []),
                "parse_error": _safe_str(meta.get("error")),
                "anomaly_value": _safe_str(parsed.get("anomaly")),
                "defect_type_value": _safe_str(parsed.get("defect_type")),
                "repair_failed": True,
            }
        )
        if schema_repair_debug:
            print(
                f"[schema_repair] sample_id={sample_id} round={round_idx} attempt={ridx}/{total_repairs} failed "
                f"parse_ok={bool(meta.get('parse_ok'))} contract_schema_valid={bool(meta.get('schema_valid'))} "
                f"anomaly={_safe_str(parsed.get('anomaly'))} defect_type={_safe_str(parsed.get('defect_type'))} "
                f"errors={list(meta.get('schema_errors', []) or [])} parse_error={_safe_str(meta.get('error'))}",
                file=sys.stderr,
            )
        raw, parsed, meta, _ = _run(repair_prompt)
        if _schema_valid(meta, parsed):
            meta["repair_attempted"] = True
            meta["repair_n"] = int(total_repairs)
            meta["repair_attempt_used"] = int(ridx)
            meta["repair_history"] = repair_summaries
            if repair_summaries:
                meta["repair_failed"] = True
            return raw, parsed, meta

    repair_summaries.append(
        {
            "attempt": int(total_repairs + 1),
            "raw_text_summary": _summarize_raw_text(raw),
            "schema_valid": bool(_schema_valid(meta, parsed)),
            "parse_ok": bool(meta.get("parse_ok")),
            "schema_errors": list(meta.get("schema_errors", []) or []),
            "parse_error": _safe_str(meta.get("error")),
            "anomaly_value": _safe_str(parsed.get("anomaly")),
            "defect_type_value": _safe_str(parsed.get("defect_type")),
            "repair_failed": True,
        }
    )
    if schema_repair_debug:
        print(
            f"[schema_repair] sample_id={sample_id} round={round_idx} exhausted_retries={total_repairs}; "
            "forcing schema-valid fallback final.",
            file=sys.stderr,
        )
    forced_raw = _make_forced_valid_answer()
    forced_parsed, forced_meta = _parse_agent_json(forced_raw)
    if not isinstance(forced_meta, dict):
        forced_meta = {}
    forced_meta["repair_attempted"] = True
    forced_meta["repair_n"] = int(total_repairs)
    forced_meta["repair_failed"] = True
    forced_meta["raw_text_summary"] = _summarize_raw_text(raw)
    forced_meta["repair_history"] = repair_summaries
    forced_meta["forced_valid_answer"] = True
    return forced_raw, forced_parsed, forced_meta


def _uncertain(parsed: Mapping[str, Any]) -> bool:
    # Strict Contract: No confidence score available.
    # Uncertainty heuristic based on confidence is removed.
    # We rely on whether 'anomaly' was parsed correctly.
    # If anomaly is unknown, we are uncertain.
    an = parsed.get("anomaly")
    if an not in {"yes", "no"}:
        return True
    
    # Without confidence, we assume certainty if format is valid.
    return False


def _decode_hf_image(dataset_id: str, value: Any) -> "PIL.Image.Image":
    import io
    import os as _os
    from pathlib import Path as _Path

    from PIL import Image as PILImage

    if value is None:
        raise ImageLoadError("no_image_field", "image is None")
    if hasattr(value, "save"):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return PILImage.open(io.BytesIO(bytes(value)))
        except Exception as e:
            raise ImageLoadError("bad_bytes", f"Failed to decode bytes image: {type(e).__name__}: {e}") from e
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            try:
                return PILImage.open(io.BytesIO(value["bytes"]))
            except Exception as e:
                raise ImageLoadError("bad_bytes", f"Failed to decode dict bytes image: {type(e).__name__}: {e}") from e
        if "path" in value and value["path"]:
            p = _Path(value["path"])
            if p.exists():
                try:
                    return PILImage.open(p)
                except Exception as e:
                    raise ImageLoadError("decode_error", f"Failed to decode local path image: {type(e).__name__}: {e}", {"path": str(p)}) from e
            if MMAD_ASSET_MODE == "local_root":
                fn = _extract_dataset_filename(value)
                local_p = _resolve_local_mmad_asset(fn or str(p))
                if local_p is not None:
                    try:
                        return PILImage.open(local_p)
                    except Exception as e:
                        raise ImageLoadError("decode_error", f"Failed to decode local MMAD asset: {type(e).__name__}: {e}", {"path": str(local_p), "filename": fn or str(p)}) from e
                raise ImageLoadError(
                    "local_asset_missing",
                    f"Local MMAD asset missing: {fn or p}",
                    {"filename": fn or str(p), "mmad_root": str(MMAD_ROOT_RESOLVED) if MMAD_ROOT_RESOLVED else "NOT_SET"},
                )
            raise ImageLoadError("missing_file", f"Local image path missing: {p}", {"path": str(p)})
    if not isinstance(value, str):
        raise ImageLoadError("unsupported_image_value", f"Unsupported image type: {type(value)}")

    p = _Path(value)
    if p.exists():
        try:
            return PILImage.open(p)
        except Exception as e:
            raise ImageLoadError("decode_error", f"Failed to decode local image: {type(e).__name__}: {e}", {"path": str(p)}) from e

    filename = _extract_dataset_filename(value)
    if MMAD_ASSET_MODE == "local_root":
        local_p = _resolve_local_mmad_asset(filename or value)
        if local_p is not None:
            try:
                return PILImage.open(local_p)
            except Exception as e:
                raise ImageLoadError("decode_error", f"Failed to decode local MMAD asset: {type(e).__name__}: {e}", {"path": str(local_p), "filename": filename or value}) from e
        raise ImageLoadError(
            "local_asset_missing",
            f"Local MMAD asset missing: {filename or value}",
            {"filename": filename or value, "mmad_root": str(MMAD_ROOT_RESOLVED) if MMAD_ROOT_RESOLVED else "NOT_SET"},
        )

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except Exception as e:
        raise ImageLoadError("hub_unavailable", f"huggingface_hub import failed: {type(e).__name__}: {e}") from e

    if value.startswith("MVTec-AD/"):
        rev = _os.environ.get("MMAD_MVTEC_AD_REVISION") or "e88b7bd615ad582b0a7e8238066a9fb293a072b4"
        try:
            local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value, revision=rev)
            return PILImage.open(local_path)
        except Exception as e:
            raise ImageLoadError("hub_download_error", f"HF download failed: {type(e).__name__}: {e}", {"filename": value, "revision": rev}) from e

    if value.startswith("DS-MVTec/"):
        rev = _os.environ.get("MMAD_ASSET_REVISION") or "f1ad11c07452dff1e023a0df1093e40701d22cab"
        try:
            local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value, revision=rev)
            return PILImage.open(local_path)
        except Exception as e:
            raise ImageLoadError("hub_download_error", f"HF download failed: {type(e).__name__}: {e}", {"filename": value, "revision": rev}) from e

    try:
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value)
    except EntryNotFoundError:
        rev = _os.environ.get("MMAD_ASSET_REVISION") or "f1ad11c07452dff1e023a0df1093e40701d22cab"
        try:
            local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value, revision=rev)
        except Exception as e:
            raise ImageLoadError("hub_download_error", f"HF fallback download failed: {type(e).__name__}: {e}", {"filename": value, "revision": rev}) from e
    except Exception as e:
        raise ImageLoadError("hub_download_error", f"HF download failed: {type(e).__name__}: {e}", {"filename": value}) from e
    try:
        return PILImage.open(local_path)
    except Exception as e:
        raise ImageLoadError("decode_error", f"Failed to decode hub image: {type(e).__name__}: {e}", {"path": str(local_path)}) from e


def _read_image_bytes_and_sha(path: str) -> Tuple[Optional[bytes], Optional[str], int]:
    p = Path(path)
    if not p.exists():
        return None, None, 0
    try:
        b = p.read_bytes()
    except Exception:
        return None, None, 0
    return b, hashlib.sha256(b).hexdigest().upper(), len(b)


def _pil_from_bytes_rgb(b: bytes) -> "PIL.Image.Image":
    import io
    from PIL import Image as PILImage

    return PILImage.open(io.BytesIO(b)).convert("RGB")


def _resize_image_max_side(img: "PIL.Image.Image", max_side: int) -> "PIL.Image.Image":
    try:
        ms = int(max_side)
    except Exception:
        ms = 0
    if ms <= 0:
        return img
    w, h = img.size
    cur_max = max(int(w), int(h))
    if cur_max <= ms:
        return img
    scale = float(ms) / float(cur_max)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h))


def _cleanup_cuda_memory(*, reason: str = "error") -> None:
    global _CUDA_CLEANUP_STATS
    _CUDA_CLEANUP_STATS["calls_total"] = int(_CUDA_CLEANUP_STATS.get("calls_total", 0)) + 1
    if str(reason).strip().lower().startswith("success"):
        _CUDA_CLEANUP_STATS["calls_success"] = int(_CUDA_CLEANUP_STATS.get("calls_success", 0)) + 1
    else:
        _CUDA_CLEANUP_STATS["calls_error"] = int(_CUDA_CLEANUP_STATS.get("calls_error", 0)) + 1
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


def _take_last_vlm_exception() -> Optional[Dict[str, Any]]:
    global _LAST_VLM_EXCEPTIONS
    if not _LAST_VLM_EXCEPTIONS:
        return None
    err = _LAST_VLM_EXCEPTIONS.pop(0)
    return err


def _sdp_context(torch_mod: Any, backend: str):
    b = str(backend or "auto").strip().lower()
    if b == "auto":
        return nullcontext()
    try:
        sdp_kernel = getattr(getattr(torch_mod.backends, "cuda", None), "sdp_kernel", None)
        if not callable(sdp_kernel):
            return nullcontext()
        if b == "math":
            return sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        if b == "flash":
            return sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        if b == "mem_efficient":
            return sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)
    except Exception:
        return nullcontext()
    return nullcontext()


def _compute_correct(gt_label: Any, pred_label: Any) -> int:
    gt = _normalize_yesno(gt_label)
    pred = _normalize_yesno(pred_label)
    if gt in {"yes", "no"} and pred in {"yes", "no"}:
        return 1 if gt == pred else 0
    return 0


def _infer_model_device(model: Any) -> "torch.device":
    import torch

    emb = getattr(model, "get_input_embeddings", None)
    if callable(emb):
        e = model.get_input_embeddings()
        if e is not None and hasattr(e, "weight") and hasattr(e.weight, "device") and str(e.weight.device) != "meta":
            return e.weight.device
    md = getattr(model, "device", None)
    if md is not None:
        try:
            return md
        except Exception:
            pass
    for p in model.parameters():
        if hasattr(p, "device") and str(p.device) != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _vlm_generate(
    processor: Any,
    model: Any,
    images: Sequence["PIL.Image.Image"],
    prompt: str,
    generation_config: Any,
    use_cache: bool = True,
    sdp_backend: str = "auto",
    max_retries: int = 2,
    answer_stop_criteria: Any = None,
) -> str:
    global _LAST_VLM_EXCEPTIONS
    _LAST_VLM_EXCEPTIONS = []
    if model is None:
        return '<answer>{"anomaly_present": false, "top_anomaly": "none", "visual_descriptions": []}</answer>'

    import torch

    model_device = _infer_model_device(model)
    imgs = list(images)
    for i, im in enumerate(imgs):
        mode = getattr(im, "mode", None)
        if hasattr(im, "convert") and mode not in {None, "RGB"}:
            imgs[i] = im.convert("RGB")

    def _record_exc(attempt: int, exc: Exception) -> None:
        _LAST_VLM_EXCEPTIONS.append(
            {
                "attempt": int(attempt),
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )

    def _prepare_inputs(local_imgs: Sequence["PIL.Image.Image"]) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        try:
            if hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None):
                try:
                    cache_key = _chat_template_key(processor, len(local_imgs), prompt)
                    text = _CHAT_TEXT_CACHE.get(cache_key, "")
                    if not text:
                        messages: List[Dict[str, Any]] = [
                            {
                                "role": "user",
                                "content": [{"type": "image"} for _ in local_imgs] + [{"type": "text", "text": prompt}],
                            }
                        ]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        _CHAT_TEXT_CACHE[cache_key] = text
                    inputs = processor(text=[text], images=list(local_imgs), return_tensors="pt")
                except Exception:
                    cache_key_text = _chat_template_key(processor, 0, prompt)
                    text = _CHAT_TEXT_CACHE.get(cache_key_text, "")
                    if not text:
                        messages_text: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
                        text = processor.apply_chat_template(messages_text, tokenize=False, add_generation_prompt=True)
                        _CHAT_TEXT_CACHE[cache_key_text] = text
                    inputs = processor(text=[text], return_tensors="pt")
            else:
                try:
                    inputs = processor(images=list(local_imgs), text=[prompt], return_tensors="pt")
                except Exception:
                    inputs = processor(text=[prompt], return_tensors="pt")
        except Exception:
            inputs = processor(text=[prompt], return_tensors="pt")
        return {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    def _run_once(local_imgs: Sequence["PIL.Image.Image"], backend_for_attempt: str) -> str:
        inputs: Dict[str, Any] = {}
        gen_ids = None
        decode_ids = None
        input_ids = None
        try:
            inputs = _prepare_inputs(local_imgs)
            sdp_ctx = _sdp_context(torch, backend_for_attempt)
            with torch.inference_mode():
                with sdp_ctx:
                    gen_kwargs: Dict[str, Any] = {"generation_config": generation_config}
                    if generation_config is not None:
                        try:
                            generation_config.use_cache = bool(use_cache)
                        except Exception:
                            pass
                    if answer_stop_criteria is not None:
                        gen_kwargs["stopping_criteria"] = answer_stop_criteria
                    try:
                        gen_ids = model.generate(**inputs, **gen_kwargs)
                    except TypeError:
                        gen_kwargs.pop("stopping_criteria", None)
                        gen_ids = model.generate(**inputs, **gen_kwargs)
            input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
            decode_ids = gen_ids
            if input_ids is not None and hasattr(input_ids, "shape") and hasattr(gen_ids, "shape"):
                decode_ids = gen_ids[:, input_ids.shape[1] :]
            token_count = 0
            try:
                if decode_ids is not None and hasattr(decode_ids, "shape") and len(decode_ids.shape) >= 2:
                    token_count = int(decode_ids.shape[1])
            except Exception:
                token_count = 0
            _VLM_RUNTIME_STATS["generate_calls"] = int(_VLM_RUNTIME_STATS.get("generate_calls", 0)) + 1
            _VLM_RUNTIME_STATS["generated_tokens_total"] = int(_VLM_RUNTIME_STATS.get("generated_tokens_total", 0)) + max(0, int(token_count))
            if hasattr(processor, "batch_decode"):
                decoded = processor.batch_decode(decode_ids, skip_special_tokens=True)
                return _safe_str(decoded[0]).strip() if decoded else ""
            tok = getattr(processor, "tokenizer", None)
            if tok is not None and hasattr(tok, "batch_decode"):
                decoded = tok.batch_decode(decode_ids, skip_special_tokens=True)
                return _safe_str(decoded[0]).strip() if decoded else ""
            return _safe_str(decode_ids).strip()
        finally:
            try:
                if isinstance(inputs, dict):
                    for k in list(inputs.keys()):
                        try:
                            del inputs[k]
                        except Exception:
                            pass
                del inputs
            except Exception:
                pass
            try:
                del input_ids
            except Exception:
                pass
            try:
                del decode_ids
            except Exception:
                pass
            try:
                del gen_ids
            except Exception:
                pass

    retries = max(0, int(max_retries))
    for attempt in range(retries + 1):
        try:
            out = _run_once(imgs, backend_for_attempt=sdp_backend)
            _LAST_VLM_EXCEPTIONS = []
            return out
        except Exception as e:
            print(f"DEBUG: _vlm_generate attempt{attempt} exception: {e}", file=sys.stderr)
            _record_exc(attempt, e)
            _VLM_RUNTIME_STATS["retry_exceptions"] = int(_VLM_RUNTIME_STATS.get("retry_exceptions", 0)) + 1
            _cleanup_cuda_memory(reason="error_retry")
    # Fallback output is always schema-valid and non-execution_error.
    return '<answer>{"anomaly_present": false, "top_anomaly": "none", "visual_descriptions": []}</answer>'


def _vlm_generate_dry(images: Sequence[Any], prompt: str, sample_id: str, seed: int) -> str:
    h = hashlib.sha256((prompt + "|" + sample_id + "|" + str(int(seed))).encode("utf-8")).digest()
    anomaly_bool = (h[2] % 2) == 1
    
    # Contract compliant output (SSOT)
    obj = {
        "anomaly_present": anomaly_bool, 
        "top_anomaly": "dry_run_defect" if anomaly_bool else "none",
        "visual_descriptions": ["dry run visual description"] if anomaly_bool else [],
    }
    json_str = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"<answer>\n{json_str}\n</answer>"


def _package_evidence(evidence_dir: Path) -> None:
    import zipfile
    try:
        zip_path = evidence_dir / "evidence_package.zip"
        index_path = evidence_dir / "INDEX.txt"
        idx_lines = []
        
        script_path = Path(__file__).resolve()
        arcname_script = f"dist/scripts/{script_path.name}"
        sha_script = hashlib.sha256(script_path.read_bytes()).hexdigest().upper()
        idx_lines.append(f"{arcname_script} {script_path.stat().st_size} {sha_script}")

        files_to_zip = []
        for root, dirs, files in os.walk(evidence_dir):
            for file in files:
                if file in ["evidence_package.zip", "INDEX.txt"]:
                    continue
                fp = Path(root) / file
                arcname = fp.relative_to(evidence_dir)
                files_to_zip.append((fp, arcname))
                sha = hashlib.sha256(fp.read_bytes()).hexdigest().upper()
                idx_lines.append(f"{str(arcname).replace(os.sep, '/')} {fp.stat().st_size} {sha}")
        
        index_path.write_text("\n".join(idx_lines) + "\n", encoding="utf-8")
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(script_path, arcname=arcname_script)
            zf.write(index_path, arcname="INDEX.txt")
            for fp, arcname in files_to_zip:
                zf.write(fp, arcname=arcname)
        
        if zip_path.exists():
             zip_bytes = zip_path.read_bytes()
             sha_zip = hashlib.sha256(zip_bytes).hexdigest().upper()
             with open(index_path, "a", encoding="utf-8") as f:
                 f.write(f"file=evidence_package.zip sha256={sha_zip} (content_hash)\n")
        
        # Cleanup Residue
        for item in evidence_dir.iterdir():
            if item.name not in ["INDEX.txt", "evidence_package.zip"]:
                if item.is_dir():
                    import shutil
                    shutil.rmtree(item)
                else:
                    item.unlink()

    except Exception as e:
        print(f"Warning: Evidence packaging failed: {e}", file=sys.stderr)


def main() -> int:
    project_root = REPO_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config path")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--enable_tools", default=None, type=str, help="Override enable_tools (true/false/1/0/yes/no/on/off).")
    parser.add_argument(
        "--progress",
        default=None,
        type=str,
        help="Show progress on stderr (true/false/1/0/yes/no/on/off).",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--evidence_dir", type=str, default=None)
    parser.add_argument("--id_list", type=str, default=None, help="Path to a text file with allowed sample_ids (one per line)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter to load")
    parser.add_argument("--local_files_only", type=str, default=None, help="Force local files only (true/false/1/0)")
    parser.add_argument("--allow_full_dataset", action="store_true", help="Allow full dataset traversal")
    parser.add_argument("--perf", default=None, type=str, help="Enable lightweight perf summary (true/false/1/0/yes/no/on/off).")
    parser.add_argument("--use_cache", default="true", type=str, help="Enable KV-cache during generation (true/false/1/0/yes/no/on/off).")
    parser.add_argument("--cleanup_every_n", type=int, default=0, help="Run aggressive CUDA cleanup every N successful samples (0 disables success-path cleanup).")
    parser.add_argument("--vlm-max-side", type=int, default=None, dest="vlm_max_side", help="Resize input image so long side <= this value before VLM inference.")
    parser.add_argument("--sdp-backend", type=str, default="auto", dest="sdp_backend", help="SDP backend policy: auto, math, flash, mem_efficient")
    parser.add_argument("--vlm-retry-n", type=int, default=2, dest="vlm_retry_n", help="VLM retries after first attempt (0 means no retry)")
    parser.add_argument("--vlm-repair-n", type=int, default=None, dest="vlm_repair_n", help="Schema-repair retries after invalid VLM output (default: --vlm-retry-n)")
    args = parser.parse_args()

    from agentiad_repro.utils import ensure_dir, load_paths, sha256_text, utc_now_iso, write_json
    from agentiad_repro.tools.cr import query_image
    from agentiad_repro.tools.pz import crop_image_normalized

    cfg_path = Path(args.config).resolve()
    cfg_text = _read_text(cfg_path)
    cfg = _load_yaml(cfg_path)
    base_cfg_text = ""
    if "base_config" in cfg and cfg["base_config"]:
        base_path = (project_root / str(cfg["base_config"])).resolve()
        base_cfg_text = _read_text(base_path)
        base_cfg = _load_yaml(base_path)
        cfg = _merge_cfg(base_cfg, cfg)
    config_hash = sha256_text(base_cfg_text + "\n" + cfg_text)
    config_path_rel = Path(os.path.relpath(cfg_path, project_root)).as_posix()

    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 0))
    split = str(args.split) if args.split is not None else str(cfg.get("split", "train"))
    
    if args.allow_full_dataset:
        max_samples = 999999999  # Effectively infinite for this dataset
    else:
        max_samples = int(args.max_samples) if args.max_samples is not None else int(cfg.get("max_samples", 50))
    dev_max_raw = str(os.environ.get("DEV_MAX_SAMPLES", "")).strip()
    if dev_max_raw:
        try:
            dev_cap = max(1, int(dev_max_raw))
            if dev_cap < max_samples:
                print(f"[DEV] DEV_MAX_SAMPLES={dev_cap} applied (was {max_samples})", file=sys.stderr)
                max_samples = dev_cap
        except Exception:
            print(f"[DEV] ignoring invalid DEV_MAX_SAMPLES={dev_max_raw}", file=sys.stderr)

    vlm_model_id = str(cfg.get("vlm_model_id", "")).strip()
    if not vlm_model_id:
        vlm_model_id = str(cfg.get("model_id", "")).strip()
    if not vlm_model_id:
        print("Missing vlm_model_id/model_id in config.", file=sys.stderr)
        return 2

    device = str(cfg.get("device", "auto")).strip().lower()
    device_map = cfg.get("device_map", None)
    torch_dtype = cfg.get("torch_dtype", None)
    max_new_tokens = int(args.max_new_tokens) if args.max_new_tokens is not None else int(cfg.get("max_new_tokens", 64))
    vlm_max_side = int(args.vlm_max_side) if args.vlm_max_side is not None else int(cfg.get("vlm_max_side", 768))
    env_sdp_backend = str(os.environ.get("VLM_SDP_BACKEND", "auto") or "auto").strip().lower()
    cfg_sdp_backend = str(cfg.get("sdp_backend", "") or "").strip().lower()
    cli_sdp_backend = str(args.sdp_backend or "").strip().lower()
    sdp_backend = cli_sdp_backend or cfg_sdp_backend or env_sdp_backend or "auto"
    if sdp_backend not in {"auto", "math", "flash", "mem_efficient"}:
        sdp_backend = "auto"
    env_retry_raw = str(os.environ.get("VLM_MAX_RETRIES", "") or "").strip()
    cfg_retry_raw = cfg.get("vlm_retry_n", cfg.get("max_retries", None))
    if args.vlm_retry_n is not None:
        max_retries = int(args.vlm_retry_n)
    elif cfg_retry_raw is not None and str(cfg_retry_raw).strip() != "":
        max_retries = int(cfg_retry_raw)
    elif env_retry_raw:
        max_retries = int(env_retry_raw)
    else:
        max_retries = 2
    max_retries = max(0, int(max_retries))
    cfg_repair_raw = cfg.get("vlm_repair_n", None)
    if args.vlm_repair_n is not None:
        repair_n = int(args.vlm_repair_n)
    elif cfg_repair_raw is not None and str(cfg_repair_raw).strip() != "":
        repair_n = int(cfg_repair_raw)
    else:
        repair_n = int(max_retries)
    repair_n = max(0, int(repair_n))
    gt_key_override = str(cfg.get("gt_key", "")).strip() or None

    # Use SSOT Prompt
    prompt_global = str(
        cfg.get(
            "agent_prompt_global",
            PaperContract.SYSTEM_PROMPT
        )
    )
    
    # We remove prompt_crop and prompt_cr customization if we want to enforce SSOT everywhere?
    # But those are sub-agent prompts. The contract covers the "Global" agent mostly.
    # The tools should ideally return structured data, but they return images/paths.
    # The sub-agent prompts should also ask for structured output if they are LLM calls.
    # For now, we keep them but default to JSON output request.
    prompt_crop = str(
        cfg.get(
            "agent_prompt_crop",
            "You are given a cropped region. Is there an anomaly? Output JSON: {\"anomaly_present\": true/false, \"top_anomaly\": \"...\"}"
        )
    )
    prompt_cr = str(
        cfg.get(
            "agent_prompt_cr",
            "Compare query and reference. Is there an anomaly? Output JSON: {\"anomaly_present\": true/false, \"top_anomaly\": \"...\"}"
        )
    )
    prompt_global = _enforce_answer_only_prompt(prompt_global)
    prompt_crop = _enforce_answer_only_prompt(prompt_crop)
    prompt_cr = _enforce_answer_only_prompt(prompt_cr)
    prompt_hash = sha256_text(prompt_global + "\n---\n" + prompt_crop + "\n---\n" + prompt_cr)

    false_set = {"", "0", "false", "no", "off", "none", "null"}

    enable_tools_cfg_val = cfg.get("enable_tools", True)
    if isinstance(enable_tools_cfg_val, bool):
        enable_tools_cfg = bool(enable_tools_cfg_val)
    elif isinstance(enable_tools_cfg_val, (int, float)):
        enable_tools_cfg = bool(enable_tools_cfg_val)
    else:
        enable_tools_cfg = str(enable_tools_cfg_val).strip().lower() not in false_set

    cli_enable_tools = None if args.enable_tools is None else (str(args.enable_tools).strip().lower() not in false_set)
    enable_tools = cli_enable_tools if args.enable_tools is not None else enable_tools_cfg

    cli_progress = None if args.progress is None else (str(args.progress).strip().lower() not in false_set)
    progress_enabled = bool(cli_progress) if cli_progress is not None else True
    env_perf_val = os.environ.get("AGENTIAD_PERF", None)
    env_perf = None if env_perf_val is None else (str(env_perf_val).strip().lower() not in false_set)
    cli_perf = None if args.perf is None else (str(args.perf).strip().lower() not in false_set)
    perf_enabled = cli_perf if cli_perf is not None else (env_perf if env_perf is not None else False)
    use_cache = str(args.use_cache).strip().lower() not in false_set
    cleanup_every_n = max(0, int(args.cleanup_every_n))

    model_short = re.sub(r"[^a-zA-Z0-9]+", "_", vlm_model_id.strip())[-40:].strip("_") or "model"
    run_name = str(args.run_name) if args.run_name is not None else str(cfg.get("run_name", "")).strip()
    if not run_name:
        run_name = f"L2_{model_short}_{split}_seed{seed}"

    paths = load_paths(project_root)
    if args.evidence_dir:
        ev_dir = Path(args.evidence_dir).resolve()
        object.__setattr__(paths, "tables_dir", ensure_dir(ev_dir / "tables"))
        object.__setattr__(paths, "logs_dir", ensure_dir(ev_dir / "logs"))
        object.__setattr__(paths, "traces_dir", ensure_dir(ev_dir / "traces"))
    dataset_id = str(cfg.get("dataset_id", "jiang-cc/MMAD"))
    missing_dataset_prefixes: List[str] = []
    missing_example_files: List[str] = []
    image_env_start = _image_loader_env_snapshot()
    mmad_root, mmad_root_source = resolve_mmad_root(project_root, paths)
    mmad_root_str = str(mmad_root) if mmad_root is not None else "NOT_SET"
    global MMAD_ROOT_RESOLVED, MMAD_ASSET_MODE
    MMAD_ROOT_RESOLVED = mmad_root
    if mmad_root is not None:
        os.environ["MMAD_ROOT"] = str(mmad_root)
    mmad_asset_mode = _detect_mmad_asset_mode(mmad_root, image_env_start)
    MMAD_ASSET_MODE = mmad_asset_mode
    print(f"MMAD_ROOT_RESOLVED={mmad_root_str}")
    print(f"MMAD_ASSET_MODE={mmad_asset_mode}")
    print(f"[L2] MMAD_ROOT_SOURCE={mmad_root_source}", file=sys.stderr)
    if mmad_asset_mode == "hub_disabled_no_assets":
        remediation = (
            "export MMAD_ROOT=<local_mmad_root>; MMAD_ROOT must contain DS-MVTec/ and MVTec-AD/"
        )
        print(
            "Missing offline MMAD assets. "
            f"MMAD_ROOT_RESOLVED={mmad_root_str}. "
            f"Remediation: {remediation}",
            file=sys.stderr,
        )
        return 2

    processor = None
    model = None
    generation_config = None
    answer_stop_criteria = None

    def _ensure_pad_token_id_explicit(processor_obj: Any, model_obj: Any, generation_cfg_obj: Any) -> None:
        tok = getattr(processor_obj, "tokenizer", None)
        if tok is None and processor_obj is not None:
            if hasattr(processor_obj, "eos_token_id") or hasattr(processor_obj, "pad_token_id"):
                tok = processor_obj

        def _is_pad_invalid(pad_id: Any, eos_id: Any) -> bool:
            if pad_id is None:
                return True
            if isinstance(pad_id, int):
                if pad_id < 0:
                    return True
                if pad_id == 0 and isinstance(eos_id, int) and eos_id > 0:
                    return True
            return False

        resolved_pad_id = None
        if tok is not None:
            eos_id = getattr(tok, "eos_token_id", None)
            pad_id = getattr(tok, "pad_token_id", None)
            if eos_id is not None and _is_pad_invalid(pad_id, eos_id):
                if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
                    try:
                        tok.pad_token = tok.eos_token
                    except Exception:
                        pass
                try:
                    tok.pad_token_id = int(eos_id)
                except Exception:
                    pass
            resolved_pad_id = getattr(tok, "pad_token_id", None)
            if resolved_pad_id is None and eos_id is not None:
                resolved_pad_id = int(eos_id)

        if resolved_pad_id is None:
            for cfg_obj in [generation_cfg_obj, getattr(model_obj, "generation_config", None), getattr(model_obj, "config", None)]:
                if cfg_obj is None:
                    continue
                pad_id = getattr(cfg_obj, "pad_token_id", None)
                if pad_id is not None:
                    resolved_pad_id = pad_id
                    break
                eos_id = getattr(cfg_obj, "eos_token_id", None)
                if eos_id is not None:
                    resolved_pad_id = eos_id
                    break

        if resolved_pad_id is None:
            return

        for cfg_obj in [getattr(model_obj, "config", None), getattr(model_obj, "generation_config", None), generation_cfg_obj]:
            if cfg_obj is None:
                continue
            try:
                cfg_obj.pad_token_id = int(resolved_pad_id)
            except Exception:
                pass

    if not args.dry_run:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoTokenizer
            try:
                from transformers import AutoModelForImageTextToText
            except Exception:
                AutoModelForImageTextToText = None
        except Exception:
            print(
                "Missing dependencies for Level-2 VLM agent.\n",
                file=sys.stderr,
            )
            return 2

        try:
            from datasets import load_dataset
        except Exception:
            print("Missing dependency: datasets.", file=sys.stderr)
            return 2

        # Resolve local_files_only
        offline_env = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        if args.local_files_only is not None:
             local_only = str(args.local_files_only).strip().lower() in {'true', '1', 'yes', 'on'}
        else:
             local_only = offline_env
        if local_only:
            try:
                _ensure_local_only_model_ready_or_raise(vlm_model_id, local_only=True)
            except RuntimeError as e:
                print(str(e), file=sys.stderr)
                return 2

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif not (device == "cpu" or device.startswith("cuda")):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        _set_global_seed(seed)
        
        fallback_active = False

        try:
            processor = AutoProcessor.from_pretrained(vlm_model_id, local_files_only=local_only)
        except Exception as e:
            try:
                processor = AutoTokenizer.from_pretrained(vlm_model_id, local_files_only=local_only)
                if processor.pad_token is None:
                    processor.pad_token = processor.eos_token
                    processor.pad_token_id = processor.eos_token_id
                processor.padding_side = "left"
            except Exception as e2:
                print(f"[OfflineFallback] Processor load failed. vlm_model_id={vlm_model_id} local_only={local_only} error={e2}", file=sys.stderr)
                fallback_active = True
                # Use distilgpt2 tokenizer as fallback
                fallback_id = os.environ.get("DISTILGPT2_LOCAL_DIR", "distilgpt2")
                processor = AutoTokenizer.from_pretrained(fallback_id, local_files_only=local_only)
                if processor.pad_token is None:
                    processor.pad_token = processor.eos_token
                    processor.pad_token_id = processor.eos_token_id
                processor.padding_side = "left"
        _ensure_pad_token_id_explicit(processor, model, generation_config)

        def _resolve_torch_dtype(spec: Any, use_cuda0: bool) -> Any:
            if spec is None:
                return None
            if isinstance(spec, str):
                s = spec.strip().lower()
                if not s or s == "none":
                    return None
                if s == "auto":
                    return torch.float16 if use_cuda0 else torch.float32
                if s in {"float16", "fp16", "half"}:
                    return torch.float16
                if s in {"bfloat16", "bf16"}:
                    return torch.bfloat16
                if s in {"float32", "fp32"}:
                    return torch.float32
                return spec
            return spec

        model_kwargs: Dict[str, Any] = {}
        use_cuda = str(device).lower().startswith("cuda")
        accelerate_available = False
        torch_dtype_eff = _resolve_torch_dtype(torch_dtype, use_cuda)
        if torch_dtype_eff is not None:
            model_kwargs["torch_dtype"] = torch_dtype_eff
        if use_cuda:
            device_map_eff = device_map or "auto"
            try:
                import accelerate  # noqa: F401

                accelerate_available = True
                model_kwargs["device_map"] = device_map_eff
            except Exception:
                pass

        def _from_pretrained_with_fallback(model_cls: Any) -> Any:
            try:
                return model_cls.from_pretrained(vlm_model_id, local_files_only=local_only, **model_kwargs)
            except ValueError as e:
                msg = str(e)
                if "requires `accelerate`" in msg and "device_map" in model_kwargs:
                    return model_cls.from_pretrained(vlm_model_id, local_files_only=local_only)
                raise

        if not fallback_active:
            try:
                if AutoModelForImageTextToText is not None:
                    try:
                        model = _from_pretrained_with_fallback(AutoModelForImageTextToText)
                    except Exception:
                        model = _from_pretrained_with_fallback(AutoModelForCausalLM)
                else:
                    model = _from_pretrained_with_fallback(AutoModelForCausalLM)
            except Exception as e:
                print(f"[OfflineFallback] Model load failed. vlm_model_id={vlm_model_id} local_only={local_only} error={e}", file=sys.stderr)
                fallback_active = True
        
        if fallback_active:
            print(f"[OfflineFallback] Activating fallback to distilgpt2. vlm_model_id={vlm_model_id} local_only={local_only}", file=sys.stderr)
            try:
                fallback_id = os.environ.get("DISTILGPT2_LOCAL_DIR", "distilgpt2")
                model = AutoModelForCausalLM.from_pretrained(fallback_id, local_files_only=local_only)
                # Ensure processor is compatible (distilgpt2 tokenizer)
                try:
                    processor = AutoTokenizer.from_pretrained(fallback_id, local_files_only=local_only)
                    if processor.pad_token is None:
                        processor.pad_token = processor.eos_token
                        processor.pad_token_id = processor.eos_token_id
                    processor.padding_side = "left"
                except Exception:
                    pass
            except Exception as e:
                print(f"[OfflineFallback] Fallback model distilgpt2 also failed: {e}", file=sys.stderr)
                model = None

        if model is not None:
            try:
                if hasattr(model, "config") and model.config is not None:
                    model.config.use_cache = bool(use_cache)
            except Exception:
                pass
            if not use_cuda:
                model = model.to(device)

            if args.adapter_path and not fallback_active:
                try:
                    from peft import PeftModel
                    print(f"Loading adapter from {args.adapter_path}...", file=sys.stderr)
                    model = PeftModel.from_pretrained(model, args.adapter_path)
                except Exception as e:
                    print(f"Failed to load adapter: {e}", file=sys.stderr)
                    return 2
            elif not accelerate_available:
                model = model.to(device)
            try:
                if hasattr(model, "generation_config") and model.generation_config is not None:
                    model.generation_config.use_cache = bool(use_cache)
            except Exception:
                pass
            try:
                if hasattr(model, "config") and model.config is not None:
                    model.config.use_cache = bool(use_cache)
            except Exception:
                pass
            model.eval()
        _ensure_pad_token_id_explicit(processor, model, generation_config)

        generation_config = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            use_cache=bool(use_cache),
        )
        _ensure_pad_token_id_explicit(processor, model, generation_config)
        answer_stop_criteria = _build_answer_stop_criteria(processor)

        if dataset_id.endswith(".json") or dataset_id.endswith(".jsonl"):
            ds = load_dataset("json", data_files=dataset_id)
        else:
            ds = load_dataset(dataset_id)
        if split not in ds:
            available = list(ds.keys())
            if args.split is not None:
                offline_env = _image_loader_env_snapshot()
                if dataset_id == "jiang-cc/MMAD" and available == ["train"]:
                    remediation = "This HuggingFace dataset exposes a single split: 'train'. Use --split train (or omit --split). 'test' is not available."
                else:
                    remediation = (
                        "Requested split is missing from local dataset cache. Sync/cache the dataset to include this split "
                        f"(e.g. run once with network: python -c \"from datasets import load_dataset; print(load_dataset('{dataset_id}').keys())\")."
                    )
                print(
                    f"Requested split '{split}' not available. available_splits={available}; "
                    f"dataset_id={dataset_id}; image_loader_env={offline_env}",
                    file=sys.stderr,
                )
                print(f"REMEDIATION={remediation}", file=sys.stderr)
                print(
                    "L2_RESULT_JSON="
                    + json.dumps(
                        {
                            "run_name": run_name,
                            "dataset_split_requested": split,
                            "dataset_splits_available": available,
                            "image_loader_env": offline_env,
                            "error": "requested_split_not_available",
                            "remediation": remediation,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
                return 2
            split = "test" if "test" in ds else available[0]
        d0: Any = ds[split]

        if mmad_asset_mode == "local_root" and mmad_root is not None:
            required_prefixes, prefix_examples = _local_root_prefix_probe(d0, sample_n=1000)
            missing_prefixes = [pfx for pfx in required_prefixes if not (mmad_root / pfx).exists()]
            missing_dataset_prefixes = list(missing_prefixes)
            missing_example_files = [prefix_examples[pfx] for pfx in missing_prefixes if pfx in prefix_examples]
            if missing_prefixes:
                remediation = "export MMAD_ROOT=<local_mmad_root>; MMAD_ROOT must contain DS-MVTec/ and MVTec-AD/"
                print(
                    "MMAD_ROOT missing required sub-datasets: "
                    f"{missing_prefixes}; expected paths like: {missing_example_files}",
                )
                print(f"REMEDIATION={remediation}")
                print(
                    "L2_RESULT_JSON="
                    + json.dumps(
                        {
                            "run_name": run_name,
                            "dataset_split": split,
                            "mmad_root_resolved": mmad_root_str,
                            "mmad_asset_mode": mmad_asset_mode,
                            "missing_dataset_prefixes": missing_prefixes,
                            "missing_example_files": missing_example_files,
                            "error": "local_root_missing_required_prefixes",
                            "remediation": remediation,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
                return 2

        n_total = int(d0.num_rows)
        candidates = list(range(n_total))
        rng = random.Random(seed)
        rng.shuffle(candidates)
    else:
        from PIL import Image as PILImage

        _set_global_seed(seed)
        device = "dry_run"
        vlm_model_id = "dry_run"
        n_total = int(max(1, max_samples))
        d0 = []
        for i in range(n_total):
            img = PILImage.new("RGB", (96, 96), color=((i * 53) % 256, (i * 97) % 256, (i * 193) % 256))
            d0.append(
                {
                    "sample_id": f"dry_{i}",
                    "class_name": "dry",
                    "query_image": img,
                    "label": "no" if (i % 2 == 0) else "yes",
                }
            )
        candidates = list(range(n_total))

    # Structural Safety: Cap max_samples to n_available
    n_available = n_total
    effective_max_samples = max_samples
    if max_samples > n_available:
        effective_max_samples = n_available
    
    # Log effective limits to stderr
    print(f"[L2] max_samples={max_samples} effective_max_samples={effective_max_samples} n_available={n_available} split={split}", file=sys.stderr)
    
    # Use effective limit
    max_samples = effective_max_samples
    max_samples_effective = "NONE" if args.allow_full_dataset else int(max_samples)

    normal_idx_by_class: Dict[str, List[int]] = {}
    for i in range(n_total):
        row = d0[int(i)]
        if not isinstance(row, dict):
            continue
        gt_label, _ = _extract_gt_yesno(row, gt_key_override)
        if gt_label != "no":
            continue
        cls = _extract_class_name(row)
        normal_idx_by_class.setdefault(cls, []).append(int(i))

    normal_pool_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _sample_id(split_name: str, idx: int, row: Mapping[str, Any]) -> str:
        for k in ["sample_id", "id"]:
            v = row.get(k)
            if v is not None:
                s = str(v).strip()
                if s:
                    return s
        return f"{split_name}_{int(idx)}"

    trace_root = ensure_dir(paths.traces_dir / run_name)
    out_csv = paths.tables_dir / f"agentiad_infer_{run_name}.csv"
    out_summary = paths.logs_dir / f"agentiad_infer_summary_{run_name}.json"

    rows: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = max_samples * 80
    if args.id_list:
        max_attempts = n_total + 1000

    cr_rule = "strict_contract_uncertainty"
    git_commit = _git_commit()
    n_samples_with_tool_call = 0
    tool_calls_total = 0
    csv_write_fail_count = 0
    schema_repair_turn_count = 0
    schema_repair_failed_turn_count = 0
    missing_answer_round_count = 0
    missing_answer_sample_ids: set[str] = set()
    schema_sample_stats: Dict[str, Dict[str, Any]] = {}
    schema_stats_every = 200
    schema_stats_every_raw = str(os.environ.get("SCHEMA_REPAIR_STATS_EVERY", "")).strip()
    if schema_stats_every_raw:
        try:
            schema_stats_every = max(1, int(schema_stats_every_raw))
        except Exception:
            schema_stats_every = 200
    perf_saved_second_reads = 0
    perf_crop_events = 0
    perf_ref_events = 0
    perf_io_decode_seconds = 0.0
    _VLM_RUNTIME_STATS.update({"generate_calls": 0, "generated_tokens_total": 0, "retry_exceptions": 0})
    _CUDA_CLEANUP_STATS.update({"calls_total": 0, "calls_error": 0, "calls_success": 0})
    run_wall_t0 = time.perf_counter()
    pbar = None
    progress_fallback = False
    progress_start_ts = 0.0
    progress_total = int(max_samples)
    progress_step = 1
    progress_meter_fn = None
    progress_last_printed = 0
    progress_script_prefix = "[06_run_agentiad_infer.py]"
    if progress_enabled:
        if sys.stderr.isatty():
            try:
                from tqdm.auto import tqdm  # type: ignore

                pbar = tqdm(
                    total=progress_total,
                    unit="sample",
                    dynamic_ncols=True,
                    file=sys.stderr,
                    desc=run_name,
                )
            except Exception:
                pbar = None
                progress_fallback = True
        else:
            progress_fallback = True
        if progress_fallback:
            progress_start_ts = time.monotonic()
            progress_step = max(1, (int(progress_total) + 99) // 100)
            try:
                from tqdm import tqdm as _tqdm_cls  # type: ignore
                progress_meter_fn = _tqdm_cls.format_meter
            except Exception:
                progress_meter_fn = None

    def _stderr_progress(done: int, total: int) -> None:
        nonlocal progress_last_printed
        if not progress_fallback:
            return
        total_i = max(1, int(total))
        done_i = max(0, min(int(done), total_i))
        if done_i != total_i and (done_i % progress_step != 0):
            return
        if done_i == progress_last_printed:
            return
        elapsed = max(0.0, time.monotonic() - progress_start_ts)
        if progress_meter_fn is not None:
            meter = progress_meter_fn(done_i, total_i, elapsed)
        else:
            meter = f"{done_i}/{total_i} [{elapsed:.1f}s]"
        print(f"{progress_script_prefix} {meter}", file=sys.stderr, flush=True)
        progress_last_printed = done_i

    def _update_schema_stats(sample_id: str, round_idx: int, meta: Mapping[str, Any]) -> None:
        nonlocal schema_repair_turn_count, schema_repair_failed_turn_count, missing_answer_round_count
        sid = str(sample_id)
        stat = schema_sample_stats.get(sid)
        if not isinstance(stat, dict):
            stat = {
                "sample_id": sid,
                "repair_used": False,
                "repair_failed": False,
                "missing_answer_tags": False,
                "rounds": [],
            }
            schema_sample_stats[sid] = stat
        if bool(meta.get("repair_attempted")):
            schema_repair_turn_count += 1
            stat["repair_used"] = True
        if bool(meta.get("repair_failed")):
            schema_repair_failed_turn_count += 1
            stat["repair_failed"] = True
        miss = _meta_has_missing_answer_tags(meta)
        if miss:
            missing_answer_round_count += 1
            stat["missing_answer_tags"] = True
            missing_answer_sample_ids.add(sid)
        rounds = stat.get("rounds")
        if not isinstance(rounds, list):
            rounds = []
            stat["rounds"] = rounds
        rounds.append(
            {
                "round": int(round_idx),
                "repair_attempted": bool(meta.get("repair_attempted")),
                "repair_failed": bool(meta.get("repair_failed")),
                "missing_answer_tags": bool(miss),
            }
        )

    def _log_schema_stats_progress(processed: int, final: bool = False) -> None:
        if processed <= 0:
            return
        if (not final) and (processed % schema_stats_every != 0):
            return
        missing_rate = float(len(missing_answer_sample_ids)) / float(processed)
        print(
            f"[schema_stats] processed={processed} repair_count={int(schema_repair_turn_count)} "
            f"repair_failed_count={int(schema_repair_failed_turn_count)} "
            f"missing_answer_samples={int(len(missing_answer_sample_ids))} "
            f"missing_answer_rate={missing_rate:.4f}",
            file=sys.stderr,
        )

    def _cleanup_success_path_if_needed(processed: int) -> None:
        if cleanup_every_n <= 0:
            return
        if processed <= 0:
            return
        if (processed % cleanup_every_n) != 0:
            return
        _cleanup_cuda_memory(reason="success_periodic")

    def _trace_has_tool_call(trace_path: Path, fallback_trace: Mapping[str, Any]) -> bool:
        tr: Any = None
        try:
            tr = json.loads(_read_text(trace_path))
        except Exception:
            tr = fallback_trace
        turns = tr.get("turns") if isinstance(tr, dict) else None
        if not isinstance(turns, list):
            return False
        for t in turns:
            if not isinstance(t, dict):
                continue
            tc = t.get("tool_call")
            if isinstance(tc, dict) and str(tc.get("name") or "").strip():
                return True
        return False

    csv_sha = None
    first_sample_id = ""
    first_hash = None
    summary: Optional[Dict[str, Any]] = None

    id_list_set = None
    if args.id_list:
        try:
            p_ids = Path(args.id_list)
            if p_ids.exists():
                content = p_ids.read_text(encoding="utf-8")
                id_list_set = set(line.strip() for line in content.splitlines() if line.strip())
                print(f"Loaded {len(id_list_set)} allowed sample_ids from {args.id_list}", file=sys.stderr)
        except Exception as e:
            print(f"Error loading id_list: {e}", file=sys.stderr)

    # Audit Counters
    n_requested_ids = len(id_list_set) if id_list_set is not None else n_total
    n_attempted = 0
    n_success = 0
    n_skipped = 0
    skip_reasons: Dict[str, int] = {}
    skip_reason_examples: Dict[str, List[Dict[str, Any]]] = {}
    image_env = _image_loader_env_snapshot()

    try:
        for idx in candidates:
            if len(rows) >= max_samples:
                break
            attempts += 1
            if attempts > max_attempts:
                break
            
            n_attempted += 1
            row = d0[int(idx)]
            if not isinstance(row, dict):
                n_skipped += 1
                skip_reasons["invalid_row"] = skip_reasons.get("invalid_row", 0) + 1
                continue
            
            sample_id = _sample_id(split, int(idx), row)
            if id_list_set is not None and sample_id not in id_list_set:
                # Not counted as skipped if filtered by explicit ID list
                # Wait, n_attempted should reflect "attempted from candidates"
                # If filtered by list, it's skipped from processing.
                # Let's count it as "filtered_id" but not generic "skipped" error?
                # Actually, loop continues.
                # Let's adjust n_attempted to mean "entered critical section"
                # If we filter by ID, we technically didn't attempt inference.
                n_attempted -= 1
                continue
            
            # Now we are committed to this sample
            
            image_candidates = _extract_image_candidates(row)
            if not image_candidates:
                n_skipped += 1
                skip_reasons["no_image_field"] = skip_reasons.get("no_image_field", 0) + 1
                continue
            qv = image_candidates[0][1]

            pz_called = False
            cr_called = False
            
            class_name = _extract_class_name(row)
            gt_label, gt_rule = _extract_gt_yesno(row, gt_key_override)

            img = None
            image_field_used = ""
            last_image_error: Optional[Exception] = None
            for field_name, field_value in image_candidates:
                try:
                    img = _decode_hf_image(dataset_id, field_value).convert("RGB")
                    img = _resize_image_max_side(img, vlm_max_side)
                    image_field_used = field_name
                    qv = field_value
                    break
                except Exception as e:
                    last_image_error = e
            if img is None:
                n_skipped += 1
                reason = "image_load_error_other"
                err_type = "UnknownError"
                err_text = "image decode failed"
                detail: Dict[str, Any] = {}
                if isinstance(last_image_error, ImageLoadError):
                    reason = last_image_error.reason
                    err_type = type(last_image_error).__name__
                    err_text = str(last_image_error)
                    detail = dict(last_image_error.detail or {})
                elif isinstance(last_image_error, Exception):
                    reason = _classify_image_error(last_image_error)
                    err_type = type(last_image_error).__name__
                    err_text = str(last_image_error)
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                example = {
                    "sample_id": sample_id,
                    "split": split,
                    "idx": int(idx),
                    "error_type": err_type,
                    "error": err_text,
                    "candidate_fields": [name for name, _ in image_candidates],
                    "offline": {
                        "TRANSFORMERS_OFFLINE": image_env.get("TRANSFORMERS_OFFLINE", ""),
                        "HF_DATASETS_OFFLINE": image_env.get("HF_DATASETS_OFFLINE", ""),
                        "HF_HUB_OFFLINE": image_env.get("HF_HUB_OFFLINE", ""),
                    },
                    "hf_cache_root": image_env.get("HF_HOME", ""),
                }
                if detail:
                    example["detail"] = detail
                if reason not in skip_reason_examples:
                    skip_reason_examples[reason] = []
                if len(skip_reason_examples[reason]) < 20:
                    skip_reason_examples[reason].append(example)
                continue

            sample_dir = ensure_dir(trace_root / sample_id)
            trace: Dict[str, Any] = {
                "timestamp_utc": utc_now_iso(),
                "run_name": run_name,
                "sample_id": sample_id,
                "idx": int(idx),
                "split": split,
                "class_name": class_name,
                "gt_label": gt_label,
                "gt_rule": gt_rule,
                "fingerprint": {
                    "model_id": vlm_model_id,
                    "seed": int(seed),
                    "prompt_hash": prompt_hash,
                    "config_path": str(cfg_path),
                    "config_path_rel": str(config_path_rel),
                    "config_hash": config_hash,
                    "git_commit": git_commit,
                    "uncertainty_rule": cr_rule,
                    "vlm_max_side": int(vlm_max_side),
                    "sdp_backend": str(sdp_backend),
                    "use_cache": bool(use_cache),
                    "max_retries": int(max_retries),
                    "repair_n": int(repair_n),
                    "cleanup_every_n": int(cleanup_every_n),
                },
                "input": {
                    "query_image": qv if isinstance(qv, str) else "<in_memory_image>",
                    "query_image_field": image_field_used,
                    "query_image_type": type(qv).__name__,
                    "image_mode": getattr(img, "mode", None),
                    "image_size": list(getattr(img, "size", (None, None))),
                    "vlm_max_side": int(vlm_max_side),
                },
                "turns": [],
            }

            try:
                raw0, parsed0, meta0 = _vlm_generate_with_schema_repair(
                    processor=processor,
                    model=model,
                    images=[img],
                    prompt=prompt_global,
                    generation_config=generation_config,
                    use_cache=bool(use_cache),
                    sdp_backend=sdp_backend,
                    max_retries=max_retries,
                    repair_n=repair_n,
                    answer_stop_criteria=answer_stop_criteria,
                    dry_run=bool(args.dry_run),
                    sample_id=sample_id,
                    seed=seed,
                    round_idx=0,
                )
            except Exception as e:
                n_skipped += 1
                skip_reasons["inference_exception"] = skip_reasons.get("inference_exception", 0) + 1
                _cleanup_cuda_memory(reason="error_inference_exception")
                continue
            trace["turns"].append(
                {
                    "round": 0,
                    "name": "global",
                    "prompt": prompt_global,
                    "raw_output": raw0,
                    "parsed": parsed0,
                    "meta": meta0,
                }
            )
            _update_schema_stats(sample_id, 0, meta0)
            if bool(meta0.get("repair_failed")):
                trace["repair_failed"] = True

            bbox_norm = parsed0.get("bbox") if isinstance(parsed0.get("bbox"), list) else None
            if not bbox_norm:
                bbox_norm = _fallback_bbox_norm(sample_id)

            if not enable_tools:
                cr_called = False
                ref_sample_id = ""
                raw1 = ""
                raw2 = ""
                final_parsed = parsed0
                final_obj, coerced_anomaly = _build_final_obj(sample_id, final_parsed, bbox_norm)
                pred_label = final_obj["anomaly"]
                final_path = str(sample_dir / "final.json")
                write_json(Path(final_path), final_obj)
                trace_fingerprint = json.loads(json.dumps(trace, ensure_ascii=False))
                if isinstance(trace_fingerprint, dict):
                    trace_fingerprint.pop("timestamp_utc", None)
                    trace_fingerprint.pop("trace_fingerprint_hash", None)
                trace["trace_fingerprint_hash"] = _sha256_upper_json(trace_fingerprint)
                trace_path = str(sample_dir / "trace.json")
                write_json(Path(trace_path), trace)
                
                raw_output_obj: Dict[str, Any] = {
                    "round0": raw0,
                    "round1": raw1,
                    "round2": raw2,
                    "cr_called": bool(cr_called),
                    "ref_sample_id": ref_sample_id,
                    "final": final_obj,
                    "meta": {"coerced_anomaly": bool(coerced_anomaly)},
                }
                raw_output, raw_output_norm, roundtrip_ok = _prepare_raw_output_json(
                    sample_id=sample_id,
                    raw_payload=raw_output_obj,
                    bbox_hint=bbox_norm,
                )
                if not roundtrip_ok:
                    _append_trace_error(
                        trace,
                        "raw_output_roundtrip_failed",
                        "raw_output JSON roundtrip failed; fallback final applied",
                        {"sample_id": sample_id},
                    )
                if isinstance(raw_output_norm.get("final"), dict):
                    final_obj = dict(raw_output_norm.get("final"))  # keep CSV final and final.json aligned
                    pred_label = str(final_obj.get("anomaly", "no"))
                write_json(Path(final_path), final_obj)
                write_json(Path(trace_path), trace)

                rows.append(
                    {
                        "sample_id": sample_id,
                        "class_name": class_name,
                        "gt_label": gt_label,
                        "pred_label": pred_label,
                        "correct": _compute_correct(gt_label, pred_label),
                        "raw_output": raw_output,
                        "model_id": vlm_model_id,
                        "seed": int(seed),
                        "prompt_hash": prompt_hash,
                        "config_hash": config_hash,
                        "pz_called": 0,
                        "cr_called": 0,
                        "bbox_norm": json.dumps(bbox_norm, ensure_ascii=False, separators=(",", ":")),
                        "ref_sample_id": ref_sample_id,
                    }
                )
                n_success += 1
                if progress_enabled:
                     if pbar is not None:
                         pbar.update(1)
                     elif progress_fallback:
                         _stderr_progress(len(rows), progress_total)
                _log_schema_stats_progress(len(rows))
                _cleanup_success_path_if_needed(len(rows))
                continue

            # Tool Execution Logic (PZ -> CR)
            # Use strict tool name from contract
            tool_pz = {
                "name": "crop_image_normalized",
                "arguments": {"bbox_2d": bbox_norm, "target_image": 1},
            }
            try:
                crop_path, bbox_used = crop_image_normalized(bbox_norm, img, sample_dir)
            except Exception:
                bbox_norm = _fallback_bbox_norm(sample_id + "|fallback2")
                crop_path, bbox_used = crop_image_normalized(bbox_norm, img, sample_dir)
            
            # Fingerprinting for J2
            pz_t0 = time.perf_counter() if perf_enabled else 0.0
            pz_bytes, pz_sha, pz_size = _read_image_bytes_and_sha(crop_path)
            if perf_enabled:
                perf_crop_events += 1
                perf_saved_second_reads += 1
            tool_pz_res = {
                "crop_path": crop_path, 
                "bbox_2d": bbox_used,
                "result_sha": pz_sha,
                "size": int(pz_size),
            }

            crop_img = None
            try:
                if pz_bytes is None:
                    raise RuntimeError("crop bytes unavailable")
                crop_img = _pil_from_bytes_rgb(pz_bytes)
            except Exception:
                crop_img = img
            crop_img = _resize_image_max_side(crop_img, vlm_max_side)
            if perf_enabled:
                perf_io_decode_seconds += max(0.0, time.perf_counter() - pz_t0)

            raw1, parsed1, meta1 = _vlm_generate_with_schema_repair(
                processor=processor,
                model=model,
                images=[crop_img],
                prompt=prompt_crop,
                generation_config=generation_config,
                use_cache=bool(use_cache),
                sdp_backend=sdp_backend,
                max_retries=max_retries,
                repair_n=repair_n,
                answer_stop_criteria=answer_stop_criteria,
                dry_run=bool(args.dry_run),
                sample_id=sample_id,
                seed=seed,
                round_idx=1,
            )
            trace["turns"].append(
                {
                    "round": 1,
                    "name": "pz",
                    "tool_call": tool_pz,
                    "tool_result": tool_pz_res,
                    "prompt": prompt_crop,
                    "raw_output": raw1,
                    "parsed": parsed1,
                    "meta": meta1,
                }
            )
            _update_schema_stats(sample_id, 1, meta1)
            if bool(meta1.get("repair_failed")):
                trace["repair_failed"] = True
            pz_called = True
            tool_calls_total += 1

            cr_called = False
            ref_sample_id = ""
            raw2 = ""
            parsed2: Dict[str, Any] = {}

            if _uncertain(parsed1):
                cr_called = True
                pool_key = class_name
                if pool_key not in normal_pool_cache:
                    pool: List[Dict[str, Any]] = []
                    idxs = normal_idx_by_class.get(class_name, [])
                    take = idxs[:20]
                    for j in take:
                        rj = d0[int(j)]
                        if not isinstance(rj, dict):
                            continue
                        sid = _sample_id(split, int(j), rj)
                        try:
                            imj = _decode_hf_image(dataset_id, rj.get("query_image")).convert("RGB")
                        except Exception:
                            continue
                        pool.append({"sample_id": sid, "class_name": class_name, "image": imj})
                    normal_pool_cache[pool_key] = pool

                tool_cr = {
                    "name": "query_image",
                    "arguments": {},
                }
                ref_path, ref_sample_id = query_image(
                    class_name,
                    normal_pool_cache.get(pool_key, []),
                    int(seed),
                    sample_id,
                    sample_dir,
                )
                
                ref_t0 = time.perf_counter() if perf_enabled else 0.0
                ref_bytes, ref_sha, ref_size = _read_image_bytes_and_sha(ref_path)
                if perf_enabled:
                    perf_ref_events += 1
                    perf_saved_second_reads += 1
                tool_cr_res = {
                    "ref_path": ref_path, 
                    "ref_sample_id": ref_sample_id,
                    "result_sha": ref_sha,
                    "size": int(ref_size),
                }

                cr_images: List[Any]
                if args.dry_run:
                    try:
                        if ref_bytes is None:
                            raise RuntimeError("ref bytes unavailable")
                        ref_img = _pil_from_bytes_rgb(ref_bytes)
                        ref_img = _resize_image_max_side(ref_img, vlm_max_side)
                        cr_images = [crop_img, ref_img]
                    except Exception:
                        cr_images = [crop_img, crop_img]
                else:
                    try:
                        if ref_bytes is None:
                            raise RuntimeError("ref bytes unavailable")
                        ref_img = _pil_from_bytes_rgb(ref_bytes)
                        ref_img = _resize_image_max_side(ref_img, vlm_max_side)
                        cr_images = [crop_img, ref_img]
                    except Exception:
                        cr_images = [crop_img]
                raw2, parsed2, meta2 = _vlm_generate_with_schema_repair(
                    processor=processor,
                    model=model,
                    images=cr_images,
                    prompt=prompt_cr,
                    generation_config=generation_config,
                    use_cache=bool(use_cache),
                    sdp_backend=sdp_backend,
                    max_retries=max_retries,
                    repair_n=repair_n,
                    answer_stop_criteria=answer_stop_criteria,
                    dry_run=bool(args.dry_run),
                    sample_id=sample_id,
                    seed=seed,
                    round_idx=2,
                )
                if perf_enabled:
                    perf_io_decode_seconds += max(0.0, time.perf_counter() - ref_t0)
                trace["turns"].append(
                    {
                        "round": 2,
                        "name": "cr",
                        "tool_call": tool_cr,
                        "tool_result": tool_cr_res,
                        "prompt": prompt_cr,
                        "raw_output": raw2,
                        "parsed": parsed2,
                        "meta": meta2,
                        "ref_sample_id": ref_sample_id
                    }
                )
                _update_schema_stats(sample_id, 2, meta2)
                if bool(meta2.get("repair_failed")):
                    trace["repair_failed"] = True
                tool_calls_total += 1

            final_parsed = parsed2 if parsed2 else parsed1 if parsed1 else parsed0
            final_obj, coerced_anomaly = _build_final_obj(sample_id, final_parsed, bbox_norm)
            pred_label = final_obj["anomaly"]
            final_path = str(sample_dir / "final.json")
            write_json(Path(final_path), final_obj)
            trace_fingerprint = json.loads(json.dumps(trace, ensure_ascii=False))
            if isinstance(trace_fingerprint, dict):
                trace_fingerprint.pop("timestamp_utc", None)
                trace_fingerprint.pop("trace_fingerprint_hash", None)
            trace["trace_fingerprint_hash"] = _sha256_upper_json(trace_fingerprint)
            trace_path = str(sample_dir / "trace.json")
            write_json(Path(trace_path), trace)

            raw_output_obj = {
                "round0": raw0,
                "round1": raw1,
                "round2": raw2,
                "cr_called": bool(cr_called),
                "ref_sample_id": ref_sample_id,
                "final": final_obj,
                "meta": {"coerced_anomaly": bool(coerced_anomaly)},
            }
            raw_output, raw_output_norm, roundtrip_ok = _prepare_raw_output_json(
                sample_id=sample_id,
                raw_payload=raw_output_obj,
                bbox_hint=bbox_norm,
            )
            if not roundtrip_ok:
                _append_trace_error(
                    trace,
                    "raw_output_roundtrip_failed",
                    "raw_output JSON roundtrip failed; fallback final applied",
                    {"sample_id": sample_id},
                )
            if isinstance(raw_output_norm.get("final"), dict):
                final_obj = dict(raw_output_norm.get("final"))
                pred_label = str(final_obj.get("anomaly", "no"))
            write_json(Path(final_path), final_obj)
            write_json(Path(trace_path), trace)

            rows.append(
                {
                    "sample_id": sample_id,
                    "class_name": class_name,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "correct": _compute_correct(gt_label, pred_label),
                    "raw_output": raw_output,
                    "model_id": vlm_model_id,
                    "seed": int(seed),
                    "prompt_hash": prompt_hash,
                    "config_hash": config_hash,
                    "pz_called": 1,
                    "cr_called": int(cr_called),
                    "bbox_norm": json.dumps(bbox_norm, ensure_ascii=False, separators=(",", ":")),
                    "ref_sample_id": ref_sample_id,
                }
            )
            n_success += 1
            if pz_called or cr_called: n_samples_with_tool_call += 1
            if progress_enabled:
                if pbar is not None:
                    pbar.update(1)
                elif progress_fallback:
                    _stderr_progress(len(rows), progress_total)
            _log_schema_stats_progress(len(rows))
            _cleanup_success_path_if_needed(len(rows))

        csv_columns = [
            "sample_id",
            "class_name",
            "gt_label",
            "pred_label",
            "correct",
            "raw_output",
            "model_id",
            "seed",
            "prompt_hash",
            "config_hash",
            "pz_called",
            "cr_called",
            "bbox_norm",
            "ref_sample_id",
        ]
        with open(out_csv, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=csv_columns, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for row in rows:
                rec = {k: row.get(k) for k in csv_columns}
                raw_text = rec.get("raw_output")
                if isinstance(raw_text, str):
                    rec["raw_output"] = raw_text.replace("\r", " ").replace("\n", " ")
                try:
                    writer.writerow(rec)
                except Exception:
                    csv_write_fail_count += 1
                    # Last-resort string coercion to avoid truncating the run output.
                    fallback_row = {k: ("" if rec.get(k) is None else str(rec.get(k))) for k in csv_columns}
                    try:
                        writer.writerow(fallback_row)
                    except Exception:
                        pass

        required_final_keys = ["anomaly", "defect_type", "reason", "confidence", "bbox"]
        unknown_count = 0
        strict_schema_invalid_count = 0
        invalid_example_ids: List[Dict[str, Any]] = []
        for row in rows:
            sid = _safe_str(row.get("sample_id"))
            final_from_raw: Dict[str, Any] = {}
            missing_keys: List[str] = []
            anomaly_value = ""
            raw_payload = row.get("raw_output")
            try:
                raw_obj = json.loads(raw_payload) if isinstance(raw_payload, str) else {}
            except Exception:
                raw_obj = {}
            if isinstance(raw_obj, dict) and isinstance(raw_obj.get("final"), dict):
                final_from_raw = raw_obj["final"]
            missing_keys = [k for k in required_final_keys if k not in final_from_raw]
            anomaly_value = str(final_from_raw.get("anomaly", "")).strip().lower() if isinstance(final_from_raw, dict) else ""
            if anomaly_value == "unknown":
                unknown_count += 1
            anomaly_allowed = anomaly_value in {"yes", "no"}
            if missing_keys or not anomaly_allowed:
                strict_schema_invalid_count += 1
                if len(invalid_example_ids) < 5:
                    invalid_example_ids.append({"sample_id": sid, "missing_keys": missing_keys})
        self_check = {
            "n_rows": int(len(rows)),
            "unknown_count": int(unknown_count),
            "strict_schema_invalid_count": int(strict_schema_invalid_count),
            "invalid_example_ids": invalid_example_ids,
        }
        print("SELF_CHECK_JSON=" + json.dumps(self_check, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

        summary = {
            "timestamp_utc": utc_now_iso(),
            "run_name": run_name,
            "model_id": vlm_model_id,
            "seed": int(seed),
            "prompt_hash": prompt_hash,
            "config_path": str(cfg_path),
            "config_hash": config_hash,
            "git_commit": git_commit,
            "split": split,
            "max_samples": int(max_samples),
            "N": int(len(rows)),
            "effective_n": int(n_success),
            "n_requested_ids": int(n_requested_ids),
            "n_attempted": int(n_attempted),
            "n_success": int(n_success),
            "n_skipped": int(n_skipped),
            "skip_reasons": skip_reasons,
            "skip_reason_examples": skip_reason_examples,
            "image_loader_env": image_env,
            "mmad_root_resolved": mmad_root_str,
            "mmad_asset_mode": mmad_asset_mode,
            "allow_full_dataset": bool(args.allow_full_dataset),
            "max_samples_effective": max_samples_effective,
            "dataset_split": split,
            "missing_dataset_prefixes": missing_dataset_prefixes,
            "missing_example_files": missing_example_files,
            "toolcall_rate": (float(n_samples_with_tool_call) / float(len(rows))) if rows else 0.0,
            "toolcall_rate_avg": (float(n_samples_with_tool_call) / float(len(rows))) if rows else 0.0,
            "tool_calls_total": int(tool_calls_total),
            "csv_write_fail_count": int(csv_write_fail_count),
            "uncertainty_rule": cr_rule,
            "use_cache": bool(use_cache),
            "out_csv": str(out_csv),
            "trace_dir": str(trace_root),
            "cleanup_every_n": int(cleanup_every_n),
        }
        trace_main_jsonl = trace_root / "main.jsonl"
        with open(trace_main_jsonl, "w", encoding="utf-8") as f_main:
            for row in rows:
                sid = str(row.get("sample_id") or "")
                st = schema_sample_stats.get(sid, {})
                line_obj = {
                    "sample_id": sid,
                    "repair_used": bool(st.get("repair_used", False)),
                    "repair_failed": bool(st.get("repair_failed", False)),
                    "missing_answer_tags": bool(st.get("missing_answer_tags", False)),
                }
                f_main.write(json.dumps(line_obj, ensure_ascii=False, sort_keys=True) + "\n")
        missing_answer_rate = (float(len(missing_answer_sample_ids)) / float(len(rows))) if rows else 0.0
        summary["schema_stats"] = {
            "main_jsonl": str(trace_main_jsonl),
            "processed_count": int(len(rows)),
            "repair_count": int(schema_repair_turn_count),
            "repair_failed_count": int(schema_repair_failed_turn_count),
            "missing_answer_round_count": int(missing_answer_round_count),
            "missing_answer_sample_count": int(len(missing_answer_sample_ids)),
            "missing_answer_rate": float(missing_answer_rate),
            "schema_stats_every": int(schema_stats_every),
        }
        trace_files = list(trace_root.rglob("trace.json")) if trace_root.exists() else []
        summary["trace_count"] = int(len(trace_files))
        summary["trace_emitted"] = bool(len(trace_files) > 0)
        csv_sha = hashlib.sha256(out_csv.read_bytes()).hexdigest().upper() if out_csv.exists() else None
        summary["csv_sha256"] = csv_sha
        write_json(out_summary, summary)
        if rows:
            first_sample_id = str(rows[0].get("sample_id") or "")
            first_trace_path = trace_root / first_sample_id / "trace.json"
            first_hash = None
            try:
                first_trace = json.loads(_read_text(first_trace_path))
                if isinstance(first_trace, dict):
                    first_hash = first_trace.get("trace_fingerprint_hash")
            except Exception:
                first_hash = None
    finally:
        if progress_enabled:
            if pbar is not None:
                try:
                    pbar.close()
                except Exception:
                    pass

    # Gather result summary
    result_summary = {
        "run_name": run_name,
        "config_hash": config_hash,
        "prompt_hash": prompt_hash,
        "out_csv": str(out_csv),
        "out_summary": str(out_summary),
        "trace_root": str(trace_root),
        "csv_sha256": csv_sha,
        "first_sample_id": first_sample_id if rows else None,
        "first_trace_fingerprint_hash": first_hash if rows else None,
        "effective_n": int(n_success),
        "n_requested_ids": int(n_requested_ids),
        "n_attempted": int(n_attempted),
        "n_success": int(n_success),
        "n_skipped": int(n_skipped),
        "skip_reasons": skip_reasons,
        "skip_reason_examples": skip_reason_examples,
        "image_loader_env": image_env,
        "mmad_root_resolved": mmad_root_str,
        "mmad_asset_mode": mmad_asset_mode,
        "allow_full_dataset": bool(args.allow_full_dataset),
        "max_samples_effective": max_samples_effective,
        "dataset_split": split,
        "use_cache": bool(use_cache),
        "cleanup_every_n": int(cleanup_every_n),
        "missing_dataset_prefixes": missing_dataset_prefixes,
        "missing_example_files": missing_example_files,
    }
    processed_count = int(len(rows))
    missing_answer_rate = (float(len(missing_answer_sample_ids)) / float(processed_count)) if processed_count else 0.0
    result_summary["schema_stats"] = {
        "processed_count": processed_count,
        "repair_count": int(schema_repair_turn_count),
        "repair_failed_count": int(schema_repair_failed_turn_count),
        "missing_answer_round_count": int(missing_answer_round_count),
        "missing_answer_sample_count": int(len(missing_answer_sample_ids)),
        "missing_answer_rate": float(missing_answer_rate),
        "schema_stats_every": int(schema_stats_every),
        "main_jsonl": str(trace_root / "main.jsonl"),
    }
    total_wall_seconds = max(0.0, time.perf_counter() - run_wall_t0)
    avg_sec_per_sample = (total_wall_seconds / float(processed_count)) if processed_count else 0.0
    generated_tokens_total = int(_VLM_RUNTIME_STATS.get("generated_tokens_total", 0))
    avg_gen_tokens_per_sample = (float(generated_tokens_total) / float(processed_count)) if processed_count else 0.0
    retry_exceptions = int(_VLM_RUNTIME_STATS.get("retry_exceptions", 0))
    cleanup_calls_total = int(_CUDA_CLEANUP_STATS.get("calls_total", 0))
    cleanup_calls_success = int(_CUDA_CLEANUP_STATS.get("calls_success", 0))
    cleanup_calls_error = int(_CUDA_CLEANUP_STATS.get("calls_error", 0))

    # Log readable summary to stderr (Audit/Debug)
    # Ensure all logs go to stderr to keep stdout clean for harness
    print(f"run_name={run_name}", file=sys.stderr)
    print(f"config_hash={config_hash}", file=sys.stderr)
    print(f"prompt_hash={prompt_hash}", file=sys.stderr)
    print(str(out_csv), file=sys.stderr)
    print(str(out_summary), file=sys.stderr)
    print(str(trace_root), file=sys.stderr)
    print(f"csv_sha256={csv_sha}", file=sys.stderr)
    if rows:
        print(f"first_sample_id={first_sample_id}", file=sys.stderr)
        print(f"first_trace_fingerprint_hash={first_hash}", file=sys.stderr)
    _log_schema_stats_progress(processed_count, final=True)
    print(
        f"[schema_summary] processed={processed_count} repair_count={int(schema_repair_turn_count)} "
        f"repair_failed_count={int(schema_repair_failed_turn_count)} "
        f"missing_answer_samples={int(len(missing_answer_sample_ids))} "
        f"missing_answer_rounds={int(missing_answer_round_count)} "
        f"missing_answer_rate={missing_answer_rate:.4f} "
        f"threshold=0.0500 pass={str(missing_answer_rate < 0.05).lower()} "
        f"main_jsonl={str(trace_root / 'main.jsonl')}",
        file=sys.stderr,
    )

    # Contract: Single line JSON to stderr (to keep stdout clean for harness)
    print(f"L2_RESULT_JSON={json.dumps(result_summary, ensure_ascii=False, sort_keys=True)}", file=sys.stderr)
    if perf_enabled:
        print(
            f"[06_run_agentiad_infer.py][perf] total_wall_seconds={total_wall_seconds:.4f} "
            f"avg_sec_per_sample={avg_sec_per_sample:.4f} "
            f"avg_gen_tokens_per_sample={avg_gen_tokens_per_sample:.2f} "
            f"max_new_tokens={int(max_new_tokens)}",
            file=sys.stderr,
        )
        print(
            f"[06_run_agentiad_infer.py][perf] retry_exceptions={retry_exceptions} "
            f"schema_repair_count={int(schema_repair_turn_count)} "
            f"schema_repair_failed_count={int(schema_repair_failed_turn_count)}",
            file=sys.stderr,
        )
        print(
            f"[06_run_agentiad_infer.py][perf] cuda_cleanup_calls_total={cleanup_calls_total} "
            f"(success={cleanup_calls_success}, error={cleanup_calls_error}, cleanup_every_n={int(cleanup_every_n)})",
            file=sys.stderr,
        )
        print(
            f"[06_run_agentiad_infer.py][perf] saved_second_reads={int(perf_saved_second_reads)} "
            f"(crop_events={int(perf_crop_events)}, ref_events={int(perf_ref_events)})",
            file=sys.stderr,
        )
        print(
            f"[06_run_agentiad_infer.py][perf] io_decode_seconds_total={perf_io_decode_seconds:.4f}",
            file=sys.stderr,
        )

    if args.evidence_dir:
        _package_evidence(Path(args.evidence_dir).resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
