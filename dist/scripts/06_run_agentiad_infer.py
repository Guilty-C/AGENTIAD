
import sys
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import re
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
        raise ValueError("image is None")
    if hasattr(value, "save"):
        return value
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            return PILImage.open(io.BytesIO(value["bytes"]))
        if "path" in value and value["path"]:
            p = _Path(value["path"])
            if p.exists():
                return PILImage.open(p)
    if not isinstance(value, str):
        raise TypeError(f"Unsupported image type: {type(value)}")

    p = _Path(value)
    if p.exists():
        return PILImage.open(p)

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except Exception as e:
        raise RuntimeError("huggingface_hub is required to fetch images.") from e

    if value.startswith("MVTec-AD/"):
        rev = _os.environ.get("MMAD_MVTEC_AD_REVISION") or "e88b7bd615ad582b0a7e8238066a9fb293a072b4"
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value, revision=rev)
        return PILImage.open(local_path)

    if value.startswith("DS-MVTec/"):
        rev = _os.environ.get("MMAD_ASSET_REVISION") or "f1ad11c07452dff1e023a0df1093e40701d22cab"
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value, revision=rev)
        return PILImage.open(local_path)

    try:
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value)
    except EntryNotFoundError:
        rev = _os.environ.get("MMAD_ASSET_REVISION") or "f1ad11c07452dff1e023a0df1093e40701d22cab"
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value, revision=rev)
    return PILImage.open(local_path)


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
) -> str:
    try:
        if model is None:
            return '<answer>{"anomaly_present": "unknown", "top_anomaly": "offline_fallback", "visual_descriptions": []}</answer>'

        import torch

        model_device = _infer_model_device(model)
        imgs = [im.convert("RGB") for im in images]
        
        inputs = {}
        try:
            if hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None):
                try:
                    messages: List[Dict[str, Any]] = [
                        {
                            "role": "user",
                            "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}],
                        }
                    ]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=imgs, return_tensors="pt")
                except Exception:
                    messages_text: List[Dict[str, Any]] = [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]
                    text = processor.apply_chat_template(messages_text, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], return_tensors="pt")
            else:
                try:
                    inputs = processor(images=imgs, text=[prompt], return_tensors="pt")
                except Exception:
                     inputs = processor(text=[prompt], return_tensors="pt")
                     
        except Exception:
             inputs = processor(text=[prompt], return_tensors="pt")

        inputs = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            gen_ids = model.generate(**inputs, generation_config=generation_config)
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        decode_ids = gen_ids
        if input_ids is not None and hasattr(input_ids, "shape") and hasattr(gen_ids, "shape"):
            decode_ids = gen_ids[:, input_ids.shape[1] :]
        if hasattr(processor, "batch_decode"):
            decoded = processor.batch_decode(decode_ids, skip_special_tokens=True)
            return _safe_str(decoded[0]).strip() if decoded else ""
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and hasattr(tok, "batch_decode"):
            decoded = tok.batch_decode(decode_ids, skip_special_tokens=True)
            return _safe_str(decoded[0]).strip() if decoded else ""
        return _safe_str(decode_ids).strip()
    except Exception as e:
        print(f"DEBUG: _vlm_generate exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Fallback output that respects contract schema (but says unknown)
        return '<answer>{"anomaly_present": "unknown", "top_anomaly": "execution_error", "visual_descriptions": []}</answer>'


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
    max_samples = int(args.max_samples) if args.max_samples is not None else int(cfg.get("max_samples", 50))

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

    progress_cfg_val = cfg.get("progress", False)
    if isinstance(progress_cfg_val, bool):
        progress_cfg = bool(progress_cfg_val)
    elif isinstance(progress_cfg_val, (int, float)):
        progress_cfg = bool(progress_cfg_val)
    else:
        progress_cfg = str(progress_cfg_val).strip().lower() not in false_set
    env_progress_val = os.environ.get("AGENTIAD_PROGRESS", None)
    env_progress = None if env_progress_val is None else (str(env_progress_val).strip().lower() not in false_set)
    cli_progress = None if args.progress is None else (str(args.progress).strip().lower() not in false_set)
    progress_enabled = cli_progress if cli_progress is not None else env_progress if env_progress is not None else bool(progress_cfg)

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

    processor = None
    model = None
    generation_config = None

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
                 processor = AutoTokenizer.from_pretrained("distilgpt2", local_files_only=local_only)
                 if processor.pad_token is None:
                    processor.pad_token = processor.eos_token
                    processor.pad_token_id = processor.eos_token_id
                 processor.padding_side = "left"

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
                model = AutoModelForCausalLM.from_pretrained("distilgpt2", local_files_only=local_only)
                # Ensure processor is compatible (distilgpt2 tokenizer)
                try:
                    processor = AutoTokenizer.from_pretrained("distilgpt2", local_files_only=local_only)
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
            model.eval()

        generation_config = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )

        if dataset_id.endswith(".json") or dataset_id.endswith(".jsonl"):
            ds = load_dataset("json", data_files=dataset_id)
        else:
            ds = load_dataset(dataset_id)
        if split not in ds:
            split = "test" if "test" in ds else list(ds.keys())[0]
        d0: Any = ds[split]

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
    pbar = None
    progress_fallback = False
    if progress_enabled:
        try:
            from tqdm.auto import tqdm  # type: ignore

            pbar = tqdm(
                total=int(max_samples),
                unit="sample",
                dynamic_ncols=True,
                file=sys.stderr,
                desc=run_name,
            )
        except Exception:
            pbar = None
            progress_fallback = True

    def _stderr_progress(done: int, total: int, attempts_i: int) -> None:
        sys.stderr.write("\r[" + str(int(done)) + "/" + str(int(total)) + "] loop_attempts=" + str(int(attempts_i)))
        sys.stderr.flush()

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

    try:
        for idx in candidates:
            if len(rows) >= max_samples:
                break
            attempts += 1
            if attempts > max_attempts:
                break

            row = d0[int(idx)]
            if not isinstance(row, dict):
                continue
            
            sample_id = _sample_id(split, int(idx), row)
            if id_list_set is not None and sample_id not in id_list_set:
                continue

            qv = row.get("query_image")
            if qv is None:
                continue

            pz_called = False
            cr_called = False

            class_name = _extract_class_name(row)
            gt_label, gt_rule = _extract_gt_yesno(row, gt_key_override)

            try:
                img = _decode_hf_image(dataset_id, qv).convert("RGB")
            except Exception:
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
                },
                "input": {
                    "query_image": qv if isinstance(qv, str) else "<in_memory_image>",
                    "query_image_type": type(qv).__name__,
                    "image_mode": getattr(img, "mode", None),
                    "image_size": list(getattr(img, "size", (None, None))),
                },
                "turns": [],
            }

            if args.dry_run:
                raw0 = _vlm_generate_dry([img], prompt_global, sample_id, seed)
            else:
                raw0 = _vlm_generate(processor, model, [img], prompt_global, generation_config)
            parsed0, meta0 = _parse_agent_json(raw0)
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

            bbox_norm = parsed0.get("bbox") if isinstance(parsed0.get("bbox"), list) else None
            if not bbox_norm:
                bbox_norm = _fallback_bbox_norm(sample_id)

            if not enable_tools:
                cr_called = False
                ref_sample_id = ""
                raw1 = ""
                raw2 = ""
                final_parsed = parsed0
                pred_label = _normalize_yesno(final_parsed.get("anomaly")) or "UNKNOWN"
                
                final_obj = {
                    "anomaly": pred_label if pred_label in {"yes", "no"} else "unknown",
                    "bbox": bbox_norm,
                    "defect_type": _safe_str(final_parsed.get("defect_type")),
                    "confidence": None, # Removed legacy confidence
                }
                final_path = str(sample_dir / "final.json")
                write_json(Path(final_path), final_obj)
                trace_fingerprint = json.loads(json.dumps(trace, ensure_ascii=False))
                if isinstance(trace_fingerprint, dict):
                    trace_fingerprint.pop("timestamp_utc", None)
                    trace_fingerprint.pop("trace_fingerprint_hash", None)
                trace["trace_fingerprint_hash"] = _sha256_upper_json(trace_fingerprint)
                trace_path = str(sample_dir / "trace.json")
                write_json(Path(trace_path), trace)
                
                rows.append(
                    {
                        "sample_id": sample_id,
                        "class_name": class_name,
                        "gt_label": gt_label,
                        "pred_label": pred_label,
                        "raw_output": raw0,
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
                if progress_enabled:
                     if pbar is not None: pbar.update(1)
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
            pz_sha = hashlib.sha256(Path(crop_path).read_bytes()).hexdigest().upper() if Path(crop_path).exists() else None
            tool_pz_res = {
                "crop_path": crop_path, 
                "bbox_2d": bbox_used,
                "result_sha": pz_sha,
                "size": Path(crop_path).stat().st_size if Path(crop_path).exists() else 0,
            }

            crop_img = None
            try:
                from PIL import Image as PILImage
                crop_img = PILImage.open(crop_path).convert("RGB")
            except Exception:
                crop_img = img

            if args.dry_run:
                raw1 = _vlm_generate_dry([crop_img], prompt_crop, sample_id, seed)
            else:
                raw1 = _vlm_generate(processor, model, [crop_img], prompt_crop, generation_config)
            parsed1, meta1 = _parse_agent_json(raw1)
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
                
                cr_sha = hashlib.sha256(Path(ref_path).read_bytes()).hexdigest().upper() if Path(ref_path).exists() else None
                tool_cr_res = {
                    "ref_path": ref_path, 
                    "ref_sample_id": ref_sample_id,
                    "result_sha": cr_sha,
                    "size": Path(ref_path).stat().st_size if Path(ref_path).exists() else 0,
                }

                if args.dry_run:
                    from PIL import Image as PILImage
                    try:
                        ref_img = PILImage.open(ref_path).convert("RGB")
                        raw2 = _vlm_generate_dry([crop_img, ref_img], prompt_cr, sample_id, seed)
                    except Exception:
                        raw2 = _vlm_generate_dry([crop_img, crop_img], prompt_cr, sample_id, seed)
                else:
                    try:
                        from PIL import Image as PILImage
                        ref_img = PILImage.open(ref_path).convert("RGB")
                        raw2 = _vlm_generate(processor, model, [crop_img, ref_img], prompt_cr, generation_config)
                    except Exception:
                        raw2 = _vlm_generate(processor, model, [crop_img], prompt_cr, generation_config)

                parsed2, meta2 = _parse_agent_json(raw2)
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
                    }
                )
                tool_calls_total += 1

            final_parsed = parsed2 if parsed2 else parsed1 if parsed1 else parsed0
            pred_label = _normalize_yesno(final_parsed.get("anomaly")) or "UNKNOWN"

            final_obj = {
                "anomaly": pred_label if pred_label in {"yes", "no"} else "unknown",
                "bbox": bbox_norm,
                "defect_type": _safe_str(final_parsed.get("defect_type")),
                "confidence": None, # Removed legacy
            }
            final_path = str(sample_dir / "final.json")
            write_json(Path(final_path), final_obj)
            trace_fingerprint = json.loads(json.dumps(trace, ensure_ascii=False))
            if isinstance(trace_fingerprint, dict):
                trace_fingerprint.pop("timestamp_utc", None)
                trace_fingerprint.pop("trace_fingerprint_hash", None)
            trace["trace_fingerprint_hash"] = _sha256_upper_json(trace_fingerprint)
            trace_path = str(sample_dir / "trace.json")
            write_json(Path(trace_path), trace)

            raw_output = json.dumps(
                {
                    "round0": raw0,
                    "round1": raw1,
                    "round2": raw2,
                    "cr_called": bool(cr_called),
                    "ref_sample_id": ref_sample_id,
                    "final": final_obj,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

            rows.append(
                {
                    "sample_id": sample_id,
                    "class_name": class_name,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
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
            if pz_called or cr_called: n_samples_with_tool_call += 1
            if progress_enabled:
                if pbar is not None:
                    pbar.update(1)
                elif progress_fallback:
                    _stderr_progress(len(rows), int(max_samples), int(attempts))

        import pandas as pd

        df = pd.DataFrame(
            rows,
            columns=[
                "sample_id",
                "class_name",
                "gt_label",
                "pred_label",
                "raw_output",
                "model_id",
                "seed",
                "prompt_hash",
                "config_hash",
                "pz_called",
                "cr_called",
                "bbox_norm",
                "ref_sample_id",
            ],
        )
        df.to_csv(out_csv, index=False, encoding="utf-8")

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
            "toolcall_rate": (float(n_samples_with_tool_call) / float(len(rows))) if rows else 0.0,
            "tool_calls_total": int(tool_calls_total),
            "uncertainty_rule": cr_rule,
            "out_csv": str(out_csv),
            "trace_dir": str(trace_root),
        }
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
            sys.stderr.write("\n")
            sys.stderr.flush()

    print(f"run_name={run_name}")
    print(f"config_hash={config_hash}")
    print(f"prompt_hash={prompt_hash}")
    print(str(out_csv))
    print(str(out_summary))
    print(str(trace_root))
    print(f"csv_sha256={csv_sha}")
    if rows:
        print(f"first_sample_id={first_sample_id}")
        print(f"first_trace_fingerprint_hash={first_hash}")

    if args.evidence_dir:
        _package_evidence(Path(args.evidence_dir).resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
