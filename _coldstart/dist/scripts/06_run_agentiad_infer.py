from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _bootstrap_src() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root


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


def _extract_first_json(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    i = text.find("{")
    j = text.rfind("}")
    if i < 0 or j < 0 or j <= i:
        return None
    return text[i : j + 1]


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
    raw = _safe_str(text).strip()
    js = _extract_first_json(raw)
    if js is None:
        return {}, {"parse_ok": False}
    try:
        obj = json.loads(js)
    except Exception:
        return {}, {"parse_ok": False}
    if not isinstance(obj, dict):
        return {}, {"parse_ok": False}

    out: Dict[str, Any] = {}
    an = _normalize_yesno(obj.get("anomaly"))
    if an in {"yes", "no"}:
        out["anomaly"] = an
    if "defect_type" in obj and obj.get("defect_type") is not None:
        out["defect_type"] = str(obj.get("defect_type"))
    conf = obj.get("confidence")
    if isinstance(conf, (int, float)):
        out["confidence"] = float(conf)
    bbox = obj.get("bbox") or obj.get("bbox_norm") or obj.get("bbox_2d")
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            b = [float(x) for x in bbox]
            if all(0.0 <= x <= 1.0 for x in b) and b[0] < b[2] and b[1] < b[3]:
                out["bbox"] = b
        except Exception:
            pass
    return out, {"parse_ok": True}


def _uncertain(parsed: Mapping[str, Any]) -> bool:
    an = parsed.get("anomaly")
    conf = parsed.get("confidence")
    if an not in {"yes", "no"}:
        return True
    if conf is None:
        return True
    try:
        return float(conf) < 0.6
    except Exception:
        return True


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
    import torch

    model_device = _infer_model_device(model)
    imgs = [im.convert("RGB") for im in images]
    if hasattr(processor, "apply_chat_template"):
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    else:
        inputs = processor(images=imgs, text=[prompt], return_tensors="pt", padding=True)
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


def _vlm_generate_dry(images: Sequence[Any], prompt: str, sample_id: str, seed: int) -> str:
    h = hashlib.sha256((prompt + "|" + sample_id + "|" + str(int(seed))).encode("utf-8")).digest()
    conf = int.from_bytes(h[:2], byteorder="big", signed=False) % 101
    confidence = float(conf) / 100.0
    anomaly = "yes" if (h[2] % 2) == 1 else "no"
    bbox = _fallback_bbox_norm(sample_id)
    obj = {"anomaly": anomaly, "confidence": confidence, "bbox": bbox, "defect_type": ""}
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    project_root = _bootstrap_src()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
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

    prompt_global = str(
        cfg.get(
            "agent_prompt_global",
            "You are an industrial anomaly inspector. Look at the image and decide if it is anomalous.\n"
            "Return a single JSON object only:\n"
            "{\"anomaly\":\"yes|no\",\"confidence\":0.0,\"bbox\":[0,0,1,1],\"defect_type\":\"\"}\n"
            "Rules:\n"
            "- bbox must be normalized [x1,y1,x2,y2] in [0,1] with x1<x2,y1<y2\n"
            "- do_sample must be false; be deterministic\n",
        )
    )
    prompt_crop = str(
        cfg.get(
            "agent_prompt_crop",
            "You are given a cropped region from the image. Decide if this region indicates anomaly.\n"
            "Return a single JSON object only:\n"
            "{\"anomaly\":\"yes|no\",\"confidence\":0.0,\"bbox\":[0,0,1,1],\"defect_type\":\"\"}\n",
        )
    )
    prompt_cr = str(
        cfg.get(
            "agent_prompt_cr",
            "You are given a query crop and a reference normal image. Compare them.\n"
            "Return a single JSON object only:\n"
            "{\"anomaly\":\"yes|no\",\"confidence\":0.0,\"bbox\":[0,0,1,1],\"defect_type\":\"\"}\n",
        )
    )
    prompt_hash = sha256_text(prompt_global + "\n---\n" + prompt_crop + "\n---\n" + prompt_cr)

    model_short = re.sub(r"[^a-zA-Z0-9]+", "_", vlm_model_id.strip())[-40:].strip("_") or "model"
    run_name = str(args.run_name) if args.run_name is not None else str(cfg.get("run_name", "")).strip()
    if not run_name:
        run_name = f"L2_{model_short}_{split}_seed{seed}"

    paths = load_paths(project_root)
    dataset_id = "jiang-cc/MMAD"

    processor = None
    model = None
    generation_config = None

    if not args.dry_run:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
            try:
                from transformers import AutoModelForImageTextToText
            except Exception:
                AutoModelForImageTextToText = None
        except Exception:
            print(
                "Missing dependencies for Level-2 VLM agent.\n"
                "Install (CPU):\n"
                "  python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
                "  python -m pip install --upgrade transformers accelerate datasets pillow\n",
                file=sys.stderr,
            )
            return 2

        try:
            from datasets import load_dataset
        except Exception:
            print("Missing dependency: datasets. Install with: python -m pip install --upgrade datasets", file=sys.stderr)
            return 2

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif not (device == "cpu" or device.startswith("cuda")):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        _set_global_seed(seed)

        processor = AutoProcessor.from_pretrained(vlm_model_id)

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
                return model_cls.from_pretrained(vlm_model_id, **model_kwargs)
            except ValueError as e:
                msg = str(e)
                if "requires `accelerate`" in msg and "device_map" in model_kwargs:
                    return model_cls.from_pretrained(vlm_model_id)
                raise

        if AutoModelForImageTextToText is not None:
            try:
                model = _from_pretrained_with_fallback(AutoModelForImageTextToText)
            except Exception:
                model = _from_pretrained_with_fallback(AutoModelForCausalLM)
        else:
            model = _from_pretrained_with_fallback(AutoModelForCausalLM)

        if not use_cuda:
            model = model.to(device)
        elif not accelerate_available:
            model = model.to(device)
        model.eval()

        generation_config = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )

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
    cr_rule = "confidence_missing_or_lt_0.6_or_pred_unknown"
    git_commit = _git_commit()

    for idx in candidates:
        if len(rows) >= max_samples:
            break
        attempts += 1
        if attempts > max_attempts:
            break

        row = d0[int(idx)]
        if not isinstance(row, dict):
            continue
        qv = row.get("query_image")
        if qv is None:
            continue

        sample_id = _sample_id(split, int(idx), row)
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
                "config_hash": config_hash,
                "git_commit": git_commit,
                "gen": {
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": int(max_new_tokens),
                },
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

        tool_pz = {
            "name": "pz.crop_image_normalized",
            "args": {"bbox_2d": bbox_norm},
        }
        try:
            crop_path, bbox_used = crop_image_normalized(bbox_norm, img, sample_dir)
        except Exception:
            bbox_norm = _fallback_bbox_norm(sample_id + "|fallback2")
            crop_path, bbox_used = crop_image_normalized(bbox_norm, img, sample_dir)
        tool_pz_res = {"crop_path": crop_path, "bbox_2d": bbox_used}

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

        cr_called = False
        ref_sample_id = ""
        ref_path = ""
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
                "name": "cr.query_image",
                "args": {"class_name": class_name, "seed": int(seed), "sample_id": sample_id},
            }
            ref_path, ref_sample_id = query_image(
                class_name,
                normal_pool_cache.get(pool_key, []),
                int(seed),
                sample_id,
                sample_dir,
            )
            tool_cr_res = {"ref_path": ref_path, "ref_sample_id": ref_sample_id}

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

        final_parsed = parsed2 if parsed2 else parsed1 if parsed1 else parsed0
        pred_label = _normalize_yesno(final_parsed.get("anomaly")) or "UNKNOWN"
        confidence = final_parsed.get("confidence")
        try:
            confidence_f = float(confidence) if confidence is not None else None
        except Exception:
            confidence_f = None

        final_obj = {
            "anomaly": pred_label if pred_label in {"yes", "no"} else "unknown",
            "bbox": bbox_norm,
            "defect_type": _safe_str(final_parsed.get("defect_type")),
            "confidence": confidence_f,
        }
        final_path = str(sample_dir / "final.json")
        write_json(Path(final_path), final_obj)
        trace_fingerprint = dict(trace)
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
        "uncertainty_rule": cr_rule,
        "out_csv": str(out_csv),
        "trace_dir": str(trace_root),
    }
    write_json(out_summary, summary)

    print(f"run_name={run_name}")
    if args.dry_run:
        print(f"device={device} torch_cuda_available=None model_device=None")
        print("hf_device_map=None")
    else:
        print(f"device={device} torch_cuda_available={torch.cuda.is_available()} model_device={_infer_model_device(model)}")
        print(f"hf_device_map={getattr(model, 'hf_device_map', None)}")
    print(f"gen_do_sample=False temperature=0 top_p=1 max_new_tokens={int(max_new_tokens)}")
    print(f"config_hash={config_hash}")
    print(f"prompt_hash={prompt_hash}")
    print(str(out_csv))
    print(str(out_summary))
    print(str(trace_root))

    csv_sha = hashlib.sha256(out_csv.read_bytes()).hexdigest().upper() if out_csv.exists() else None
    print(f"csv_sha256={csv_sha}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
