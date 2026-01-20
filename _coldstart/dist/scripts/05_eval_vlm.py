from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _decode_hf_image(dataset_id: str, value: Any) -> "PIL.Image.Image":
    import io
    import os
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
        rev = os.environ.get("MMAD_MVTEC_AD_REVISION") or "e88b7bd615ad582b0a7e8238066a9fb293a072b4"
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value, revision=rev)
        return PILImage.open(local_path)

    if value.startswith("DS-MVTec/"):
        rev = os.environ.get("MMAD_ASSET_REVISION") or "f1ad11c07452dff1e023a0df1093e40701d22cab"
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value, revision=rev)
        return PILImage.open(local_path)

    try:
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value)
    except EntryNotFoundError:
        rev = os.environ.get("MMAD_ASSET_REVISION") or "f1ad11c07452dff1e023a0df1093e40701d22cab"
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value, revision=rev)
    return PILImage.open(local_path)


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


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


def _extract_gt_yesno(row: dict, gt_key_override: str | None) -> tuple[str, str]:
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


def main() -> int:
    project_root = _bootstrap_src()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    from agentiad_repro.utils import load_paths, sha256_text, utc_now_iso, write_json

    cfg_path = Path(args.config).resolve()
    cfg_text = _read_text(cfg_path)
    cfg = _load_yaml(cfg_path)
    config_hash = sha256_text(cfg_text)

    baseline_name = str(cfg.get("baseline_name", "vlm_yesno_v1"))
    seed = int(cfg.get("seed", 0))
    split = str(cfg.get("split", "train"))
    max_samples = int(cfg.get("max_samples", 200))
    vlm_model_id = str(cfg.get("vlm_model_id", "")).strip()
    device = str(cfg.get("device", "cpu")).strip().lower()
    batch = int(cfg.get("batch", 4))
    max_new_tokens = int(cfg.get("max_new_tokens", 8))
    prompt_template = str(cfg.get("prompt_template", "")).strip()
    gen = cfg.get("gen", {}) if isinstance(cfg.get("gen", {}), dict) else {}
    require_query_prefix = str(cfg.get("require_query_prefix", "")).strip()
    gt_key_override = str(cfg.get("gt_key", "")).strip()
    device_map = cfg.get("device_map", None)
    torch_dtype = cfg.get("torch_dtype", None)

    if not vlm_model_id:
        print(
            "Missing vlm_model_id in dist/configs/model.yaml (set dist/configs/model.yaml: vlm_model_id).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not prompt_template:
        print("Missing prompt_template in configs/model.yaml.", file=sys.stderr)
        return 2

    try:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText
        except Exception:
            AutoModelForImageTextToText = None
        from transformers import GenerationConfig
        from transformers import AutoModelForCausalLM
    except Exception:
        print(
            "Missing dependencies for Level-1 VLM baseline.\n"
            "Install (CPU):\n"
            "  python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
            "  python -m pip install --upgrade transformers datasets pillow\n",
            file=sys.stderr,
        )
        return 2

    paths = load_paths(project_root)
    dataset_id = "jiang-cc/MMAD"
    prompt_hash = sha256_text(prompt_template)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif not (device == "cpu" or device.startswith("cuda")):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

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
            if device_map is not None and str(device_map).strip().lower() not in {"", "none", "auto"}:
                print("accelerate is required for device_map; falling back to model.to(cuda) may be slow/oom")

    def _from_pretrained_with_fallback(model_cls: Any) -> Any:
        try:
            return model_cls.from_pretrained(vlm_model_id, **model_kwargs)
        except ValueError as e:
            msg = str(e)
            if "requires `accelerate`" in msg and "device_map" in model_kwargs:
                return model_cls.from_pretrained(vlm_model_id)
            raise

    model: Any
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

    try:
        from datasets import load_dataset
    except Exception:
        print("Missing dependency: datasets. Install with: python -m pip install --upgrade datasets", file=sys.stderr)
        return 2

    ds = load_dataset(dataset_id)
    if split not in ds:
        split = "test" if "test" in ds else list(ds.keys())[0]
    d0 = ds[split]
    print(list(d0.features.keys()))

    n_total = int(d0.num_rows)
    candidates = list(range(n_total))
    rng.shuffle(candidates)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(project_root))
        except Exception:
            return str(p)

    rows: List[Dict[str, Any]] = []
    correct_n = 0
    with_gt_n = 0
    gt_rule_used_counts: Dict[str, int] = {}
    attempts = 0
    max_attempts = max_samples * 50

    def _pick_batch_ids() -> List[int]:
        out: List[int] = []
        nonlocal attempts
        for idx in candidates:
            if len(rows) + len(out) >= max_samples:
                break
            attempts += 1
            if attempts > max_attempts:
                break
            row = d0[int(idx)]
            qv = row.get("query_image")
            if require_query_prefix and isinstance(qv, str) and not qv.startswith(require_query_prefix):
                continue
            out.append(int(idx))
            if len(out) >= max(1, int(batch)):
                break
        return out

    _ = gen
    generation_config = GenerationConfig(max_new_tokens=int(max_new_tokens), do_sample=False)

    def _infer_model_device(m: Any) -> "torch.device":
        emb = getattr(m, "get_input_embeddings", None)
        if callable(emb):
            e = m.get_input_embeddings()
            if (
                e is not None
                and hasattr(e, "weight")
                and hasattr(e.weight, "device")
                and str(e.weight.device) != "meta"
            ):
                return e.weight.device
        md = getattr(m, "device", None)
        if md is not None:
            try:
                return md
            except Exception:
                pass
        for p in m.parameters():
            if hasattr(p, "device") and str(p.device) != "meta":
                return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_device = _infer_model_device(model)
    print(f"device={device} torch_cuda_available={torch.cuda.is_available()} model_device={model_device}")
    print(f"hf_device_map={getattr(model, 'hf_device_map', None)}")
    print(f"max_new_tokens={int(max_new_tokens)}")

    while len(rows) < max_samples and attempts <= max_attempts:
        batch_ids = _pick_batch_ids()
        if not batch_ids:
            break

        images: List["PIL.Image.Image"] = []
        prompts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for idx in batch_ids:
            row = d0[int(idx)]
            qv = row.get("query_image")
            tv = row.get("template_image")
            try:
                img = _decode_hf_image(dataset_id, qv).convert("RGB")
            except Exception:
                continue
            images.append(img)
            prompts.append(prompt_template)
            metas.append(
                {
                    "idx": int(idx),
                    "query_image": qv,
                    "template_image": tv,
                    "row": row,
                }
            )

        if not images:
            continue

        if hasattr(processor, "apply_chat_template"):
            texts: List[str] = []
            for p in prompts:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": p},
                        ],
                    }
                ]
                texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        else:
            inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True)
        inputs = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            gen_ids = model.generate(**inputs, generation_config=generation_config)

        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        decode_ids = gen_ids
        if input_ids is not None and hasattr(input_ids, "shape") and hasattr(gen_ids, "shape"):
            decode_ids = gen_ids[:, input_ids.shape[1] :]

        if hasattr(processor, "batch_decode"):
            decoded = processor.batch_decode(decode_ids, skip_special_tokens=True)
        else:
            tok = getattr(processor, "tokenizer", None)
            if tok is None:
                decoded = [str(x) for x in decode_ids]
            else:
                decoded = tok.batch_decode(decode_ids, skip_special_tokens=True)

        for meta, out_text in zip(metas, decoded):
            raw_output = _safe_str(out_text).strip()
            pred_label = "UNKNOWN"
            parse_ok = False
            js = _extract_first_json(raw_output)
            if js is not None:
                try:
                    obj = json.loads(js)
                    an = obj.get("anomaly") if isinstance(obj, dict) else None
                    norm = _normalize_yesno(an)
                    if norm in {"yes", "no"}:
                        pred_label = norm
                        parse_ok = True
                except Exception:
                    parse_ok = False
                    pred_label = "UNKNOWN"

            row_obj = meta.get("row", {})
            if not isinstance(row_obj, dict):
                row_obj = {}
            gt_label, gt_rule_used = _extract_gt_yesno(row_obj, gt_key_override or None)
            gt_rule_used_counts[gt_rule_used] = gt_rule_used_counts.get(gt_rule_used, 0) + 1
            correct = 0
            if gt_label:
                with_gt_n += 1
                if pred_label in {"yes", "no"}:
                    correct = int(gt_label == pred_label)
                    correct_n += int(correct)

            rows.append(
                {
                    "idx": int(meta["idx"]),
                    "split": split,
                    "query_image": meta.get("query_image"),
                    "template_image": meta.get("template_image"),
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "raw_output": raw_output,
                    "parse_ok": int(parse_ok),
                    "model_id": vlm_model_id,
                    "seed": int(seed),
                    "prompt_hash": prompt_hash,
                    "config_hash": config_hash,
                }
            )

    acc = float(correct_n / with_gt_n) if with_gt_n > 0 else None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = paths.tables_dir / f"baseline_vlm_{ts}.csv"
    import pandas as pd

    df = pd.DataFrame(
        rows,
        columns=[
            "idx",
            "split",
            "query_image",
            "template_image",
            "gt_label",
            "pred_label",
            "raw_output",
            "parse_ok",
            "model_id",
            "seed",
            "prompt_hash",
            "config_hash",
        ],
    )
    df.to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "timestamp": utc_now_iso(),
        "acc": acc,
        "N_total": int(len(rows)),
        "N_with_gt": int(with_gt_n),
        "gt_rule_used_counts": gt_rule_used_counts,
        "vlm_model_id": vlm_model_id,
        "seed": int(seed),
        "prompt_hash": prompt_hash,
        "config_hash": config_hash,
        "split": split,
        "max_samples": int(max_samples),
        "baseline_name": baseline_name,
        "subset_filter": f"query_image startswith {require_query_prefix}" if require_query_prefix else "NONE",
        "out_csv": _rel(out_csv),
    }
    out_summary = paths.logs_dir / "baseline_vlm_summary.json"
    out_summary_ts = paths.logs_dir / f"baseline_vlm_summary_{ts}.json"
    write_json(out_summary, summary)
    write_json(out_summary_ts, summary)

    print(f"N_total={len(rows)}")
    print(f"N_with_gt={with_gt_n}")
    if acc is None:
        print("acc=null")
    else:
        print(f"acc={acc:.6f}")
    print(_rel(out_csv))
    print(_rel(out_summary))
    print(f"config_hash={config_hash}")
    print(f"prompt_hash={prompt_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
