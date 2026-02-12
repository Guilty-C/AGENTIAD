# [REMOTE EXECUTION DETECTED? CHECK GUIDELINES]
# This project enforces a strict "Zero-Pollution" remote execution protocol for shared lab servers.
# See REMOTE_EXECUTION_GUIDE.txt for the mandatory "Upload -> Tmp Run -> Cleanup" workflow.

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from subprocess import CalledProcessError, run
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


def _decode_query_image(dataset_id: str, value: Any) -> "PIL.Image.Image":
    import os
    from pathlib import Path as _Path

    from PIL import Image as PILImage

    if value is None:
        raise ValueError("query_image is None")

    if hasattr(value, "save"):
        return value

    if isinstance(value, dict):
        if "path" in value and value["path"]:
            p = _Path(value["path"])
            if p.exists():
                return PILImage.open(p)

    if not isinstance(value, str):
        raise TypeError(f"Unsupported query_image type: {type(value)}")

    p = _Path(value)
    if p.exists():
        return PILImage.open(p)

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except Exception as e:
        raise RuntimeError("huggingface_hub is required to fetch images.") from e

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


def _option_letters(n: int) -> List[str]:
    letters = []
    for i in range(n):
        letters.append(chr(ord("A") + i))
    return letters


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    colset = set(columns)
    for name in candidates:
        if name in colset:
            return name
    return None


def _git_commit(project_root: Path) -> Optional[str]:
    try:
        out = run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip() or None


def _parse_options(options: Any) -> List[str]:
    if isinstance(options, list):
        return [str(x) for x in options]

    if isinstance(options, str):
        lines = [ln.strip() for ln in options.splitlines() if ln.strip()]
        out: List[str] = []
        for ln in lines:
            m = re.match(r"^[A-Z]\s*[\)\:\.\-]\s*(.*)$", ln)
            if m:
                out.append(m.group(1).strip())
            else:
                out.append(ln)
        return out

    return []


def main() -> int:
    project_root = _bootstrap_src()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup_n", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    from agentiad_repro.utils import ensure_dir, load_paths, sha256_text, utc_now_iso, write_json

    cfg_path = Path(args.config).resolve()
    cfg_text = _read_text(cfg_path)
    cfg = _load_yaml(cfg_path)
    config_hash = sha256_text(cfg_text)

    baseline_name = str(cfg.get("baseline_name", "clip_mcq_v1"))
    model_id = str(cfg["model_id"])
    seed = int(cfg.get("seed", 0))
    split = str(cfg.get("split", "train"))
    max_samples = int(cfg.get("max_samples", 200))
    prompt_template = str(cfg.get("prompt_template", cfg.get("text_template", "{question}\n{option}")))
    prompt_hash = sha256_text(prompt_template)

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception:
        print(
            "Missing dependencies for Level-1 CLIP baseline.\n"
            "Install (CPU):\n"
            "  python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
            "  python -m pip install --upgrade transformers\n"
            "Install (GPU, CUDA 12.1 example):\n"
            "  python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
            "  python -m pip install --upgrade transformers\n",
            file=sys.stderr,
        )
        return 2

    paths = load_paths(project_root)
    dataset_id = "jiang-cc/MMAD"
    ds = load_dataset(dataset_id)
    if split not in ds:
        split = "test" if "test" in ds else list(ds.keys())[0]
    d0 = ds[split]
    columns0 = list(d0.column_names)
    label_field = _first_existing(columns0, ["label", "anomaly", "is_anomaly"])
    class_field = _first_existing(columns0, ["category", "class_name", "object", "cls"])
    sample_id_field = _first_existing(columns0, ["id", "sample_id", "index"])

    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    rng = random.Random(seed)
    n_total = int(d0.num_rows)
    candidates = list(range(n_total))
    rng.shuffle(candidates)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(project_root))
        except Exception:
            return str(p)

    cache_root = paths.cache_dir / "clip" / sha256_text(model_id + "|" + config_hash)
    cache_img_dir = ensure_dir(cache_root / "image" / split)
    cache_txt_dir = ensure_dir(cache_root / "text" / split)

    def _load_npy(path: Path) -> Optional["torch.Tensor"]:
        if not path.exists():
            return None
        try:
            import numpy as np

            arr = np.load(path)
            t = torch.from_numpy(arr)
            if t.dtype != torch.float32:
                t = t.float()
            return t
        except Exception:
            return None

    def _save_npy(path: Path, t: "torch.Tensor") -> None:
        import numpy as np

        ensure_dir(path.parent)
        arr = t.detach().cpu().float().numpy()
        np.save(path, arr)

    if args.warmup:
        warm_n = int(args.warmup_n) if int(args.warmup_n) > 0 else int(max_samples)
        warmed = 0
        attempts = 0
        max_attempts = max(2000, warm_n * 50)
        for idx in candidates:
            if warmed >= warm_n or attempts >= max_attempts:
                break
            attempts += 1
            row = d0[int(idx)]
            qv = row.get("query_image")
            if isinstance(qv, str) and not qv.startswith("DS-MVTec/"):
                continue
            try:
                _ = _decode_query_image(dataset_id, qv)
                warmed += 1
            except Exception:
                continue
        print(f"warmup_ok={warmed}")
        return 0

    rows: List[Dict[str, Any]] = []
    correct_n = 0
    attempts = 0
    max_attempts = max_samples * 50

    for idx in candidates:
        if len(rows) >= max_samples:
            break
        attempts += 1
        if attempts > max_attempts:
            break

        row = d0[int(idx)]
        question = _safe_str(row.get("question"))
        options = _parse_options(row.get("options"))
        answer = row.get("answer")
        sample_id = row.get(sample_id_field) if sample_id_field else None
        class_name = row.get(class_field) if class_field else None
        gt_label = row.get(label_field) if label_field else None
        query_image_val = row.get("query_image")

        if not isinstance(options, list) or len(options) == 0:
            continue

        if isinstance(query_image_val, str) and not query_image_val.startswith("DS-MVTec/"):
            continue

        try:
            image = _decode_query_image(dataset_id, query_image_val).convert("RGB")
        except Exception:
            continue

        texts = []
        for opt in options:
            texts.append(prompt_template.format(question=question, option=_safe_str(opt)))

        img_cache_path = cache_img_dir / f"{int(idx)}.npy"
        image_emb = _load_npy(img_cache_path)
        if image_emb is None:
            img_inputs = processor(images=image, return_tensors="pt")
            img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
            with torch.no_grad():
                image_emb = model.get_image_features(**img_inputs)
                image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
            _save_npy(img_cache_path, image_emb)
        image_emb = image_emb.to(device)

        texts_key = sha256_text(json.dumps(texts, ensure_ascii=False))
        txt_cache_path = cache_txt_dir / f"{int(idx)}_{texts_key}.npy"
        text_emb = _load_npy(txt_cache_path)
        if text_emb is None:
            txt_inputs = processor(text=texts, return_tensors="pt", padding=True)
            txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}
            with torch.no_grad():
                text_emb = model.get_text_features(**txt_inputs)
                text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
            _save_npy(txt_cache_path, text_emb)
        text_emb = text_emb.to(device)

        with torch.no_grad():
            sims = (image_emb @ text_emb.T).squeeze(0)

        scores = sims.detach().cpu().tolist()
        pred_i = int(sims.argmax().detach().cpu().item())
        letters = _option_letters(len(options))
        pred = letters[pred_i] if pred_i < len(letters) else str(pred_i)

        ans = answer
        correct = 0
        if isinstance(ans, str) and ans.strip().upper() in set(letters):
            correct = int(pred == ans.strip().upper())
        else:
            correct = int(_safe_str(pred).strip() == _safe_str(ans).strip())

        correct_n += correct
        raw_output = {
            "scores": scores,
            "pred": pred,
            "options": options,
        }
        rows.append(
            {
                "sample_id": sample_id if sample_id is not None else int(idx),
                "class_name": class_name,
                "gt_label": gt_label,
                "pred_label": pred,
                "raw_output": json.dumps(raw_output, ensure_ascii=False),
                "idx": int(idx),
                "split": split,
                "question": question,
                "answer": ans,
                "pred": pred,
                "correct": int(correct),
                "scores_json": json.dumps(scores, ensure_ascii=False),
                "model_id": model_id,
                "seed": seed,
                "prompt_hash": prompt_hash,
                "config_path": str(cfg_path),
                "config_hash": config_hash,
                "git_commit": _git_commit(project_root),
                "baseline_name": baseline_name,
                "device": device,
            }
        )

    acc = float(correct_n / len(rows)) if rows else 0.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = paths.tables_dir / f"baseline_vlm_{ts}.csv"
    import pandas as pd

    df = pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "class_name",
            "gt_label",
            "pred_label",
            "raw_output",
            "idx",
            "split",
            "question",
            "answer",
            "pred",
            "correct",
            "scores_json",
            "model_id",
            "seed",
            "prompt_hash",
            "config_path",
            "config_hash",
            "git_commit",
            "baseline_name",
            "device",
        ],
    )
    df.to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "timestamp": utc_now_iso(),
        "acc": acc,
        "N": int(len(rows)),
        "model_id": model_id,
        "seed": seed,
        "prompt_hash": prompt_hash,
        "config_path": str(cfg_path),
        "config_hash": config_hash,
        "git_commit": _git_commit(project_root),
        "split": split,
        "max_samples": int(max_samples),
        "baseline_name": baseline_name,
        "subset_filter": "query_image startswith DS-MVTec/",
        "cache_dir_rel": _rel(cache_root),
    }
    out_summary = paths.logs_dir / "baseline_vlm_summary.json"
    out_summary_ts = paths.logs_dir / f"baseline_vlm_summary_{ts}.json"
    write_json(out_summary, summary)
    write_json(out_summary_ts, summary)

    print(f"acc={acc:.6f}")
    print(_rel(out_csv))
    print(_rel(out_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
