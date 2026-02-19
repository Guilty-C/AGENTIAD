# [REMOTE EXECUTION DETECTED? CHECK GUIDELINES]
# This project enforces a strict "Zero-Pollution" remote execution protocol for shared lab servers.
# See REMOTE_EXECUTION_GUIDE.txt for the mandatory "Upload -> Tmp Run -> Cleanup" workflow.

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


def _merge_cfg(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k == "base_config":
            continue
        out[k] = v
    return out


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
            value = value["path"]

    if not isinstance(value, str):
        raise TypeError(f"Unsupported image type: {type(value)}")

    p = _Path(value)
    if p.exists():
        return PILImage.open(p)

    mmad_root = os.environ.get("MMAD_ROOT", "").strip()
    rel = value.replace("\\", "/")
    if mmad_root and rel.startswith(("DS-MVTec/", "MVTec-AD/")):
        local_abs = _Path(mmad_root) / _Path(*rel.split("/"))
        if local_abs.exists():
            return PILImage.open(local_abs)

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


def _option_letters(n: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(n)]


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


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


def _find_pz_boxes(
    query_rgb: "PIL.Image.Image",
    template_rgb: "PIL.Image.Image",
    topk: int,
    min_area_orig: int,
    pad_frac: float,
    max_side: int = 256,
) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, Any]]:
    import numpy as np

    q = query_rgb
    t = template_rgb
    if q.size != t.size:
        t = t.resize(q.size)

    w0, h0 = q.size
    scale = 1.0
    if max(w0, h0) > max_side:
        scale = float(max_side) / float(max(w0, h0))
        w1 = max(1, int(round(w0 * scale)))
        h1 = max(1, int(round(h0 * scale)))
        q = q.resize((w1, h1))
        t = t.resize((w1, h1))

    q_arr = np.asarray(q).astype(np.float32)
    t_arr = np.asarray(t).astype(np.float32)
    diff = np.mean(np.abs(q_arr - t_arr), axis=2)
    thr = float(np.quantile(diff, 0.98))
    mask = diff >= thr

    h, w = mask.shape
    boxes_small: List[Tuple[int, int, int, int, int]] = []

    min_area_small = int(round(float(min_area_orig) * (scale**2)))
    if min_area_small < 1:
        min_area_small = 1

    impl = "python"
    try:
        import cv2  # type: ignore

        impl = "opencv"
        cc = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
        _, labels, stats, _ = cc
        for i in range(1, int(stats.shape[0])):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area_small:
                continue
            boxes_small.append((x, y, x + bw, y + bh, area))
    except Exception:
        visited = np.zeros((h, w), dtype=bool)
        for y0 in range(h):
            for x0 in range(w):
                if not mask[y0, x0] or visited[y0, x0]:
                    continue
                qd = [(y0, x0)]
                visited[y0, x0] = True
                area = 0
                y_min = y0
                y_max = y0
                x_min = x0
                x_max = x0
                while qd:
                    y, x = qd.pop()
                    area += 1
                    if y < y_min:
                        y_min = y
                    if y > y_max:
                        y_max = y
                    if x < x_min:
                        x_min = x
                    if x > x_max:
                        x_max = x
                    y1 = y - 1
                    y2 = y + 1
                    x1 = x - 1
                    x2 = x + 1
                    if y1 >= 0 and mask[y1, x] and not visited[y1, x]:
                        visited[y1, x] = True
                        qd.append((y1, x))
                    if y2 < h and mask[y2, x] and not visited[y2, x]:
                        visited[y2, x] = True
                        qd.append((y2, x))
                    if x1 >= 0 and mask[y, x1] and not visited[y, x1]:
                        visited[y, x1] = True
                        qd.append((y, x1))
                    if x2 < w and mask[y, x2] and not visited[y, x2]:
                        visited[y, x2] = True
                        qd.append((y, x2))
                if area >= min_area_small:
                    boxes_small.append((x_min, y_min, x_max + 1, y_max + 1, area))

    boxes_small.sort(key=lambda b: (-b[4], b[0], b[1]))
    boxes_small = boxes_small[: max(1, int(topk))]

    out: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2, area in boxes_small:
        if scale != 1.0:
            x1 = int(round(x1 / scale))
            y1 = int(round(y1 / scale))
            x2 = int(round(x2 / scale))
            y2 = int(round(y2 / scale))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        px = int(round(bw * float(pad_frac)))
        py = int(round(bh * float(pad_frac)))
        x1p = max(0, x1 - px)
        y1p = max(0, y1 - py)
        x2p = min(w0, x2 + px)
        y2p = min(h0, y2 + py)
        if x2p > x1p and y2p > y1p:
            out.append((int(x1p), int(y1p), int(x2p), int(y2p)))

    info = {
        "query_size": [int(w0), int(h0)],
        "scale": float(scale),
        "thr": float(thr),
        "min_area_small": int(min_area_small),
        "found_components": int(len(boxes_small)),
        "impl": impl,
    }
    return out, info


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
    base_cfg_text = ""
    if "base_config" in cfg and cfg["base_config"]:
        base_path = (project_root / str(cfg["base_config"])).resolve()
        base_cfg_text = _read_text(base_path)
        base_cfg = _load_yaml(base_path)
        cfg = _merge_cfg(base_cfg, cfg)
    config_hash = sha256_text(base_cfg_text + "\n" + cfg_text)

    agent_name = str(cfg.get("agent_name", "clip_pzcr_v1"))
    model_id = str(cfg["model_id"])
    seed = int(cfg.get("seed", 0))
    split = str(cfg.get("split", "train"))
    max_samples = int(cfg.get("max_samples", 200))
    text_template = str(cfg.get("text_template", "{question}\n{option}"))
    require_query_prefix = str(cfg.get("require_query_prefix", "DS-MVTec/")).strip()
    trigger_margin = float(cfg.get("trigger_margin", 0.02))
    pz_topk = int(cfg.get("pz_topk", 2))
    pz_min_box_area = int(cfg.get("pz_min_box_area", 800))
    pz_pad = float(cfg.get("pz_pad", 0.10))
    cr_agg = str(cfg.get("cr_agg", "max")).strip().lower()
    if cr_agg not in {"max", "mean"}:
        raise ValueError("cr_agg must be 'max' or 'mean'")

    try:
        import numpy as np
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception:
        print(
            "Missing dependencies for CLIP PZ/CR agent.\n"
            "Install (CPU):\n"
            "  python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
            "  python -m pip install --upgrade transformers\n"
            "  python -m pip install --upgrade pillow numpy\n",
            file=sys.stderr,
        )
        return 2

    paths = load_paths(project_root)
    dataset_id = "jiang-cc/MMAD"
    ds = load_dataset(dataset_id)
    if split not in ds:
        split = "test" if "test" in ds else list(ds.keys())[0]
    d0 = ds[split]

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
    cache_img_full_dir = ensure_dir(cache_root / "image_full" / split)
    cache_img_crop_dir = ensure_dir(cache_root / "image_crop" / split)
    cache_txt_dir = ensure_dir(cache_root / "text" / split)

    def _load_npy(path: Path) -> Optional["torch.Tensor"]:
        if not path.exists():
            return None
        try:
            arr = np.load(path)
            t = torch.from_numpy(arr)
            if t.dtype != torch.float32:
                t = t.float()
            return t
        except Exception:
            return None

    def _save_npy(path: Path, t: "torch.Tensor") -> None:
        ensure_dir(path.parent)
        arr = t.detach().cpu().float().numpy()
        np.save(path, arr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    def _get_text_emb(idx: int, texts: List[str]) -> "torch.Tensor":
        texts_key = sha256_text(json.dumps(texts, ensure_ascii=False))
        path = cache_txt_dir / f"{int(idx)}_{texts_key}.npy"
        t = _load_npy(path)
        if t is not None:
            return t.to(device)
        txt_inputs = processor(text=texts, return_tensors="pt", padding=True)
        txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}
        with torch.no_grad():
            t = model.get_text_features(**txt_inputs)
            t = t / t.norm(p=2, dim=-1, keepdim=True)
        _save_npy(path, t)
        return t

    def _get_image_emb_full(idx: int, image_rgb: "PIL.Image.Image") -> "torch.Tensor":
        path = cache_img_full_dir / f"{int(idx)}.npy"
        t = _load_npy(path)
        if t is not None:
            return t.to(device)
        img_inputs = processor(images=image_rgb, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        with torch.no_grad():
            t = model.get_image_features(**img_inputs)
            t = t / t.norm(p=2, dim=-1, keepdim=True)
        _save_npy(path, t)
        return t

    def _bbox_hash(idx: int, bbox: Tuple[int, int, int, int], size: Tuple[int, int]) -> str:
        x1, y1, x2, y2 = bbox
        w, h = size
        return sha256_text(f"{idx}:{x1},{y1},{x2},{y2}:{w}x{h}")

    def _get_image_emb_crop(idx: int, bbox: Tuple[int, int, int, int], crop_rgb: "PIL.Image.Image") -> "torch.Tensor":
        hsh = _bbox_hash(idx, bbox, crop_rgb.size)
        path = cache_img_crop_dir / f"{int(idx)}_{hsh}.npy"
        t = _load_npy(path)
        if t is not None:
            return t.to(device)
        img_inputs = processor(images=crop_rgb, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        with torch.no_grad():
            t = model.get_image_features(**img_inputs)
            t = t / t.norm(p=2, dim=-1, keepdim=True)
        _save_npy(path, t)
        return t

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
            tv = row.get("template_image")
            if tv is None:
                continue
            if require_query_prefix and isinstance(qv, str) and not qv.startswith(require_query_prefix):
                continue
            try:
                _ = _decode_hf_image(dataset_id, qv)
                _ = _decode_hf_image(dataset_id, tv)
                warmed += 1
            except Exception:
                continue
        print(f"warmup_ok={warmed}")
        return 0

    rows: List[Dict[str, Any]] = []
    margins: List[float] = []
    triggered_n = 0
    correct_n = 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = paths.tables_dir / f"agent_pzcr_{ts}.csv"
    out_trace = paths.traces_dir / f"agent_pzcr_{ts}.jsonl"

    trace_f = out_trace.open("w", encoding="utf-8")
    try:
        attempts = 0
        max_attempts = max_samples * 80

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
            qv = row.get("query_image")
            tv = row.get("template_image")

            if not options:
                continue
            if tv is None:
                continue
            if require_query_prefix and isinstance(qv, str) and not qv.startswith(require_query_prefix):
                continue

            try:
                query_img = _decode_hf_image(dataset_id, qv).convert("RGB")
                templ_img = _decode_hf_image(dataset_id, tv).convert("RGB")
            except Exception:
                continue

            texts = [text_template.format(question=question, option=_safe_str(opt)) for opt in options]
            text_emb = _get_text_emb(int(idx), texts)
            img_emb = _get_image_emb_full(int(idx), query_img)

            with torch.no_grad():
                baseline_scores_t = (img_emb @ text_emb.T).squeeze(0)

            baseline_scores = baseline_scores_t.detach().cpu().tolist()
            order = sorted(range(len(baseline_scores)), key=lambda i: (-baseline_scores[i], i))
            top1 = float(baseline_scores[order[0]])
            top2 = float(baseline_scores[order[1]]) if len(order) > 1 else float("-inf")
            margin = float(top1 - top2) if top2 != float("-inf") else float("inf")
            margins.append(float(margin))
            triggered = bool(margin < float(trigger_margin))
            if triggered:
                triggered_n += 1

            boxes: List[Tuple[int, int, int, int]] = []
            pz_info: Dict[str, Any] = {}
            final_scores_t = baseline_scores_t

            if triggered:
                boxes, pz_info = _find_pz_boxes(
                    query_rgb=query_img,
                    template_rgb=templ_img,
                    topk=pz_topk,
                    min_area_orig=pz_min_box_area,
                    pad_frac=pz_pad,
                )
                crop_scores_list: List["torch.Tensor"] = []
                if boxes:
                    crop_embs: List[Optional["torch.Tensor"]] = [None for _ in boxes]
                    missing_crops: List["PIL.Image.Image"] = []
                    missing_paths: List[Path] = []
                    missing_is: List[int] = []

                    for i_box, b in enumerate(boxes):
                        crop = query_img.crop(b)
                        hsh = _bbox_hash(int(idx), b, crop.size)
                        path = cache_img_crop_dir / f"{int(idx)}_{hsh}.npy"
                        t = _load_npy(path)
                        if t is not None:
                            crop_embs[i_box] = t.to(device)
                            continue
                        missing_is.append(i_box)
                        missing_crops.append(crop)
                        missing_paths.append(path)

                    if missing_crops:
                        img_inputs = processor(images=missing_crops, return_tensors="pt")
                        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
                        with torch.no_grad():
                            feats = model.get_image_features(**img_inputs)
                            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                        for j, i_box in enumerate(missing_is):
                            t = feats[j : j + 1]
                            _save_npy(missing_paths[j], t)
                            crop_embs[i_box] = t

                    for t in crop_embs:
                        if t is None:
                            continue
                        with torch.no_grad():
                            crop_scores_list.append((t @ text_emb.T).squeeze(0))

                if crop_scores_list:
                    if cr_agg == "max":
                        crop_agg_t = torch.stack(crop_scores_list, dim=0).amax(dim=0)
                        final_scores_t = torch.maximum(baseline_scores_t, crop_agg_t)
                    else:
                        crop_agg_t = torch.stack(crop_scores_list, dim=0).mean(dim=0)
                        final_scores_t = (baseline_scores_t + crop_agg_t) * 0.5

            final_scores = final_scores_t.detach().cpu().tolist()
            pred_i = int(final_scores_t.argmax().detach().cpu().item())
            letters = _option_letters(len(options))
            pred = letters[pred_i] if pred_i < len(letters) else str(pred_i)

            correct = 0
            if isinstance(answer, str) and answer.strip().upper() in set(letters):
                correct = int(pred == answer.strip().upper())
            else:
                correct = int(_safe_str(pred).strip() == _safe_str(answer).strip())

            correct_n += int(correct)
            rows.append(
                {
                    "idx": int(idx),
                    "split": split,
                    "answer": answer,
                    "pred": pred,
                    "correct": int(correct),
                    "triggered": int(triggered),
                    "margin": float(margin),
                    "top1": float(top1),
                    "top2": float(top2),
                    "n_rois": int(len(boxes)),
                    "baseline_scores_json": json.dumps(baseline_scores, ensure_ascii=False),
                    "final_scores_json": json.dumps(final_scores, ensure_ascii=False),
                    "model_id": model_id,
                    "seed": seed,
                    "config_hash": config_hash,
                    "agent_name": agent_name,
                    "device": device,
                }
            )

            trace_f.write(
                json.dumps(
                    {
                        "idx": int(idx),
                        "split": split,
                        "subset_prefix": require_query_prefix if require_query_prefix else "NONE",
                        "triggered": bool(triggered),
                        "margin": float(margin),
                        "pz_info": pz_info,
                        "boxes": [list(map(int, b)) for b in boxes],
                        "baseline_scores": baseline_scores,
                        "final_scores": final_scores,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    finally:
        trace_f.close()

    acc = float(correct_n / len(rows)) if rows else 0.0
    trigger_rate = float(triggered_n / len(rows)) if rows else 0.0

    def _quantiles(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {}
        arr = np.asarray(xs, dtype=np.float64)
        qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        out = {}
        for q in qs:
            out[f"q{int(q*100):02d}"] = float(np.quantile(arr, q))
        return out

    import pandas as pd

    df = pd.DataFrame(
        rows,
        columns=[
            "idx",
            "split",
            "answer",
            "pred",
            "correct",
            "triggered",
            "margin",
            "top1",
            "top2",
            "n_rois",
            "baseline_scores_json",
            "final_scores_json",
            "model_id",
            "seed",
            "config_hash",
            "agent_name",
            "device",
        ],
    )
    df.to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "timestamp": utc_now_iso(),
        "acc": float(acc),
        "trigger_rate": float(trigger_rate),
        "N": int(len(rows)),
        "model_id": model_id,
        "seed": int(seed),
        "config_hash": config_hash,
        "split": split,
        "max_samples": int(max_samples),
        "text_template": text_template,
        "trigger_margin": float(trigger_margin),
        "pz_topk": int(pz_topk),
        "pz_min_box_area": int(pz_min_box_area),
        "pz_pad": float(pz_pad),
        "cr_agg": cr_agg,
        "agent_name": agent_name,
        "subset_filter": f"query_image startswith {require_query_prefix}" if require_query_prefix else "NONE",
        "margin_quantiles": _quantiles(margins),
        "cache_root_rel": _rel(cache_root),
        "out_csv": _rel(out_csv),
        "out_trace": _rel(out_trace),
    }

    out_summary = paths.logs_dir / "agent_pzcr_summary.json"
    out_summary_ts = paths.logs_dir / f"agent_pzcr_summary_{ts}.json"
    write_json(out_summary, summary)
    write_json(out_summary_ts, summary)

    print(f"acc={acc:.6f} trigger_rate={trigger_rate:.3f}")
    print(_rel(out_csv))
    print(_rel(out_summary))
    print(_rel(out_trace))
    print(f"config_hash={config_hash}")
    print(f"cache_root_rel={_rel(cache_root)}")
    print(f"subset_filter={summary['subset_filter']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
