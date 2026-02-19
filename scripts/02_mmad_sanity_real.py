# [REMOTE EXECUTION DETECTED? CHECK GUIDELINES]
# This project enforces a strict "Zero-Pollution" remote execution protocol for shared lab servers.
# See REMOTE_EXECUTION_GUIDE.txt for the mandatory "Upload -> Tmp Run -> Cleanup" workflow.

from __future__ import annotations

import io
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _bootstrap_src() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root


def _first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def _find_image_field(ds_split) -> Optional[str]:
    columns = list(ds_split.column_names)
    if "image" in columns:
        return "image"

    features = getattr(ds_split, "features", None)
    if features is None:
        return None

    try:
        from datasets import Image
    except Exception:
        return None

    for name, feat in features.items():
        if isinstance(feat, Image):
            return name
    return None


def _decode_image(value: Any):
    try:
        from PIL import Image as PILImage
    except Exception as e:
        raise RuntimeError("Pillow is required to decode images.") from e

    if value is None:
        raise ValueError("image value is None")

    if hasattr(value, "save"):
        return value

    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            return PILImage.open(io.BytesIO(value["bytes"]))
        if "path" in value and value["path"]:
            return PILImage.open(value["path"])

    raise TypeError(f"Unsupported image type: {type(value)}")


def _decode_image_from_dataset_path(dataset_id: str, value: Any):
    try:
        from PIL import Image as PILImage
    except Exception as e:
        raise RuntimeError("Pillow is required to decode images.") from e

    mmad_root = os.environ.get("MMAD_ROOT", "").strip()

    if isinstance(value, str):
        rel = value.replace("\\", "/")
        if mmad_root and rel.startswith(("DS-MVTec/", "MVTec-AD/")):
            local_abs = Path(mmad_root) / Path(*rel.split("/"))
            if local_abs.exists():
                return PILImage.open(local_abs)

        p = Path(value)
        if p.exists():
            return PILImage.open(p)

        if value.startswith("DS-MVTec/"):
            try:
                from huggingface_hub import hf_hub_download

                rev = os.environ.get("MMAD_ASSET_REVISION") or "f1ad11c07452dff1e023a0df1093e40701d22cab"
                local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value, revision=rev)
                return PILImage.open(local_path)
            except Exception as e:
                raise RuntimeError(f"Failed to fetch DS-MVTec asset from hub: {value}") from e

        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError

            try:
                local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value)
            except EntryNotFoundError:
                fallback_rev = os.environ.get("MMAD_ASSET_REVISION") or "f1ad11c07452dff1e023a0df1093e40701d22cab"
                local_path = hf_hub_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    filename=value,
                    revision=fallback_rev,
                )
            return PILImage.open(local_path)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch image file from hub: {value}") from e

    return _decode_image(value)


def _count_column(ds_split, field: str, max_rows: Optional[int] = None) -> Counter:
    c: Counter = Counter()
    limit = int(max_rows) if max_rows is not None else None
    seen = 0
    for batch in ds_split.iter(batch_size=1024):
        vals = batch.get(field, [])
        for v in vals:
            c[str(v)] += 1
        seen += len(vals)
        if limit is not None and seen >= limit:
            break
    return c


def main() -> int:
    project_root = _bootstrap_src()

    from datasets import load_dataset

    from agentiad_repro.utils import ensure_dir, load_paths, utc_now_iso, write_json

    paths = load_paths(project_root)

    dataset_id = "jiang-cc/MMAD"
    ds = load_dataset(dataset_id)
    splits = list(ds.keys())
    split_sizes = {s: int(ds[s].num_rows) for s in splits}

    if "train" in splits:
        preferred_split = "train"
    elif "test" in splits:
        preferred_split = "test"
    else:
        preferred_split = splits[0]
    d0 = ds[preferred_split]

    columns0 = list(d0.column_names)
    query_field = "query_image" if "query_image" in columns0 else _find_image_field(d0)
    template_field = "template_image" if "template_image" in columns0 else None
    mask_field = "mask" if "mask" in columns0 else None

    label_field = _first_existing(columns0, ["label", "anomaly", "is_anomaly"])
    category_field = _first_existing(columns0, ["category", "class_name", "object", "cls"])

    label_dist_by_split: Dict[str, Dict[str, int]] = {}
    category_dist_by_split: Dict[str, Dict[str, int]] = {}
    for s in splits:
        if label_field is not None and label_field in ds[s].column_names:
            label_dist_by_split[s] = dict(_count_column(ds[s], label_field))
        if category_field is not None and category_field in ds[s].column_names:
            category_dist_by_split[s] = dict(_count_column(ds[s], category_field))

    category_top20: Dict[str, List[Tuple[str, int]]] = {}
    for s, dist in category_dist_by_split.items():
        top = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        category_top20[s] = [(k, int(v)) for k, v in top]

    samples_dir = ensure_dir(paths.traces_dir / "mmad_samples")
    for p in samples_dir.glob("*.png"):
        try:
            p.unlink()
        except Exception:
            pass
    saved_samples: List[Dict[str, Any]] = []
    saved_query_count = 0
    rng = random.Random(0)
    n_total = int(d0.num_rows)
    candidates = list(range(n_total))
    rng.shuffle(candidates)

    max_attempts = 2000
    attempts = 0
    for idx in candidates:
        if saved_query_count >= 5:
            break
        attempts += 1
        if attempts > max_attempts:
            break

        row = d0[int(idx)]
        files: List[str] = []
        if query_field is None:
            continue

        try:
            img = _decode_image_from_dataset_path(dataset_id, row.get(query_field))
            out_path = samples_dir / f"{preferred_split}_{int(idx)}_query.png"
            img.save(out_path)
            files.append(str(out_path))
            saved_query_count += 1
        except Exception:
            continue

        if template_field is not None:
            try:
                img = _decode_image_from_dataset_path(dataset_id, row.get(template_field))
                out_path = samples_dir / f"{preferred_split}_{int(idx)}_template.png"
                img.save(out_path)
                files.append(str(out_path))
            except Exception:
                pass

        if mask_field is not None:
            try:
                img = _decode_image_from_dataset_path(dataset_id, row.get(mask_field))
                out_path = samples_dir / f"{preferred_split}_{int(idx)}_mask.png"
                img.save(out_path)
                files.append(str(out_path))
            except Exception:
                pass

        saved_samples.append({"split": preferred_split, "idx": int(idx), "files": files})

    sanity = {
        "timestamp_utc": utc_now_iso(),
        "dataset_id": dataset_id,
        "splits": splits,
        "split_sizes": split_sizes,
        "preferred_split": preferred_split,
        "samples_dir": str(samples_dir),
        "detected_fields": {
            "query_image_field": query_field,
            "template_image_field": template_field,
            "mask_field": mask_field,
            "label_field": label_field,
            "category_field": category_field,
        },
        "label_distribution": label_dist_by_split,
        "category_top20": {s: [{"category": k, "count": v} for k, v in top] for s, top in category_top20.items()},
        "samples": saved_samples,
        "ok": True if (saved_query_count >= 5) else False,
    }

    out_path = paths.logs_dir / "mmad_sanity_real.json"
    write_json(out_path, sanity)
    print(str(out_path))
    return 0 if sanity["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
