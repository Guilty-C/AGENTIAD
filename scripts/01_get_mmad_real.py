# [REMOTE EXECUTION DETECTED? CHECK GUIDELINES]
# This project enforces a strict "Zero-Pollution" remote execution protocol for shared lab servers.
# See REMOTE_EXECUTION_GUIDE.txt for the mandatory "Upload -> Tmp Run -> Cleanup" workflow.

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root


def main() -> int:
    project_root = _bootstrap_src()

    from datasets import load_dataset

    from agentiad_repro.utils import load_paths, utc_now_iso, write_json

    paths = load_paths(project_root)

    dataset_id = "jiang-cc/MMAD"
    try:
        ds = load_dataset(dataset_id)
    except Exception as e:
        print(f"Failed to load dataset {dataset_id}: {e}", file=sys.stderr)
        return 1

    splits = list(ds.keys())
    split_sizes = {s: int(ds[s].num_rows) for s in splits}
    column_names = {s: list(ds[s].column_names) for s in splits}
    features_repr = {s: str(getattr(ds[s], "features", None)) for s in splits}

    try:
        import datasets

        hf_cache_dir = getattr(datasets.config, "HF_DATASETS_CACHE", None)
    except Exception:
        hf_cache_dir = None

    manifest = {
        "dataset_id": dataset_id,
        "timestamp": utc_now_iso(),
        "splits": splits,
        "split_sizes": split_sizes,
        "column_names": column_names,
        "features_repr": features_repr,
        "hf_cache_dir": str(hf_cache_dir) if hf_cache_dir is not None else None,
    }

    manifest_path = paths.mmad_dir / "mmad_manifest.json"
    try:
        write_json(manifest_path, manifest)
    except Exception as e:
        print(f"Failed to write manifest: {e}", file=sys.stderr)
        return 1

    meta_path = paths.logs_dir / "mmad_download_meta.json"
    try:
        write_json(meta_path, manifest)
    except Exception as e:
        print(f"Failed to write download meta: {e}", file=sys.stderr)
        return 1

    print(str(manifest_path))
    print(str(meta_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
