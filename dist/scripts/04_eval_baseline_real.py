# [REMOTE EXECUTION DETECTED? CHECK GUIDELINES]
# This project enforces a strict "Zero-Pollution" remote execution protocol for shared lab servers.
# See REMOTE_EXECUTION_GUIDE.txt for the mandatory "Upload -> Tmp Run -> Cleanup" workflow.

from __future__ import annotations

import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _bootstrap_src() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root


def _first_existing(columns, candidates):
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def _infer_label_mapping(ds_split, label_field: str) -> Tuple[Dict[Any, int], str]:
    features = getattr(ds_split, "features", None) or {}
    feat = features.get(label_field)

    if feat is not None and hasattr(feat, "names") and isinstance(getattr(feat, "names", None), list):
        names = [str(x) for x in feat.names]
        normal_ids = [i for i, n in enumerate(names) if n.strip().lower() in {"normal", "ok", "good"}]
        if len(normal_ids) == 1:
            mapping = {i: (0 if i == normal_ids[0] else 1) for i in range(len(names))}
            return mapping, f"classlabel_names={names}, normal_id={normal_ids[0]}"
        mapping = {i: (0 if i == 0 else 1) for i in range(len(names))}
        return mapping, f"classlabel_names={names}, normal_id=0"

    mapping: Dict[Any, int] = {}
    rule = "fallback_mapping=first_seen_is_normal(0), others=1"
    return mapping, rule


def _to_binary_label(y: Any, mapping: Dict[Any, int]) -> int:
    if y is None:
        return 0
    if isinstance(y, bool):
        return int(bool(y))
    if isinstance(y, int):
        if y in (0, 1):
            return int(y)
        if y in mapping:
            return int(mapping[y])
        return int(1 if y != 0 else 0)
    if isinstance(y, str):
        s = y.strip().lower()
        if s in {"0", "normal", "ok", "good", "negative"}:
            return 0
        if s in {"1", "anomaly", "abnormal", "defect", "positive"}:
            return 1
    if y in mapping:
        return int(mapping[y])
    return 0


def main() -> int:
    project_root = _bootstrap_src()

    import pandas as pd
    from datasets import load_dataset

    from agentiad_repro.utils import load_paths, utc_now_iso, write_json

    paths = load_paths(project_root)

    dataset_id = "jiang-cc/MMAD"
    ds = load_dataset(dataset_id)
    splits = list(ds.keys())
    split = "test" if "test" in splits else splits[0]
    d0 = ds[split]

    columns0 = list(d0.column_names)
    label_field = _first_existing(columns0, ["label", "anomaly", "is_anomaly"])
    answer_field = "answer" if "answer" in columns0 else None
    options_field = "options" if "options" in columns0 else None
    label_field_used = label_field if label_field is not None else answer_field

    if label_field_used is None:
        raise RuntimeError(f"Cannot find label or answer field in columns: {d0.column_names}")

    label_mapping, mapping_rule = ({}, "not_used")
    if label_field is not None:
        label_mapping, mapping_rule = _infer_label_mapping(d0, label_field)

    max_n = 200
    n_total = int(d0.num_rows)
    n = min(max_n, n_total)
    rng = random.Random(0)
    indices = list(range(n_total))
    rng.shuffle(indices)
    indices = sorted(indices[:n])

    rows = []
    correct_n = 0
    for idx in indices:
        row = d0[int(idx)]
        if label_field is not None:
            y_raw = row.get(label_field)
            y = _to_binary_label(y_raw, label_mapping)
        else:
            y = row.get(answer_field) if answer_field is not None else None

        if isinstance(y, int):
            yhat = 0
        elif isinstance(y, str):
            opts = row.get(options_field) if options_field is not None else None
            if isinstance(opts, list) and len(opts) > 0:
                yhat = opts[0]
            else:
                yhat = "A"
        else:
            yhat = None

        if yhat is None or y is None:
            correct = 0
        elif isinstance(y, int) and isinstance(yhat, int):
            correct = int(yhat == y)
        else:
            correct = int(str(yhat).strip() == str(y).strip())
        correct_n += correct
        rows.append(
            {
                "split": split,
                "idx": int(idx),
                "y": y,
                "yhat": yhat,
                "correct": int(correct),
            }
        )

    acc = float(correct_n / len(rows)) if rows else 0.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = paths.tables_dir / f"baseline_real_{ts}.csv"
    df = pd.DataFrame(rows, columns=["split", "idx", "y", "yhat", "correct"])
    df.to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "timestamp": utc_now_iso(),
        "split": split,
        "N": int(len(rows)),
        "acc": acc,
        "label_field_used": label_field_used,
    }

    summary_path = paths.logs_dir / "baseline_real_summary.json"
    write_json(summary_path, summary)

    print(f"acc={acc:.6f}")
    print(str(out_csv))
    print(str(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
