from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


def _bootstrap_src() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root


def main() -> int:
    project_root = _bootstrap_src()

    from agentiad_repro.utils import load_paths, read_jsonl, write_csv

    paths = load_paths(project_root)
    mmad_jsonl = paths.mmad_dir / "mmad_dummy.jsonl"
    items = read_jsonl(mmad_jsonl)

    rows = []
    for it in items:
        gt = str(it.get("answer", ""))
        pred = "UNKNOWN"
        rows.append(
            {
                "question_id": it.get("id", ""),
                "task": it.get("task", ""),
                "prediction": pred,
                "ground_truth": gt,
                "correct": int(pred.strip().lower() == gt.strip().lower()),
            }
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = paths.tables_dir / f"baseline_dummy_{ts}.csv"
    write_csv(
        out_path,
        rows=rows,
        fieldnames=["question_id", "task", "prediction", "ground_truth", "correct"],
    )
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

