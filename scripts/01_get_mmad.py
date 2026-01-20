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

    from agentiad_repro.utils import load_paths, utc_now_iso, write_json, write_jsonl

    paths = load_paths(project_root)

    dataset = [
        {
            "id": "mmad_dummy_0001",
            "task": "arithmetic",
            "question": "1 + 1 = ?",
            "answer": "2",
        },
        {
            "id": "mmad_dummy_0002",
            "task": "string",
            "question": "Return the first letter of 'apple'.",
            "answer": "a",
        },
        {
            "id": "mmad_dummy_0003",
            "task": "logic",
            "question": "If all cats are mammals and Tom is a cat, Tom is a ____.",
            "answer": "mammal",
        },
    ]

    mmad_jsonl = paths.mmad_dir / "mmad_dummy.jsonl"
    write_jsonl(mmad_jsonl, dataset)

    meta = {"timestamp_utc": utc_now_iso(), "items": len(dataset), "mmad_jsonl": str(mmad_jsonl)}
    meta_path = paths.logs_dir / "mmad_download.json"
    write_json(meta_path, meta)

    print(str(mmad_jsonl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

