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

    from agentiad_repro.utils import load_paths, read_jsonl, utc_now_iso, write_json

    paths = load_paths(project_root)
    mmad_jsonl = paths.mmad_dir / "mmad_dummy.jsonl"
    items = read_jsonl(mmad_jsonl)

    ids = [it.get("id") for it in items]
    unique_ids = len(set(ids))

    sanity = {
        "timestamp_utc": utc_now_iso(),
        "mmad_jsonl": str(mmad_jsonl),
        "exists": mmad_jsonl.exists(),
        "items": len(items),
        "unique_ids": unique_ids,
        "all_have_question": all(bool(it.get("question")) for it in items),
        "all_have_answer": all(bool(it.get("answer")) for it in items),
        "ok": mmad_jsonl.exists()
        and len(items) > 0
        and unique_ids == len(items)
        and all(bool(it.get("question")) for it in items)
        and all(bool(it.get("answer")) for it in items),
    }

    out_path = paths.logs_dir / "mmad_sanity.json"
    write_json(out_path, sanity)
    print(str(out_path))
    return 0 if sanity["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

