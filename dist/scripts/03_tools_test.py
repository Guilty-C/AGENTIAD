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

    from agentiad_repro.utils import load_paths, utc_now_iso, write_json

    paths = load_paths(project_root)

    trace = {
        "timestamp_utc": utc_now_iso(),
        "checks": [
            {"name": "paths_exist", "ok": paths.logs_dir.exists() and paths.tables_dir.exists()},
            {"name": "can_write_trace", "ok": True},
        ],
    }

    out_path = paths.traces_dir / "tools_trace.json"
    write_json(out_path, trace)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

