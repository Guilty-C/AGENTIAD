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

    from agentiad_repro.utils import get_env_snapshot, load_paths, write_json

    paths = load_paths(project_root)
    env = get_env_snapshot()
    env["paths"] = {k: str(getattr(paths, k)) for k in paths.__dataclass_fields__.keys()}

    try:
        import numpy as np

        selfcheck = {
            "path": paths.project_root,
            "np_int": np.int64(7),
            "np_float": np.float32(1.25),
            "np_bool": np.bool_(True),
            "tuple": (1, 2),
            "set": {"a", "b"},
        }
        write_json(paths.logs_dir / "json_encoder_selfcheck.json", selfcheck)
        env["json_encoder_selfcheck_ok"] = True
    except Exception as e:
        env["json_encoder_selfcheck_ok"] = False
        env["json_encoder_selfcheck_error"] = str(e)

    out_path = paths.logs_dir / "env.json"
    write_json(out_path, env)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
