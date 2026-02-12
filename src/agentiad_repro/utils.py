# [REMOTE EXECUTION DETECTED? CHECK GUIDELINES]
# This project enforces a strict "Zero-Pollution" remote execution protocol for shared lab servers.
# See REMOTE_EXECUTION_GUIDE.txt for the mandatory "Upload -> Tmp Run -> Cleanup" workflow.

from __future__ import annotations

import csv
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

try:
    import yaml
except ImportError:
    raise RuntimeError("PyYAML missing. Install: python -m pip install pyyaml (or conda install pyyaml)")


@dataclass(frozen=True)
class ResolvedPaths:
    project_root: Path
    outputs_dir: Path
    logs_dir: Path
    tables_dir: Path
    traces_dir: Path
    data_dir: Path
    mmad_dir: Path
    trajectories_dir: Path
    cache_dir: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (set, tuple)):
        return list(obj)

    try:
        import numpy as np

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    except Exception:
        pass

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def load_paths(project_root: Path, config_path: Optional[Path] = None) -> ResolvedPaths:
    if config_path is None:
        config_path = project_root / "configs" / "paths.yaml"

    data: Dict[str, Any] = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    paths = (data.get("paths") or {}) if isinstance(data, dict) else {}

    def r(rel: str) -> Path:
        return (project_root / rel).resolve()

    outputs_dir = r(paths.get("outputs_dir", "outputs"))
    logs_dir = r(paths.get("logs_dir", "outputs/logs"))
    tables_dir = r(paths.get("tables_dir", "outputs/tables"))
    traces_dir = r(paths.get("traces_dir", "outputs/traces"))
    data_dir = r(paths.get("data_dir", "data"))
    mmad_dir = r(paths.get("mmad_dir", "data/mmad"))
    trajectories_dir = r(paths.get("trajectories_dir", "data/trajectories"))
    cache_dir = r(paths.get("cache_dir", "data/cache"))

    for p in [
        outputs_dir,
        logs_dir,
        tables_dir,
        traces_dir,
        data_dir,
        mmad_dir,
        trajectories_dir,
        cache_dir,
    ]:
        ensure_dir(p)

    return ResolvedPaths(
        project_root=project_root.resolve(),
        outputs_dir=outputs_dir,
        logs_dir=logs_dir,
        tables_dir=tables_dir,
        traces_dir=traces_dir,
        data_dir=data_dir,
        mmad_dir=mmad_dir,
        trajectories_dir=trajectories_dir,
        cache_dir=cache_dir,
    )


def get_env_snapshot() -> Dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "cwd": str(Path.cwd().resolve()),
        "env": {
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        return items
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(dict(it), ensure_ascii=False) + "\n")


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
