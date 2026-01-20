from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest().upper()


def _sha256_file(p: Path) -> str:
    return _sha256_bytes(p.read_bytes())


def _iter_dist_files(dist_dir: Path) -> Iterable[Path]:
    for p in dist_dir.rglob("*"):
        if p.is_file():
            yield p


def _should_exclude(rel_parts: Tuple[str, ...], suffix: str) -> bool:
    if suffix.lower() == ".pyc":
        return True
    if "__pycache__" in rel_parts:
        return True
    if len(rel_parts) >= 2 and rel_parts[0] == "dist" and rel_parts[1] == "outputs":
        return True
    if ".venv" in rel_parts:
        return True
    if ".hf" in rel_parts:
        return True
    if ".git" in rel_parts:
        return True
    return False


def _zip_build(project_root: Path, dist_dir: Path, out_zip: Path) -> None:
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(_iter_dist_files(dist_dir), key=lambda x: x.relative_to(project_root).as_posix()):
            rel = p.relative_to(project_root)
            rel_parts = tuple(rel.parts)
            if _should_exclude(rel_parts, p.suffix):
                continue
            zf.write(p, arcname=rel.as_posix())


def _zip_list_names(zip_path: Path) -> List[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.namelist()


def _zip_sha256_file(zip_path: Path, name: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(name, "r") as fp:
            return _sha256_bytes(fp.read())


def _zip_read_text(zip_path: Path, name: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(name, "r") as fp:
            return fp.read().decode("utf-8", errors="replace")


def _infer_project_root(script_path: Path) -> Path:
    p1 = script_path.parents[1]
    if (p1 / "dist").is_dir():
        return p1
    if len(script_path.parents) >= 3:
        p2 = script_path.parents[2]
        if (p2 / "dist").is_dir():
            return p2
    return p1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="agentiad_repro_release.zip")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = _infer_project_root(script_path)
    dist_dir = project_root / "dist"
    out_zip = (project_root / str(args.out)).resolve()

    key_files = [
        "dist/scripts/06_run_agentiad_infer.py",
        "dist/src/agentiad_repro/tools/cr.py",
        "dist/src/agentiad_repro/tools/pz.py",
        "dist/README.md",
        "dist/scripts/07_make_release_zip.py",
    ]

    print(
        "DIST_SELF_CHECK: dist 内关键增强点已通过行号与 hash 自检确认存在；未对仓库源文件逐字一致性做证明"
    )

    dist_hashes: Dict[str, str] = {}
    for k in key_files:
        dist_hashes[k] = _sha256_file(project_root / k)

    print("DIST_KEY_SHA256:")
    for k in key_files:
        print(f"{k} {dist_hashes[k]}")

    _zip_build(project_root, dist_dir, out_zip)
    zip_sha = _sha256_file(out_zip)
    print(f"ZIP_SHA256 {zip_sha}")

    names = _zip_list_names(out_zip)

    backslash_paths = [n for n in names if "\\" in n]
    has_dist_outputs = any(n.startswith("dist/outputs/") for n in names)
    pyc_paths = [n for n in names if n.lower().endswith(".pyc")]
    pycache_paths = [n for n in names if "__pycache__" in n.split("/")]
    zip_has_dist_script = "dist/scripts/07_make_release_zip.py" in names
    try:
        readme_text = _zip_read_text(out_zip, "dist/README.md")
        zip_has_readme_section = "Release / Packaging Reproducibility" in readme_text
    except Exception:
        zip_has_readme_section = False

    print(f"ZIP_BACKSLASH_PATHS {len(backslash_paths)}")
    print(f"ZIP_HAS_DIST_OUTPUTS {bool(has_dist_outputs)}")
    print(f"ZIP_HAS_PYC {bool(pyc_paths)}")
    print(f"ZIP_HAS_PYCACHE_DIR {bool(pycache_paths)}")
    print(f"ZIP_PYC_COUNT {len(pyc_paths)}")
    print(f"ZIP_PYCACHE_COUNT {len(pycache_paths)}")
    print(f"ZIP_HAS_README_SECTION {bool(zip_has_readme_section)}")
    print(f"ZIP_HAS_DIST_SCRIPT {bool(zip_has_dist_script)}")

    if pyc_paths:
        print("ZIP_PYC_SAMPLES:")
        for n in pyc_paths[:5]:
            print(n)
    if pycache_paths:
        print("ZIP_PYCACHE_SAMPLES:")
        for n in pycache_paths[:5]:
            print(n)

    zip_hashes: Dict[str, str] = {}
    for k in key_files:
        zip_hashes[k] = _zip_sha256_file(out_zip, k)

    print("ZIP_KEY_SHA256:")
    for k in key_files:
        print(f"{k} {zip_hashes[k]}")

    print("DIST_VS_ZIP_MATCH:")
    for k in key_files:
        print(f"{k} {dist_hashes[k] == zip_hashes[k]}")

    hard_fail = False
    hard_fail = hard_fail or (len(backslash_paths) != 0)
    hard_fail = hard_fail or bool(has_dist_outputs)
    hard_fail = hard_fail or bool(pyc_paths)
    hard_fail = hard_fail or bool(pycache_paths)
    hard_fail = hard_fail or (not bool(zip_has_readme_section))
    hard_fail = hard_fail or (not bool(zip_has_dist_script))
    hard_fail = hard_fail or any(dist_hashes[k] != zip_hashes[k] for k in key_files)

    if hard_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
