import os
from pathlib import Path

from datasets import load_dataset

os.environ["HF_HOME"] = r"D:\hf_offline"

splits = ["train", "test"]

MMAD_IMAGE_FIELDS = ["query_image", "template_image", "mask"]
CANDIDATE_FIELDS = MMAD_IMAGE_FIELDS + ["image", "img", "rgb", "image_path"]
_MMAD_ROOT = None
_MMAD_ROOT_LOGGED = False


def get_mmad_root() -> Path:
    global _MMAD_ROOT, _MMAD_ROOT_LOGGED
    if _MMAD_ROOT is None:
        for key in ("MMAD_ROOT", "MMAD_DATA_ROOT", "DATASET_ROOT", "DATA_ROOT"):
            value = os.environ.get(key, "").strip()
            if value:
                _MMAD_ROOT = Path(value).expanduser()
                break
        if _MMAD_ROOT is None:
            _MMAD_ROOT = Path(__file__).resolve().parent / "data" / "MMAD_ROOT"
    if not _MMAD_ROOT_LOGGED:
        print(f"[MMAD] MMAD_ROOT={_MMAD_ROOT}")
        _MMAD_ROOT_LOGGED = True
    return _MMAD_ROOT


def resolve_mmad_image_path(path_value):
    if not isinstance(path_value, str) or not path_value:
        return None
    if path_value.startswith(("DS-MVTec/", "MVTec-AD/")):
        rel = Path(*path_value.split("/"))
        return get_mmad_root() / rel
    return None


def touch_mmad_image(x) -> bool:
    path_value = x.get("path") if isinstance(x, dict) else x
    abs_path = resolve_mmad_image_path(path_value)
    if abs_path is None or not abs_path.exists():
        return False
    _ = abs_path.stat().st_size
    return True


def touch_image(x) -> bool:
    if x is None:
        return False
    _ = x
    return True


for split in splits:
    print(f"[prefetch] loading split={split} ...")
    ds = load_dataset("jiang-cc/MMAD", split=split)

    n = len(ds)
    ok = 0
    miss = 0

    for i in range(n):
        row = ds[i]
        hit = False
        for f in CANDIDATE_FIELDS:
            if f in row and row[f] is not None:
                try:
                    if f in MMAD_IMAGE_FIELDS and touch_mmad_image(row[f]):
                        hit = True
                        break
                    touch_image(row[f])
                    hit = True
                    break
                except Exception:
                    pass
        if hit:
            ok += 1
        else:
            miss += 1

        if (i + 1) % 1000 == 0:
            print(f"[prefetch] {split}: {i+1}/{n}, ok={ok}, miss={miss}")

    print(f"[prefetch] done split={split}, ok={ok}, miss={miss}")
