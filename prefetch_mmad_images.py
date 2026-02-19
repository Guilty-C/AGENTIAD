import os
from datasets import load_dataset

# 强制使用你指定的缓存
os.environ["HF_HOME"] = r"D:\hf_offline"

# 注意：你 Phase1 用的是 split=test，但日志里 sample_id 叫 train_xxx
# 这里建议先 prefetch train（量最大），再 prefetch test
splits = ["train", "test"]

# MMAD 里常见候选字段名（你日志里出现 query_image）
CANDIDATE_FIELDS = ["query_image", "image", "img", "rgb", "image_path"]

def touch_image(x):
    # datasets 的 Image feature：访问一次就会触发下载/解码
    # 这里只做“访问”，不保存、不推理
    if x is None:
        return False
    # 有的直接是 PIL.Image；有的是 dict/bytes
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
import os
from pathlib import Path

_MMAD_ROOT_LOGGED = False


def resolve_mmad_root() -> Path:
    """
    Resolve MMAD root with SSOT precedence:
    1) MMAD_ROOT
    2) legacy keys (MMAD_DATA_ROOT, DATASET_ROOT, DATA_ROOT)
    3) default: <repo_root>/data/MMAD_ROOT
    """
    for key in ("MMAD_ROOT", "MMAD_DATA_ROOT", "DATASET_ROOT", "DATA_ROOT"):
        value = os.environ.get(key)
        if value and value.strip():
            return Path(value).expanduser()
    return Path(__file__).resolve().parent / "data" / "MMAD_ROOT"


def mmad_path(relative_path: str) -> Path:
    """
    Join MMAD-relative dataset entries against the resolved MMAD root.
    Example entries:
      - DS-MVTec/bottle/image/...
      - MVTec-AD/bottle/train/good/...
    """
    global _MMAD_ROOT_LOGGED
    root = resolve_mmad_root()
    if not _MMAD_ROOT_LOGGED:
        print(f"[MMAD] Using MMAD_ROOT={root}")
        _MMAD_ROOT_LOGGED = True
    return root / relative_path


# Backward-compatible SSOT constant for existing call sites in this module.
MMAD_ROOT = resolve_mmad_root()


def get_mmad_root() -> Path:
    global _MMAD_ROOT_LOGGED
    if not _MMAD_ROOT_LOGGED:
        print(f"[MMAD] Using MMAD_ROOT={MMAD_ROOT}")
        _MMAD_ROOT_LOGGED = True
    return MMAD_ROOT
