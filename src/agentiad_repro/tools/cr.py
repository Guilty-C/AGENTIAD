# [REMOTE EXECUTION DETECTED? CHECK GUIDELINES]
# This project enforces a strict "Zero-Pollution" remote execution protocol for shared lab servers.
# See REMOTE_EXECUTION_GUIDE.txt for the mandatory "Upload -> Tmp Run -> Cleanup" workflow.

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple


def query_image(
    class_name: str,
    normal_pool: Sequence[Mapping[str, Any]],
    seed: int,
    sample_id: str,
    out_dir: Path,
) -> Tuple[str, str]:
    try:
        from PIL import Image as PILImage
    except Exception as e:
        raise RuntimeError("Pillow is required.") from e

    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not normal_pool:
        raise ValueError("normal_pool is empty")

    class_name = "" if class_name is None else str(class_name)
    seed_i = int(seed)
    sample_id = "" if sample_id is None else str(sample_id)

    candidates = [x for x in normal_pool if str(x.get("class_name", "")) == class_name]
    if not candidates:
        candidates = list(normal_pool)

    h = hashlib.sha256(f"{seed_i}|{sample_id}".encode("utf-8")).digest()
    sub_seed = int.from_bytes(h[:8], byteorder="big", signed=False) % 1_000_000_000
    rng = random.Random(sub_seed)
    chosen = candidates[int(rng.randrange(len(candidates)))]

    ref_sample_id = str(chosen.get("sample_id") or chosen.get("id") or chosen.get("idx") or "")
    img = chosen.get("image")
    if img is None and chosen.get("image_path"):
        img = PILImage.open(str(chosen.get("image_path")))
    if img is None:
        raise ValueError("normal_pool item must contain 'image' or 'image_path'")

    if hasattr(img, "convert"):
        img = img.convert("RGB")

    ref_path = out_dir / "ref.png"
    img.save(ref_path)
    return str(ref_path), ref_sample_id
