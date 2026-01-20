from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple


def crop_image_normalized(
    bbox_2d: Sequence[float],
    image: "PIL.Image.Image",
    out_dir: Path,
) -> Tuple[str, List[float]]:
    try:
        from PIL import Image as PILImage
    except Exception as e:
        raise RuntimeError("Pillow is required.") from e

    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b = list(bbox_2d) if bbox_2d is not None else None
    if b is None or len(b) != 4:
        raise ValueError("bbox_2d must be length-4: [x1,y1,x2,y2]")

    x1, y1, x2, y2 = [float(x) for x in b]
    for v in (x1, y1, x2, y2):
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"bbox values must be in [0,1], got {bbox_2d}")
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"bbox must satisfy x1<x2 and y1<y2, got {bbox_2d}")

    if image is None or not hasattr(image, "size"):
        raise ValueError("image must be a PIL.Image")
    if not isinstance(image, PILImage.Image):
        raise ValueError("image must be a PIL.Image")

    w, h = image.size
    px1 = int(round(x1 * w))
    py1 = int(round(y1 * h))
    px2 = int(round(x2 * w))
    py2 = int(round(y2 * h))

    if px1 < 0:
        px1 = 0
    if py1 < 0:
        py1 = 0
    if px2 > w:
        px2 = w
    if py2 > h:
        py2 = h

    if px2 <= px1:
        px2 = min(w, px1 + 1)
    if py2 <= py1:
        py2 = min(h, py1 + 1)

    crop = image.crop((px1, py1, px2, py2))
    crop_path = out_dir / "crop.png"
    crop.save(crop_path)
    return str(crop_path), [x1, y1, x2, y2]
