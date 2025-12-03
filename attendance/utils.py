# attendance/utils.py
from __future__ import annotations

import base64
import re
from io import BytesIO
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np
from PIL import Image

# ------------ Matching thresholds ------------
THRESH: float = 0.75        # cosine similarity accept threshold
MARGIN: float = 0.05        # best - second_best similarity margin
DEBOUNCE_SEC: int = 45      # not used here

# ------------ Quality gates (face ROI) ------------
# Slightly more lenient than before
MIN_DET_SCORE: float = 0.50    # was 0.60 - allow slightly weaker detections
MIN_FACE_RATIO: float = 0.03   # was 0.05 - allow smaller faces (~3% of image)
MIN_BLUR: float = 25.0         # was 40.0 - allow softer images
MIN_BRIGHTNESS: float = 20.0   # was 30.0 - allow darker images
MAX_BRIGHTNESS: float = 240.0  # was 230.0 - allow slightly brighter

DATAURL_RE = re.compile(
    r"^data:image\/(jpeg|jpg|png);base64,",
    re.IGNORECASE,
)


def dataurl_to_bytes(dataurl: str) -> bytes:
    """
    Strip data URL header and base64-decode safely.
    """
    payload = DATAURL_RE.sub("", dataurl or "")
    return base64.b64decode(payload)


def get_image_bytes(file_or_image: Any) -> bytes:
    """
    Return raw bytes from InMemoryUploadedFile/TemporaryUploadedFile or PIL.Image.Image.
    """
    if hasattr(file_or_image, "read"):  # UploadedFile
        file_or_image.seek(0)
        return file_or_image.read()

    if isinstance(file_or_image, Image.Image):  # PIL Image
        buf = BytesIO()
        file_or_image.save(buf, format="JPEG")
        return buf.getvalue()

    raise ValueError(f"Unsupported image type: {type(file_or_image)}")


def pil_to_bgr_array_from_bytes(raw_bytes: bytes) -> np.ndarray:
    """
    Convert raw bytes → OpenCV BGR numpy array. Safe for PNG/JPG.
    """
    try:
        image = Image.open(BytesIO(raw_bytes))
        image = image.convert("RGB")  # ensure RGB
        arr = np.array(image)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid image bytes: {exc}") from exc
