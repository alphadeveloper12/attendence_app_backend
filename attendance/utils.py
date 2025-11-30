# attendance/utils.py
import re
import numpy as np
from PIL import Image, ImageOps
import base64, cv2
from io import BytesIO
# ------------ Matching thresholds ------------
THRESH = 0.75        # cosine similarity accept threshold
MARGIN = 0.05        # best - second_best similarity margin
DEBOUNCE_SEC = 45    # not used here (your flow creates one record/day)

# ------------ Quality gates (face ROI) ------------
MIN_DET_SCORE = 0.60    # detector confidence (0..1)
MIN_FACE_RATIO = 0.05   # face area / image area (>= 5%)
MIN_BLUR = 40.0         # variance of Laplacian on ROI
MIN_BRIGHTNESS = 30.0   # too dark if below
MAX_BRIGHTNESS = 230.0  # too bright if above

DATAURL_RE = re.compile(r"^data:image\/(jpeg|jpg|png);base64,", re.IGNORECASE)

def dataurl_to_bytes(dataurl: str) -> bytes:
    """Strip data URL header and base64-decode safely."""
    payload = DATAURL_RE.sub("", dataurl or "")
    return base64.b64decode(payload)

def get_image_bytes(file_or_image):
    """Return raw bytes from InMemoryUploadedFile/TemporaryUploadedFile or PIL.Image.Image."""
    if hasattr(file_or_image, "read"):  # UploadedFile
        file_or_image.seek(0)
        return file_or_image.read()
    if isinstance(file_or_image, Image.Image):  # PIL Image
        buf = BytesIO()
        file_or_image.save(buf, format="JPEG")
        return buf.getvalue()
    raise ValueError(f"Unsupported image type: {type(file_or_image)}")

def pil_to_bgr_array_from_bytes(raw_bytes: bytes):
    """Convert raw bytes → OpenCV BGR numpy array. Safe for PNG/JPG."""
    try:
        image = Image.open(BytesIO(raw_bytes))
        image = image.convert("RGB")  # ensure RGB
        arr = np.array(image)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as e:
        raise ValueError(f"Invalid image bytes: {e}")
