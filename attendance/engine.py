# attendance/engine.py
import numpy as np
import faiss
from threading import Lock
from typing import Tuple, Dict, Any, Optional

from .utils import (
    MIN_DET_SCORE, MIN_FACE_RATIO, MIN_BLUR, MIN_BRIGHTNESS, MAX_BRIGHTNESS
)

CPU_PROVIDERS = ["CPUExecutionProvider"]

def _as_bgr_uint8_contig(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected HxWx3 BGR image")
    return np.ascontiguousarray(arr)

class FaceEngine:
    """
    InsightFace engine with robust embedding:
      - original + mild upscale
      - 90° and 270° rotation fallbacks (each with mild upscale if needed)
      - quality diagnostics (det_score, face_ratio, blur, brightness)
    """
    def __init__(self, providers=None, det_size=(640, 640)):
        self.lock = Lock()
        self.providers = providers or CPU_PROVIDERS
        self.det_size = det_size

        self._app = None
        self._app_name = None
        self.index = None
        self.ids = []   # [(template_id, employee_id)]
        self.dim = 512

    # ---------------- loader ----------------
    def _load_app_once(self):
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        errors = []
        for name in ("antelopev2", "buffalo_l"):
            try:
                app = FaceAnalysis(name=name, providers=self.providers)
                app.prepare(ctx_id=0, det_size=self.det_size)
                self._app = app
                self._app_name = name
                return
            except Exception as e:
                errors.append(f"{name}: {e!r}")
        raise RuntimeError("Failed to load InsightFace models.\n" + "\n".join(errors))

    @property
    def app(self):
        self._load_app_once()
        return self._app

    # --------------- helpers ----------------
    @staticmethod
    def _lap_var(gray: np.ndarray) -> float:
        try:
            import cv2
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            gx = np.diff(gray.astype(np.float32), axis=1)
            gy = np.diff(gray.astype(np.float32), axis=0)
            return float(np.var(gx) + np.var(gy))

    @staticmethod
    def _rotate_bgr(bgr: np.ndarray, k: int) -> np.ndarray:
        return _as_bgr_uint8_contig(np.rot90(bgr, k=k))

    @staticmethod
    def _upscale_bgr(bgr: np.ndarray, scale: float) -> np.ndarray:
        from PIL import Image
        h, w = bgr.shape[:2]
        rgb = bgr[:, :, ::-1]
        img = Image.fromarray(rgb, mode="RGB").resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        up = np.asarray(img, dtype=np.uint8)[:, :, ::-1]
        return _as_bgr_uint8_contig(up)

    def _analyze_once(self, bgr: np.ndarray):
        bgr = _as_bgr_uint8_contig(bgr)
        faces = self.app.get(bgr)
        if not faces:
            return None, None

        def _area(f):
            x1, y1, x2, y2 = getattr(f, "bbox", [0, 0, 0, 0])
            return max(1.0, float(x2 - x1) * float(y2 - y1))

        faces.sort(key=lambda f: (float(getattr(f, "det_score", 0.0)), _area(f)), reverse=True)
        return faces[0], faces

    def _compute_meta(self, bgr: np.ndarray, face) -> Dict[str, Any]:
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = getattr(face, "bbox", [0, 0, 0, 0])

        # Face ROI with 10% padding for quality metrics
        pad_x = 0.10 * (x2 - x1)
        pad_y = 0.10 * (y2 - y1)
        rx1 = max(0, int(np.floor(x1 - pad_x)))
        ry1 = max(0, int(np.floor(y1 - pad_y)))
        rx2 = min(w, int(np.ceil(x2 + pad_x)))
        ry2 = min(h, int(np.ceil(y2 + pad_y)))
        roi = bgr[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            roi = bgr

        # area
        area_face = max(1.0, float(x2 - x1) * float(y2 - y1))
        area_img  = float(h * w)
        face_ratio = area_face / max(1.0, area_img)
        det_score = float(getattr(face, "det_score", 0.0))

        # brightness & blur on ROI
        rgb_roi = roi[:, :, ::-1]
        gray = np.dot(rgb_roi[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        brightness = float(gray.mean())
        blur = self._lap_var(gray)

        ok, reason = True, None
        # hard gates
        if det_score < MIN_DET_SCORE:
            ok, reason = False, f"det_score<{MIN_DET_SCORE:.2f}"
        elif face_ratio < MIN_FACE_RATIO:
            ok, reason = False, f"face_too_small<{MIN_FACE_RATIO*100:.0f}%"
        elif brightness < MIN_BRIGHTNESS:
            ok, reason = False, f"too_dark<{MIN_BRIGHTNESS:.0f}"
        elif brightness > MAX_BRIGHTNESS:
            ok, reason = False, f"too_bright>{MAX_BRIGHTNESS:.0f}"
        else:
            # blur gate becomes soft if detection is strong
            if blur < MIN_BLUR:
                if det_score >= 0.75:
                    ok, reason = True, f"soft_blurry<{MIN_BLUR:.0f}"
                else:
                    ok, reason = False, f"blurry<{MIN_BLUR:.0f}"

        return {
            "ok": ok,
            "reason": reason,
            "det_score": det_score,
            "face_ratio": face_ratio,
            "blur": blur,
            "brightness": brightness,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "img_size": [int(h), int(w)],
        }

    def _extract_embedding(self, face) -> Optional[np.ndarray]:
        emb = getattr(face, "normed_embedding", None)
        if emb is None or (hasattr(emb, "size") and getattr(emb, "size", 0) == 0):
            emb = getattr(face, "embedding", None)
        if emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(emb))
        if not np.isfinite(n) or n < 1e-6:
            return None
        return emb / n

    # ---------------- public: embed + quality ----------------
    def embed_best_face(self, bgr_image) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
        bgr_image = _as_bgr_uint8_contig(bgr_image)
        h0, w0 = bgr_image.shape[:2]

        attempts = [("orig", bgr_image)]
        if max(h0, w0) < 900:
            attempts.append(("orig_up", self._upscale_bgr(bgr_image, 1.6)))

        rot90 = self._rotate_bgr(bgr_image, 1)
        attempts.append(("rot90", rot90))
        if max(rot90.shape[:2]) < 900:
            attempts.append(("rot90_up", self._upscale_bgr(rot90, 1.6)))

        rot270 = self._rotate_bgr(bgr_image, 3)
        attempts.append(("rot270", rot270))
        if max(rot270.shape[:2]) < 900:
            attempts.append(("rot270_up", self._upscale_bgr(rot270, 1.6)))

        last_meta: Dict[str, Any] = {"ok": False, "reason": "no_face", "img_size": [int(h0), int(w0)]}

        for tag, img in attempts:
            face, _ = self._analyze_once(img)
            if face is None:
                last_meta = {"ok": False, "reason": f"no_face_{tag}", "img_size": [int(img.shape[0]), int(img.shape[1])]}
                continue

            meta = self._compute_meta(img, face)
            emb = self._extract_embedding(face)
            if emb is None:
                meta.update({"ok": False, "reason": "no_embedding", "attempt": tag})
                last_meta = meta
                continue

            # heuristic quality (informational)
            quality = meta["det_score"] * (meta["face_ratio"] ** 0.5) * max(0.5, min(1.5, meta["blur"] / 150.0))
            meta.update({"attempt": tag})
            return emb, float(quality), meta

        return None, 0.0, last_meta

    # ---------------- FAISS gallery ----------------
    def rebuild_index(self, templates):
        with self.lock:
            xs = [t[2].astype("float32") for t in templates]
            if not xs:
                self.index = None
                self.ids = []
                return
            xb = np.vstack(xs)
            self.dim = xb.shape[1]
            index = faiss.IndexFlatIP(self.dim)
            index.add(xb)
            self.index = index
            self.ids = [(t[0], t[1]) for t in templates]

    def search(self, q: np.ndarray, k: int = 10):
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return [], []
            D, I = self.index.search(q[None, :].astype("float32"), k)
        return D[0].tolist(), I[0].tolist()

ENGINE = FaceEngine()
