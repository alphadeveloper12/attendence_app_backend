# attendance/engine.py
from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np

from .utils import (
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
    MIN_BLUR,
    MIN_DET_SCORE,
    MIN_FACE_RATIO,
)

CPU_PROVIDERS: List[str] = ["CPUExecutionProvider"]


def _as_bgr_uint8_contig(arr: np.ndarray) -> np.ndarray:
    """
    Ensure an array is uint8, HxWx3 and contiguous in BGR layout.
    """
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected HxWx3 BGR image")

    return np.ascontiguousarray(arr)


class FaceEngine:
    """
    InsightFace engine with robust embedding:
      - original + mild upscale
      - 90°, 180° and 270° rotation fallbacks (each with mild upscale)
      - quality diagnostics (det_score, face_ratio, blur, brightness)
      - FAISS gallery with incremental updates
    """

    def __init__(
        self,
        providers: Optional[List[str]] = None,
        det_size: Tuple[int, int] = (640, 640),
    ) -> None:
        self.lock = Lock()
        self.providers: List[str] = providers or ["CPUExecutionProvider"]
        self.det_size: Tuple[int, int] = det_size

        self._app: Any = None
        self._app_name: Optional[str] = None

        self.index: Optional[faiss.IndexFlatIP] = None
        self.ids: List[Tuple[int, int]] = []  # [(template_id, employee_id)]
        self.dim: int = 512

        # template_id -> (employee_id, embedding)
        self._templates: Dict[int, Tuple[int, np.ndarray]] = {}

    # ---------------- loader ----------------
    def _load_app_once(self) -> None:
        if self._app is not None:
            return

        from insightface.app import FaceAnalysis

        print("Loading models...")

        errors: List[str] = []
        # Single model for now; extend tuple if you want fallbacks
        for name in ("buffalo_l",):
            print(f"Trying to load {name}...")
            try:
                app = FaceAnalysis(name=name, providers=self.providers)
                print(f"Model {name} loaded successfully.")
                app.prepare(ctx_id=0, det_size=self.det_size)
                self._app = app
                self._app_name = name
                return
            except Exception as exc:  # noqa: BLE001
                msg = f"{name}: {exc}"
                errors.append(msg)
                print(f"Failed to load model {name}: {exc}")

        raise RuntimeError(
            "Failed to load InsightFace models.\n" + "\n".join(errors),
        )

    @property
    def app(self) -> Any:
        self._load_app_once()
        return self._app

    # --------------- helpers ----------------
    @staticmethod
    def _lap_var(gray: np.ndarray) -> float:
        """
        Variance of Laplacian; falls back to simple gradient variance
        if OpenCV is not available.
        """
        try:
            import cv2  # type: ignore[import-not-found]

            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:  # noqa: BLE001
            gx = np.diff(gray.astype(np.float32), axis=1)
            gy = np.diff(gray.astype(np.float32), axis=0)
            return float(np.var(gx) + np.var(gy))

    @staticmethod
    def _rotate_bgr(bgr: np.ndarray, k: int) -> np.ndarray:
        return _as_bgr_uint8_contig(np.rot90(bgr, k=k))

    @staticmethod
    def _upscale_bgr(bgr: np.ndarray, scale: float) -> np.ndarray:
        """
        Upscale with PIL (LANCZOS). For very small images, force a
        higher scale to give the detector more pixels to work with.
        """
        from PIL import Image

        h, w = bgr.shape[:2]

        # If the image is small (like 256x256), be more aggressive.
        if max(h, w) < 300:
            scale = max(scale, 2.5)

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        rgb = bgr[:, :, ::-1]
        img = Image.fromarray(rgb, mode="RGB").resize(
            (new_w, new_h),
            Image.LANCZOS,
        )
        up = np.asarray(img, dtype=np.uint8)[:, :, ::-1]
        return _as_bgr_uint8_contig(up)

    def _analyze_once(
        self,
        bgr: np.ndarray,
    ) -> Tuple[Optional[Any], Optional[List[Any]]]:
        """
        Run face detection/embedding on a single BGR image.
        Returns (best_face, all_faces) or (None, None).
        """
        bgr = _as_bgr_uint8_contig(bgr)
        faces = self.app.get(bgr)
        if not faces:
            return None, None

        def _area(f: Any) -> float:
            x1, y1, x2, y2 = getattr(f, "bbox", [0, 0, 0, 0])
            return max(1.0, float(x2 - x1) * float(y2 - y1))

        faces.sort(
            key=lambda f: (float(getattr(f, "det_score", 0.0)), _area(f)),
            reverse=True,
        )
        return faces[0], faces

    def _compute_meta(self, bgr: np.ndarray, face: Any) -> Dict[str, Any]:
        """
        Compute quality metrics for the detected face.
        """
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
        area_img = float(h * w)
        face_ratio = area_face / max(1.0, area_img)
        det_score = float(getattr(face, "det_score", 0.0))

        # brightness & blur on ROI
        rgb_roi = roi[:, :, ::-1]
        gray = np.dot(rgb_roi[..., :3], [0.2989, 0.5870, 0.1140]).astype(
            np.uint8,
        )
        brightness = float(gray.mean())
        blur = self._lap_var(gray)

        ok = True
        reason: Optional[str] = None

        # hard gates
        if det_score < MIN_DET_SCORE:
            ok, reason = False, f"det_score<{MIN_DET_SCORE:.2f}"
        elif face_ratio < MIN_FACE_RATIO:
            ok, reason = False, f"face_too_small<{MIN_FACE_RATIO * 100:.0f}%"
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

    @staticmethod
    def _extract_embedding(face: Any) -> Optional[np.ndarray]:
        """
        Extract normalized embedding from InsightFace's Face object.
        """
        emb = getattr(face, "normed_embedding", None)
        if emb is None or (hasattr(emb, "size") and getattr(emb, "size", 0) == 0):
            emb = getattr(face, "embedding", None)
        if emb is None:
            return None

        emb_arr = np.asarray(emb, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(emb_arr))
        if not np.isfinite(n) or n < 1e-6:
            return None

        return emb_arr / n

    # ---------------- public: embed + quality ----------------
    def embed_best_face(
        self,
        bgr_image: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
        """
        Try multiple orientations and upscales to find a usable face embedding.
        Returns (embedding | None, quality_score, meta).
        meta["ok"] indicates passing quality gates; meta["reason"] explains failure.
        """
        bgr_image = _as_bgr_uint8_contig(bgr_image)
        h0, w0 = bgr_image.shape[:2]

        attempts: List[Tuple[str, np.ndarray]] = [("orig", bgr_image)]
        if max(h0, w0) < 900:
            attempts.append(("orig_up", self._upscale_bgr(bgr_image, 1.6)))

        # 180°
        rot180 = self._rotate_bgr(bgr_image, 2)
        attempts.append(("rot180", rot180))
        if max(rot180.shape[:2]) < 900:
            attempts.append(("rot180_up", self._upscale_bgr(rot180, 1.6)))

        # 90°
        rot90 = self._rotate_bgr(bgr_image, 1)
        attempts.append(("rot90", rot90))
        if max(rot90.shape[:2]) < 900:
            attempts.append(("rot90_up", self._upscale_bgr(rot90, 1.6)))

        # 270°
        rot270 = self._rotate_bgr(bgr_image, 3)
        attempts.append(("rot270", rot270))
        if max(rot270.shape[:2]) < 900:
            attempts.append(("rot270_up", self._upscale_bgr(rot270, 1.6)))

        last_meta: Dict[str, Any] = {
            "ok": False,
            "reason": "no_face",
            "img_size": [int(h0), int(w0)],
        }

        for tag, img in attempts:
            face, _ = self._analyze_once(img)
            if face is None:
                last_meta = {
                    "ok": False,
                    "reason": f"no_face_{tag}",
                    "img_size": [int(img.shape[0]), int(img.shape[1])],
                }
                continue

            meta = self._compute_meta(img, face)
            emb = self._extract_embedding(face)
            if emb is None:
                meta.update({"ok": False, "reason": "no_embedding", "attempt": tag})
                last_meta = meta
                continue

            # heuristic quality (informational)
            quality = (
                meta["det_score"]
                * (meta["face_ratio"] ** 0.5)
                * max(0.5, min(1.5, meta["blur"] / 150.0))
            )
            meta.update({"attempt": tag})
            return emb, float(quality), meta

        return None, 0.0, last_meta

    # ---------------- FAISS gallery ----------------
    def _rebuild_index_locked(self) -> None:
        """
        Rebuild FAISS index using self._templates.
        Caller MUST hold self.lock.
        """
        if not self._templates:
            self.index = None
            self.ids = []
            self.dim = 0
            return

        xs: List[np.ndarray] = []
        ids: List[Tuple[int, int]] = []

        for template_id, (employee_id, emb) in self._templates.items():
            emb_arr = np.asarray(emb, dtype=np.float32).reshape(-1)
            xs.append(emb_arr)
            ids.append((template_id, employee_id))

        xb = np.vstack(xs).astype("float32")
        self.dim = xb.shape[1]

        index = faiss.IndexFlatIP(self.dim)
        index.add(xb)

        self.index = index
        self.ids = ids

    def rebuild_index(
        self,
        templates: Iterable[Tuple[int, int, np.ndarray]],
    ) -> None:
        """
        Fully rebuild the index from provided (template_id, employee_id, embedding) tuples.
        Called on startup from apps.ready().
        """
        with self.lock:
            self._templates.clear()
            for template_id, employee_id, emb in templates:
                emb_arr = np.asarray(emb, dtype=np.float32).reshape(-1)
                self._templates[template_id] = (employee_id, emb_arr)

            self._rebuild_index_locked()

    def update_or_add(
        self,
        template_id: int,
        employee_id: int,
        embedding: np.ndarray,
    ) -> None:
        """
        Add a new template or update an existing one, then rebuild the index.
        Used by FaceTemplate post_save signal.
        """
        emb_arr = np.asarray(embedding, dtype=np.float32).reshape(-1)

        with self.lock:
            if self.dim and emb_arr.shape[0] != self.dim:
                raise ValueError(
                    f"Embedding dimension {emb_arr.shape[0]} does not match "
                    f"index dimension {self.dim}",
                )

            self._templates[template_id] = (employee_id, emb_arr)
            self._rebuild_index_locked()

    def remove(self, template_id: int) -> None:
        """
        Remove a template by its id (if present), then rebuild the index.
        Used by FaceTemplate post_delete signal.
        """
        with self.lock:
            if template_id not in self._templates:
                # Nothing to do; avoid raising to keep signal safe.
                return

            self._templates.pop(template_id)
            self._rebuild_index_locked()

    def search(
        self,
        q: np.ndarray,
        k: int = 10,
    ) -> Tuple[List[float], List[int]]:
        """
        Search top-k nearest neighbors for a query embedding q.
        Returns (scores, indices_in_ids_list).
        """
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return [], []

            D, I = self.index.search(q[None, :].astype("float32"), k)
        return D[0].tolist(), I[0].tolist()


ENGINE = FaceEngine()
