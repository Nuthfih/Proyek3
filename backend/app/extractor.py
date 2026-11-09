import os
import base64
import urllib.request
from dataclasses import dataclass
from typing import Tuple, List, Optional
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import MEDIAPIPE_TASK_PATH, MEDIAPIPE_TASK_URL, FEATURE_SIZE


@dataclass
class Landmark3D:
    x: float
    y: float
    z: float


# Koneksi 21 titik sendi tangan untuk keperluan visualisasi
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Ibu Jari
    (0, 5), (5, 6), (6, 7), (7, 8),        # Telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),   # Jari Tengah
    (9, 13), (13, 14), (14, 15), (15, 16), # Jari Manis
    (13, 17), (17, 18), (18, 19), (19, 20),# Kelingking
    (0, 17)                                 # Telapak
]


def ensure_mediapipe_model() -> str:
    """Memastikan berkas hand_landmarker.task tersedia di folder backend/models/."""
    os.makedirs(os.path.dirname(MEDIAPIPE_TASK_PATH), exist_ok=True)
    if not os.path.exists(MEDIAPIPE_TASK_PATH):
        print(f"[INFO] Mengunduh model MediaPipe Hands ke {MEDIAPIPE_TASK_PATH}...")
        urllib.request.urlretrieve(MEDIAPIPE_TASK_URL, MEDIAPIPE_TASK_PATH)
        print("[INFO] Model MediaPipe Hands selesai diunduh.")
    return MEDIAPIPE_TASK_PATH


def normalize_landmarks_wrist(raw_coords: np.ndarray) -> np.ndarray:
    """
    Kebutuhan F-03: Menormalisasi 21 koordinat landmark 3D (63 fitur)
    terhadap titik pergelangan tangan (wrist / indeks 0) serta menskalakannya
    agar tahan terhadap perubahan posisi dan variasi jarak (30 cm vs 60 cm).

    Input:
        raw_coords: array berukuran (21, 3) atau (63,)
    Output:
        normalized: array 1D berukuran (63,) bertipe float32
    """
    coords = np.array(raw_coords, dtype=np.float32).reshape(21, 3)

    # Titik acuan: pergelangan tangan (indeks 0)
    wrist = coords[0].copy()
    relative_coords = coords - wrist

    # Normalisasi skala berdasarkan jarak maksimum dari pergelangan tangan
    max_val = np.max(np.abs(relative_coords))
    if max_val > 1e-6:
        relative_coords = relative_coords / max_val

    return relative_coords.flatten()


class HandLandmarkExtractor:
    """
    Modul Pengolahan Citra (OpenCV) dan Ekstraksi Fitur 21 Landmark 3D
    menggunakan Google MediaPipe Hands (Kebutuhan F-02 & F-03).
    """

    def __init__(self, max_hands: int = 1, min_confidence: float = 0.5):
        model_path = ensure_mediapipe_model()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def decode_base64_image(image_base64: str) -> Optional[np.ndarray]:
        """Mengubah string Base64 dari peramban menjadi matriks citra BGR OpenCV."""
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            img_bytes = base64.b64decode(image_base64)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def extract_from_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[bool, List[Landmark3D], np.ndarray]:
        """
        Mengekstraksi 21 landmark tangan dari citra BGR OpenCV.
        Jika tangan tidak terdeteksi, mengembalikan (False, [], zeros(63)) tanpa error (Uji T-02).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, [], np.zeros(FEATURE_SIZE, dtype=np.float32)

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return False, [], np.zeros(FEATURE_SIZE, dtype=np.float32)

        # Ambil tangan pertama yang terdeteksi
        hand_lms = result.hand_landmarks[0]
        raw_points = []
        landmark_objects = []

        for lm in hand_lms:
            raw_points.append([lm.x, lm.y, lm.z])
            landmark_objects.append(
                Landmark3D(x=float(lm.x), y=float(lm.y), z=float(lm.z))
            )

        normalized_63 = normalize_landmarks_wrist(np.array(raw_points, dtype=np.float32))
        return True, landmark_objects, normalized_63
