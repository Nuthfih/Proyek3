import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_TASK_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def ensure_model_exists():
    """Memastikan file model hand_landmarker.task tersedia, jika belum ada otomatis diunduh."""
    if not os.path.exists(MODEL_TASK_PATH):
        print(f"[INFO] Mengunduh model pelacak tangan MediaPipe (7.8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_TASK_PATH)
        print("[INFO] Model berhasil diunduh!")

def create_hand_detector():
    """Inisialisasi HandLandmarker dari MediaPipe modern Tasks API."""
    ensure_model_exists()
    base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.HandLandmarker.create_from_options(options)

# Koneksi 21 titik sendi tangan
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Jempol
    (0, 5), (5, 6), (6, 7), (7, 8),        # Jari Telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),   # Jari Tengah
    (9, 13), (13, 14), (14, 15), (15, 16), # Jari Manis
    (13, 17), (17, 18), (18, 19), (19, 20),# Kelingking
    (0, 17)                                 # Telapak
]

def mediapipe_detection(image, detector):
    """
    Konversi frame BGR OpenCV ke format MediaPipe Image,
    lalu lakukan deteksi titik sendi tangan.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)
    return image, detection_result

def draw_styled_landmarks(image, detection_result):
    """
    Menggambar titik landmark dan garis koneksi sendi tangan
    dengan warna yang jelas (Cyan untuk kiri, Oranye/Merah muda untuk kanan).
    """
    if not detection_result.hand_landmarks:
        return
    
    h, w, _ = image.shape
    for idx, landmarks in enumerate(detection_result.hand_landmarks):
        hand_label = "Right"
        if detection_result.handedness and idx < len(detection_result.handedness):
            hand_label = detection_result.handedness[idx][0].category_name
        
        # Pembeda warna tangan kiri dan kanan
        line_color = (255, 180, 0) if hand_label == "Left" else (50, 100, 255)
        dot_color = (255, 255, 0) if hand_label == "Left" else (0, 255, 255)
        
        # Konversi koordinat normalisasi (0.0 - 1.0) ke koordinat piksel
        points = []
        for lm in landmarks:
            cx = int(np.clip(lm.x * w, 0, w - 1))
            cy = int(np.clip(lm.y * h, 0, h - 1))
            points.append((cx, cy))
            
        # Gambar garis tulang sendi
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(image, points[start_idx], points[end_idx], line_color, 2, cv2.LINE_AA)
            
        # Gambar bulatan titik sendi
        for pt in points:
            cv2.circle(image, pt, 4, dot_color, -1, cv2.LINE_AA)
            cv2.circle(image, pt, 5, (20, 20, 20), 1, cv2.LINE_AA)

def extract_keypoints(detection_result):
    """
    Mengekstrak titik koordinat tangan kiri (63) dan tangan kanan (63)
    sehingga menghasilkan total 126 fitur per frame.
    """
    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)
    
    if detection_result.hand_landmarks:
        for idx, landmarks in enumerate(detection_result.hand_landmarks):
            hand_label = "Right"
            if detection_result.handedness and idx < len(detection_result.handedness):
                hand_label = detection_result.handedness[idx][0].category_name
            
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            if hand_label == "Left":
                left_hand = coords
            else:
                right_hand = coords
                
    return np.concatenate([left_hand, right_hand])

def draw_text_with_bg(img, text, position, font_scale=0.7, text_color=(255, 255, 255), bg_color=(20, 20, 20), thickness=2, padding=8):
    """
    Menggambar teks berlatar belakang kotak kontras (HUD pill).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    
    x1 = max(0, x - padding)
    y1 = max(0, y - text_height - padding)
    x2 = min(img.shape[1], x + text_width + padding)
    y2 = min(img.shape[0], y + baseline + padding)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
