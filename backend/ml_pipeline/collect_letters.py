"""
[Jalur Abjad - Kontribusi Alif Satria Raghib]
Skrip pengumpulan data landmark abjad SIBI (gestur statis).
Menyimpan hasil ekstraksi 21 landmark ternormalisasi (63 fitur) ke berkas CSV:
backend/data/letters/letters_landmarks.csv (Sesuai Tabel 3.1 Proposal).
"""

import os
import sys
import csv
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import LETTERS_CSV_PATH, DEFAULT_LETTER_LABELS, FEATURE_SIZE
from app.extractor import HandLandmarkExtractor, HAND_CONNECTIONS


def init_csv(csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        headers = ["label"] + [
            f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")
        ]
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def main():
    init_csv(LETTERS_CSV_PATH)
    extractor = HandLandmarkExtractor(max_hands=1, min_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera web tidak dapat dibuka.")
        return

    current_idx = 0
    saved_count = 0

    print("=" * 65)
    print("  PENGUMPULAN DATASET ABJAD SIBI (63 FITUR TERNORMALISASI)  ")
    print("=" * 65)
    print("Kontrol Keyboard:")
    print("  [SPASI] : Simpan 1 sampel landmark huruf aktif ke CSV")
    print("  [N / P] : Pindah ke huruf Berikutnya (Next) / Sebelumnya (Prev)")
    print("  [Q]     : Keluar")
    print("=" * 65)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        current_letter = DEFAULT_LETTER_LABELS[current_idx]

        hand_detected, landmarks, features_63 = extractor.extract_from_frame(frame)

        # Gambar visualisasi 21 landmark jika terdeteksi
        if hand_detected:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, pts[s], pts[e], (0, 200, 120), 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

        # HUD Informasi
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 30, 25), -1)
        cv2.putText(
            frame,
            f"Target Huruf: [{current_letter}] | Tersimpan Sesi Ini: {saved_count}",
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 180),
            2,
        )
        status_txt = "Tangan Terdeteksi (Tekan SPASI untuk simpan)" if hand_detected else "Tangan Tidak Terdeteksi"
        cv2.putText(
            frame,
            status_txt,
            (15, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255) if hand_detected else (100, 100, 255),
            1,
        )

        cv2.imshow("IsyaratKu - Koleksi Dataset Abjad SIBI", frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("n") or key == ord("N"):
            current_idx = (current_idx + 1) % len(DEFAULT_LETTER_LABELS)
        elif key == ord("p") or key == ord("P"):
            current_idx = (current_idx - 1) % len(DEFAULT_LETTER_LABELS)
        elif key == ord(" ") and hand_detected and len(features_63) == FEATURE_SIZE:
            with open(LETTERS_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([current_letter] + [round(float(v), 6) for v in features_63])
            saved_count += 1
            print(f"[SAVED] Huruf '{current_letter}' berhasil ditambahkan ke {LETTERS_CSV_PATH}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

