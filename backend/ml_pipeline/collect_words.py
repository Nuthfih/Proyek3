"""
[Jalur Kata Dasar - Kontribusi Zahra Nur'azijah Lutfiani]
Skrip pengumpulan urutan gerakan 30 bingkai (30 x 63 fitur ternormalisasi)
untuk kata dasar SIBI ke folder backend/data/words/<KATA>/seq_<N>.npy.
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import WORDS_DATA_DIR, DEFAULT_WORD_LABELS, SEQUENCE_LENGTH, FEATURE_SIZE
from app.extractor import HandLandmarkExtractor, HAND_CONNECTIONS


def main():
    for word in DEFAULT_WORD_LABELS:
        os.makedirs(os.path.join(WORDS_DATA_DIR, word), exist_ok=True)

    extractor = HandLandmarkExtractor(max_hands=1, min_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera web tidak dapat dibuka.")
        return

    current_idx = 0
    print("=" * 65)
    print("  PENGUMPULAN URUTAN GERAKAN KATA DASAR SIBI (30 BINGKAI)  ")
    print("=" * 65)
    print("Kontrol Keyboard:")
    print("  [R]     : Rekam 1 urutan gerakan (30 bingkai) untuk kata aktif")
    print("  [N / P] : Pindah ke kata Berikutnya (Next) / Sebelumnya (Prev)")
    print("  [Q]     : Keluar")
    print("=" * 65)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        current_word = DEFAULT_WORD_LABELS[current_idx]
        word_dir = os.path.join(WORDS_DATA_DIR, current_word)
        existing_count = len([f for f in os.listdir(word_dir) if f.endswith(".npy")])

        hand_detected, landmarks, _ = extractor.extract_from_frame(frame)
        if hand_detected:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, pts[s], pts[e], (0, 200, 120), 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 0), (w, 80), (20, 30, 25), -1)
        cv2.putText(
            frame,
            f"Target Kata: [{current_word}] | Jumlah Rekaman: {existing_count}",
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 180),
            2,
        )
        cv2.putText(
            frame,
            "Tekan [R] untuk rekam 30 bingkai | [N]/[P] ganti kata | [Q] keluar",
            (15, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
        )

        cv2.imshow("IsyaratKu - Koleksi Kata Dasar SIBI (30 Bingkai)", frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("n") or key == ord("N"):
            current_idx = (current_idx + 1) % len(DEFAULT_WORD_LABELS)
        elif key == ord("p") or key == ord("P"):
            current_idx = (current_idx - 1) % len(DEFAULT_WORD_LABELS)
        elif key == ord("r") or key == ord("R"):
            # Hitung mundur 2 detik
            t_start = time.time()
            while time.time() - t_start < 2.0:
                ret, f_cd = cap.read()
                if not ret:
                    break
                f_cd = cv2.flip(f_cd, 1)
                rem = 2.0 - (time.time() - t_start)
                cv2.rectangle(f_cd, (0, 0), (w, 80), (0, 60, 120), -1)
                cv2.putText(
                    f_cd,
                    f"Siap-siap memeragakan [{current_word}] dalam {rem:.1f} dtk...",
                    (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("IsyaratKu - Koleksi Kata Dasar SIBI (30 Bingkai)", f_cd)
                cv2.waitKey(10)

            # Rekam 30 bingkai
            seq_frames = []
            for f_idx in range(SEQUENCE_LENGTH):
                ret, f_rec = cap.read()
                if not ret:
                    break
                f_rec = cv2.flip(f_rec, 1)
                det, lms, feat_63 = extractor.extract_from_frame(f_rec)
                if det:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                    for s, e in HAND_CONNECTIONS:
                        cv2.line(f_rec, pts[s], pts[e], (0, 200, 120), 2, cv2.LINE_AA)
                seq_frames.append(feat_63)

                cv2.rectangle(f_rec, (0, 0), (w, 80), (20, 20, 120), -1)
                cv2.putText(
                    f_rec,
                    f"MEREKAM [{current_word}] : Bingkai {f_idx + 1}/{SEQUENCE_LENGTH}",
                    (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("IsyaratKu - Koleksi Kata Dasar SIBI (30 Bingkai)", f_rec)
                cv2.waitKey(15)

            if len(seq_frames) == SEQUENCE_LENGTH:
                save_file = os.path.join(word_dir, f"seq_{existing_count:03d}.npy")
                np.save(save_file, np.array(seq_frames, dtype=np.float32))
                print(f"[SAVED] Urutan (30, {FEATURE_SIZE}) disimpan ke {save_file}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

