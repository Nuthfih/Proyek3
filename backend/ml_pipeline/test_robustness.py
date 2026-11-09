"""
[Pengujian T-06, T-07, T-08, T-09]
Skrip pengujian ketahanan sistem pada variasi jarak (30 cm vs 60 cm),
kondisi pencahayaan (normal vs rendah), serta pengukuran FPS dan latensi inferensi.
Hasil disimpan ke backend/evaluation/robustness_test_results.csv.
"""

import os
import sys
import csv
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import EVALUATION_DIR
from app.extractor import HandLandmarkExtractor
from app.predictor import SignPredictorManager


def run_benchmark_session(condition_name: str, target_label: str, num_frames: int = 100):
    extractor = HandLandmarkExtractor(max_hands=1, min_confidence=0.5)
    predictor = SignPredictorManager()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak ditemukan.")
        return

    os.makedirs(EVALUATION_DIR, exist_ok=True)
    correct = 0
    detected_frames = 0
    total_latency_ms = 0.0
    t_session_start = time.perf_counter()

    for i in range(num_frames):
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        det, _, feat_63 = extractor.extract_from_frame(frame)
        if det:
            detected_frames += 1
            pred_label, conf, _ = predictor.predict_letter(feat_63)
            if pred_label == target_label:
                correct += 1
        frame_ms = (time.perf_counter() - t0) * 1000.0
        total_latency_ms += frame_ms

        cv2.putText(
            frame,
            f"Uji [{condition_name}] Target: {target_label} ({i+1}/{num_frames})",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        cv2.imshow("IsyaratKu - Pengujian Ketahanan (T-06 / T-07 / T-09)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed_sec = time.perf_counter() - t_session_start
    cap.release()
    cv2.destroyAllWindows()

    fps = num_frames / max(elapsed_sec, 1e-6)
    avg_latency = total_latency_ms / max(num_frames, 1)
    accuracy = correct / max(detected_frames, 1)

    out_csv = os.path.join(EVALUATION_DIR, "robustness_test_results.csv")
    write_header = not os.path.exists(out_csv)
    with open(out_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Condition", "Target_Label", "Frames", "Detected_Frames", "Accuracy", "Avg_Latency_ms", "FPS"])
        writer.writerow([condition_name, target_label, num_frames, detected_frames, f"{accuracy:.4f}", f"{avg_latency:.2f}", f"{fps:.2f}"])

    print(f"[HASIL {condition_name}] Akurasi: {accuracy*100:.1f}% | Latensi: {avg_latency:.2f} ms | FPS: {fps:.1f}")


if __name__ == "__main__":
    print("Contoh penggunaan: menguji kondisi 'Jarak_30cm_Cahaya_Normal' untuk huruf 'A'")
    run_benchmark_session("Jarak_30cm_Cahaya_Normal", "A", num_frames=100)

