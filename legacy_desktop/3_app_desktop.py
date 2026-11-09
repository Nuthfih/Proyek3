import cv2
import os
import time
import threading
import subprocess
import numpy as np
import torch
import torch.nn.functional as F

from config import ACTIONS, MODEL_PATH, SEQUENCE_LENGTH, FEATURE_SIZE
from model import SignLanguageLSTM
from utils import create_hand_detector, mediapipe_detection, draw_styled_landmarks, extract_keypoints, draw_text_with_bg

def speak_text(text):
    """
    Menjalankan Text-to-Speech bawaan Windows melalui thread terpisah agar video webcam tidak freeze.
    """
    def _run_tts():
        if not text.strip():
            return
        cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\');"'
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    thread = threading.Thread(target=_run_tts, daemon=True)
    thread.start()

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model belum ditemukan di: {MODEL_PATH}")
        print("Silakan jalankan:")
        print("  1. 'python 1_collect_data.py' (untuk merekam dataset)")
        print("  2. 'python 2_train_model.py' (untuk melatih model)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageLSTM(
        input_size=FEATURE_SIZE,
        hidden_size=64,
        num_layers=2,
        num_classes=len(ACTIONS),
        dropout=0.2
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("[INFO] Model PyTorch LSTM berhasil dimuat!")

    # Inisialisasi Hand Detector MediaPipe
    detector = create_hand_detector()

    sequence = []
    sentence = []
    predictions = []
    last_added_word = ""
    last_added_time = 0
    COOLDOWN_SECONDS = 1.8  # Jeda waktu minimal sebelum kata baru ditambahkan
    THRESHOLD = 0.75        # Ambang batas keyakinan (75%)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam tidak dapat dibuka.")
        return

    print("=" * 60)
    print("      PENERJEMAH BAHASA ISYARAT REAL-TIME (DESKTOP)      ")
    print("=" * 60)
    print("Kontrol Keyboard:")
    print("  [C]         : Hapus semua kalimat")
    print("  [Backspace] : Hapus kata terakhir")
    print("  [T / Enter] : Bacakan kalimat (Text-to-Speech)")
    print("  [Q]         : Keluar dari aplikasi")
    print("=" * 60)

    current_action = "-"
    confidence = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Mirror horizontal
        h, w, _ = frame.shape

        # Deteksi MediaPipe
        image, results = mediapipe_detection(frame, detector)
        draw_styled_landmarks(image, results)

        # Ekstraksi koordinat
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        # Prediksi sekuens frame
        if len(sequence) == SEQUENCE_LENGTH:
            input_tensor = torch.tensor(np.array([sequence]), dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
                predicted_idx = np.argmax(probabilities)
                confidence = probabilities[predicted_idx]
                current_action = ACTIONS[predicted_idx]

            predictions.append(predicted_idx)
            predictions = predictions[-10:]

            most_frequent_idx = max(set(predictions), key=predictions.count)
            if predictions.count(most_frequent_idx) >= 7 and confidence > THRESHOLD:
                candidate_word = ACTIONS[most_frequent_idx]
                now = time.time()
                
                if candidate_word != last_added_word or (now - last_added_time > COOLDOWN_SECONDS):
                    sentence.append(candidate_word)
                    last_added_word = candidate_word
                    last_added_time = now
                    if len(sentence) > 6:
                        sentence = sentence[-6:]

        # ================= RENDER INTERFACE / HUD =================
        top_bar_h = 75
        cv2.rectangle(image, (0, 0), (w, top_bar_h), (25, 25, 25), -1)
        cv2.line(image, (0, top_bar_h), (w, top_bar_h), (60, 60, 60), 1)

        cv2.putText(image, "KATA TERDETEKSI:", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(image, f"{current_action}", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        # Confidence Bar
        bar_x = 260
        bar_y = 35
        bar_w = 200
        bar_h = 18
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * confidence)
        color_bar = (0, 200, 0) if confidence > THRESHOLD else (0, 165, 255)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color_bar, -1)
        cv2.putText(image, f"{confidence*100:.0f}%", (bar_x + bar_w + 10, bar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Panel Bawah Kalimat
        bot_bar_h = 85
        cv2.rectangle(image, (0, h - bot_bar_h), (w, h), (18, 18, 18), -1)
        cv2.line(image, (0, h - bot_bar_h), (w, h - bot_bar_h), (60, 60, 60), 1)

        kalimat_str = " ".join(sentence) if sentence else "..."
        cv2.putText(image, "HASIL TERJEMAHAN KALIMAT:", (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(image, kalimat_str, (15, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, "[T] Baca Suara | [C] Hapus | [Q] Keluar", (w - 330, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("Penerjemah Bahasa Isyarat SIBI", image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('c') or key == ord('C'):
            sentence.clear()
            last_added_word = ""
        elif key == 8:  # Backspace
            if sentence:
                sentence.pop()
        elif key == ord('t') or key == ord('T') or key == 13:  # T atau Enter (Speak)
            if sentence:
                speak_text(" ".join(sentence))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
