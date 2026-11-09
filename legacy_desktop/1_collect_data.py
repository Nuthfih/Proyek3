import cv2
import os
import time
import numpy as np
from config import ACTIONS, DATA_PATH, NO_SEQUENCES, SEQUENCE_LENGTH
from utils import create_hand_detector, mediapipe_detection, draw_styled_landmarks, extract_keypoints, draw_text_with_bg

def main():
    # Pastikan folder dataset untuk setiap kata sudah siap
    for action in ACTIONS:
        os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka webcam. Pastikan kamera terpasang dan tidak sedang dipakai aplikasi lain.")
        return

    print("=" * 60)
    print("   PENGUMPUL DATASET BAHASA ISYARAT (SIBI) DENGAN WEBCAM   ")
    print("=" * 60)
    print(f"Daftar kata yang akan direkam: {list(ACTIONS)}")
    print(f"Target sampel per kata: {NO_SEQUENCES} video")
    print(f"Panjang per sampel: {SEQUENCE_LENGTH} frame (~1 detik)")
    print("\nPetunjuk:")
    print("- Tekan 'S' di jendela kamera untuk memulai proses perekaman.")
    print("- Tekan 'Q' kapan saja untuk membatalkan dan keluar.")
    print("=" * 60)

    # Inisialisasi Hand Detector MediaPipe
    detector = create_hand_detector()
    
    # State tunggu user tekan 'S' pertama kali
    started = False
    while not started:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1) # Mirror horizontal
        
        draw_text_with_bg(frame, "PENGUMPULAN DATASET ISYARAT", (20, 40), font_scale=0.8, bg_color=(40, 40, 40))
        draw_text_with_bg(frame, "Tekan tombol 'S' pada keyboard untuk MULAI", (20, 90), font_scale=0.7, text_color=(0, 255, 255), bg_color=(20, 20, 20))
        draw_text_with_bg(frame, "Tekan 'Q' untuk KELUAR", (20, 130), font_scale=0.6, text_color=(180, 180, 180), bg_color=(20, 20, 20))
        
        cv2.imshow("Koleksi Data SIBI", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s') or key == ord('S'):
            started = True
        elif key == ord('q') or key == ord('Q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    # Loop setiap kata
    for action_idx, action in enumerate(ACTIONS):
        existing_files = [f for f in os.listdir(os.path.join(DATA_PATH, action)) if f.endswith(".npy")]
        start_seq = len(existing_files)
        
        if start_seq >= NO_SEQUENCES:
            print(f"[INFO] Kata '{action}' sudah memiliki {start_seq} sampel. Melanjutkan ke kata berikutnya...")
            continue

        for sequence in range(start_seq, NO_SEQUENCES):
            # Fase hitung mundur / persiapan (2 detik)
            countdown_start = time.time()
            while time.time() - countdown_start < 2.0:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, detector)
                draw_styled_landmarks(image, results)
                
                sisa_waktu = 2.0 - (time.time() - countdown_start)
                draw_text_with_bg(image, f"KATA: [{action}] ({action_idx + 1}/{len(ACTIONS)})", (20, 40), font_scale=0.8, text_color=(0, 255, 0), bg_color=(20, 40, 20))
                draw_text_with_bg(image, f"Sampel #{sequence + 1} dari {NO_SEQUENCES}", (20, 80), font_scale=0.6, bg_color=(30, 30, 30))
                draw_text_with_bg(image, f"SIAP-SIAP DALAM: {sisa_waktu:.1f} detik...", (20, 130), font_scale=0.7, text_color=(0, 255, 255), bg_color=(0, 0, 120))
                
                cv2.imshow("Koleksi Data SIBI", image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("[INFO] Perekaman dibatalkan oleh pengguna.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Fase perekaman 30 frame
            recorded_frames = []
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, detector)
                draw_styled_landmarks(image, results)
                
                # Simpan 126 koordinat titik tangan
                keypoints = extract_keypoints(results)
                recorded_frames.append(keypoints)
                
                # Indikator perekaman merah (REC)
                cv2.circle(image, (image.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                draw_text_with_bg(image, "REC", (image.shape[1] - 85, 36), font_scale=0.6, text_color=(0, 0, 255), bg_color=(20, 20, 20))
                
                draw_text_with_bg(image, f"MEREKAM: [{action}]", (20, 40), font_scale=0.8, text_color=(0, 255, 255), bg_color=(50, 20, 20))
                draw_text_with_bg(image, f"Frame {frame_num + 1}/{SEQUENCE_LENGTH}", (20, 80), font_scale=0.7, text_color=(255, 255, 255), bg_color=(30, 30, 30))
                
                cv2.imshow("Koleksi Data SIBI", image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("[INFO] Perekaman dibatalkan oleh pengguna.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Simpan 1 sequence (30, 126) ke file .npy
            if len(recorded_frames) == SEQUENCE_LENGTH:
                save_path = os.path.join(DATA_PATH, action, f"seq_{sequence}.npy")
                np.save(save_path, np.array(recorded_frames))
                print(f"  [SAVED] {action} - Sampel #{sequence + 1}/{NO_SEQUENCES} tersimpan.")

    print("\n" + "=" * 60)
    print("   SEMUA SAMPEL BERHASIL DIREKAM!   ")
    print(f"Data tersimpan di direktori: {DATA_PATH}")
    print("Langkah selanjutnya: Jalankan 'python 2_train_model.py' untuk melatih model.")
    print("=" * 60)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
