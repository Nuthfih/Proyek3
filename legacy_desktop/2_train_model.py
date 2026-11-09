import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from config import ACTIONS, DATA_PATH, MODEL_PATH, SEQUENCE_LENGTH, FEATURE_SIZE
from model import SignLanguageLSTM

def load_dataset():
    sequences = []
    labels = []
    
    label_map = {action: idx for idx, action in enumerate(ACTIONS)}
    
    print("\n[INFO] Membaca file rekaman dataset...")
    for action in ACTIONS:
        action_dir = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_dir):
            continue
            
        files = [f for f in os.listdir(action_dir) if f.endswith(".npy")]
        print(f"  - {action}: {len(files)} sampel ditemukan.")
        
        for file in files:
            file_path = os.path.join(action_dir, file)
            res = np.load(file_path)
            # Pastikan bentuknya (30, 126)
            if res.shape == (SEQUENCE_LENGTH, FEATURE_SIZE):
                sequences.append(res)
                labels.append(label_map[action])
            else:
                print(f"    [Peringatan] Dimensi tidak cocok pada {file}: {res.shape}, diabaikan.")

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)

def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    X, y = load_dataset()
    if len(X) == 0:
        print("[ERROR] Dataset belum ada atau kosong!")
        print("Silakan jalankan 'python 1_collect_data.py' terlebih dahulu untuk merekam dataset.")
        return

    print(f"\nTotal sampel valid: {len(X)}")
    print(f"Dimensi fitur masukan: {X.shape} (Jumlah Sampel, Sequence Length, Fitur per Frame)")

    # Split dataset menjadi data latih (80%) dan data uji (20%)
    if len(X) >= 10:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        # Jika data masih sedikit sekali untuk uji coba coba-coba
        X_train, X_val, y_train, y_val = X, X, y, y

    # Konversi ke PyTorch DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    batch_size = 8 if len(X_train) < 32 else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Inisialisasi Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device yang digunakan: {device}")

    model = SignLanguageLSTM(
        input_size=FEATURE_SIZE,
        hidden_size=64,
        num_layers=2,
        num_classes=len(ACTIONS),
        dropout=0.2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 60
    best_val_acc = 0.0

    print("\n" + "=" * 50)
    print("        MEMULAI PELATIHAN MODEL LSTM        ")
    print("=" * 50)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

        epoch_loss = total_loss / total_train
        train_acc = correct_train / total_train

        # Evaluasi Validasi
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)

        val_acc = correct_val / total_val if total_val > 0 else 0.0

        # Simpan checkpoint model terbaik
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {epoch_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

    print("=" * 50)
    print(f"[SUKSES] Pelatihan Selesai!")
    print(f"Akurasi Validasi Tertinggi: {best_val_acc * 100:.2f}%")
    print(f"Model tersimpan di: {MODEL_PATH}")

    # Laporan Klasifikasi Akhir
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_targets = []
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(batch_y.numpy())

        print("\nLaporan Evaluasi per Kata:")
        target_names = [ACTIONS[i] for i in sorted(list(set(val_targets)))]
        print(classification_report(val_targets, val_preds, target_names=target_names, zero_division=0))

    print("\nLangkah selanjutnya: Jalankan 'python 3_app_desktop.py' untuk menjalankan penerjemah real-time!")

if __name__ == "__main__":
    main()
