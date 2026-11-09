"""
[Jalur Kata Dasar - Kontribusi Zahra Nur'azijah Lutfiani]
Skrip pelatihan dan evaluasi model LSTM untuk pengenalan gestur dinamis (Kata Dasar SIBI)
berdasarkan urutan 30 bingkai x 63 fitur landmark ternormalisasi.
"""

import os
import sys
import csv
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    WORDS_DATA_DIR,
    LSTM_WORD_MODEL_PATH,
    LABELS_WORD_JSON,
    EVALUATION_DIR,
    SEQUENCE_LENGTH,
    FEATURE_SIZE,
)
from app.predictor import SignLSTM


def load_word_sequences(words_dir: str):
    if not os.path.exists(words_dir):
        return np.empty((0, SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32), []

    sequences = []
    labels = []
    for word_label in sorted(os.listdir(words_dir)):
        sub_dir = os.path.join(words_dir, word_label)
        if not os.path.isdir(sub_dir):
            continue
        for fname in os.listdir(sub_dir):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(sub_dir, fname))
                if arr.shape == (SEQUENCE_LENGTH, FEATURE_SIZE):
                    sequences.append(arr)
                    labels.append(word_label)

    return np.array(sequences, dtype=np.float32), labels


def main():
    os.makedirs(os.path.dirname(LSTM_WORD_MODEL_PATH), exist_ok=True)
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    X, raw_labels = load_word_sequences(WORDS_DATA_DIR)
    if len(X) < 10:
        print(f"[ERROR] Data urutan kata dasar pada {WORDS_DATA_DIR} belum cukup ({len(X)} sampel).")
        print("Silakan jalankan 'python backend/ml_pipeline/collect_words.py' terlebih dahulu.")
        return

    unique_words = sorted(list(set(raw_labels)))
    word_to_idx = {w: idx for idx, w in enumerate(unique_words)}
    y = np.array([word_to_idx[w] for w in raw_labels], dtype=np.int64)

    with open(LABELS_WORD_JSON, "w", encoding="utf-8") as f:
        json.dump(unique_words, f, indent=2, ensure_ascii=False)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLSTM(
        input_size=FEATURE_SIZE,
        hidden_size=64,
        num_layers=2,
        num_classes=len(unique_words),
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=8,
        shuffle=True,
    )

    best_val_acc = 0.0
    print(f"[INFO] Melatih Model LSTM untuk {len(unique_words)} kata dasar...")
    for epoch in range(1, 61):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(torch.tensor(X_val, device=device))
            val_preds = torch.argmax(val_out, dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), LSTM_WORD_MODEL_PATH)

    model.load_state_dict(torch.load(LSTM_WORD_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        test_out = model(torch.tensor(X_test, device=device))
        lstm_preds = torch.argmax(test_out, dim=1).cpu().numpy()
    latency_ms = ((time.perf_counter() - t0) * 1000.0) / len(X_test)

    acc = accuracy_score(y_test, lstm_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, lstm_preds, average="weighted", zero_division=0
    )

    eval_csv = os.path.join(EVALUATION_DIR, "word_lstm_evaluation.csv")
    with open(eval_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1_Score", "Avg_Latency_ms"])
        writer.writerow(["LSTM", f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{latency_ms:.3f}"])

    np.savetxt(
        os.path.join(EVALUATION_DIR, "confusion_matrix_lstm_word.csv"),
        confusion_matrix(y_test, lstm_preds),
        delimiter=",",
        fmt="%d",
    )

    print("\n" + "=" * 65)
    print("  HASIL EVALUASI MODEL KATA DASAR LSTM (Uji T-04)  ")
    print("=" * 65)
    print(f"LSTM -> Akurasi: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Latensi: {latency_ms:.3f} ms")
    print(f"Laporan disimpan di: {eval_csv}")


if __name__ == "__main__":
    main()

