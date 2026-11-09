"""
[Jalur Abjad - Kontribusi Alif Satria Raghib]
Skrip pelatihan dan perbandingan model Random Forest vs Feedforward Neural Network (FFNN)
untuk pengenalan gestur statis Abjad SIBI (63 fitur landmark ternormalisasi).
Menghasilkan berkas model (.pkl & .pt) serta laporan evaluasi di backend/evaluation/.
"""

import os
import sys
import csv
import json
import time
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    LETTERS_CSV_PATH,
    RF_LETTER_MODEL_PATH,
    FFNN_LETTER_MODEL_PATH,
    LABELS_LETTER_JSON,
    EVALUATION_DIR,
    FEATURE_SIZE,
)
from app.predictor import SignFFNN


def load_letter_csv(csv_path: str):
    if not os.path.exists(csv_path):
        return np.empty((0, FEATURE_SIZE), dtype=np.float32), []

    features = []
    labels = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) == FEATURE_SIZE + 1:
                labels.append(row[0])
                features.append([float(x) for x in row[1:]])

    return np.array(features, dtype=np.float32), labels


def main():
    os.makedirs(os.path.dirname(RF_LETTER_MODEL_PATH), exist_ok=True)
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    X, raw_labels = load_letter_csv(LETTERS_CSV_PATH)
    if len(X) < 10:
        print(f"[ERROR] Data pada {LETTERS_CSV_PATH} belum cukup ({len(X)} sampel).")
        print("Silakan jalankan 'python backend/ml_pipeline/collect_letters.py' terlebih dahulu.")
        return

    unique_labels = sorted(list(set(raw_labels)))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    y = np.array([label_to_idx[lbl] for lbl in raw_labels], dtype=np.int64)

    with open(LABELS_LETTER_JSON, "w", encoding="utf-8") as f:
        json.dump(unique_labels, f, indent=2, ensure_ascii=False)

    # Pembagian data 70% latih, 15% validasi, 15% uji (Subbab 3.7 Proposal)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"[INFO] Kelas Abjad ({len(unique_labels)}): {unique_labels}")
    print(f"[INFO] Split Data -> Latih: {len(X_train)}, Validasi: {len(X_val)}, Uji: {len(X_test)}")

    # =========================================================================
    # 1. PELATIHAN & EVALUASI RANDOM FOREST
    # =========================================================================
    print("\n--- Melatih Model Random Forest (Abjad) ---")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    rf_clf.fit(X_train, y_train)
    # Simpan dengan nama kelas string agar langsung mudah dipakai predictor
    rf_clf_named = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    rf_clf_named.fit(X_train, np.array([unique_labels[i] for i in y_train]))
    joblib.dump(rf_clf_named, RF_LETTER_MODEL_PATH)

    t0 = time.perf_counter()
    rf_preds = rf_clf.predict(X_test)
    rf_latency_ms = ((time.perf_counter() - t0) * 1000.0) / len(X_test)

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(
        y_test, rf_preds, average="weighted", zero_division=0
    )

    # =========================================================================
    # 2. PELATIHAN & EVALUASI FFNN (Feedforward Neural Network)
    # =========================================================================
    print("--- Melatih Model FFNN (Abjad) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ffnn = SignFFNN(input_size=FEATURE_SIZE, num_classes=len(unique_labels), dropout=0.25).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ffnn.parameters(), lr=0.001)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=16,
        shuffle=True,
    )

    best_val_acc = 0.0
    for epoch in range(1, 61):
        ffnn.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(ffnn(bx), by)
            loss.backward()
            optimizer.step()

        ffnn.eval()
        with torch.no_grad():
            val_out = ffnn(torch.tensor(X_val, device=device))
            val_preds = torch.argmax(val_out, dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(ffnn.state_dict(), FFNN_LETTER_MODEL_PATH)

    ffnn.load_state_dict(torch.load(FFNN_LETTER_MODEL_PATH, map_location=device, weights_only=True))
    ffnn.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        test_out = ffnn(torch.tensor(X_test, device=device))
        ffnn_preds = torch.argmax(test_out, dim=1).cpu().numpy()
    ffnn_latency_ms = ((time.perf_counter() - t0) * 1000.0) / len(X_test)

    ffnn_acc = accuracy_score(y_test, ffnn_preds)
    ffnn_prec, ffnn_rec, ffnn_f1, _ = precision_recall_fscore_support(
        y_test, ffnn_preds, average="weighted", zero_division=0
    )

    # =========================================================================
    # 3. SIMPAN LAPORAN PERBANDINGAN METRIK & CONFUSION MATRIX
    # =========================================================================
    comp_csv = os.path.join(EVALUATION_DIR, "letter_models_comparison.csv")
    with open(comp_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1_Score", "Avg_Latency_ms"])
        writer.writerow(["Random Forest", f"{rf_acc:.4f}", f"{rf_prec:.4f}", f"{rf_rec:.4f}", f"{rf_f1:.4f}", f"{rf_latency_ms:.3f}"])
        writer.writerow(["FFNN", f"{ffnn_acc:.4f}", f"{ffnn_prec:.4f}", f"{ffnn_rec:.4f}", f"{ffnn_f1:.4f}", f"{ffnn_latency_ms:.3f}"])

    np.savetxt(
        os.path.join(EVALUATION_DIR, "confusion_matrix_rf_letter.csv"),
        confusion_matrix(y_test, rf_preds),
        delimiter=",",
        fmt="%d",
    )
    np.savetxt(
        os.path.join(EVALUATION_DIR, "confusion_matrix_ffnn_letter.csv"),
        confusion_matrix(y_test, ffnn_preds),
        delimiter=",",
        fmt="%d",
    )

    print("\n" + "=" * 65)
    print("  HASIL PERBANDINGAN MODEL ABJAD (Uji T-03)  ")
    print("=" * 65)
    print(f"1. Random Forest -> Akurasi: {rf_acc*100:.2f}% | F1: {rf_f1:.4f} | Latensi: {rf_latency_ms:.3f} ms/frame")
    print(f"2. FFNN          -> Akurasi: {ffnn_acc*100:.2f}% | F1: {ffnn_f1:.4f} | Latensi: {ffnn_latency_ms:.3f} ms/frame")
    print(f"Laporan disimpan di: {comp_csv}")


if __name__ == "__main__":
    main()

