import os
import json
import time
from typing import Tuple, List
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    MODELS_DIR,
    RF_LETTER_MODEL_PATH,
    FFNN_LETTER_MODEL_PATH,
    LSTM_WORD_MODEL_PATH,
    LABELS_LETTER_JSON,
    LABELS_WORD_JSON,
    DEFAULT_LETTER_LABELS,
    DEFAULT_WORD_LABELS,
    FEATURE_SIZE,
    SEQUENCE_LENGTH,
)


class SignFFNN(nn.Module):
    """
    Model Feedforward Neural Network (FFNN) untuk klasifikasi gestur statis (Abjad SIBI).
    Masukan: Vektor 63 fitur landmark ternormalisasi.
    """

    def __init__(self, input_size: int = FEATURE_SIZE, num_classes: int = 26, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SignLSTM(nn.Module):
    """
    Model Long Short-Term Memory (LSTM) untuk klasifikasi gestur dinamis (Kata Dasar SIBI).
    Masukan: Urutan bingkai berukuran (batch, 30, 63).
    """

    def __init__(
        self,
        input_size: int = FEATURE_SIZE,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class SignPredictorManager:
    """
    Modul Klasifikasi: Memuat model Random Forest (.pkl), FFNN (.pt), dan LSTM (.pt)
    satu kali saat server dijalankan (Subbab 3.7 Proposal).
    """

    def __init__(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.letter_labels: List[str] = self._load_or_init_labels(
            LABELS_LETTER_JSON, DEFAULT_LETTER_LABELS
        )
        self.word_labels: List[str] = self._load_or_init_labels(
            LABELS_WORD_JSON, DEFAULT_WORD_LABELS
        )

        self.rf_model = None
        self.ffnn_model = None
        self.lstm_model = None

        self.reload_models()

    @staticmethod
    def _load_or_init_labels(json_path: str, default_labels: List[str]) -> List[str]:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(default_labels, f, indent=2, ensure_ascii=False)
        return default_labels

    def reload_models(self):
        """Memuat ulang berkas model dari folder backend/models/."""
        self.letter_labels = self._load_or_init_labels(
            LABELS_LETTER_JSON, DEFAULT_LETTER_LABELS
        )
        self.word_labels = self._load_or_init_labels(
            LABELS_WORD_JSON, DEFAULT_WORD_LABELS
        )

        # 1. Muat Random Forest (Abjad)
        if os.path.exists(RF_LETTER_MODEL_PATH):
            try:
                self.rf_model = joblib.load(RF_LETTER_MODEL_PATH)
                print(f"[INFO] Model Random Forest dimuat dari {RF_LETTER_MODEL_PATH}")
            except Exception as e:
                print(f"[WARN] Gagal memuat Random Forest: {e}")

        # 2. Muat FFNN (Abjad)
        self.ffnn_model = SignFFNN(
            input_size=FEATURE_SIZE, num_classes=len(self.letter_labels)
        ).to(self.device)
        if os.path.exists(FFNN_LETTER_MODEL_PATH):
            try:
                self.ffnn_model.load_state_dict(
                    torch.load(FFNN_LETTER_MODEL_PATH, map_location=self.device, weights_only=True)
                )
                print(f"[INFO] Model FFNN dimuat dari {FFNN_LETTER_MODEL_PATH}")
            except Exception as e:
                print(f"[WARN] Gagal memuat bobot FFNN: {e}")
        self.ffnn_model.eval()

        # 3. Muat LSTM (Kata Dasar)
        self.lstm_model = SignLSTM(
            input_size=FEATURE_SIZE, num_classes=len(self.word_labels)
        ).to(self.device)
        if os.path.exists(LSTM_WORD_MODEL_PATH):
            try:
                self.lstm_model.load_state_dict(
                    torch.load(LSTM_WORD_MODEL_PATH, map_location=self.device, weights_only=True)
                )
                print(f"[INFO] Model LSTM dimuat dari {LSTM_WORD_MODEL_PATH}")
            except Exception as e:
                print(f"[WARN] Gagal memuat bobot LSTM: {e}")
        self.lstm_model.eval()

    def predict_letter(
        self, features_63: np.ndarray, model_type: str = "random_forest"
    ) -> Tuple[str, float, float]:
        """
        Klasifikasi gestur statis (abjad SIBI) dari vektor 63 fitur.
        Mengembalikan: (label, confidence, inference_time_ms)
        """
        t0 = time.perf_counter()
        x = np.array(features_63, dtype=np.float32).reshape(1, FEATURE_SIZE)

        if model_type == "random_forest" and self.rf_model is not None:
            probs = self.rf_model.predict_proba(x)[0]
            classes = self.rf_model.classes_
            best_idx = int(np.argmax(probs))
            label = str(classes[best_idx])
            confidence = float(probs[best_idx])
        elif model_type == "ffnn" and os.path.exists(FFNN_LETTER_MODEL_PATH):
            with torch.no_grad():
                tensor_x = torch.tensor(x, dtype=torch.float32, device=self.device)
                logits = self.ffnn_model(tensor_x)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                best_idx = int(np.argmax(probs))
                label = self.letter_labels[best_idx]
                confidence = float(probs[best_idx])
        else:
            # Heuristik geometris dasar saat berkas model belum dilatih
            label, confidence = self._heuristic_letter(features_63)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return label, confidence, elapsed_ms

    def predict_word(self, sequence_30x63: np.ndarray) -> Tuple[str, float, float]:
        """
        Klasifikasi gestur dinamis (kata dasar SIBI) dari urutan (30, 63) fitur.
        Mengembalikan: (label, confidence, inference_time_ms)
        """
        t0 = time.perf_counter()
        seq = np.array(sequence_30x63, dtype=np.float32)

        # Jika jumlah bingkai kurang/lebih dari 30, sesuaikan ke 30 bingkai
        if len(seq) < SEQUENCE_LENGTH:
            pad = np.repeat(seq[-1:], SEQUENCE_LENGTH - len(seq), axis=0)
            seq = np.vstack([seq, pad])
        elif len(seq) > SEQUENCE_LENGTH:
            seq = seq[-SEQUENCE_LENGTH:]

        seq_batch = seq.reshape(1, SEQUENCE_LENGTH, FEATURE_SIZE)

        if os.path.exists(LSTM_WORD_MODEL_PATH):
            with torch.no_grad():
                tensor_x = torch.tensor(seq_batch, dtype=torch.float32, device=self.device)
                logits = self.lstm_model(tensor_x)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                best_idx = int(np.argmax(probs))
                label = self.word_labels[best_idx]
                confidence = float(probs[best_idx])
        else:
            label, confidence = self._heuristic_word(seq)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return label, confidence, elapsed_ms

    @staticmethod
    def _heuristic_letter(features_63: np.ndarray) -> Tuple[str, float]:
        """Fallback aturan jari dasar agar UI dapat langsung diuji sebelum model dilatih."""
        pts = features_63.reshape(21, 3)
        # Cek jari terangkat (ujung jari lebih tinggi / y lebih kecil dari sendi PIP)
        index_up = pts[8, 1] < pts[6, 1]
        middle_up = pts[12, 1] < pts[10, 1]
        ring_up = pts[16, 1] < pts[14, 1]
        pinky_up = pts[20, 1] < pts[18, 1]
        thumb_out = abs(pts[4, 0] - pts[2, 0]) > 0.22

        up_count = sum([index_up, middle_up, ring_up, pinky_up])
        if up_count == 0 and thumb_out:
            return "A", 0.92
        if up_count == 4 and not thumb_out:
            return "B", 0.94
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "D", 0.90
        if not index_up and not middle_up and not ring_up and pinky_up:
            return "I", 0.91
        if index_up and not middle_up and not ring_up and not pinky_up and thumb_out:
            return "L", 0.93
        if index_up and middle_up and not ring_up and not pinky_up:
            dist_uv = abs(pts[8, 0] - pts[12, 0])
            return ("V", 0.92) if dist_uv > 0.15 else ("U", 0.89)
        if index_up and middle_up and ring_up and not pinky_up:
            return "W", 0.91
        if not index_up and not middle_up and not ring_up and pinky_up and thumb_out:
            return "Y", 0.90
        return "S", 0.85

    @staticmethod
    def _heuristic_word(seq_30x63: np.ndarray) -> Tuple[str, float]:
        """Fallback aturan gerakan dasar kata sebelum model LSTM dilatih."""
        first = seq_30x63[0].reshape(21, 3)
        last = seq_30x63[-1].reshape(21, 3)
        movement = float(np.linalg.norm(last - first))
        if movement > 0.45:
            return "HALO", 0.91
        if movement > 0.25:
            return "TERIMA KASIH", 0.88
        return "SAYA", 0.86

