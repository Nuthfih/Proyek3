import os
import numpy as np

# Daftar kata/gestur SIBI yang akan dikenali
ACTIONS = np.array([
    "HALO",
    "TERIMA KASIH",
    "SAYA",
    "MAKAN",
    "SAMA_SAMA"
])

# Parameter video dan dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model_lstm.pt")

# Jumlah video/sampel per kata (misal 30 rekaman per kata)
NO_SEQUENCES = 30

# Panjang frame dalam 1 sequence gerakan (30 frame ~ 1 detik)
SEQUENCE_LENGTH = 30

# Dimensi fitur per frame:
# 21 titik tangan kiri (x, y, z) = 63
# 21 titik tangan kanan (x, y, z) = 63
# Total = 126 fitur per frame
FEATURE_SIZE = 126
