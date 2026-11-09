import os

# Base directory: folder backend/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Direktori penyimpanan lokal server (Sesuai Tabel 3.1 Proposal)
DATA_DIR = os.path.join(BASE_DIR, "data")
LETTERS_DATA_DIR = os.path.join(DATA_DIR, "letters")
WORDS_DATA_DIR = os.path.join(DATA_DIR, "words")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVALUATION_DIR = os.path.join(BASE_DIR, "evaluation")

# Berkas dataset & model
LETTERS_CSV_PATH = os.path.join(LETTERS_DATA_DIR, "letters_landmarks.csv")
MEDIAPIPE_TASK_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
MEDIAPIPE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Berkas model klasifikasi (Sesuai Tabel 3.1 Proposal)
RF_LETTER_MODEL_PATH = os.path.join(MODELS_DIR, "rf_letter_model.pkl")
FFNN_LETTER_MODEL_PATH = os.path.join(MODELS_DIR, "ffnn_letter_model.pt")
LSTM_WORD_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_word_model.pt")

# Berkas pemetaan label JSON
LABELS_LETTER_JSON = os.path.join(MODELS_DIR, "labels_letter.json")
LABELS_WORD_JSON = os.path.join(MODELS_DIR, "labels_word.json")

# Spesifikasi fitur (Sesuai Kebutuhan F-02, F-03, F-05 Proposal)
NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
FEATURE_SIZE = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63 fitur (x0,y0,z0 .. x20,y20,z20)
SEQUENCE_LENGTH = 30  # 30 bingkai untuk gestur dinamis / kata dasar

# Ambang kepercayaan default (Kebutuhan F-07)
DEFAULT_CONFIDENCE_THRESHOLD = 0.70

# Daftar default Abjad SIBI (A-Z)
DEFAULT_LETTER_LABELS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

# Daftar default Kata Dasar SIBI
DEFAULT_WORD_LABELS = [
    "HALO",
    "SAYA",
    "MAKAN",
    "MINUM",
    "TOLONG",
    "TERIMA KASIH",
    "SAMA-SAMA",
    "SELAMAT PAGI",
]

