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

# Daftar Abjad SIBI beserta deskripsi posisi jari (Sesuai Gambar 3.6 Proposal)
SIBI_LETTER_GUIDE = {
    "A": "Kepalan tangan dengan ibu jari menempel tegak di samping telunjuk",
    "B": "Keempat jari lurus rapat ke atas, ibu jari ditekuk di depan telapak",
    "C": "Kelima jari melengkung membentuk setengah lingkaran huruf C",
    "D": "Telunjuk lurus ke atas, jari tengah-manis-kelingking menyentuh ibu jari",
    "E": "Kelima jari ditekuk merapat ke arah telapak tangan",
    "F": "Ujung telunjuk dan ibu jari bertemu, tiga jari lainnya lurus ke atas",
    "G": "Telunjuk dan ibu jari lurus mengarah ke samping",
    "H": "Telunjuk dan jari tengah lurus rapat mengarah ke samping",
    "I": "Jari kelingking lurus ke atas, keempat jari lainnya mengepal",
    "J": "Jari kelingking lurus lalu digerakkan mengayun membentuk huruf J",
    "K": "Telunjuk dan jari tengah terbuka ke atas, ibu jari di antara keduanya",
    "L": "Telunjuk lurus ke atas dan ibu jari terentang ke samping (sudut 90°)",
    "M": "Ibu jari diselipkan di bawah tiga jari (telunjuk, tengah, manis)",
    "N": "Ibu jari diselipkan di bawah dua jari (telunjuk dan tengah)",
    "O": "Seluruh ujung jari bertemu ujung ibu jari membentuk bulatan O",
    "P": "Posisi seperti huruf K namun diarahkan condong ke bawah",
    "Q": "Telunjuk dan ibu jari mengarah ke bawah",
    "R": "Jari telunjuk dan jari tengah lurus ke atas saling menyilang",
    "S": "Kepalan tangan rapat dengan ibu jari menyilang di depan jari-jari",
    "T": "Ibu jari diselipkan di antara jari telunjuk dan jari tengah",
    "U": "Jari telunjuk dan jari tengah lurus rapat ke atas",
    "V": "Jari telunjuk dan jari tengah lurus terbuka membentuk huruf V",
    "W": "Tiga jari (telunjuk, tengah, manis) lurus terbuka membentuk huruf W",
    "X": "Jari telunjuk ditekuk membentuk kait, jari lainnya mengepal",
    "Y": "Ibu jari dan kelingking terentang keluar, tiga jari tengah ditekuk",
    "Z": "Jari telunjuk lurus menggambar pola zig-zag huruf Z di udara",
}

# Daftar Kata Dasar SIBI beserta panduan gerakan (30 bingkai)
SIBI_WORD_GUIDE = {
    "HALO": "Telapak tangan terbuka (5 jari) digerakkan melambai ke kiri dan kanan",
    "SAYA": "Jari telunjuk atau telapak tangan diarahkan menunjuk ke dada sendiri",
    "MAKAN": "Ujung kelima jari menguncup rapat lalu digerakkan ke arah mulut",
    "MINUM": "Tangan membentuk huruf C (seperti memegang gelas) digerakkan ke arah mulut",
    "TOLONG": "Ibu jari tegak di atas kepalan tangan lalu digerakkan naik ke atas",
    "TERIMA KASIH": "Telapak tangan terbuka dari dekat dagu digerakkan maju ke depan",
    "SAMA-SAMA": "Tangan dengan ibu jari & kelingking terbuka (Y) digerakkan maju-mundur",
    "SELAMAT PAGI": "Telapak tangan terbuka diangkat menyapa dari bawah ke atas",
}

DEFAULT_LETTER_LABELS = list(SIBI_LETTER_GUIDE.keys())
DEFAULT_WORD_LABELS = list(SIBI_WORD_GUIDE.keys())
