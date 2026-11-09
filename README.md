# 🤟 IsyaratKu — Sistem Penerjemah Bahasa Isyarat SIBI Berbasis Web

**Pengembangan Sistem Penerjemah Bahasa Isyarat Berbasis Google MediaPipe dan Machine Learning sebagai Alat Bantu Komunikasi Tunarungu**  
*(Proyek III — Program Studi D-IV Teknik Informatika, Universitas Logistik dan Bisnis Internasional)*

### 👥 Tim Pengusul
1. **Alif Satria Raghib (714240005)** — *Jalur Abjad (Gestur Statis: Random Forest & FFNN), Normalisasi 63 Fitur Landmark, Endpoint `/api/predict/letter`, Mode Abjad, Penyangga Teks, Riwayat `LocalStorage`, Pengukuran FPS/Latensi.*
2. **Zahra Nur’azijah Lutfiani (714240052)** — *Jalur Kata Dasar (Gestur Dinamis: LSTM 30 Bingkai), Kerangka Server FastAPI, Endpoint `/api/predict/word`, Antarmuka Web (Beranda, Izin Kamera, Visualisasi 21 Landmark, Mode Kata Dasar).*

---

## 📂 Struktur Proyek (Modular Sesuai Proposal Bab III)

```text
Proyek3/
├── backend/                          # Layanan Aplikasi (Server Python FastAPI & Machine Learning)
│   ├── app/
│   │   ├── main.py                   # Entrypoint FastAPI (/api/predict/letter, /api/predict/word, /api/health)
│   │   ├── config.py                 # Konfigurasi 63 fitur, urutan 30 bingkai, label, & ambang kepercayaan
│   │   ├── schemas.py                # Validasi JSON Request/Response (Keamanan API)
│   │   ├── extractor.py              # OpenCV + MediaPipe Hands (21 landmark 3D) + Normalisasi Pergelangan Tangan
│   │   └── predictor.py              # Modul Klasifikasi (Random Forest .pkl, FFNN .pt, LSTM .pt)
│   │
│   ├── ml_pipeline/                  # Skrip Pengembang (Eksperimen, Koleksi Data, Pelatihan & Evaluasi)
│   │   ├── collect_letters.py        # [Jalur Abjad - Alif] Koleksi dataset abjad SIBI ke CSV (63 fitur)
│   │   ├── train_letters.py          # [Jalur Abjad - Alif] Pelatihan & perbandingan Random Forest vs FFNN
│   │   ├── collect_words.py          # [Jalur Kata Dasar - Zahra] Koleksi urutan 30 bingkai kata dasar (.npy)
│   │   ├── train_words.py            # [Jalur Kata Dasar - Zahra] Pelatihan & evaluasi model LSTM
│   │   └── test_robustness.py        # [Uji T-06, T-07, T-09] Pengujian jarak (30cm vs 60cm), cahaya, & FPS
│   │
│   ├── data/                         # Penyimpanan dataset lokal server
│   │   ├── letters/                  # Dataset landmark abjad (letters_landmarks.csv)
│   │   └── words/                    # Dataset urutan gerakan kata dasar (30 bingkai x 63 fitur)
│   │
│   ├── models/                       # Penyimpanan berkas model terlatih (.pkl, .pt, .task, & JSON label)
│   ├── evaluation/                   # Hasil metrik evaluasi (CSV akurasi/precision/recall/F1 & confusion matrix)
│   └── requirements.txt              # Dependensi Python Backend
│
├── frontend/                         # Antarmuka Web "IsyaratKu" (React + Vite + Tailwind CSS)
│   ├── src/
│   │   ├── components/
│   │   │   └── Navbar.jsx            # Navigasi utama & indikator "Sistem Siap"
│   │   ├── pages/
│   │   │   ├── HomePage.jsx          # Gambar 3.4: Halaman Beranda
│   │   │   ├── CameraPermission.jsx  # Gambar 3.5: Layar Izin Akses Kamera (Uji T-01)
│   │   │   ├── TranslatorPage.jsx    # Gambar 3.6: Halaman Penerjemah (Abjad & Kata Dasar + 21 Landmark)
│   │   │   ├── HistoryPage.jsx       # Gambar 3.7 & 3.8: Halaman Riwayat Terjemahan (Web LocalStorage)
│   │   │   └── AboutPage.jsx         # Gambar 3.9: Halaman Tentang IsyaratKu
│   │   └── services/
│   │       ├── api.js                # Klien HTTP REST API ke Backend FastAPI
│   │       └── storage.js            # Manajemen Riwayat Terjemahan pada LocalStorage peramban
│   └── package.json
│
└── legacy_desktop/                   # Arsip prototipe desktop awal
```

---

## 🚀 Cara Menjalankan Sistem

### 1. Menjalankan Backend (Python FastAPI)
Buka terminal pertama di direktori `backend/`:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- Server API berjalan di: `http://127.0.0.1:8000`
- Dokumentasi interaktif Swagger UI (untuk pengujian endpoint `/api/predict/letter` dan `/api/predict/word`): `http://127.0.0.1:8000/docs`

### 2. Menjalankan Frontend (React + Vite)
Buka terminal kedua di direktori `frontend/`:
```bash
cd frontend
npm install
npm run dev
```
- Buka peramban pada alamat: `http://localhost:5173`
