import time
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    FEATURE_SIZE,
    SEQUENCE_LENGTH,
    RF_LETTER_MODEL_PATH,
    FFNN_LETTER_MODEL_PATH,
    LSTM_WORD_MODEL_PATH,
)
from .schemas import (
    LetterPredictRequest,
    WordPredictRequest,
    PredictResponse,
)
from .extractor import HandLandmarkExtractor
from .predictor import SignPredictorManager

app = FastAPI(
    title="IsyaratKu API - Sistem Penerjemah Bahasa Isyarat SIBI",
    description=(
        "Layanan Backend REST API berbasis Python FastAPI, OpenCV, Google MediaPipe Hands, "
        "serta model Random Forest, FFNN, dan LSTM untuk penerjemahan gestur abjad dan kata dasar SIBI."
    ),
    version="1.0.0",
)

# Izinkan akses dari Frontend Web (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi Ekstraktor MediaPipe & Manajer Model satu kali saat server dijalankan (Subbab 3.7)
extractor = HandLandmarkExtractor(max_hands=1, min_confidence=0.5)
predictor = SignPredictorManager()


@app.get("/api/health")
def health_check():
    """Memeriksa kesiapan server, ketersediaan model, dan daftar label."""
    return {
        "status": "ready",
        "app_name": "IsyaratKu",
        "feature_size": FEATURE_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "models_loaded": {
            "random_forest_letter": os.path.exists(RF_LETTER_MODEL_PATH),
            "ffnn_letter": os.path.exists(FFNN_LETTER_MODEL_PATH),
            "lstm_word": os.path.exists(LSTM_WORD_MODEL_PATH),
        },
        "letter_labels": predictor.letter_labels,
        "word_labels": predictor.word_labels,
    }


@app.post("/api/predict/letter", response_model=PredictResponse)
def predict_letter_endpoint(payload: LetterPredictRequest):
    """
    Endpoint klasifikasi gestur statis / Abjad SIBI (Kontribusi Jalur Abjad).
    Menerima citra Base64 dari kamera web atau vektor 63 fitur langsung.
    """
    t_start = time.perf_counter()
    landmarks_list = []

    if payload.features is not None:
        if len(payload.features) != FEATURE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Panjang fitur harus {FEATURE_SIZE} koordinat (21 landmark x 3).",
            )
        features_63 = np.array(payload.features, dtype=np.float32)
        hand_detected = True
    elif payload.image_base64:
        frame = extractor.decode_base64_image(payload.image_base64)
        if frame is None:
            raise HTTPException(status_code=400, detail="Format citra Base64 tidak valid.")
        hand_detected, landmarks_list, features_63 = extractor.extract_from_frame(frame)
    else:
        raise HTTPException(
            status_code=400,
            detail="Harus menyertakan 'image_base64' atau 'features'.",
        )

    # Jika tangan tidak terdeteksi, lewati klasifikasi tanpa error (Uji T-02 & Diagram Activity)
    if not hand_detected:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return PredictResponse(
            hand_detected=False,
            label=None,
            confidence=0.0,
            passed_threshold=False,
            model_used=payload.model_type,
            inference_time_ms=round(total_ms, 2),
            landmarks=[],
            normalized_features=[],
            message="Tangan tidak terdeteksi pada bingkai.",
        )

    label, confidence, _ = predictor.predict_letter(features_63, model_type=payload.model_type)
    total_ms = (time.perf_counter() - t_start) * 1000.0
    passed = confidence >= payload.threshold

    return PredictResponse(
        hand_detected=True,
        label=label if passed else None,
        confidence=round(confidence, 4),
        passed_threshold=passed,
        model_used=payload.model_type,
        inference_time_ms=round(total_ms, 2),
        landmarks=landmarks_list,
        normalized_features=features_63.tolist(),
        message="Prediksi berhasil" if passed else "Di bawah ambang kepercayaan",
    )


@app.post("/api/predict/word", response_model=PredictResponse)
def predict_word_endpoint(payload: WordPredictRequest):
    """
    Endpoint klasifikasi gestur dinamis / Kata Dasar SIBI dengan LSTM (Kontribusi Jalur Kata Dasar).
    Dapat mengekstrak bingkai tunggal untuk buffer urutan, atau memprediksi urutan 30x63 fitur.
    """
    t_start = time.perf_counter()
    landmarks_list = []
    features_63 = np.zeros(FEATURE_SIZE, dtype=np.float32)
    hand_detected = False

    # Jika mengirim citra tunggal untuk ekstraksi landmark sebelum buffer penuh 30 bingkai
    if payload.image_base64:
        frame = extractor.decode_base64_image(payload.image_base64)
        if frame is None:
            raise HTTPException(status_code=400, detail="Format citra Base64 tidak valid.")
        hand_detected, landmarks_list, features_63 = extractor.extract_from_frame(frame)

    # Jika belum menyertakan sequence_features lengkap (hanya ekstraksi bingkai)
    if not payload.sequence_features:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return PredictResponse(
            hand_detected=hand_detected,
            label=None,
            confidence=0.0,
            passed_threshold=False,
            model_used="lstm",
            inference_time_ms=round(total_ms, 2),
            landmarks=landmarks_list,
            normalized_features=features_63.tolist() if hand_detected else [],
            message="Bingkai diekstraksi untuk buffer urutan kata dasar.",
        )

    # Prediksi urutan (sequence_features) menggunakan model LSTM
    seq_arr = np.array(payload.sequence_features, dtype=np.float32)
    if seq_arr.ndim != 2 or seq_arr.shape[1] != FEATURE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Dimensi sequence_features harus (N, {FEATURE_SIZE}).",
        )

    label, confidence, _ = predictor.predict_word(seq_arr)
    total_ms = (time.perf_counter() - t_start) * 1000.0
    passed = confidence >= payload.threshold

    return PredictResponse(
        hand_detected=True,
        label=label if passed else None,
        confidence=round(confidence, 4),
        passed_threshold=passed,
        model_used="lstm",
        inference_time_ms=round(total_ms, 2),
        landmarks=landmarks_list,
        normalized_features=features_63.tolist() if hand_detected else [],
        message="Prediksi kata dasar berhasil" if passed else "Di bawah ambang kepercayaan",
    )

