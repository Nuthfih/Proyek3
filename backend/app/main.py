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
    SIBI_LETTER_GUIDE,
    SIBI_WORD_GUIDE,
)
from .schemas import (
    LandmarkPoint,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gunakan ambang deteksi tangan 0.35 agar tetap responsif saat tangan melambai atau cahaya ruangan redup
extractor = HandLandmarkExtractor(max_hands=1, min_confidence=0.35)
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
        "letter_guide": SIBI_LETTER_GUIDE,
        "word_guide": SIBI_WORD_GUIDE,
    }


@app.get("/api/gestures")
def get_gestures_guide():
    """Mengembalikan daftar lengkap gestur Abjad dan Kata Dasar SIBI beserta panduannya."""
    return {
        "letters": [
            {"label": k, "description": v} for k, v in SIBI_LETTER_GUIDE.items()
        ],
        "words": [
            {"label": k, "description": v} for k, v in SIBI_WORD_GUIDE.items()
        ],
    }


@app.post("/api/predict/letter", response_model=PredictResponse)
def predict_letter_endpoint(payload: LetterPredictRequest):
    """
    Endpoint klasifikasi gestur statis / Abjad SIBI (Kontribusi Jalur Abjad).
    Menerima citra Base64 dari kamera web atau vektor 63 fitur langsung.
    """
    t_start = time.perf_counter()
    landmarks_out = []

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
        hand_detected, landmarks_raw, features_63 = extractor.extract_from_frame(frame)
        landmarks_out = [
            LandmarkPoint(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks_raw
        ]
    else:
        raise HTTPException(
            status_code=400,
            detail="Harus menyertakan 'image_base64' atau 'features'.",
        )

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
        landmarks=landmarks_out,
        normalized_features=features_63.tolist(),
        message="Prediksi berhasil" if passed else "Di bawah ambang kepercayaan",
    )


@app.post("/api/predict/word", response_model=PredictResponse)
def predict_word_endpoint(payload: WordPredictRequest):
    """
    Endpoint klasifikasi gestur dinamis / Kata Dasar SIBI dengan LSTM (Kontribusi Jalur Kata Dasar).
    Mengekstrak bingkai kamera terbaru dan langsung memprediksi kata dasar dari urutan bingkai.
    """
    t_start = time.perf_counter()
    landmarks_out = []
    features_63 = np.zeros(FEATURE_SIZE, dtype=np.float32)
    hand_detected = False

    if payload.image_base64:
        frame = extractor.decode_base64_image(payload.image_base64)
        if frame is None:
            raise HTTPException(status_code=400, detail="Format citra Base64 tidak valid.")
        hand_detected, landmarks_raw, features_63 = extractor.extract_from_frame(frame)
        landmarks_out = [
            LandmarkPoint(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks_raw
        ]

    # Gabungkan riwayat sequence_features dengan bingkai terbaru jika tangan terdeteksi
    seq_list = list(payload.sequence_features) if payload.sequence_features else []
    if hand_detected:
        seq_list.append(features_63.tolist())

    # Jika tidak ada tangan pada bingkai ini dan tidak mengirim sequence khusus
    if not hand_detected and not payload.sequence_features:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return PredictResponse(
            hand_detected=False,
            label=None,
            confidence=0.0,
            passed_threshold=False,
            model_used="lstm",
            inference_time_ms=round(total_ms, 2),
            landmarks=[],
            normalized_features=[],
            message="Tangan tidak terdeteksi pada bingkai.",
        )

    # Jika urutan sudah memiliki minimal 6 bingkai (~0.6 detik gerakan), jalankan prediksi kata
    if len(seq_list) >= 6:
        seq_arr = np.array(seq_list[-SEQUENCE_LENGTH:], dtype=np.float32)
        if seq_arr.ndim == 2 and seq_arr.shape[1] == FEATURE_SIZE:
            label, confidence, _ = predictor.predict_word(seq_arr)
            total_ms = (time.perf_counter() - t_start) * 1000.0
            passed = confidence >= payload.threshold
            return PredictResponse(
                hand_detected=hand_detected or True,
                label=label if passed else None,
                confidence=round(confidence, 4),
                passed_threshold=passed,
                model_used="lstm",
                inference_time_ms=round(total_ms, 2),
                landmarks=landmarks_out,
                normalized_features=features_63.tolist() if hand_detected else [],
                message="Prediksi kata dasar berhasil" if passed else "Di bawah ambang kepercayaan",
            )

    total_ms = (time.perf_counter() - t_start) * 1000.0
    return PredictResponse(
        hand_detected=hand_detected,
        label=None,
        confidence=0.0,
        passed_threshold=False,
        model_used="lstm",
        inference_time_ms=round(total_ms, 2),
        landmarks=landmarks_out,
        normalized_features=features_63.tolist() if hand_detected else [],
        message=f"Mengumpulkan urutan bingkai ({len(seq_list)}/6)...",
    )
