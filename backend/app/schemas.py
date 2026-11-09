from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class LandmarkPoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    x: float
    y: float
    z: float


class LetterPredictRequest(BaseModel):
    """
    Request payload untuk endpoint POST /api/predict/letter.
    Dapat menerima bingkai gambar Base64 dari peramban (getUserMedia)
    atau vektor 63 fitur landmark secara langsung untuk pengujian.
    """
    image_base64: Optional[str] = Field(
        None, description="Bingkai gambar kamera dalam format Base64 (JPEG/PNG)"
    )
    features: Optional[List[float]] = Field(
        None, description="Opsional: Vektor 63 fitur landmark yang sudah dinormalisasi"
    )
    model_type: Literal["random_forest", "ffnn"] = Field(
        "random_forest", description="Model klasifikasi abjad yang digunakan"
    )
    threshold: float = Field(
        0.70, ge=0.0, le=1.0, description="Ambang kepercayaan (confidence threshold)"
    )


class WordPredictRequest(BaseModel):
    """
    Request payload untuk endpoint POST /api/predict/word.
    Dapat menerima urutan bingkai fitur beserta bingkai kamera terbaru.
    """
    sequence_features: Optional[List[List[float]]] = Field(
        None, description="Urutan bingkai x 63 fitur landmark ternormalisasi"
    )
    image_base64: Optional[str] = Field(
        None, description="Bingkai gambar kamera dalam format Base64 untuk diekstraksi"
    )
    threshold: float = Field(
        0.70, ge=0.0, le=1.0, description="Ambang kepercayaan (confidence threshold)"
    )


class PredictResponse(BaseModel):
    """
    Response JSON standar dari endpoint prediksi (Sesuai Gambar 3.1 Proposal).
    """
    hand_detected: bool
    label: Optional[str] = None
    confidence: float = 0.0
    passed_threshold: bool = False
    model_used: str
    inference_time_ms: float
    landmarks: List[LandmarkPoint] = []
    normalized_features: List[float] = []
    message: Optional[str] = None
