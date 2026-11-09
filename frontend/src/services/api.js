const API_BASE = import.meta.env.VITE_API_URL || '';

export async function checkServerHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error('Server tidak merespons');
  return res.json();
}

export async function predictLetter({ imageBase64, modelType = 'random_forest', threshold = 0.70 }) {
  const res = await fetch(`${API_BASE}/api/predict/letter`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_base64: imageBase64,
      model_type: modelType,
      threshold,
    }),
  });
  if (!res.ok) throw new Error('Gagal memproses prediksi abjad');
  return res.json();
}

export async function predictWord({ imageBase64 = null, sequenceFeatures = null, threshold = 0.70 }) {
  const res = await fetch(`${API_BASE}/api/predict/word`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_base64: imageBase64,
      sequence_features: sequenceFeatures,
      threshold,
    }),
  });
  if (!res.ok) throw new Error('Gagal memproses prediksi kata dasar');
  return res.json();
}

