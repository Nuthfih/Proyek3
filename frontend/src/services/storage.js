const STORAGE_KEY = 'isyaratku_history_v1';

export function getTranslationHistory() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function addTranslationHistory({ text, mode, confidence = 0.9 }) {
  const cleanText = (text || '').trim();
  if (!cleanText) return getTranslationHistory();

  const current = getTranslationHistory();
  const newItem = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    text: cleanText,
    mode: mode === 'word' ? 'Kata Dasar' : 'Abjad',
    confidence: Math.round(confidence * 100),
    charCount: cleanText.length,
    createdAt: new Date().toISOString(),
  };

  const updated = [newItem, ...current].slice(0, 100);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  return updated;
}

export function deleteHistoryItem(id) {
  const current = getTranslationHistory().filter((item) => item.id !== id);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(current));
  return current;
}

export function clearAllHistory() {
  window.localStorage.removeItem(STORAGE_KEY);
  return [];
}

