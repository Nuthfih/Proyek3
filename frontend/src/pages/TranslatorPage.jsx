import React, { useEffect, useRef, useState } from 'react';
import {
  Camera,
  Square,
  Copy,
  RotateCcw,
  Delete,
  Check,
  Lock,
  Info,
  BookmarkPlus,
} from 'lucide-react';
import { predictLetter, predictWord } from '../services/api';
import { addTranslationHistory, getTranslationHistory } from '../services/storage';

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export default function TranslatorPage({ onViewAllHistory }) {
  const videoRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const captureCanvasRef = useRef(null);

  const [mode, setMode] = useState('letter'); // 'letter' | 'word'
  const [modelType, setModelType] = useState('random_forest'); // 'random_forest' | 'ffnn'
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [showLandmarks, setShowLandmarks] = useState(true);

  // State hasil deteksi real-time
  const [handDetected, setHandDetected] = useState(false);
  const [detectedLabel, setDetectedLabel] = useState('-');
  const [confidence, setConfidence] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [fps, setFps] = useState(0);

  // Penyangga huruf/kata (Kebutuhan F-07)
  const [translatedText, setTranslatedText] = useState('');
  const [copied, setCopied] = useState(false);
  const [recentHistory, setRecentHistory] = useState(() => getTranslationHistory().slice(0, 3));

  const wordBufferRef = useRef([]);
  const stabilityBufferRef = useRef([]);
  const lastAppendedRef = useRef({ label: '', time: 0 });
  const showLandmarksRef = useRef(showLandmarks);
  showLandmarksRef.current = showLandmarks;

  useEffect(() => {
    navigator.mediaDevices
      ?.enumerateDevices()
      .then((list) => {
        const cams = list.filter((d) => d.kind === 'videoinput');
        setDevices(cams);
        if (cams.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(cams[0].deviceId);
        }
      })
      .catch(() => {});
  }, [selectedDeviceId]);

  const startCamera = async (deviceIdOverride) => {
    setCameraError('');
    const targetDeviceId =
      deviceIdOverride !== undefined ? deviceIdOverride : selectedDeviceId;

    try {
      // Hentikan stream lama jika masih ada sebelum membuka stream baru
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      }

      const constraints = {
        video: targetDeviceId
          ? { deviceId: { ideal: targetDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
          : { width: { ideal: 640 }, height: { ideal: 480 } },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (!videoRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      videoRef.current.srcObject = stream;
      try {
        await videoRef.current.play();
      } catch (playErr) {
        // Abaikan AbortError akibat double-mount React StrictMode
        if (playErr?.name !== 'AbortError') {
          throw playErr;
        }
      }

      setCameraError('');
      setCameraActive(true);
    } catch (err) {
      if (err?.name === 'AbortError') return;
      setCameraError('Kamera tidak dapat diakses. Pastikan izin kamera telah diberikan.');
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setHandDetected(false);
    clearOverlayCanvas();
  };

  useEffect(() => {
    startCamera();
    return () => stopCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const clearOverlayCanvas = () => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const drawLandmarksOnCanvas = (landmarks) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!showLandmarksRef.current || !landmarks || landmarks.length === 0) return;

    const w = canvas.width;
    const h = canvas.height;
    const pts = landmarks.map((lm) => ({ x: lm.x * w, y: lm.y * h }));

    // Bounding box hijau di sekitar tangan
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const minX = Math.max(0, Math.min(...xs) - 18);
    const maxX = Math.min(w, Math.max(...xs) + 18);
    const minY = Math.max(0, Math.min(...ys) - 18);
    const maxY = Math.min(h, Math.max(...ys) + 18);

    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);

    // Garis kerangka 21 landmark
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 2.5;
    HAND_CONNECTIONS.forEach(([i, j]) => {
      if (pts[i] && pts[j]) {
        ctx.beginPath();
        ctx.moveTo(pts[i].x, pts[i].y);
        ctx.lineTo(pts[j].x, pts[j].y);
        ctx.stroke();
      }
    });

    // Titik sendi
    pts.forEach((p, idx) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, idx === 0 ? 5 : 4, 0, Math.PI * 2);
      ctx.fillStyle = idx === 0 ? '#f59e0b' : '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#0f766e';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  };

  // Loop inferensi real-time (~10-12 FPS sesuai kriteria keberhasilan proposal)
  useEffect(() => {
    if (!cameraActive) return;

    let isRunning = true;
    let inFlight = false;
    let lastFrameTime = performance.now();

    const interval = setInterval(async () => {
      if (!isRunning || inFlight || !videoRef.current) return;
      const video = videoRef.current;
      if (video.readyState < 2) return;

      inFlight = true;
      const tNow = performance.now();
      const currentFps = 1000 / Math.max(tNow - lastFrameTime, 1);
      lastFrameTime = tNow;

      try {
        const capCanvas = captureCanvasRef.current;
        capCanvas.width = 320;
        capCanvas.height = 240;
        const capCtx = capCanvas.getContext('2d');
        // Flip horizontal agar natural seperti cermin
        capCtx.save();
        capCtx.scale(-1, 1);
        capCtx.drawImage(video, -320, 0, 320, 240);
        capCtx.restore();

        const imageBase64 = capCanvas.toDataURL('image/jpeg', 0.72);

        if (mode === 'letter') {
          const res = await predictLetter({
            imageBase64,
            modelType,
            threshold: 0.70,
          });
          if (!isRunning) return;

          setHandDetected(res.hand_detected);
          setLatencyMs(res.inference_time_ms || 0);
          setFps(Math.min(30, Math.round(currentFps)));
          drawLandmarksOnCanvas(res.landmarks);

          if (res.hand_detected && res.label) {
            setDetectedLabel(res.label);
            setConfidence(Math.round((res.confidence || 0) * 100));

            // Penyangga huruf (stabil selama 3 siklus berturut-turut)
            stabilityBufferRef.current.push(res.label);
            if (stabilityBufferRef.current.length > 4) {
              stabilityBufferRef.current.shift();
            }
            const allSame =
              stabilityBufferRef.current.length >= 3 &&
              stabilityBufferRef.current.every((val) => val === res.label);

            const nowMs = Date.now();
            if (
              allSame &&
              (lastAppendedRef.current.label !== res.label ||
                nowMs - lastAppendedRef.current.time > 1800)
            ) {
              setTranslatedText((prev) => prev + res.label);
              lastAppendedRef.current = { label: res.label, time: nowMs };
            }
          } else if (!res.hand_detected) {
            stabilityBufferRef.current = [];
          }
        } else {
          // Mode Kata Dasar (30 bingkai urutan)
          const extractRes = await predictWord({ imageBase64, threshold: 0.70 });
          if (!isRunning) return;

          setHandDetected(extractRes.hand_detected);
          setLatencyMs(extractRes.inference_time_ms || 0);
          setFps(Math.min(30, Math.round(currentFps)));
          drawLandmarksOnCanvas(extractRes.landmarks);

          if (extractRes.hand_detected && extractRes.normalized_features?.length === 63) {
            wordBufferRef.current.push(extractRes.normalized_features);
            if (wordBufferRef.current.length > 30) {
              wordBufferRef.current.shift();
            }

            if (wordBufferRef.current.length === 30) {
              const wordRes = await predictWord({
                sequenceFeatures: wordBufferRef.current,
                threshold: 0.70,
              });
              if (wordRes.label) {
                setDetectedLabel(wordRes.label);
                setConfidence(Math.round((wordRes.confidence || 0) * 100));
                const nowMs = Date.now();
                if (
                  lastAppendedRef.current.label !== wordRes.label ||
                  nowMs - lastAppendedRef.current.time > 2500
                ) {
                  setTranslatedText((prev) =>
                    prev ? `${prev} ${wordRes.label}` : wordRes.label
                  );
                  lastAppendedRef.current = { label: wordRes.label, time: nowMs };
                }
              }
              wordBufferRef.current = wordBufferRef.current.slice(15);
            }
          }
        }
      } catch {
        // Abaikan kegagalan jaringan sesaat agar loop tetap berjalan
      } finally {
        inFlight = false;
      }
    }, 110);

    return () => {
      isRunning = false;
      clearInterval(interval);
    };
  }, [cameraActive, mode, modelType]);

  const handleBackspace = () => {
    setTranslatedText((prev) => {
      if (mode === 'word') {
        const parts = prev.trim().split(' ');
        parts.pop();
        return parts.join(' ');
      }
      return prev.slice(0, -1);
    });
  };

  const handleSaveToHistory = () => {
    if (!translatedText.trim()) return;
    const updated = addTranslationHistory({
      text: translatedText,
      mode,
      confidence: confidence / 100 || 0.92,
    });
    setRecentHistory(updated.slice(0, 3));
  };

  const handleReset = () => {
    if (translatedText.trim()) {
      handleSaveToHistory();
    }
    setTranslatedText('');
    setDetectedLabel('-');
    setConfidence(0);
    wordBufferRef.current = [];
    stabilityBufferRef.current = [];
  };

  const handleCopy = async () => {
    if (!translatedText) return;
    await navigator.clipboard.writeText(translatedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
      <canvas ref={captureCanvasRef} className="hidden" />

      <div className="grid gap-6 lg:grid-cols-12">
        {/* ================= KOLOM KIRI: KAMERA & KONTROL (Gambar 3.6) ================= */}
        <div className="space-y-4 lg:col-span-7">
          {/* Baris Pilihan Mode, Kamera, Model, dan Status */}
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
            <div className="inline-flex rounded-lg bg-slate-100 p-1">
              <button
                type="button"
                onClick={() => {
                  setMode('letter');
                  wordBufferRef.current = [];
                }}
                className={`rounded-md px-3 py-1.5 text-xs font-bold transition ${
                  mode === 'letter'
                    ? 'bg-teal-700 text-white shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                A Abjad
              </button>
              <button
                type="button"
                onClick={() => {
                  setMode('word');
                  wordBufferRef.current = [];
                }}
                className={`rounded-md px-3 py-1.5 text-xs font-bold transition ${
                  mode === 'word'
                    ? 'bg-teal-700 text-white shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                ✋ Kata Dasar
              </button>
            </div>

            {mode === 'letter' ? (
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs font-medium text-slate-700"
              >
                <option value="random_forest">Model: Random Forest</option>
                <option value="ffnn">Model: FFNN</option>
              </select>
            ) : (
              <span className="rounded-lg border border-teal-200 bg-teal-50 px-2.5 py-1.5 text-xs font-medium text-teal-800">
                Model: LSTM (30 Bingkai)
              </span>
            )}

            <select
              value={selectedDeviceId}
              onChange={(e) => {
                setSelectedDeviceId(e.target.value);
                startCamera(e.target.value);
              }}
              className="max-w-[180px] truncate rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs text-slate-700"
            >
              {devices.length === 0 && <option value="">Kamera Web Default</option>}
              {devices.map((d, i) => (
                <option key={d.deviceId || i} value={d.deviceId}>
                  {d.label || `Kamera Web ${i + 1}`}
                </option>
              ))}
            </select>

            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${
                cameraActive
                  ? 'bg-emerald-50 text-emerald-700'
                  : 'bg-slate-100 text-slate-600'
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  cameraActive ? 'bg-emerald-500' : 'bg-slate-400'
                }`}
              />
              {cameraActive ? 'Kamera Aktif' : 'Nonaktif'}
            </span>
          </div>

          {/* Kotak Pratinjau Video + Overlay 21 Landmark */}
          <div className="relative overflow-hidden rounded-2xl border border-slate-200 bg-slate-950 shadow-md aspect-[4/3]">
            <video
              ref={videoRef}
              playsInline
              muted
              className="h-full w-full object-cover -scale-x-100"
            />
            <canvas
              ref={overlayCanvasRef}
              width={640}
              height={480}
              className="pointer-events-none absolute inset-0 h-full w-full"
            />

            {/* Badge Status Kiri Atas */}
            <div className="absolute left-3 top-3 flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-900/80 px-3 py-1 text-xs font-medium text-white backdrop-blur">
                <span
                  className={`h-2 w-2 rounded-full ${
                    handDetected ? 'bg-emerald-400' : 'bg-amber-400'
                  }`}
                />
                {handDetected
                  ? `Deteksi SIBI Aktif (${mode === 'letter' ? 'Abjad' : 'Kata Dasar'})`
                  : 'Menunggu Tangan di Depan Kamera...'}
              </span>
            </div>

            {/* Badge FPS & Latensi Kanan Atas (Uji T-09) */}
            {cameraActive && (
              <div className="absolute right-3 top-3 rounded-full bg-slate-900/80 px-3 py-1 text-[11px] font-mono text-emerald-300 backdrop-blur">
                {fps} FPS • {latencyMs} ms
              </div>
            )}

            {/* Badge Bawah Video */}
            <div className="absolute bottom-3 left-3 rounded-lg bg-slate-900/80 px-3 py-1 text-xs text-slate-200 backdrop-blur">
              Isyarat Terakhir: <span className="font-bold text-emerald-300">{detectedLabel}</span>
            </div>

            {cameraError && !cameraActive && (
              <div className="absolute inset-0 flex items-center justify-center bg-slate-900/90 p-6 text-center text-sm text-red-300">
                {cameraError}
              </div>
            )}
          </div>

          {/* Banner Panduan Posisi Gestur */}
          <div className="flex items-center gap-3 rounded-xl border border-amber-200 bg-amber-50/80 px-4 py-3 text-xs text-amber-900">
            <Info className="h-4 w-4 shrink-0 text-amber-600" />
            <div>
              <span className="font-bold">Panduan Posisi Gestur: </span>
              Posisikan tangan sekitar 30 cm dari kamera dengan pencahayaan cukup dan latar belakang
              polos.
            </div>
          </div>

          {/* Tombol Mulai/Hentikan Kamera & Saklar Tampilkan Landmark */}
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white p-3.5 shadow-sm">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => startCamera()}
                disabled={cameraActive}
                className="inline-flex items-center gap-1.5 rounded-lg bg-teal-700 px-4 py-2 text-xs font-semibold text-white transition hover:bg-teal-800 disabled:opacity-50"
              >
                <Camera className="h-3.5 w-3.5" />
                Mulai Kamera
              </button>
              <button
                type="button"
                onClick={stopCamera}
                disabled={!cameraActive}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
              >
                <Square className="h-3.5 w-3.5" />
                Hentikan Kamera
              </button>
            </div>

            <label className="flex cursor-pointer items-center gap-2 text-xs font-medium text-slate-700">
              <input
                type="checkbox"
                checked={showLandmarks}
                onChange={(e) => setShowLandmarks(e.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-teal-700 focus:ring-teal-600"
              />
              Tampilkan Landmark (21 Titik)
            </label>
          </div>

          <div className="flex items-center justify-center gap-1.5 text-xs text-slate-500">
            <Lock className="h-3.5 w-3.5 text-teal-700" />
            <span>Video diproses langsung dan tidak disimpan</span>
          </div>
        </div>

        {/* ================= KOLOM KANAN: HASIL, TEKS, & RIWAYAT (Gambar 3.6) ================= */}
        <div className="space-y-4 lg:col-span-5">
          {/* Kartu 1: Hasil Terdeteksi */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-slate-800">Hasil Terdeteksi</h2>
              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-0.5 text-[11px] font-bold text-emerald-700">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                REAL-TIME
              </span>
            </div>

            <div className="mt-4 flex items-center gap-4 rounded-xl border border-slate-100 bg-slate-50 p-4">
              <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-xl bg-teal-700 text-2xl font-extrabold text-white shadow-sm">
                {detectedLabel.slice(0, 2)}
              </div>
              <div className="min-w-0 flex-1">
                <div className="truncate text-base font-bold text-slate-900">
                  {detectedLabel === '-'
                    ? 'Belum Ada Isyarat'
                    : mode === 'letter'
                    ? `Huruf ${detectedLabel} (SIBI)`
                    : `Kata "${detectedLabel}" (SIBI)`}
                </div>
                <div className="mt-0.5 text-xs text-slate-500">
                  {mode === 'letter'
                    ? 'Jalur Abjad • Gestur Statis (63 Fitur)'
                    : 'Jalur Kata Dasar • Gestur Dinamis (30 Bingkai)'}
                </div>
              </div>
            </div>

            <div className="mt-4">
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="font-medium text-slate-600">Tingkat kepercayaan</span>
                <span className="font-bold text-teal-800">{confidence}%</span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                <div
                  className="h-full rounded-full bg-teal-600 transition-all duration-200"
                  style={{ width: `${Math.min(100, Math.max(0, confidence))}%` }}
                />
              </div>
            </div>
          </div>

          {/* Kartu 2: Teks Terjemahan */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-slate-800">Teks Terjemahan</h2>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setTranslatedText((prev) => `${prev} `)}
                  className="rounded border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] font-medium text-slate-600 hover:bg-slate-100"
                >
                  + Spasi
                </button>
                <span className="text-xs text-slate-400">{translatedText.length} karakter</span>
              </div>
            </div>

            <div className="mt-3 flex min-h-[84px] items-center rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-xl font-bold tracking-wide text-slate-900">
              {translatedText || (
                <span className="text-sm font-normal text-slate-400">
                  Peragakan isyarat di depan kamera untuk menyusun teks...
                </span>
              )}
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
              <button
                type="button"
                onClick={handleBackspace}
                className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
              >
                <Delete className="h-3.5 w-3.5" />
                Hapus
              </button>
              <button
                type="button"
                onClick={handleReset}
                className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Reset
              </button>
              <button
                type="button"
                onClick={handleSaveToHistory}
                className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-teal-200 bg-teal-50 px-3 py-2 text-xs font-semibold text-teal-800 hover:bg-teal-100"
              >
                <BookmarkPlus className="h-3.5 w-3.5" />
                Simpan
              </button>
              <button
                type="button"
                onClick={handleCopy}
                className="inline-flex items-center justify-center gap-1.5 rounded-lg bg-teal-700 px-3 py-2 text-xs font-semibold text-white hover:bg-teal-800"
              >
                {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                {copied ? 'Tersalin' : 'Salin Teks'}
              </button>
            </div>
          </div>

          {/* Kartu 3: Riwayat Terbaru */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-slate-800">Riwayat Terbaru</h2>
              <button
                type="button"
                onClick={onViewAllHistory}
                className="text-xs font-semibold text-teal-700 hover:underline"
              >
                Lihat Semua
              </button>
            </div>

            {recentHistory.length === 0 ? (
              <p className="mt-4 text-center text-xs text-slate-400 py-4">
                Belum ada riwayat terjemahan yang disimpan.
              </p>
            ) : (
              <div className="mt-3 space-y-2.5">
                {recentHistory.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50 px-3.5 py-2.5"
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-slate-900">{item.text}</span>
                        <span className="rounded bg-teal-100 px-1.5 py-0.5 text-[10px] font-semibold text-teal-800">
                          {item.mode}
                        </span>
                        <span className="rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-800">
                          {item.confidence}%
                        </span>
                      </div>
                      <div className="mt-0.5 text-[11px] text-slate-400">
                        {new Date(item.createdAt).toLocaleTimeString('id-ID', {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}{' '}
                        • {item.charCount} karakter
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={() => navigator.clipboard.writeText(item.text)}
                      className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-200 hover:text-slate-700"
                      title="Salin"
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

