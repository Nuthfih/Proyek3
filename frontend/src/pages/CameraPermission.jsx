import React, { useState } from 'react';
import { Camera, Sun, Maximize2, Grid, Lock, AlertCircle } from 'lucide-react';

export default function CameraPermission({ onGranted }) {
  const [errorMessage, setErrorMessage] = useState('');
  const [requesting, setRequesting] = useState(false);

  const handleRequestPermission = async () => {
    setErrorMessage('');
    setRequesting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      // Tutup stream sementara; halaman Penerjemah akan membukanya kembali
      stream.getTracks().forEach((t) => t.stop());
      onGranted();
    } catch (err) {
      // Uji T-01: Menampilkan pesan yang jelas tanpa crash saat izin ditolak
      setErrorMessage(
        'Izin akses kamera ditolak atau kamera tidak ditemukan. Silakan aktifkan izin kamera pada pengaturan peramban Anda untuk melanjutkan.'
      );
    } finally {
      setRequesting(false);
    }
  };

  return (
    <div className="mx-auto flex max-w-2xl flex-col items-center justify-center px-4 py-14 sm:px-6">
      <div className="w-full rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
        <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-teal-50 text-teal-700">
          <Camera className="h-8 w-8" />
        </div>

        <h1 className="text-2xl font-bold text-slate-900">Izinkan Akses Kamera</h1>
        <p className="mx-auto mt-2 max-w-md text-sm text-slate-600">
          Kamera digunakan hanya untuk mendeteksi gestur tangan Anda. Video tidak disimpan.
        </p>

        {errorMessage && (
          <div className="mt-5 flex items-start gap-2.5 rounded-xl border border-red-200 bg-red-50 p-4 text-left text-xs text-red-700">
            <AlertCircle className="h-4 w-4 shrink-0 text-red-600 mt-0.5" />
            <span>{errorMessage}</span>
          </div>
        )}

        <button
          type="button"
          disabled={requesting}
          onClick={handleRequestPermission}
          className="mt-6 inline-flex items-center gap-2 rounded-lg bg-teal-700 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-teal-800 disabled:opacity-60"
        >
          <Camera className="h-4 w-4" />
          {requesting ? 'Meminta Izin Kamera...' : 'Izinkan Kamera'}
        </button>

        {/* Tiga Petunjuk Penggunaan (Gambar 3.5) */}
        <div className="mt-8 grid gap-4 border-t border-slate-100 pt-6 sm:grid-cols-3">
          <div className="flex flex-col items-center text-center">
            <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-amber-50 text-amber-600">
              <Sun className="h-4 w-4" />
            </div>
            <div className="text-xs font-semibold text-slate-800">Pencahayaan cukup</div>
            <div className="mt-0.5 text-[11px] text-slate-500">Hindari cahaya dari belakang</div>
          </div>

          <div className="flex flex-col items-center text-center">
            <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-teal-50 text-teal-700">
              <Maximize2 className="h-4 w-4" />
            </div>
            <div className="text-xs font-semibold text-slate-800">Jarak tangan sekitar 30 cm</div>
            <div className="mt-0.5 text-[11px] text-slate-500">Pastikan seluruh jari terlihat</div>
          </div>

          <div className="flex flex-col items-center text-center">
            <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600">
              <Grid className="h-4 w-4" />
            </div>
            <div className="text-xs font-semibold text-slate-800">Latar belakang polos</div>
            <div className="mt-0.5 text-[11px] text-slate-500">Memudahkan deteksi tangan</div>
          </div>
        </div>

        <div className="mt-6 flex items-center justify-center gap-1.5 text-xs text-slate-500">
          <Lock className="h-3.5 w-3.5 text-teal-700" />
          <span>Video diproses langsung dan tidak disimpan</span>
        </div>
      </div>
    </div>
  );
}

