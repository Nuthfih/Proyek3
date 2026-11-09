import React from 'react';
import { AlertTriangle, Lock } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-6 px-4 py-10 sm:px-6">
      {/* Header Tentang IsyaratKu (Gambar 3.9) */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <span className="inline-block rounded-full bg-teal-50 px-3 py-1 text-[11px] font-bold uppercase tracking-wider text-teal-800">
          TENTANG APLIKASI
        </span>
        <h1 className="mt-2 text-2xl font-bold text-slate-900">Tentang IsyaratKu</h1>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          IsyaratKu adalah alat bantu komunikasi berbasis web yang menerjemahkan gestur abjad dan
          kata dasar SIBI (Sistem Isyarat Bahasa Indonesia) menjadi teks secara{' '}
          <span className="font-medium text-slate-800">real-time</span> menggunakan kamera web
          standar.
        </p>
      </div>

      {/* Cara Menggunakan */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-sm font-bold text-slate-900">Cara Menggunakan</h2>
        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          {[
            {
              no: '1',
              step: 'LANGKAH 01',
              text: 'Izinkan akses kamera web pada peramban Anda.',
            },
            {
              no: '2',
              step: 'LANGKAH 02',
              text: 'Posisikan tangan sekitar 30 cm dari kamera dengan pencahayaan cukup.',
            },
            {
              no: '3',
              step: 'LANGKAH 03',
              text: 'Peragakan isyarat dan baca teks hasil terjemahan secara langsung.',
            },
          ].map((item) => (
            <div
              key={item.no}
              className="rounded-xl border border-slate-100 bg-slate-50 p-4"
            >
              <div className="mb-2 flex h-7 w-7 items-center justify-center rounded-md bg-teal-700 text-xs font-bold text-white">
                {item.no}
              </div>
              <p className="text-xs leading-relaxed text-slate-700">{item.text}</p>
              <div className="mt-3 text-[10px] font-bold uppercase tracking-wider text-slate-400">
                {item.step}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Teknologi */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-sm font-bold text-slate-900">Teknologi</h2>
        <div className="mt-3 flex flex-wrap gap-2">
          {[
            'OpenCV',
            'Google MediaPipe Hands (21 Landmark 3D)',
            'Random Forest & FFNN (Abjad)',
            'LSTM (Kata Dasar)',
            'Python FastAPI',
            'React + Tailwind CSS',
          ].map((tech) => (
            <span
              key={tech}
              className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-700"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-teal-600" />
              {tech}
            </span>
          ))}
        </div>
      </div>

      {/* Keterbatasan */}
      <div className="rounded-2xl border border-amber-200 bg-amber-50/60 p-6">
        <div className="flex items-center gap-2 text-sm font-bold text-amber-900">
          <AlertTriangle className="h-4 w-4 text-amber-600" />
          <span>Keterbatasan</span>
        </div>
        <ul className="mt-3 space-y-2 text-xs text-amber-900/90">
          <li>• Hanya mengenali abjad dan kata dasar SIBI tertentu yang telah dilatih.</li>
          <li>• Akurasi dapat menurun pada kondisi cahaya rendah atau jarak tangan terlalu jauh (&gt; 60 cm).</li>
          <li>• Alat bantu komunikasi, bukan pengganti juru bahasa isyarat profesional.</li>
        </ul>
      </div>

      {/* Privasi */}
      <div className="flex items-center gap-3 rounded-2xl border border-teal-200 bg-teal-50/60 p-5">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-teal-700 text-white">
          <Lock className="h-5 w-5" />
        </div>
        <div>
          <div className="text-[11px] font-bold uppercase tracking-wider text-teal-800">
            PRIVASI
          </div>
          <div className="text-xs font-semibold text-slate-800">
            Video diproses langsung secara real-time dan tidak disimpan di server.
          </div>
        </div>
      </div>
    </div>
  );
}

