import React from 'react';
import { Camera, Zap, Monitor, Clock, Lock, Users, MessageSquare } from 'lucide-react';

export default function HomePage({ onStartTranslate }) {
  return (
    <div className="space-y-16 pb-16">
      {/* Bagian Hero (Gambar 3.4) */}
      <section className="mx-auto max-w-6xl px-4 pt-10 sm:px-6 lg:pt-14">
        <div className="grid items-center gap-10 lg:grid-cols-2">
          <div className="space-y-6">
            <span className="inline-flex items-center gap-2 rounded-full border border-teal-200 bg-teal-50 px-3.5 py-1 text-xs font-semibold text-teal-800">
              <span className="h-2 w-2 rounded-full bg-teal-600" />
              Bahasa Isyarat SIBI • Real-Time
            </span>

            <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 sm:text-4xl lg:text-5xl">
              Jembatani Komunikasi dengan{' '}
              <span className="text-teal-700">Bahasa Isyarat</span>
            </h1>

            <p className="text-base leading-relaxed text-slate-600 sm:text-lg">
              IsyaratKu menerjemahkan gestur abjad dan kata dasar SIBI menjadi teks secara{' '}
              <span className="font-medium text-slate-800">real-time</span>, cukup dengan kamera web
              standar di peramban Anda.
            </p>

            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={onStartTranslate}
                className="inline-flex items-center gap-2 rounded-lg bg-teal-700 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-teal-800"
              >
                <Camera className="h-4 w-4" />
                Mulai Menerjemahkan
              </button>
              <a
                href="#cara-kerja"
                className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
              >
                Lihat Cara Kerja
              </a>
            </div>

            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Lock className="h-3.5 w-3.5 text-teal-700" />
              <span>Video diproses langsung dan tidak disimpan di server</span>
            </div>
          </div>

          {/* Kartu Ilustrasi Pratinjau Aplikasi */}
          <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-lg">
            <div className="mb-3 flex items-center justify-between text-xs text-slate-500">
              <span className="font-semibold text-slate-700">MODE PENERJEMAH AKTIF</span>
              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-0.5 font-medium text-emerald-700">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                21 Landmark Terdeteksi
              </span>
            </div>
            <div className="relative flex h-64 items-center justify-center rounded-xl bg-slate-900 text-white">
              <div className="h-40 w-40 rounded-xl border-2 border-dashed border-emerald-400/70 p-4 flex flex-col items-center justify-center">
                <span className="mb-2 text-[11px] font-medium text-emerald-300">
                  Kerangka Tangan 3D (63 Fitur)
                </span>
                <svg viewBox="0 0 120 120" className="h-24 w-24 stroke-emerald-400 fill-amber-400">
                  <line x1="60" y1="105" x2="35" y2="75" strokeWidth="2.5" />
                  <line x1="35" y1="75" x2="22" y2="55" strokeWidth="2.5" />
                  <line x1="60" y1="105" x2="48" y2="60" strokeWidth="2.5" />
                  <line x1="48" y1="60" x2="45" y2="25" strokeWidth="2.5" />
                  <line x1="60" y1="105" x2="60" y2="58" strokeWidth="2.5" />
                  <line x1="60" y1="58" x2="60" y2="20" strokeWidth="2.5" />
                  <line x1="60" y1="105" x2="72" y2="62" strokeWidth="2.5" />
                  <line x1="72" y1="62" x2="75" y2="28" strokeWidth="2.5" />
                  <line x1="60" y1="105" x2="85" y2="70" strokeWidth="2.5" />
                  <line x1="85" y1="70" x2="92" y2="42" strokeWidth="2.5" />
                  <circle cx="60" cy="105" r="4" />
                  <circle cx="22" cy="55" r="3.5" />
                  <circle cx="45" cy="25" r="3.5" />
                  <circle cx="60" cy="20" r="3.5" />
                  <circle cx="75" cy="28" r="3.5" />
                  <circle cx="92" cy="42" r="3.5" />
                </svg>
              </div>
            </div>
            <div className="mt-3 flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50 px-4 py-3">
              <div className="flex items-center gap-3">
                <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-teal-700 text-lg font-bold text-white">
                  A
                </span>
                <div>
                  <div className="text-xs text-slate-500">TERDETEKSI • TINGKAT KEPERCAYAAN 95%</div>
                  <div className="text-base font-bold text-slate-900">HALO</div>
                </div>
              </div>
              <span className="rounded-md bg-teal-100 px-2.5 py-1 text-xs font-semibold text-teal-800">
                SIBI
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Bagian Cara Kerja */}
      <section id="cara-kerja" className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="text-center">
          <div className="text-xs font-bold uppercase tracking-wider text-teal-700">
            LANGKAH SEDERHANA
          </div>
          <h2 className="mt-1 text-2xl font-bold text-slate-900">Cara Kerja</h2>
          <p className="mt-2 text-sm text-slate-600">
            Tiga langkah mudah untuk menerjemahkan bahasa isyarat SIBI langsung dari peramban Anda.
          </p>
        </div>

        <div className="mt-8 grid gap-6 sm:grid-cols-3">
          {[
            {
              step: '1',
              title: 'Izinkan akses kamera',
              desc: 'Berikan izin kamera web pada peramban Anda. Video hanya diproses secara lokal/real-time dan tidak disimpan.',
            },
            {
              step: '2',
              title: 'Peragakan isyarat di depan kamera',
              desc: 'Posisikan tangan sekitar 30 cm di depan kamera dengan pencahayaan cukup untuk mode Abjad atau Kata Dasar.',
            },
            {
              step: '3',
              title: 'Baca teks hasil terjemahan',
              desc: 'Sistem mengekstraksi 21 titik landmark dan menampilkan huruf/kata beserta tingkat kepercayaannya secara langsung.',
            },
          ].map((item) => (
            <div
              key={item.step}
              className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm"
            >
              <div className="mb-4 flex h-8 w-8 items-center justify-center rounded-lg bg-teal-700 text-sm font-bold text-white">
                {item.step}
              </div>
              <h3 className="text-base font-bold text-slate-900">{item.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Bagian Fitur */}
      <section className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="text-center">
          <div className="text-xs font-bold uppercase tracking-wider text-teal-700">
            KEMAMPUAN SISTEM
          </div>
          <h2 className="mt-1 text-2xl font-bold text-slate-900">Fitur</h2>
        </div>

        <div className="mt-8 grid gap-6 sm:grid-cols-2">
          {[
            {
              icon: MessageSquare,
              title: 'Abjad dan Kata Dasar SIBI',
              desc: 'Mengenali gestur statis abjad (Random Forest & FFNN) serta gestur dinamis kata dasar SIBI (LSTM 30 bingkai).',
            },
            {
              icon: Zap,
              title: 'Real-Time',
              desc: 'Ekstraksi 21 landmark 3D (63 koordinat ternormalisasi) dengan waktu inferensi cepat tanpa memerlukan GPU khusus.',
            },
            {
              icon: Monitor,
              title: 'Cukup Kamera Web',
              desc: 'Tidak memerlukan sarung tangan sensor atau kamera kedalaman yang mahal, cukup kamera laptop standar.',
            },
            {
              icon: Clock,
              title: 'Riwayat Terjemahan',
              desc: 'Menyimpan hasil terjemahan di LocalStorage peramban pengguna yang dapat dicari, disalin, atau dihapus kapan saja.',
            },
          ].map((feat) => {
            const Icon = feat.icon;
            return (
              <div
                key={feat.title}
                className="flex gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
              >
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-teal-50 text-teal-700">
                  <Icon className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">{feat.title}</h3>
                  <p className="mt-1 text-sm text-slate-600">{feat.desc}</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Bagian Untuk Siapa & CTA */}
      <section className="mx-auto max-w-6xl space-y-10 px-4 sm:px-6">
        <div className="text-center">
          <div className="text-xs font-bold uppercase tracking-wider text-teal-700">
            AKSESIBILITAS BERSAMA
          </div>
          <h2 className="mt-1 text-2xl font-bold text-slate-900">Untuk Siapa</h2>
        </div>

        <div className="grid gap-6 sm:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-white p-6 text-center shadow-sm">
            <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-teal-50 text-teal-700">
              <Users className="h-5 w-5" />
            </div>
            <h3 className="text-base font-bold text-slate-900">
              Penyandang tunarungu dan tunawicara
            </h3>
            <p className="mt-2 text-sm text-slate-600">
              Membantu menyampaikan pesan dalam isyarat dasar SIBI secara cepat menjadi teks yang
              mudah dibaca lawan bicara.
            </p>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white p-6 text-center shadow-sm">
            <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-teal-50 text-teal-700">
              <MessageSquare className="h-5 w-5" />
            </div>
            <h3 className="text-base font-bold text-slate-900">
              Masyarakat umum, pendidik, dan layanan publik
            </h3>
            <p className="mt-2 text-sm text-slate-600">
              Memahami isyarat dasar tanpa harus menguasai seluruh bahasa isyarat terlebih dahulu,
              sekaligus media latihan SIBI.
            </p>
          </div>
        </div>

        <div className="rounded-2xl bg-teal-800 px-6 py-10 text-center text-white shadow-md">
          <h2 className="text-2xl font-bold">Siap mencoba?</h2>
          <p className="mx-auto mt-2 max-w-xl text-sm text-teal-100">
            Mulai gunakan IsyaratKu sekarang juga langsung dari peramban Anda. Sistem ini merupakan
            alat bantu komunikasi dan tidak menggantikan peran juru bahasa isyarat profesional.
          </p>
          <button
            type="button"
            onClick={onStartTranslate}
            className="mt-6 inline-flex items-center gap-2 rounded-lg bg-white px-5 py-2.5 text-sm font-bold text-teal-900 shadow transition hover:bg-teal-50"
          >
            <Camera className="h-4 w-4" />
            Mulai Menerjemahkan
          </button>
        </div>
      </section>
    </div>
  );
}

