import React, { useState } from 'react';
import { Trash2, Copy, Search, Hand, ArrowRight } from 'lucide-react';
import {
  getTranslationHistory,
  deleteHistoryItem,
  clearAllHistory,
} from '../services/storage';

export default function HistoryPage({ onStartTranslate }) {
  const [items, setItems] = useState(() => getTranslationHistory());
  const [searchQuery, setSearchQuery] = useState('');
  const [filterMode, setFilterMode] = useState('Semua'); // 'Semua' | 'Abjad' | 'Kata Dasar'

  const filteredItems = items.filter((item) => {
    const matchMode = filterMode === 'Semua' || item.mode === filterMode;
    const matchSearch = item.text.toLowerCase().includes(searchQuery.toLowerCase());
    return matchMode && matchSearch;
  });

  const handleDeleteOne = (id) => {
    setItems(deleteHistoryItem(id));
  };

  const handleClearAll = () => {
    setItems(clearAllHistory());
  };

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      {/* Header Riwayat (Gambar 3.7 & 3.8) */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Riwayat Terjemahan</h1>
          <p className="mt-1 text-xs text-slate-500">
            Riwayat disimpan di peramban ini (Web LocalStorage) dan tidak dikirim ke server
          </p>
        </div>

        {items.length > 0 && (
          <button
            type="button"
            onClick={handleClearAll}
            className="inline-flex items-center gap-1.5 rounded-lg border border-red-200 bg-red-50 px-3.5 py-2 text-xs font-semibold text-red-700 transition hover:bg-red-100"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Hapus Semua
          </button>
        )}
      </div>

      {/* Bar Pencarian & Filter */}
      <div className="mt-6 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Cari teks..."
            className="w-full rounded-lg border border-slate-200 bg-slate-50 py-1.5 pl-9 pr-3 text-xs text-slate-800 focus:border-teal-600 focus:outline-none"
          />
        </div>

        <div className="flex items-center gap-1">
          {['Semua', 'Abjad', 'Kata Dasar'].map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setFilterMode(m)}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                filterMode === m
                  ? 'bg-teal-700 text-white'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Kondisi Kosong (Gambar 3.8) vs Daftar Riwayat (Gambar 3.7) */}
      {filteredItems.length === 0 ? (
        <div className="mt-6 flex flex-col items-center justify-center rounded-2xl border border-slate-200 bg-white px-6 py-16 text-center shadow-sm">
          <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-teal-50 text-teal-700">
            <Hand className="h-7 w-7" />
          </div>
          <h2 className="text-base font-bold text-slate-900">Belum ada riwayat terjemahan</h2>
          <p className="mt-1 max-w-sm text-xs text-slate-500">
            Hasil terjemahan Anda akan muncul di sini setelah Anda menyimpannya pada halaman
            Penerjemah.
          </p>
          <button
            type="button"
            onClick={onStartTranslate}
            className="mt-5 inline-flex items-center gap-2 rounded-lg bg-teal-700 px-5 py-2.5 text-xs font-semibold text-white shadow-sm transition hover:bg-teal-800"
          >
            Mulai Menerjemahkan
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>
      ) : (
        <div className="mt-6 space-y-3">
          {filteredItems.map((item) => (
            <div
              key={item.id}
              className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-5 py-4 shadow-sm"
            >
              <div>
                <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                  <span className="rounded bg-teal-50 px-2 py-0.5 font-semibold text-teal-800">
                    {item.mode}
                  </span>
                  <span className="rounded bg-emerald-50 px-2 py-0.5 font-semibold text-emerald-700">
                    • {item.confidence}%
                  </span>
                  <span>
                    {new Date(item.createdAt).toLocaleString('id-ID', {
                      dateStyle: 'medium',
                      timeStyle: 'short',
                    })}
                  </span>
                  <span>• {item.charCount} karakter</span>
                </div>
                <div className="mt-1.5 text-base font-bold text-slate-900">{item.text}</div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => navigator.clipboard.writeText(item.text)}
                  className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
                >
                  <Copy className="h-3.5 w-3.5" />
                  Salin
                </button>
                <button
                  type="button"
                  onClick={() => handleDeleteOne(item.id)}
                  className="inline-flex items-center rounded-lg border border-slate-200 p-1.5 text-slate-400 hover:border-red-200 hover:bg-red-50 hover:text-red-600"
                  title="Hapus"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

