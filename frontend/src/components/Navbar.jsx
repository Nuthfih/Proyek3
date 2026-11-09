import React from 'react';
import { Hand } from 'lucide-react';

export default function Navbar({ activePage, onNavigate, backendReady }) {
  const navItems = [
    { id: 'home', label: 'Beranda' },
    { id: 'translator', label: 'Penerjemah' },
    { id: 'history', label: 'Riwayat' },
    { id: 'about', label: 'Tentang' },
  ];

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <button
          type="button"
          onClick={() => onNavigate('home')}
          className="flex items-center gap-2.5 text-left"
        >
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-teal-700 text-white shadow-sm">
            <Hand className="h-5 w-5" />
          </div>
          <div>
            <div className="text-base font-bold leading-tight text-slate-900">IsyaratKu</div>
            <div className="text-[11px] text-slate-500">Penerjemah Bahasa Isyarat SIBI</div>
          </div>
        </button>

        <nav className="flex items-center gap-1 sm:gap-2">
          {navItems.map((item) => {
            const isActive =
              activePage === item.id ||
              (item.id === 'translator' && activePage === 'permission');
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onNavigate(item.id)}
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                  isActive
                    ? 'bg-teal-700 text-white shadow-sm'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="hidden items-center gap-2 sm:flex">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium ${
              backendReady
                ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                : 'border-amber-200 bg-amber-50 text-amber-700'
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                backendReady ? 'bg-emerald-500' : 'bg-amber-500'
              }`}
            />
            {backendReady ? 'Sistem Siap' : 'Mode Offline / Demo'}
          </span>
        </div>
      </div>
    </header>
  );
}

