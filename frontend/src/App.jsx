import React, { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import CameraPermission from './pages/CameraPermission';
import TranslatorPage from './pages/TranslatorPage';
import HistoryPage from './pages/HistoryPage';
import AboutPage from './pages/AboutPage';
import { checkServerHealth } from './services/api';

export default function App() {
  const [activePage, setActivePage] = useState('home'); // 'home' | 'permission' | 'translator' | 'history' | 'about'
  const [cameraPermitted, setCameraPermitted] = useState(false);
  const [backendReady, setBackendReady] = useState(false);

  useEffect(() => {
    checkServerHealth()
      .then(() => setBackendReady(true))
      .catch(() => setBackendReady(false));
  }, []);

  const handleNavigate = (target) => {
    if (target === 'translator' && !cameraPermitted) {
      setActivePage('permission');
    } else {
      setActivePage(target);
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
      <Navbar
        activePage={activePage}
        onNavigate={handleNavigate}
        backendReady={backendReady}
      />

      <main className="flex-1">
        {activePage === 'home' && (
          <HomePage onStartTranslate={() => handleNavigate('translator')} />
        )}
        {activePage === 'permission' && (
          <CameraPermission
            onGranted={() => {
              setCameraPermitted(true);
              setActivePage('translator');
            }}
          />
        )}
        {activePage === 'translator' && (
          <TranslatorPage onViewAllHistory={() => handleNavigate('history')} />
        )}
        {activePage === 'history' && (
          <HistoryPage onStartTranslate={() => handleNavigate('translator')} />
        )}
        {activePage === 'about' && <AboutPage />}
      </main>

      <footer className="border-t border-slate-200 bg-white py-5 text-center text-xs text-slate-400">
        © 2026 IsyaratKu — Sistem Penerjemah Bahasa Isyarat SIBI (Proyek III D-IV Teknik Informatika ULBI)
      </footer>
    </div>
  );
}

