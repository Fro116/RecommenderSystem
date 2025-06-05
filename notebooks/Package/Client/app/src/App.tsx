// src/App.tsx
import React, { useState, useEffect } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import HomePage from './HomePage';
import ViewPage from './ViewPage';
import NotFoundPage from './NotFoundPage';
import './App.css';
import {
    enable as enableDarkMode,
    disable as disableDarkMode,
    setFetchMethod,
    isEnabled as isDarkReaderEnabled,
    type DynamicThemeFix
} from 'darkreader';

const App: React.FC = () => {
  const [isMobile, setIsMobile] = useState<boolean>(false);
  const location = useLocation();

  // viewport height and mobile detection
  useEffect(() => {
    const vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--vh', `${vh}px`);
    const handleResize = () => {
      const vh = window.innerHeight * 0.01;
      document.documentElement.style.setProperty('--vh', `${vh}px`);
    };
    window.addEventListener('resize', handleResize);
    setIsMobile(window.matchMedia?.('(hover: none)').matches ?? false);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // DarkReader
  useEffect(() => {
    if (typeof window !== 'undefined' && typeof window.fetch === 'function') {
      setFetchMethod(window.fetch);
    }
    const fixes: DynamicThemeFix = {
      css: `
        header input::placeholder {
          color: #8c8c8c !important;
        }
      `,
      invert: [],
      ignoreInlineStyle: [],
      ignoreImageAnalysis: [],
      disableStyleSheetsProxy: false
    };
    enableDarkMode({}, fixes);
    return () => {
      if (isDarkReaderEnabled()) {
        disableDarkMode();
      }
    };
  }, []);

  const isHomePage = location.pathname === '/';

  const containerClass = isHomePage ? 'container homepage' : 'container';
  const containerStyle: React.CSSProperties = isHomePage
    ? { height: 'calc(var(--vh, 1vh) * 100)', overflowY: 'hidden' }
    : { display: 'flex', flexDirection: 'column', minHeight: 'calc(var(--vh, 1vh) * 100)' };


  return (
    <div className={containerClass} style={containerStyle}>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/user/:source/:username" element={<ViewPage isMobile={isMobile} />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </div>
  );
};

export default App;
