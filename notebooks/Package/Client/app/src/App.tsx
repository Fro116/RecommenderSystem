// src/App.tsx
import React, { useState, useEffect } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import HomePage from './HomePage';
import ViewPage from './ViewPage';
import './App.css';

const App: React.FC = () => {
  const [isMobile, setIsMobile] = useState<boolean>(false);
  const location = useLocation();

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

  const isHomePage = location.pathname === '/';
  const containerClass = isHomePage ? 'container homepage' : 'container';
  const containerStyle: React.CSSProperties = isHomePage
    ? { height: 'calc(var(--vh, 1vh) * 100)', overflowY: 'hidden' }
    : {};


  return (
    <div className={containerClass} style={containerStyle}>
      <Routes>
        <Route path="/" element={<HomePage />} />
        {/* Updated Route Path */}
        <Route path="/user/:source/:username" element={<ViewPage isMobile={isMobile} />} />
        {/* Other routes */}
      </Routes>
    </div>
  );
};

export default App;