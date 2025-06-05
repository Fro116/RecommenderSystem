// src/NotFoundPage.tsx
import React, { useState, useEffect } from 'react'; // Import useState and useEffect
import { useNavigate } from 'react-router-dom';
import { backgroundImages } from './types';
import './App.css';

const NotFoundPage: React.FC = () => {
  const navigate = useNavigate();
  const [currentBgImage, setCurrentBgImage] = useState<string>('');

  useEffect(() => {
    const originalTitle = document.title;
    document.title = 'Page Not Found | Recs☆Moe';
    const r = Math.random();
    let bgImages: string[];
    if (r < 0.9 * 0.99) {
      bgImages = backgroundImages.notfound_main;
    } else if (r < 0.9) {
      bgImages = backgroundImages.notfound_backup;
    } else if (r < 0.9 + 0.1 * 0.99) {
      bgImages = backgroundImages.loading_main;
    } else {
      bgImages = backgroundImages.loading_backup;
    }
    const randomBgIndex = Math.floor(Math.random() * bgImages.length);
    setCurrentBgImage(bgImages[randomBgIndex]);
    const metaRobots = document.createElement('meta');
    metaRobots.name = 'robots';
    metaRobots.content = 'noindex';
    document.head.appendChild(metaRobots);
    return () => {
      document.title = originalTitle;
      if (document.head.contains(metaRobots)) {
        document.head.removeChild(metaRobots);
      }
    };  
  }, []);

  const handleTextClick = () => {
    navigate('/');
  };

  return (
    <div
      className="not-found-container"
      style={{ backgroundImage: currentBgImage ? `url('${currentBgImage}')` : 'none' }}
    >
      <div
        className="not-found-text-content"
        onClick={handleTextClick}
        role="button"
        tabIndex={0}
        onKeyPress={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            handleTextClick();
          }
        }}
      >
        <h1 className="not-found-title">404</h1>
        <p className="not-found-message">Page not found. Return to home.</p>
      </div>
    </div>
  );
};

export default NotFoundPage;
