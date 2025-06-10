// src/NotFoundPage.tsx
import React, { useState, useEffect } from 'react'; // Import useState and useEffect
import { useNavigate } from 'react-router-dom';
import './App.css';

const backgroundImages: Record<"loading" | "notfound", string[]> = {
  loading: Array.from({ length: 23 }, (_, i) =>
    `https://cdn.recs.moe/images/backgrounds/loading/${i + 1}.large.webp`
  ),
  notfound: Array.from({ length: 35 }, (_, i) =>
    `https://cdn.recs.moe/images/backgrounds/notfound/${i + 1}.large.webp`
  ),
};

const NotFoundPage: React.FC = () => {
  const navigate = useNavigate();
  const [currentBgImage, setCurrentBgImage] = useState<string>('');

  useEffect(() => {
    const originalTitle = document.title;
    document.title = 'Page Not Found | Recs☆Moe';
    const r = Math.random();
    let bgImages: string[];
    if (r < 0.9) {
      bgImages = backgroundImages.notfound;
    } else {
      bgImages = backgroundImages.loading;
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
