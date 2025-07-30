import "./NotFoundPage.css";
// src/NotFoundPage.tsx
import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";

const backgroundImages: Record<"loading" | "notfound", string[]> = {
  loading: Array.from(
    { length: 23 },
    (_, i) =>
      `https://cdn.recs.moe/images/backgrounds/loading/${i + 1}.large.webp`,
  ),
  notfound: Array.from(
    { length: 35 },
    (_, i) =>
      `https://cdn.recs.moe/images/backgrounds/notfound/${i + 1}.large.webp`,
  ),
};

const NotFoundPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [bgImageUrls, setBgImageUrls] = useState<string[]>([]);
  const [isPortrait, setIsPortrait] = useState(
    window.innerHeight > window.innerWidth,
  );

  const isItemPath = location.pathname.startsWith("/item/");

  useEffect(() => {
    const handleResize = () =>
      setIsPortrait(window.innerHeight > window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    const pageTitle = isItemPath ? "Feature In Development" : "Page Not Found";
    const originalTitle = document.title;
    document.title = `${pageTitle} | Recs☆Moe`;
    const r = Math.random();
    let bgImages: string[];
    if (r < 0.9) {
      bgImages = backgroundImages.notfound;
    } else {
      bgImages = backgroundImages.loading;
    }

    const numImages = isPortrait ? 4 : 1;
    const shuffled = [...bgImages].sort(() => 0.5 - Math.random());
    setBgImageUrls(shuffled.slice(0, numImages));

    const metaRobots = document.createElement("meta");
    metaRobots.name = "robots";
    metaRobots.content = "noindex";
    document.head.appendChild(metaRobots);

    return () => {
      document.title = originalTitle;
      if (document.head.contains(metaRobots)) {
        document.head.removeChild(metaRobots);
      }
    };
  }, [isItemPath, isPortrait]);

  const handleTextClick = () => {
    navigate("/");
  };

  return (
    <div className="not-found-container">
      <div className="not-found-background-stack">
        {bgImageUrls.map((url, index) => (
          <div
            key={index}
            className="not-found-background-image"
            style={{ backgroundImage: `url('${url}')` }}
          />
        ))}
      </div>
      <div className="not-found-content-overlay">
        <div
          className="not-found-text-content"
          onClick={handleTextClick}
          role="button"
          tabIndex={0}
          onKeyPress={(e) => {
            if (e.key === "Enter" || e.key === " ") handleTextClick();
          }}
        >
          {isItemPath ? (
            <>
              <h1 className="not-found-title">Coming Soon</h1>
              <p className="not-found-message">
                This feature is currently in development. Stay tuned for
                updates!
              </p>
            </>
          ) : (
            <>
              <h1 className="not-found-title">404</h1>
              <p className="not-found-message">
                Page not found. Return to home.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default NotFoundPage;
