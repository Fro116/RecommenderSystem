import React, { useState, useEffect } from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import HomePage from "./HomePage";
import ViewPage from "./ViewPage";
import AboutPage from "./AboutPage";
import NotFoundPage from "./NotFoundPage";
import "./Global.css";

const App: React.FC = () => {
  const [isMobile, setIsMobile] = useState<boolean>(false);
  const location = useLocation();

  useEffect(() => {
    const vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty("--vh", `${vh}px`);
    const handleResize = () => {
      const vh = window.innerHeight * 0.01;
      document.documentElement.style.setProperty("--vh", `${vh}px`);
    };
    window.addEventListener("resize", handleResize);
    setIsMobile(window.matchMedia?.("(hover: none)").matches ?? false);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const isHomePage = location.pathname === "/";
  const isAboutPage = location.pathname === "/about";

  const containerClass = isHomePage ? "container homepage" : "container";
  let containerStyle: React.CSSProperties;
  if (isHomePage) {
    containerStyle = {
      height: "calc(var(--vh, 1vh) * 100)",
      overflowY: "hidden",
    };
  } else if (isAboutPage) {
    containerStyle = {
      display: "flex",
      flexDirection: "column",
      height: "auto",
      minHeight: "calc(var(--vh, 1vh) * 100)",
    };
  } else {
    containerStyle = {
      display: "flex",
      flexDirection: "column",
      minHeight: "calc(var(--vh, 1vh) * 100)",
    };
  }

  return (
    <div className={containerClass} style={containerStyle}>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route
          path="/user/:source/:username"
          element={<ViewPage isMobile={isMobile} />}
        />
        <Route
          path="/item/:itemType/:source/:itemid"
          element={<ViewPage isMobile={isMobile} />}
        />
        <Route path="/about" element={<AboutPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </div>
  );
};

export default App;
