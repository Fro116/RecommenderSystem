import "./CardImage.css";
// src/components/CardImage.tsx
import React, { useState, useEffect } from "react";
import { Result, getBiggestImageUrl } from "../types"; // Import types and helper

interface CardImageProps {
  item: Result;
  onClick: () => void;
}

const CardImage: React.FC<CardImageProps> = ({ item, onClick }) => {
  const initialImageUrl = getBiggestImageUrl(item.image);
  const initialMissingImageUrl = getBiggestImageUrl(item.missing_image);
  const [src, setSrc] = useState<string>(
    initialImageUrl || initialMissingImageUrl || "",
  );
  const [isFallback, setIsFallback] = useState<boolean>(!initialImageUrl);

  useEffect(() => {
    const newSrc =
      getBiggestImageUrl(item.image) || getBiggestImageUrl(item.missing_image);
    setSrc(newSrc || ""); // Ensure src is always a string
    setIsFallback(!getBiggestImageUrl(item.image));
  }, [item.image, item.missing_image]);

  const targetWidth = 300;
  const targetHeight = 426;

  return (
    <div className="card-front" onClick={onClick}>
      <img
        loading="lazy"
        className="card-image"
        src={src}
        alt={item.title}
        width={targetWidth}
        height={targetHeight}
        onError={(e) => {
          e.currentTarget.onerror = null;
          setSrc(initialMissingImageUrl || ""); // Fallback
          setIsFallback(true);
        }}
      />
      {isFallback &&
        src && ( // Show overlay only if it's a fallback AND there's a src (even if broken)
          <div className="card-placeholder-overlay">Missing Image</div>
        )}
      {!src && ( // Show overlay if there's no image URL at all
        <div className="card-placeholder-overlay">No Image Available</div>
      )}
    </div>
  );
};

export default CardImage;
