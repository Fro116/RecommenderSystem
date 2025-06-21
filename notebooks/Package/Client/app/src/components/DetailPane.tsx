// src/components/DetailPane.tsx
import './DetailPane.css';
import React from 'react';
import { Result, CardType, getBiggestImageUrl, stringToHslColor } from '../types';

interface DetailPaneProps {
  item: Result | null;
  cardType: CardType;
  onClose: () => void;
  isMobile: boolean;
}

const DetailPane: React.FC<DetailPaneProps> = ({ item, cardType, isMobile, onClose }) => {
  const [mouseDownPos, setMouseDownPos] = React.useState<{ x: number; y: number } | null>(null);

  // Handle closing the pane with the Escape key
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    // Cleanup the event listener when the component unmounts
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [onClose]);

  if (!item) {
    return null;
  }

  // Stop background scroll when pane is open
  React.useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, []);

  const handleContentMouseDown = (e: React.MouseEvent) => {
    if (!isMobile) return;
    setMouseDownPos({ x: e.clientX, y: e.clientY });
  };

  const handleContentMouseUp = (e: React.MouseEvent) => {
    if (!isMobile || !mouseDownPos) return;

    // Don't close if user is clicking a link
    let target = e.target as HTMLElement;
    while (target && target !== e.currentTarget) {
        if (target.tagName === 'A') {
            setMouseDownPos(null);
            return;
        }
        target = target.parentElement as HTMLElement;
    }
    
    // Check if the mouse moved more than a few pixels (i.e., dragging to select text)
    const deltaX = Math.abs(e.clientX - mouseDownPos.x);
    const deltaY = Math.abs(e.clientY - mouseDownPos.y);
    if (deltaX < 5 && deltaY < 5) {
        onClose();
    }
    setMouseDownPos(null);
  };

  const DetailsSection = () => (
    <div className="detail-pane-details-section">
      <h4>Details</h4>
      <div className="details-table">
        {cardType === 'Anime' ? (
          <>
            <div><strong>Type</strong><span>{item.type || '-'}</span></div>
            <div><strong>Season</strong><span>{item.season || '-'}</span></div>
            <div><strong>Source</strong><span>{item.source || '-'}</span></div>
            <div><strong>Episodes</strong><span>{item.episodes ?? '-'}</span></div>
            <div><strong>Duration</strong><span>{item.duration || '-'}</span></div>
            <div><strong>Studio</strong><span>{item.studios || '-'}</span></div>
          </>
        ) : (
          <>
            <div><strong>Type</strong><span>{item.type || '-'}</span></div>
            <div><strong>Year</strong><span>{item.startdate ? item.startdate.substring(0, 4) : '-'}</span></div>
            <div><strong>Status</strong><span>{item.status || '-'}</span></div>
            <div><strong>Volumes</strong><span>{item.volumes ?? '-'}</span></div>
            <div><strong>Chapters</strong><span>{item.chapters ?? '-'}</span></div>
            <div><strong>Magazine</strong><span>{item.studios || '-'}</span></div>
          </>
        )}
      </div>
    </div>
  );

  const TagsSection = () => (
    item.genres ? (
      <div className="detail-pane-tags-section">
        <h4>Tags</h4>
        <div className="tags-container">
          {item.genres.split(', ').map(tag => {
              const backgroundColor = stringToHslColor(tag, 70, 85);
              return (
                <span key={tag} className="tag" style={{ backgroundColor }}>
                  {tag}
                </span>
              );
            })}
        </div>
      </div>
    ) : null
  );
  
  const SynopsisSection = () => (
    <div className="detail-pane-synopsis-section">
        <h4>Synopsis</h4>
        <p>{item.synopsis || 'No synopsis available.'}</p>
    </div>
  );


  return (
    <div className="detail-pane-overlay" onClick={onClose}>
      <div 
        className="detail-pane-content"
        onClick={(e) => {
          if (!isMobile) {
            e.stopPropagation();
          }
        }}
        onMouseDown={handleContentMouseDown}
        onMouseUp={handleContentMouseUp}
      >
        <button className="detail-pane-close" onClick={onClose}>&times;</button>
        
        {isMobile ? (
          <>
            <div className="detail-pane-header-mobile">
              <a href={item.url} target="_blank" rel="noopener noreferrer" className="detail-pane-title-link">
                <h2 className="detail-pane-title">{item.title}</h2>
              </a>
              {item.english_title && <h3 className="detail-pane-english-title">{item.english_title}</h3>}
            </div>
            <DetailsSection />
            <TagsSection />
            <SynopsisSection />
          </>
        ) : (
          <>
            <div className="detail-pane-grid">
              <div className="detail-pane-left">
                <img 
                  src={getBiggestImageUrl(item.image) || getBiggestImageUrl(item.missing_image) || ''} 
                  alt={item.title} 
                  className="detail-pane-image"
                />
              </div>
              <div className="detail-pane-right">
                <a href={item.url} target="_blank" rel="noopener noreferrer" className="detail-pane-title-link">
                  <h2 className="detail-pane-title">{item.title}</h2>
                </a>
                {item.english_title && <h3 className="detail-pane-english-title">{item.english_title}</h3>}
                <DetailsSection />
                <TagsSection />
              </div>
            </div>
            <SynopsisSection />
          </>
        )}
      </div>
    </div>
  );
};

export default DetailPane;