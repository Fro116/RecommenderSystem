import React, { useState, useEffect, useRef, FormEvent, ChangeEvent } from 'react';
import pako from 'pako';
import './App.css';

// Define the base API URL so it's only written once
const API_BASE = 'https://api.recs.moe';
const UPDATE_URL = `${API_BASE}/update`;

// Helper function to pick the biggest image URL from an array of image objects.
const getBiggestImageUrl = (images: any): string => {
  if (Array.isArray(images) && images.length > 0) {
    return images.reduce((prev: any, curr: any) => {
      return (prev.width * prev.height) >= (curr.width * curr.height) ? prev : curr;
    }).url;
  }
  return images || '';
};

//
// Existing types and payloads remain unchanged
//
interface Result {
  title: string;
  english_title?: string;
  missing_image: any;
  url: string;
  source: string | null;
  type: string;
  episodes?: number | null;
  duration?: string | null;
  volumes?: number | null;
  chapters?: number | null;
  status?: string;
  season?: string;
  startdate?: string;
  enddate?: string;
  studios?: string;
  genres?: string;
  synopsis?: string;
  image: any;
}

type SourceType = 'MyAnimeList' | 'AniList' | 'Kitsu' | 'Anime-Planet';
type CardType = 'Anime' | 'Manga';

interface AddUserPayload {
  state: string;
  action: {
    type: 'add_user';
    source: string;
    username: string;
  };
}

interface MediaTypePayload {
  state: string;
  action: {
    type: 'set_media';
    medium: CardType;
  };
}

type Payload = AddUserPayload | MediaTypePayload;

// Updated interface for autocomplete results including the new "matched" field.
interface AutocompleteItem {
  username: string;
  avatar: string | null;
  missing_avatar: string | null;
  matched: boolean[];
}

//
// ManualScrollDiv Component with momentum scrolling and boundary pass-through
//
const ManualScrollDiv: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ children, ...props }) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const startY = useRef<number>(0);
  const startScrollTop = useRef<number>(0);
  const lastY = useRef<number>(0);
  const lastTime = useRef<number>(0);
  const velocity = useRef<number>(0);
  const momentumFrame = useRef<number | null>(null);

  const handleTouchStart = (e: TouchEvent) => {
    if (momentumFrame.current) {
      cancelAnimationFrame(momentumFrame.current);
      momentumFrame.current = null;
    }
    startY.current = e.touches[0].clientY;
    lastY.current = e.touches[0].clientY;
    startScrollTop.current = scrollRef.current ? scrollRef.current.scrollTop : 0;
    lastTime.current = e.timeStamp;
    velocity.current = 0;
    e.stopPropagation();
  };

  const handleTouchMove = (e: TouchEvent) => {
    const currentY = e.touches[0].clientY;
    const currentTime = e.timeStamp;
    const deltaY = currentY - startY.current;

    if (scrollRef.current) {
      const newScrollTop = startScrollTop.current - deltaY;
      const maxScrollTop = scrollRef.current.scrollHeight - scrollRef.current.clientHeight;
      if (newScrollTop > 0 && newScrollTop < maxScrollTop) {
        scrollRef.current.scrollTop = newScrollTop;
        e.preventDefault();
        e.stopPropagation();
      } else {
        if (newScrollTop < 0) {
          scrollRef.current.scrollTop = 0;
        } else if (newScrollTop > maxScrollTop) {
          scrollRef.current.scrollTop = maxScrollTop;
        }
      }
    }
    const dt = currentTime - lastTime.current;
    if (dt > 0) {
      velocity.current = (lastY.current - currentY) / dt;
    }
    lastY.current = currentY;
    lastTime.current = currentTime;
  };

  const momentum = () => {
    if (!scrollRef.current) return;
    const friction = 0.95;
    if (Math.abs(velocity.current) < 0.02) return;
    scrollRef.current.scrollTop += velocity.current * 16;
    velocity.current *= friction;
    if (
      scrollRef.current.scrollTop <= 0 ||
      scrollRef.current.scrollTop >= scrollRef.current.scrollHeight - scrollRef.current.clientHeight
    ) {
      return;
    }
    momentumFrame.current = requestAnimationFrame(momentum);
  };

  const handleTouchEnd = (_e: TouchEvent) => {
    momentumFrame.current = requestAnimationFrame(momentum);
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.addEventListener('touchstart', handleTouchStart, { passive: false });
    el.addEventListener('touchmove', handleTouchMove, { passive: false });
    el.addEventListener('touchend', handleTouchEnd, { passive: false });
    return () => {
      el.removeEventListener('touchstart', handleTouchStart);
      el.removeEventListener('touchmove', handleTouchMove);
      el.removeEventListener('touchend', handleTouchEnd);
      if (momentumFrame.current) {
        cancelAnimationFrame(momentumFrame.current);
      }
    };
  }, []);
  return (
    <div ref={scrollRef} {...props}>
      {children}
    </div>
  );
};

//
// Updated CardImage Component using a native <img> element with lazy loading
//
const CardImage: React.FC<{ item: Result; onClick: () => void }> = ({ item, onClick }) => {
  const initialImageUrl = getBiggestImageUrl(item.image);
  const initialMissingImageUrl = getBiggestImageUrl(item.missing_image);
  const [src, setSrc] = useState<string>(initialImageUrl || initialMissingImageUrl || '');
  const [isFallback, setIsFallback] = useState<boolean>(!initialImageUrl);

  useEffect(() => {
    const newSrc = getBiggestImageUrl(item.image) || getBiggestImageUrl(item.missing_image);
    setSrc(newSrc);
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
          setSrc(initialMissingImageUrl || '');
          setIsFallback(true);
        }}
      />
      {isFallback && (
        <div className="card-placeholder-overlay">Missing Image</div>
      )}
    </div>
  );
};

//
// The App component
//
const App: React.FC = () => {
  const [query, setQuery] = useState<string>('');
  const [activeSource, setActiveSource] = useState<SourceType>('MyAnimeList');
  const [results, setResults] = useState<Result[]>([]);
  const [totalResults, setTotalResults] = useState<number>(0);
  const [showButtons, setShowButtons] = useState<boolean>(true);
  const [cardType, setCardType] = useState<CardType>('Anime');
  const [apiState, setApiState] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [loadingMore, setLoadingMore] = useState<boolean>(false);
  const [showSynopsis, setShowSynopsis] = useState<{ [index: number]: boolean }>({});
  const [hasSearched, setHasSearched] = useState<boolean>(false);
  const [isMobile, setIsMobile] = useState<boolean>(false);
  const [cardRotation, setCardRotation] = useState<{ [index: number]: number }>({});
  const [flipActive, setFlipActive] = useState<boolean>(false);
  // New state for autocomplete results and a flag to disable fetching when updated via autocomplete click.
  const [autocompleteResults, setAutocompleteResults] = useState<AutocompleteItem[]>([]);
  const [autocompleteDisabled, setAutocompleteDisabled] = useState<boolean>(false);

  const searchRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const buttonContainerRef = useRef<HTMLDivElement>(null);
  const gridViewRef = useRef<HTMLDivElement>(null);
  const loadMoreRef = useRef<HTMLDivElement>(null);
  // New ref for the search container (input + autocomplete)
  const searchContainerRef = useRef<HTMLDivElement>(null);

  const placeholders: Record<SourceType, string> = {
    MyAnimeList: 'Type a MyAnimeList username',
    AniList: 'Type an AniList username',
    Kitsu: 'Type a Kitsu username',
    'Anime-Planet': 'Type an Anime-Planet username',
  };

  useEffect(() => {
    const vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--vh', `${vh}px`);
    setIsMobile(window.matchMedia && window.matchMedia('(hover: none)').matches);
  }, []);

  // Hide autocomplete when the input loses focus
  const handleInputBlur = () => {
    setAutocompleteResults([]);
  };

  // Debounced effect to fetch autocomplete suggestions (50ms delay)
  useEffect(() => {
    if (autocompleteDisabled) return;
    if (query.trim() === '') {
      setAutocompleteResults([]);
      return;
    }
    const handler = setTimeout(() => {
      fetchAutocomplete();
    }, 50);
    return () => clearTimeout(handler);
  }, [query, activeSource, autocompleteDisabled]);

  // Using POST with JSON as requested.
  const fetchAutocomplete = async () => {
    const source_map: Record<SourceType, string> = {
      MyAnimeList: 'mal',
      AniList: 'anilist',
      Kitsu: 'kitsu',
      'Anime-Planet': 'animeplanet',
    };
    try {
      const response = await fetch(`${API_BASE}/autocomplete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "user",
          source: source_map[activeSource],
          prefix: query,
        }),
      });
      if (response.ok) {
        const data = await response.json();
        if (data && data.autocompletes) {
          setAutocompleteResults(data.autocompletes);
        } else {
          setAutocompleteResults([]);
        }
      }
    } catch (e) {
      console.error(e);
      setAutocompleteResults([]);
    }
  };

  const currentPayloadForPagination = (): Payload => {
    if (results.length > 0) {
      return { state: apiState, action: { type: 'set_media', medium: cardType } };
    }
    return { state: '', action: { type: 'add_user', source: activeSource, username: query } };
  };

  const resetCardStates = () => {
    setCardRotation({});
    setShowSynopsis({});
    setFlipActive(false);
  };

  // Updated infinite scrolling useEffect with rootMargin and unconditional observer creation
  useEffect(() => {
    if (!loadMoreRef.current) return;
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !loadingMore && results.length < totalResults) {
            setLoadingMore(true);
            const offset = results.length;
            const payload = currentPayloadForPagination();
            fetchResults(payload, offset, true).finally(() => setLoadingMore(false));
          }
        });
      },
      { root: gridViewRef.current || null, rootMargin: '100px', threshold: 0.1 }
    );
    observer.observe(loadMoreRef.current);
    return () => {
      if (loadMoreRef.current) observer.unobserve(loadMoreRef.current);
    };
  }, [results, totalResults, loadingMore]);

  const fetchResults = (payload: Payload, offset: number = 0, append: boolean = false) => {
    const limit = isMobile ? 10 : 25;
    const extendedPayload = { ...payload, pagination: { offset, limit } };
    const payloadString = JSON.stringify(extendedPayload);
    const compressedPayload = pako.gzip(payloadString);
    return fetch(UPDATE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip' },
      body: compressedPayload,
    })
      .then((response) => {
        if (!response.ok) throw new Error('Failed to fetch results.');
        return response.json();
      })
      .then((data) => {
        setApiState(data.state);
        setTotalResults(data.total);
        if (append) {
          setResults(prev => [...prev, ...data.view]);
        } else {
          setResults(data.view);
          if (gridViewRef.current) gridViewRef.current.scrollTop = 0;
          if (extendedPayload.action.type === 'add_user') setCardType('Anime');
        }
        setErrorMessage('');
        setHasSearched(true);
        if (extendedPayload.action.type === 'add_user') resetCardStates();
        if (extendedPayload.action.type === 'set_media') setShowSynopsis({});
        if (data.followup_action) {
          const followupPayload = { state: data.state, action: data.followup_action, pagination: extendedPayload.pagination };
          const followupString = JSON.stringify(followupPayload);
          const compressedFollowup = pako.gzip(followupString);
          fetch(UPDATE_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip' },
            body: compressedFollowup,
          })
            .then((response) => {
              if (!response.ok) throw new Error('Failed to fetch followup action results.');
              return response.json();
            })
            .then((followupData) => {
              if (!followupData || Object.keys(followupData).length === 0) return;
              setApiState(followupData.state);
              setTotalResults(followupData.total);
              setResults(followupData.view);
              if (followupData.medium === 'Anime' || followupData.medium === 'Manga') {
                setCardType(followupData.medium);
              }
              setShowSynopsis({});
            })
            .catch((error) => console.error('Error during followup action:', error));
        }
        return data;
      })
      .catch((error) => {
        console.error('Error:', error);
        setErrorMessage('Oops! Something went wrong. Please try again.');
      });
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
    setAutocompleteDisabled(false);
  };

  const handleSearch = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!query.trim()) return;
    const source_map: Record<SourceType, string> = {
      MyAnimeList: 'mal',
      AniList: 'anilist',
      Kitsu: 'kitsu',
      'Anime-Planet': 'animeplanet',
    };
    const payload: AddUserPayload = { state: '', action: { type: 'add_user', source: source_map[activeSource], username: query } };
    fetchResults(payload, 0, false).then(() => setQuery(''));
    setShowButtons(false);
    if (inputRef.current) inputRef.current.blur();
  };

  const handleButtonClick = (source: SourceType) => setActiveSource(source);

  const handleMediaTypeChange = (type: CardType) => {
    setCardType(type);
    const payload: MediaTypePayload = { state: apiState, action: { type: 'set_media', medium: type } };
    setShowSynopsis({});
    fetchResults(payload, 0, false);
  };

  const handleFlipToggle = (e: React.MouseEvent<HTMLButtonElement>) => {
    const newFlipActive = !flipActive;
    setFlipActive(newFlipActive);
    setCardRotation(prev => {
      const newMapping: { [index: number]: number } = {};
      for (let i = 0; i < results.length; i++) {
        const currentRotation = prev[i] || 0;
        newMapping[i] = newFlipActive
          ? (currentRotation % 360 === 0 ? currentRotation + 180 : currentRotation)
          : (currentRotation % 360 === 180 ? currentRotation + 180 : currentRotation);
      }
      return newMapping;
    });
    e.currentTarget.blur();
  };

  const handleCardFrontClick = (index: number) => {
    if (isMobile) {
      setCardRotation(prev => ({ ...prev, [index]: (prev[index] || 0) + 180 }));
    }
  };

  const handleCardBackClick = (index: number) => {
    if (isMobile && !flipActive && showSynopsis[index]) {
      setCardRotation(prev => ({ ...prev, [index]: (prev[index] || 0) + 180 }));
      setTimeout(() => {
        setShowSynopsis(prev => ({ ...prev, [index]: false }));
      }, 300);
    } else if (isMobile && flipActive && showSynopsis[index]) {
      setShowSynopsis(prev => ({ ...prev, [index]: false }));
    } else if (isMobile && showSynopsis[index]) {
      setCardRotation(prev => ({ ...prev, [index]: (prev[index] || 0) + 180 }));
      setShowSynopsis(prev => ({ ...prev, [index]: false }));
    } else {
      setShowSynopsis(prev => ({ ...prev, [index]: !prev[index] }));
    }
  };

  const containerStyle: React.CSSProperties = results.length === 0
    ? { height: 'calc(var(--vh, 1vh) * 100)', overflowY: 'hidden' }
    : {};

  return (
    <div className={`container ${results.length === 0 ? 'homepage' : ''}`} style={containerStyle}>
      <header ref={searchRef} className={hasSearched ? 'header--toggle' : ''}>
        {hasSearched ? (
          <div className="header-toggle">
            <button className={`header-toggle-button ${cardType === 'Anime' ? 'selected' : ''}`}
              onClick={() => handleMediaTypeChange('Anime')}>Anime</button>
            <button className={`header-toggle-button ${cardType === 'Manga' ? 'selected' : ''}`}
              onClick={() => handleMediaTypeChange('Manga')}>Manga</button>
            <button className={`flip-all-button ${flipActive ? 'selected' : ''}`} onClick={handleFlipToggle}>
              <img src={flipActive ? "/flip-icon-selected.webp" : "/flip-icon.webp"} alt="flip" />
            </button>
          </div>
        ) : (
          <form onSubmit={handleSearch}>
            <div ref={searchContainerRef} style={{ position: 'relative', width: '80%' }}>
              <input 
                ref={inputRef} 
                type="text" 
                placeholder={showButtons ? placeholders[activeSource] : ''} 
                value={query} 
                onChange={handleInputChange} 
                onFocus={() => { 
                  setShowButtons(true); 
                  setErrorMessage(''); 
                  setHasSearched(false); 
                  if (query.trim() !== '') { 
                    fetchAutocomplete(); 
                  }
                }}
                onBlur={handleInputBlur}
              />
              {autocompleteResults.length > 0 && (
                <div className="autocomplete-container">
                  {autocompleteResults.map((item, index) => (
                    <div 
                      key={index} 
                      className="autocomplete-item"
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => {
                        // When an autocomplete is clicked, disable further autocomplete fetching
                        setAutocompleteDisabled(true);
                        setQuery(item.username);
                        setAutocompleteResults([]);
                        const source_map: Record<SourceType, string> = {
                          MyAnimeList: 'mal',
                          AniList: 'anilist',
                          Kitsu: 'kitsu',
                          'Anime-Planet': 'animeplanet',
                        };
                        const payload: AddUserPayload = { state: '', action: { type: 'add_user', source: source_map[activeSource], username: item.username } };
                        fetchResults(payload, 0, false).then(() => setQuery(''));
                        setShowButtons(false);
                        if (inputRef.current) inputRef.current.blur();
                      }}
                    >
                      {(item.avatar || item.missing_avatar) && (
                        <img 
                          className="autocomplete-avatar" 
                          src={item.avatar || item.missing_avatar || ''} 
                          alt={item.username}
                          onError={(e) => {
                            if (e.currentTarget.src !== (item.missing_avatar || '')) {
                              e.currentTarget.src = item.missing_avatar || '';
                            }
                          }}
                        />
                      )}
                      <span>
                        {item.username.split('').map((char, idx) => (
                          <span key={idx} className={item.matched[idx] ? "autocomplete-match" : "autocomplete-unmatch"}>
                            {char}
                          </span>
                        ))}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </form>
        )}
      </header>

      {errorMessage && (
        <div className="error-banner">
          <span>{errorMessage}</span>
          <button onClick={() => setErrorMessage('')}>&times;</button>
        </div>
      )}

      {showButtons && !hasSearched && (
        <div className="source-buttons" ref={buttonContainerRef}>
          {(['MyAnimeList', 'AniList', 'Kitsu', 'Anime-Planet'] as SourceType[]).map(source => (
            <button key={source} className={`source-button ${activeSource === source ? 'selected' : ''}`}
              onClick={() => handleButtonClick(source)}>
              {source}
            </button>
          ))}
        </div>
      )}

      {results.length > 0 ? (
        <>
          {!hasSearched && (
            <div className="card-toggle">
              <button className={`toggle-button ${cardType === 'Anime' ? 'selected' : ''}`}
                onClick={() => handleMediaTypeChange('Anime')}>Anime</button>
              <button className={`toggle-button ${cardType === 'Manga' ? 'selected' : ''}`}
                onClick={() => handleMediaTypeChange('Manga')}>Manga</button>
            </div>
          )}
          <div className="grid-container" ref={gridViewRef}>
            <div className="grid-view">
              {results.map((item, index) => {
                const currentRotation = cardRotation.hasOwnProperty(index)
                  ? cardRotation[index]
                  : (flipActive ? 180 : 0);
                return (
                  <div key={index} className="card"
                    onMouseEnter={!isMobile && !flipActive ? () => {
                      const r = cardRotation[index] || 0;
                      if (r % 360 === 0) setCardRotation(prev => ({ ...prev, [index]: r + 180 }));
                    } : undefined}
                    onMouseLeave={!isMobile && !flipActive ? () => {
                      const r = cardRotation[index] || 0;
                      if (r % 360 === 180) setCardRotation(prev => ({ ...prev, [index]: r - 180 }));
                    } : undefined}
                  >
                    <div className="card-inner" style={{ transform: `rotateY(${currentRotation}deg)` }}>
                      <CardImage item={item} onClick={() => handleCardFrontClick(index)} />
                      <div className="card-back" onClick={() => handleCardBackClick(index)}>
                        <div className="card-back-bg" style={{ backgroundImage: `url(${getBiggestImageUrl(item.image) || getBiggestImageUrl(item.missing_image)})` }}></div>
                        <div className="card-back-container">
                          <div className="card-back-header" onClick={(e) => {
                            e.stopPropagation();
                            window.open(item.url, '_blank', 'noopener,noreferrer');
                          }}>
                            <div>{item.title}</div>
                            {item.english_title && (<div className="card-back-english-title">{item.english_title}</div>)}
                          </div>
                          <ManualScrollDiv className="card-back-body">
                            {showSynopsis[index] ? (
                              <p style={{ whiteSpace: 'pre-line' }}>{item.synopsis}</p>
                            ) : (
                              cardType === 'Anime' ? (
                                <div className="card-details">
                                  <table>
                                    <tbody>
                                      {item.type && (<tr><td><strong>Medium:</strong></td><td>{item.type}</td></tr>)}
                                      {item.season && (<tr><td><strong>Season:</strong></td><td>{item.season}</td></tr>)}
                                      {item.episodes != null && (<tr><td><strong>Episodes:</strong></td><td>{item.episodes}</td></tr>)}
                                      {item.status && (<tr><td><strong>Status:</strong></td><td>{item.status}</td></tr>)}
                                      {item.startdate && (<tr><td><strong>Start&nbsp;Date:</strong></td><td>{item.startdate}</td></tr>)}
                                      {item.enddate && (<tr><td><strong>End Date:</strong></td><td>{item.enddate}</td></tr>)}
                                      {item.source && (<tr><td><strong>Source:</strong></td><td>{item.source}</td></tr>)}
                                      {item.duration && (<tr><td><strong>Duration:</strong></td><td>{item.duration}</td></tr>)}
                                      {item.studios && (<tr><td><strong>Studio:</strong></td><td>{item.studios}</td></tr>)}
                                      {item.genres && (<tr><td><strong>Genres:</strong></td><td>{item.genres}</td></tr>)}
                                    </tbody>
                                  </table>
                                </div>
                              ) : (
                                <div className="card-details">
                                  <table>
                                    <tbody>
                                      {item.type && (<tr><td><strong>Medium:</strong></td><td>{item.type}</td></tr>)}
                                      {item.status && (<tr><td><strong>Status:</strong></td><td>{item.status}</td></tr>)}
                                      {item.startdate && (<tr><td><strong>Start&nbsp;Date:</strong></td><td>{item.startdate}</td></tr>)}
                                      {item.enddate && (<tr><td><strong>End Date:</strong></td><td>{item.enddate}</td></tr>)}
                                      {item.volumes != null && (<tr><td><strong>Volumes:</strong></td><td>{item.volumes}</td></tr>)}
                                      {item.chapters != null && (<tr><td><strong>Chapters:</strong></td><td>{item.chapters}</td></tr>)}
                                      {item.source && (<tr><td><strong>Source:</strong></td><td>{item.source}</td></tr>)}
                                      {item.studios && (<tr><td><strong>Studio:</strong></td><td>{item.studios}</td></tr>)}
                                      {item.genres && (<tr><td><strong>Genres:</strong></td><td>{item.genres}</td></tr>)}
                                    </tbody>
                                  </table>
                                </div>
                              )
                            )}
                          </ManualScrollDiv>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
              <div ref={loadMoreRef} style={{ height: '1px' }}></div>
            </div>
          </div>
        </>
      ) : (
        <div className="home-overlay">
          Recs☆Moe is a recommender system for anime and manga
        </div>
      )}
    </div>
  );
};

export default App;
