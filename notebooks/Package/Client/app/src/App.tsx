import React, { useState, useEffect, useRef, FormEvent, ChangeEvent } from 'react';
import pako from 'pako';
import './App.css';

interface Result {
  title: string;
  english_title?: string;
  missing_image?: string;
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
  image: string | null;
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

const UPDATE_URL = 'https://api.recs.moe/update';

// Component to handle image loading and fallback
const CardImage: React.FC<{ item: Result; onClick: () => void }> = ({ item, onClick }) => {
  const [src, setSrc] = useState<string>(item.image || item.missing_image || '');
  const [isFallback, setIsFallback] = useState<boolean>(!item.image);

  useEffect(() => {
    const newSrc = item.image || item.missing_image || '';
    setSrc(newSrc);
    setIsFallback(!item.image);
  }, [item.image, item.missing_image]);

  return (
    <div className="card-front" onClick={onClick}>
      <img
        className="card-image"
        src={src}
        alt={item.title}
        loading="lazy"
        onError={(e) => {
          e.currentTarget.onerror = null;
          setSrc(item.missing_image || '');
          setIsFallback(true);
        }}
      />
      {isFallback && (
        <div className="card-placeholder-overlay">Missing Image</div>
      )}
    </div>
  );
};

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
  // Use cumulative rotation (in degrees) for flipping
  const [cardRotation, setCardRotation] = useState<{ [index: number]: number }>({});
  // Track whether the flip button is active (disabling hover behavior)
  const [flipActive, setFlipActive] = useState<boolean>(false);

  const searchRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const buttonContainerRef = useRef<HTMLDivElement>(null);
  const gridViewRef = useRef<HTMLDivElement>(null);
  const loadMoreRef = useRef<HTMLDivElement>(null);

  const placeholders: Record<SourceType, string> = {
    MyAnimeList: 'Type a MyAnimeList username',
    AniList: 'Type an AniList username',
    Kitsu: 'Type a Kitsu username',
    'Anime-Planet': 'Type an Anime-Planet username',
  };

  useEffect(() => {
    setIsMobile(window.matchMedia && window.matchMedia('(hover: none)').matches);
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (results.length > 0) {
        if (
          searchRef.current &&
          !searchRef.current.contains(event.target as Node) &&
          buttonContainerRef.current &&
          !buttonContainerRef.current.contains(event.target as Node)
        ) {
          setShowButtons(false);
        }
      }
    };
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, [results]);

  useEffect(() => {
    if (!loadMoreRef.current || results.length >= totalResults) return;
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !loadingMore) {
            setLoadingMore(true);
            const offset = results.length;
            const payload = currentPayloadForPagination();
            fetchResults(payload, offset, true).finally(() => setLoadingMore(false));
          }
        });
      },
      {
        root: gridViewRef.current,
        threshold: 0.1,
      }
    );
    observer.observe(loadMoreRef.current);
    return () => {
      if (loadMoreRef.current) observer.unobserve(loadMoreRef.current);
    };
  }, [results, totalResults, loadingMore]);

  const currentPayloadForPagination = (): Payload => {
    if (results.length > 0) {
      return {
        state: apiState,
        action: {
          type: 'set_media',
          medium: cardType,
        },
      };
    }
    return {
      state: '',
      action: {
        type: 'add_user',
        source: activeSource,
        username: query,
      },
    };
  };

  // Reset rotation, synopsis state, and flip button when a new user is added
  const resetCardStates = () => {
    setCardRotation({});
    setShowSynopsis({});
    setFlipActive(false);
  };

  const fetchResults = (payload: Payload, offset: number = 0, append: boolean = false) => {
    const limit = isMobile ? 10 : 25;
    const extendedPayload = { ...payload, pagination: { offset, limit } };
    const payloadString = JSON.stringify(extendedPayload);
    const compressedPayload = pako.gzip(payloadString);

    return fetch(UPDATE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Encoding': 'gzip'
      },
      body: compressedPayload,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error('Failed to fetch results.');
        }
        return response.json();
      })
      .then((data) => {
        setApiState(data.state);
        setTotalResults(data.total);
        if (append) {
          setResults((prev) => [...prev, ...data.view]);
        } else {
          setResults(data.view);
          if (gridViewRef.current) {
            gridViewRef.current.scrollTop = 0;
          }
          if (extendedPayload.action.type === 'add_user') {
            setCardType('Anime');
          }
        }
        setErrorMessage('');
        setHasSearched(true);
        if (extendedPayload.action.type === 'add_user') {
          resetCardStates();
        }
        if (extendedPayload.action.type === 'set_media') {
          setShowSynopsis({});
        }

        if (data.followup_action) {
          const followupPayload = {
            state: data.state,
            action: data.followup_action,
            pagination: extendedPayload.pagination,
          };
          const followupString = JSON.stringify(followupPayload);
          const compressedFollowup = pako.gzip(followupString);

          fetch(UPDATE_URL, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Content-Encoding': 'gzip'
            },
            body: compressedFollowup,
          })
            .then((response) => {
              if (!response.ok) {
                throw new Error('Failed to fetch followup action results.');
              }
              return response.json();
            })
            .then((followupData) => {
              if (!followupData || Object.keys(followupData).length === 0) {
                return;
              }
              setApiState(followupData.state);
              setTotalResults(followupData.total);
              setResults(followupData.view);
              if (followupData.medium === 'Anime' || followupData.medium === 'Manga') {
                setCardType(followupData.medium);
              }
              setShowSynopsis({});
            })
            .catch((error) => {
              console.error('Error during followup action:', error);
            });
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
  };

  const handleSearch = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!query.trim()) return;
    const currentQuery = query;
    const source_map: Record<SourceType, string> = {
      MyAnimeList: 'mal',
      AniList: 'anilist',
      Kitsu: 'kitsu',
      'Anime-Planet': 'animeplanet',
    };

    const payload: AddUserPayload = {
      state: '',
      action: {
        type: 'add_user',
        source: source_map[activeSource],
        username: currentQuery,
      },
    };

    fetchResults(payload, 0, false).then(() => {
      setQuery('');
    });
    setShowButtons(false);
    if (inputRef.current) {
      inputRef.current.blur();
    }
  };

  const handleButtonClick = (source: SourceType) => {
    setActiveSource(source);
  };

  const handleMediaTypeChange = (type: CardType) => {
    setCardType(type);
    const payload: MediaTypePayload = {
      state: apiState,
      action: {
        type: 'set_media',
        medium: type,
      },
    };
    setShowSynopsis({});
    fetchResults(payload, 0, false);
  };

  // Flip button toggle handler that updates flipActive and rotates cards accordingly.
  const handleFlipToggle = (e: React.MouseEvent<HTMLButtonElement>) => {
    const newFlipActive = !flipActive;
    setFlipActive(newFlipActive);
    setCardRotation(prev => {
      const newMapping: { [index: number]: number } = {};
      for (let i = 0; i < results.length; i++) {
        const currentRotation = prev[i] || 0;
        if (newFlipActive) {
          // Activating flip: if card is showing its front, rotate by +180° to show the back.
          newMapping[i] = (currentRotation % 360 === 0) ? currentRotation + 180 : currentRotation;
        } else {
          // Deactivating flip: if card is showing its back, rotate by +180° to show the front.
          newMapping[i] = (currentRotation % 360 === 180) ? currentRotation + 180 : currentRotation;
        }
      }
      return newMapping;
    });
    e.currentTarget.blur();
  };

  // For mobile: clicking the card front rotates it by 180°
  const handleCardFrontClick = (index: number, item: Result) => {
    if (isMobile) {
      setCardRotation(prev => ({ ...prev, [index]: (prev[index] || 0) + 180 }));
    } else {
      // On desktop, clicking the card front opens the URL.
      window.open(item.url, '_blank', 'noopener,noreferrer');
    }
  };

  // On mobile, clicking the card back when synopsis is shown rotates it by 180°
  const handleCardBackClick = (index: number) => {
    if (isMobile && showSynopsis[index]) {
      setCardRotation(prev => ({ ...prev, [index]: (prev[index] || 0) + 180 }));
      setShowSynopsis(prev => ({ ...prev, [index]: false }));
    } else {
      setShowSynopsis(prev => ({ ...prev, [index]: !prev[index] }));
    }
  };

  return (
    <div className="container">
      <header ref={searchRef} className={hasSearched ? 'header--toggle' : ''}>
        {hasSearched ? (
          <div className="header-toggle">
            <button
              className={`header-toggle-button ${cardType === 'Anime' ? 'selected' : ''}`}
              onClick={() => handleMediaTypeChange('Anime')}
            >
              Anime
            </button>
            <button
              className={`header-toggle-button ${cardType === 'Manga' ? 'selected' : ''}`}
              onClick={() => handleMediaTypeChange('Manga')}
            >
              Manga
            </button>
            <button className={`flip-all-button ${flipActive ? 'selected' : ''}`} onClick={handleFlipToggle}>
              ⇄
            </button>
          </div>
        ) : (
          <form onSubmit={handleSearch}>
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
              }}
            />
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
          {(['MyAnimeList', 'AniList', 'Kitsu', 'Anime-Planet'] as SourceType[]).map((source) => (
            <button
              key={source}
              className={`source-button ${activeSource === source ? 'selected' : ''}`}
              onClick={() => handleButtonClick(source)}
            >
              {source}
            </button>
          ))}
        </div>
      )}

      {results.length > 0 ? (
        <>
          {!hasSearched && (
            <div className="card-toggle">
              <button
                className={`toggle-button ${cardType === 'Anime' ? 'selected' : ''}`}
                onClick={() => handleMediaTypeChange('Anime')}
              >
                Anime
              </button>
              <button
                className={`toggle-button ${cardType === 'Manga' ? 'selected' : ''}`}
                onClick={() => handleMediaTypeChange('Manga')}
              >
                Manga
              </button>
            </div>
          )}
          <div className="grid-view" ref={gridViewRef}>
            {results.map((item, index) => (
              <div
                key={index}
                className="card"
                // For desktop and when flip button is not active, use hover events.
                onMouseEnter={
                  !isMobile && !flipActive
                    ? () => {
                        const currentRotation = cardRotation[index] || 0;
                        if (currentRotation % 360 === 0) {
                          setCardRotation(prev => ({ ...prev, [index]: currentRotation + 180 }));
                        }
                      }
                    : undefined
                }
                onMouseLeave={
                  !isMobile && !flipActive
                    ? () => {
                        const currentRotation = cardRotation[index] || 0;
                        // On hover off, rotate in the opposite direction (-180) instead of adding +180.
                        if (currentRotation % 360 === 180) {
                          setCardRotation(prev => ({ ...prev, [index]: currentRotation - 180 }));
                        }
                      }
                    : undefined
                }
              >
                <div
                  className="card-inner"
                  style={cardRotation.hasOwnProperty(index) ? { transform: `rotateY(${cardRotation[index]}deg)` } : {}}
                >
                  <CardImage item={item} onClick={() => handleCardFrontClick(index, item)} />
                  <div className="card-back" onClick={() => handleCardBackClick(index)}>
                    <div className="card-back-bg" style={{ backgroundImage: `url(${item.image || item.missing_image || ''})` }}></div>
                    <div className="card-back-container">
                      <div
                        className="card-back-header"
                        onClick={(e) => {
                          e.stopPropagation();
                          window.open(item.url, '_blank', 'noopener,noreferrer');
                        }}
                      >
                        <div>{item.title}</div>
                        {item.english_title && (
                          <div className="card-back-english-title">{item.english_title}</div>
                        )}
                      </div>
                      <div className="card-back-body">
                        {showSynopsis[index] ? (
                          <p style={{ whiteSpace: 'pre-line' }}>{item.synopsis}</p>
                        ) : (
                          <div className="card-details">
                            {cardType === 'Anime' ? (
                              <>
                                {item.type && <p><strong>Medium:</strong> {item.type}</p>}
                                {item.season && <p><strong>Season:</strong> {item.season}</p>}
                                {item.episodes != null && <p><strong>Episodes:</strong> {item.episodes}</p>}
                                {item.status && <p><strong>Status:</strong> {item.status}</p>}
                                {item.startdate && <p><strong>Start Date:</strong> {item.startdate}</p>}
                                {item.enddate && <p><strong>End Date:</strong> {item.enddate}</p>}
                                {item.source && <p><strong>Source:</strong> {item.source}</p>}
                                {item.duration && <p><strong>Duration:</strong> {item.duration}</p>}
                                {item.studios && <p><strong>Studio:</strong> {item.studios}</p>}
                                {item.genres && <p><strong>Genres:</strong> {item.genres}</p>}
                              </>
                            ) : (
                              <>
                                {item.type && <p><strong>Medium:</strong> {item.type}</p>}
                                {item.status && <p><strong>Status:</strong> {item.status}</p>}
                                {item.startdate && <p><strong>Start Date:</strong> {item.startdate}</p>}
                                {item.enddate && <p><strong>End Date:</strong> {item.enddate}</p>}
                                {item.volumes != null && <p><strong>Volumes:</strong> {item.volumes}</p>}
                                {item.chapters != null && <p><strong>Chapters:</strong> {item.chapters}</p>}
                                {item.source && <p><strong>Source:</strong> {item.source}</p>}
                                {item.studios && <p><strong>Studio:</strong> {item.studios}</p>}
                                {item.genres && <p><strong>Genres:</strong> {item.genres}</p>}
                              </>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            <div ref={loadMoreRef} style={{ height: '1px' }}></div>
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
