// src/ViewPage.tsx
import './Header.css';
import './ViewPage.css';
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom';
import pako from 'pako';
import { Result, CardType, MediaTypePayload, Payload, AddUserPayload, getBiggestImageUrl, UPDATE_URL, AddItemPayload } from './types';
import CardImage from './components/CardImage';
import ManualScrollDiv from './components/ManualScrollDiv';

interface ViewPageProps {
  isMobile: boolean;
}

const loadingMessages = [
  'Dusting off some hidden gems...',
  'Consulting the head maid...',
  'Polishing your recommendations...',
  'Accessing your profile...',
  'Analyzing your watch history...',
  'Tidying up the final selections...',
  'Please wait a moment...',
  'Scrubbing away fanservice...',
  'Loading recommendations...',
];

const ViewPage: React.FC<ViewPageProps> = ({ isMobile }) => {
  // Hooks and State
  const navigate = useNavigate();
  const { source, username, itemType, itemid } = useParams<{
    source: string;
    username?: string;
    itemType?: 'anime' | 'manga';
    itemid?: string;
  }>();
  const [results, setResults] = useState<Result[]>([]);
  const [apiState, setApiState] = useState<string>('');
  const [totalResults, setTotalResults] = useState<number>(0);
  const [cardType, setCardType] = useState<CardType>('Anime');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [loadingText, setLoadingText] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [loadingMore, setLoadingMore] = useState<boolean>(false);
  const [showSynopsis, setShowSynopsis] = useState<{ [index: number]: boolean }>({});
  const [cardRotation, setCardRotation] = useState<{ [index: number]: number }>({});
  const [flipState, setFlipState] = useState<'none' | 'selected-details' | 'selected-synopsis'>('none');

  // Refs
  const gridViewRef = useRef<HTMLDivElement>(null);
  const loadMoreRef = useRef<HTMLDivElement>(null);
  const initialFetchStartedRef = useRef<boolean>(false);

  const cardRotationRef = useRef(cardRotation);
  useEffect(() => {
    cardRotationRef.current = cardRotation;
  }, [cardRotation]);

  const isUserMode = !!(source && username);
  const isItemMode = !!(source && itemType && itemid);

  useEffect(() => {
    const siteDefaultTitle = document.title;
    if (isUserMode) {
      document.title = `${username}'s Room | Recs☆Moe`;
    } else if (isItemMode) {
      document.title = `Recommendations | Recs☆Moe`;
    }
    return () => {
      document.title = siteDefaultTitle;
    };
  }, [isUserMode, isItemMode, username]);


  // --- Functions ---
  const fetchResults = useCallback(
    (payload: Payload, offset: number = 0, append: boolean = false, isInitialLoad: boolean = false) => {
      if (isInitialLoad) {
        setIsLoading(true);
        setError('');
        setResults([]);
        resetCardStates();
      } else if (append) {
        setLoadingMore(true);
        setError('');
      } else {
        setIsLoading(true);
        setError('');
      }

      const limit = isMobile ? 12 : 24;
      const extendedPayload = { ...payload, pagination: { offset, limit } };
      const payloadString = JSON.stringify(extendedPayload);
      const compressedPayload = pako.gzip(payloadString);

      fetch(UPDATE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip' },
        body: compressedPayload,
      })
        .then((response) => {
          if (response.status === 404) {
            navigate('/404', { replace: true });
            return null;
          }
          if (!response.ok) {
            return response.text().then((text) => {
              throw new Error(`Workspace failed: ${response.statusText} (${response.status}) ${text || ''}`);
            });
          }
          return response.json();
        })
        .then((data) => {
          if (!data) return;
          if (!data || data.view === undefined || data.state === undefined || data.total === undefined) {
            throw new Error('Invalid data structure received from server.');
          }
          const newResults = data.view;

          if (!append && !isInitialLoad) {
            const targetSide = flipState === 'none' ? 0 : 180;
            const currentRotations = cardRotationRef.current;
            const nextCardRotationState: { [index: number]: number } = {};
            newResults.forEach((_item: Result, index: number) => {
              const currentActualRotation = currentRotations[index] || 0;
              const isVisuallyFlipped = Math.round(currentActualRotation / 180) % 2 !== 0;
              const isTargetFlipped = targetSide === 180;
              nextCardRotationState[index] =
                isVisuallyFlipped !== isTargetFlipped ? currentActualRotation + 180 : currentActualRotation;
            });
            setCardRotation(nextCardRotationState);
            setShowSynopsis({});
          }

          setApiState(data.state);
          setTotalResults(data.total);
          setResults(append ? (prev) => [...prev, ...newResults] : newResults);
          setCardType(data.medium);
          setError('');

          if (isInitialLoad) setIsLoading(false);
          if (append) setLoadingMore(false);
          if (!isInitialLoad && !append) setIsLoading(false);

          if (isInitialLoad && data.followup_action) {
            const followupPayload = { state: data.state, action: data.followup_action, pagination: extendedPayload.pagination };
            fetch(UPDATE_URL, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip' },
              body: pako.gzip(JSON.stringify(followupPayload)),
            })
              .then((res) => {
                if (!res.ok) {
                  res.text().then((text) => console.warn(`Async Followup fetch failed (${res.status}): ${text}`));
                  return null;
                }
                return res.json();
              })
              .then((followupData) => {
                if (
                  followupData &&
                  followupData.view !== undefined &&
                  followupData.state !== undefined &&
                  followupData.total !== undefined
                ) {
                  setApiState(followupData.state);
                  setTotalResults(followupData.total);
                  setResults(followupData.view);
                  setCardType(followupData.medium);
                  resetCardStates();
                  gridViewRef.current?.scrollTo(0, 0);
                } else {
                  console.warn('Async Followup data invalid or empty. No state update.', followupData);
                }
              })
              .catch((followupError) => {
                console.error('ERROR during background followup fetch:', followupError);
              });
          }

          if (!append) {
            gridViewRef.current?.scrollTo(0, 0);
          }
        })
        .catch((err) => {
          console.error('Error in main fetchResults promise chain:', err);
          setError(`Failed to load results: ${err.message}. Please try again.`);
          if (isInitialLoad) setResults([]);
          setIsLoading(false);
          setLoadingMore(false);
        });
    },
    [isMobile, flipState, navigate]
  );

  // --- Other useEffects ---
  useEffect(() => {
    if (initialFetchStartedRef.current) return;

    let payload: Payload | null = null;

    if (isUserMode) {
        payload = { state: '', action: { type: 'add_user', source: source!, username: username! } };
    } else if (isItemMode) {
        payload = {
            state: '',
            action: {
                type: 'add_item',
                medium: itemType === 'anime' ? 'Anime' : 'Manga',
                source: source!,
                itemid: itemid!
            }
        };
    }

    if (payload) {
        const randomTextIndex = Math.floor(Math.random() * loadingMessages.length);
        setLoadingText(loadingMessages[randomTextIndex]);
        initialFetchStartedRef.current = true;
        fetchResults(payload, 0, false, true);
    } else {
        console.warn('Missing or invalid URL parameters. Redirecting home.');
        setError('Invalid URL. Please start a search from the homepage.');
        setIsLoading(false);
        const timer = setTimeout(() => navigate('/'), 3000);
        return () => clearTimeout(timer);
    }
}, [fetchResults, navigate, source, username, itemType, itemid, isUserMode, isItemMode]);

  useEffect(() => {
    const currentLoadMoreRef = loadMoreRef.current;
    if (!currentLoadMoreRef || isLoading || loadingMore || !apiState || results.length === 0) return;
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !loadingMore && results.length < totalResults) {
            const offset = results.length;
            const payload: MediaTypePayload = { state: apiState, action: { type: 'set_media', medium: cardType } };
            fetchResults(payload, offset, true, false);
          }
        });
      },
      { root: gridViewRef.current, rootMargin: '400px', threshold: 0.01 }
    );
    observer.observe(currentLoadMoreRef);
    return () => {
      if (currentLoadMoreRef) observer.unobserve(currentLoadMoreRef);
    };
  }, [results.length, totalResults, loadingMore, apiState, cardType, isLoading, fetchResults]);

  // --- Card Interaction Handlers ---
  const resetCardStates = () => {
    setCardRotation({});
    setShowSynopsis({});
    setFlipState('none');
  };

  const toggleMediaType = () => {
    // Guard against toggling while a fetch is in progress.
    if (isLoading || loadingMore || !apiState) return;

    // Determine the new type by flipping the current one.
    const newType = cardType === 'Anime' ? 'Manga' : 'Anime';

    // Set the new state and fetch the results for the new type.
    setCardType(newType);
    setShowSynopsis({}); // Reset synopsis view on toggle
    fetchResults({ state: apiState, action: { type: 'set_media', medium: newType } }, 0, false, false);
  };

  const handleFlipToggle = (e: React.MouseEvent<HTMLButtonElement>) => {
    let newState: 'none' | 'selected-details' | 'selected-synopsis';
    const newMapping: { [index: number]: number } = {};
    if (flipState === 'none') {
      newState = 'selected-details';
      // Force details view: flip any card facing front and set synopsis to false.
      for (let i = 0; i < results.length; i++) {
        const currentRotation = cardRotation[i] || 0;
        newMapping[i] = currentRotation % 360 === 0 ? currentRotation + 180 : currentRotation;
      }
      const detailsMapping: { [index: number]: boolean } = {};
      for (let i = 0; i < results.length; i++) {
        detailsMapping[i] = false;
      }
      setShowSynopsis(detailsMapping);
    } else if (flipState === 'selected-details') {
      newState = 'selected-synopsis';
      // Force synopsis view: flip any card facing front and set synopsis to true.
      for (let i = 0; i < results.length; i++) {
        const currentRotation = cardRotation[i] || 0;
        newMapping[i] = currentRotation % 360 === 0 ? currentRotation + 180 : currentRotation;
      }
      const synopsisMapping: { [index: number]: boolean } = {};
      for (let i = 0; i < results.length; i++) {
        synopsisMapping[i] = true;
      }
      setShowSynopsis(synopsisMapping);
    } else {
      newState = 'none';
      // When transitioning from selected-synopsis to none, continue the rotation.
      for (let i = 0; i < results.length; i++) {
        const currentRotation = cardRotation[i] || 0;
        newMapping[i] = currentRotation % 360 === 180 ? currentRotation + 180 : currentRotation;
      }
      // Delay clearing the synopsis until after the flip animation completes to avoid a jarring flash.
      setTimeout(() => {
        setShowSynopsis({});
      }, 300);
    }
    setFlipState(newState);
    setCardRotation(newMapping);
    e.currentTarget.blur();
  };

  // Desktop hover handlers (only when no global override)
  const handleMouseEnter = (index: number) => {
    if (!isMobile && flipState === 'none') {
      const r = cardRotation[index] || 0;
      if (r % 360 === 0) {
        setCardRotation((prev) => ({ ...prev, [index]: r + 180 }));
      }
    }
  };

  const handleMouseLeave = (index: number) => {
    if (!isMobile && flipState === 'none') {
      const r = cardRotation[index] || 0;
      if (r % 360 === 180) {
        setCardRotation((prev) => ({ ...prev, [index]: r - 180 }));
      }
    }
  };

  // Mobile tap handlers (always allowed)
  const handleCardFrontClick = (index: number) => {
    if (isMobile) {
      setCardRotation((prev) => ({ ...prev, [index]: (prev[index] || 0) + 180 }));
    }
  };

  const handleCardBackClick = (index: number) => {
    if (isMobile && showSynopsis[index]) {
      setCardRotation((prev) => ({ ...prev, [index]: (prev[index] || 0) + 180 }));
      setTimeout(() => {
        setShowSynopsis((prev) => ({ ...prev, [index]: false }));
      }, 300);
    } else {
      setShowSynopsis((prev) => ({ ...prev, [index]: !prev[index] }));
    }
  };

  // --- Render Logic ---
  if (isLoading && results.length === 0) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
	<p>{loadingText}</p>
      </div>
    );
  }

  if (error && results.length === 0) {
    return (
      <div className="error-banner" style={{ margin: '20px auto' }}>
        <span>{error}</span>
      </div>
    );
  }

  const isLoadingMediaType = isLoading && !loadingMore && results.length > 0;

  // Determine the text for the flip button based on the current state.
  let flipButtonText = "Show Titles";
  if (flipState === 'selected-details') {
    flipButtonText = "Show Synopsis";
  } else if (flipState === 'selected-synopsis') {
    flipButtonText = "Show Thumbnails";
  }

  // Generate profile URL based on the source
  let profileUrl = '';
  if (isUserMode && source === 'mal') {
    profileUrl = `https://myanimelist.net/profile/${username}`;
  } else if (isUserMode && source === 'anilist') {
    profileUrl = `https://anilist.co/user/${username}`;
  } else if (isUserMode && source === 'animeplanet') {
    profileUrl = `https://www.anime-planet.com/users/${username}`;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Header */}
      <header className="header--toggle">
        <div className="header-toggle">
          <div className="recommendations-title">
          <h2>
              <Link to="/" className="recsmoe-brand-link">Recs☆Moe</Link>'s picks
              {isUserMode && (
                <>
                  {' for '}
                  {profileUrl ? (
                    <a href={profileUrl} target="_blank" rel="noopener noreferrer" className="profile-link">
                      {username}
                    </a>
                  ) : (
                    <span className="profile-no-link">{username}</span>
                  )}
                </>
              )}
              {isItemMode && ' based on your selection'}
            </h2>
          </div>
          <div className="media-toggle" onClick={toggleMediaType}>
            <div
              className={`toggle-option ${cardType === 'Anime' ? 'active' : ''}`}
            >
              Anime
            </div>
            <div
              className={`toggle-option ${cardType === 'Manga' ? 'active' : ''}`}
            >
              Manga
            </div>
            <div className={`toggle-slider ${cardType}`}></div>
          </div>
          <button
            className={`flip-all-button ${flipState !== 'none' ? 'selected' : ''}`}
            onClick={handleFlipToggle}
            disabled={isLoading || loadingMore}
          >
            {flipButtonText}
          </button>
        </div>
      </header>
      {/* Error Banner */}
      {error && results.length > 0 && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError('')}>&times;</button>
        </div>
      )}
      {/* Results Grid */}
      {results.length > 0 ? (
        <div className="grid-container" ref={gridViewRef}>
          <div className="grid-view">
            {results.map((item, index) => {
              const currentRotation = cardRotation[index] ?? (flipState === 'none' ? 0 : 180);
              const localShowSynopsis =
                typeof showSynopsis[index] !== 'undefined' ? showSynopsis[index] : flipState === 'selected-synopsis';
              return (
                <div
                  key={`${apiState}-${item.url}-${index}`}
                  className="card"
                  onMouseEnter={() => handleMouseEnter(index)}
                  onMouseLeave={() => handleMouseLeave(index)}
                >
                  <div className="card-inner" style={{ transform: `rotateY(${currentRotation}deg)` }}>
                    <CardImage item={item} onClick={() => handleCardFrontClick(index)} />
                    <div className="card-back" onClick={() => handleCardBackClick(index)}>
                      <div
                        className="card-back-bg"
                        style={{
                          backgroundImage: `url(${getBiggestImageUrl(item.image) || getBiggestImageUrl(item.missing_image)})`,
                        }}
                      ></div>
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
                        <ManualScrollDiv className="card-back-body">
                          {isLoadingMediaType ? (
                            <div className="card-details-loading">Loading...</div>
                          ) : localShowSynopsis ? (
                            <p style={{ whiteSpace: 'pre-line' }}>{item.synopsis || 'No synopsis available.'}</p>
                          ) : (
                            <div className="card-details">
                              <table>
                                <tbody>
                                  {item.type && (
                                    <tr>
                                      <td>
                                        <strong>Medium:</strong>
                                      </td>
                                      <td>{item.type}</td>
                                    </tr>
                                  )}
                                  {cardType === 'Anime' ? (
                                    <>
                                      {item.season && (
                                        <tr>
                                          <td>
                                            <strong>Season:</strong>
                                          </td>
                                          <td>{item.season}</td>
                                        </tr>
                                      )}
                                      {item.episodes != null && (
                                        <tr>
                                          <td>
                                            <strong>Episodes:</strong>
                                          </td>
                                          <td>{item.episodes}</td>
                                        </tr>
                                      )}
                                      {item.duration && (
                                        <tr>
                                          <td>
                                            <strong>Duration:</strong>
                                          </td>
                                          <td>{item.duration}</td>
                                        </tr>
                                      )}
                                      {item.studios && (
                                        <tr>
                                          <td>
                                            <strong>Studio:</strong>
                                          </td>
                                          <td>{item.studios}</td>
                                        </tr>
                                      )}
                                    </>
                                  ) : (
                                    <>
                                      {item.volumes != null && (
                                        <tr>
                                          <td>
                                            <strong>Volumes:</strong>
                                          </td>
                                          <td>{item.volumes}</td>
                                        </tr>
                                      )}
                                      {item.chapters != null && (
                                        <tr>
                                          <td>
                                            <strong>Chapters:</strong>
                                          </td>
                                          <td>{item.chapters}</td>
                                        </tr>
                                      )}
                                      {item.studios && (
                                        <tr>
                                          <td>
                                            <strong>Author:</strong>
                                          </td>
                                          <td>{item.studios}</td>
                                        </tr>
                                      )}
                                    </>
                                  )}
                                  {item.status && (
                                    <tr>
                                      <td>
                                        <strong>Status:</strong>
                                      </td>
                                      <td>{item.status}</td>
                                    </tr>
                                  )}
                                  {item.startdate && (
                                    <tr>
                                      <td>
                                        <strong>Start&nbsp;Date:</strong>
                                      </td>
                                      <td>{item.startdate}</td>
                                    </tr>
                                  )}
                                  {item.enddate && (
                                    <tr>
                                      <td>
                                        <strong>End Date:</strong>
                                      </td>
                                      <td>{item.enddate}</td>
                                    </tr>
                                  )}
                                  {item.source && (
                                    <tr>
                                      <td>
                                        <strong>Source:</strong>
                                      </td>
                                      <td>{item.source}</td>
                                    </tr>
                                  )}
                                  {item.genres && (
                                    <tr>
                                      <td>
                                        <strong>Genres:</strong>
                                      </td>
                                      <td>{item.genres}</td>
                                    </tr>
                                  )}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </ManualScrollDiv>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
            <div ref={loadMoreRef} style={{ height: '10px', gridColumn: '1 / -1', visibility: 'hidden' }}></div>
            {loadingMore && (
              <div style={{ textAlign: 'center', gridColumn: '1 / -1', padding: '20px' }}>Loading More...</div>
            )}
            {!loadingMore && results.length > 0 && results.length >= totalResults && (
              <div style={{ textAlign: 'center', gridColumn: '1 / -1', padding: '20px', color: '#777' }}>
                End of results.
              </div>
            )}
          </div>
        </div>
      ) : (
        !isLoading &&
        !error && (
          <div style={{ textAlign: 'center', padding: '40px', flexGrow: 1 }}>No recommendations found.</div>
        )
      )}
    </div>
  );
};

export default ViewPage;