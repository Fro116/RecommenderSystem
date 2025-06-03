// src/ViewPage.tsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import pako from 'pako';
import { Result, CardType, MediaTypePayload, Payload, AddUserPayload, getBiggestImageUrl, UPDATE_URL } from './types';
import CardImage from './components/CardImage';
import ManualScrollDiv from './components/ManualScrollDiv';
import './App.css';

interface ViewPageProps {
  isMobile: boolean;
}

const ViewPage: React.FC<ViewPageProps> = ({ isMobile }) => {
  // Hooks and State
  const navigate = useNavigate();
  const { source, username } = useParams<{ source: string; username: string }>();
  const [results, setResults] = useState<Result[]>([]);
  const [apiState, setApiState] = useState<string>('');
  const [totalResults, setTotalResults] = useState<number>(0);
  const [cardType, setCardType] = useState<CardType>('Anime');
  const [isLoading, setIsLoading] = useState<boolean>(true);
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

  useEffect(() => {
    const siteDefaultTitle = document.title;
    document.title = `${username}'s Room | Recs☆Moe`;
    return () => {
      document.title = siteDefaultTitle;
    };
  }, [source, username]);

  // --- Functions ---
  const determineCardTypeFromResult = useCallback(
    (view: Result[] | undefined, action?: Payload['action']): CardType => {
      if (view && view.length > 0) {
        const firstItemType = view[0].type?.toUpperCase();
        if (['TV', 'MOVIE', 'OVA', 'SPECIAL', 'ONA', 'MUSIC'].includes(firstItemType || ''))
          return 'Anime';
        if (
          ['MANGA', 'NOVEL', 'LIGHT NOVEL', 'ONE-SHOT', 'DOUJINSHI', 'MANHWA', 'MANHUA', 'OEL'].includes(
            firstItemType || ''
          )
        )
          return 'Manga';
      }
      if (action?.type === 'set_media') return action.medium;
      return 'Anime';
    },
    []
  );

  const fetchResults = useCallback(
    (payload: Payload, offset: number = 0, append: boolean = false, isInitialLoad: boolean = false) => {
      console.log(`WorkspaceResults called (isInitial: ${isInitialLoad}, append: ${append})`, payload);

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
      console.log('Sending fetch payload:', extendedPayload);

      fetch(UPDATE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip' },
        body: compressedPayload,
      })
        .then((response) => {
          console.log(`Workspace status: ${response.status}`);
          if (!response.ok) {
            return response.text().then((text) => {
              throw new Error(`Workspace failed: ${response.statusText} (${response.status}) ${text || ''}`);
            });
          }
          return response.json();
        })
        .then((data) => {
          console.log('Received data:', data);
          if (!data || data.view === undefined || data.state === undefined || data.total === undefined) {
            throw new Error('Invalid data structure received from server.');
          }
          console.log('Setting state based on received data.');
          const resolvedMedium = (data.medium as CardType) || determineCardTypeFromResult(data.view, payload.action);
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
          setCardType(resolvedMedium);
          setError('');

          if (isInitialLoad) setIsLoading(false);
          if (append) setLoadingMore(false);
          if (!isInitialLoad && !append) setIsLoading(false);

          if (isInitialLoad && data.followup_action) {
            console.log('Followup action detected. Triggering in background:', data.followup_action);
            const followupPayload = { state: data.state, action: data.followup_action, pagination: extendedPayload.pagination };
            console.log('Sending followup payload (async):', followupPayload);
            fetch(UPDATE_URL, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip' },
              body: pako.gzip(JSON.stringify(followupPayload)),
            })
              .then((res) => {
                console.log(`Async Followup fetch status: ${res.status}`);
                if (!res.ok) {
                  res.text().then((text) => console.warn(`Async Followup fetch failed (${res.status}): ${text}`));
                  return null;
                }
                return res.json();
              })
              .then((followupData) => {
                console.log('Received async followup data:', followupData);
                if (
                  followupData &&
                  followupData.view !== undefined &&
                  followupData.state !== undefined &&
                  followupData.total !== undefined
                ) {
                  console.log('Async Followup data is valid. Updating state in background.');
                  const followupResolvedMedium =
                    (followupData.medium as CardType) || determineCardTypeFromResult(followupData.view, followupPayload.action);
                  setApiState(followupData.state);
                  setTotalResults(followupData.total);
                  setResults(followupData.view);
                  setCardType(followupResolvedMedium);
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
    [isMobile, determineCardTypeFromResult, flipState]
  );

  // --- Other useEffects ---
  useEffect(() => {
    if (initialFetchStartedRef.current || !source || !username) {
      if (!source || !username) {
        console.warn('Missing source or username in URL. Redirecting home.');
        setError('Invalid URL. Please start a search from the homepage.');
        setIsLoading(false);
        const timer = setTimeout(() => navigate('/'), 3000);
        return () => clearTimeout(timer);
      }
      return;
    }
    console.log(`ViewPage initial effect running. Source: ${source}, Username: ${username}`);
    const initialPayload: AddUserPayload = { state: '', action: { type: 'add_user', source: source, username: username } };
    initialFetchStartedRef.current = true;
    fetchResults(initialPayload, 0, false, true);
  }, [fetchResults, navigate, source, username]);

  useEffect(() => {
    const currentLoadMoreRef = loadMoreRef.current;
    if (!currentLoadMoreRef || isLoading || loadingMore || !apiState || results.length === 0) return;
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !loadingMore && results.length < totalResults) {
            console.log('Load more triggered');
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

  const handleMediaTypeChange = (type: CardType) => {
    if (type === cardType || isLoading || loadingMore || !apiState) return;
    console.log(`Changing media type to: ${type}`);
    setCardType(type);
    setShowSynopsis({});
    const payload: MediaTypePayload = { state: apiState, action: { type: 'set_media', medium: type } };
    fetchResults(payload, 0, false, false);
  };

  // Updated handleFlipToggle with delayed clearing of synopsis for a smooth transition:
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
        <p>Loading Recommendations...</p>
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

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Header */}
      <header className="header--toggle">
        <div className="header-toggle">
          <button
            className={`header-toggle-button ${cardType === 'Anime' ? 'selected' : ''}`}
            onClick={() => handleMediaTypeChange('Anime')}
            disabled={isLoading || loadingMore}
          >
            Anime
          </button>
          <button
            className={`header-toggle-button ${cardType === 'Manga' ? 'selected' : ''}`}
            onClick={() => handleMediaTypeChange('Manga')}
            disabled={isLoading || loadingMore}
          >
            Manga
          </button>
          <button
            className={`flip-all-button ${flipState !== 'none' ? 'selected' : ''}`}
            onClick={handleFlipToggle}
            disabled={isLoading || loadingMore}
          >
            <img
              src={
                flipState === 'selected-details'
                  ? '/flip-icon-details.webp'
                  : flipState === 'selected-synopsis'
                  ? '/flip-icon-synopsis.webp'
                  : '/flip-icon.webp'
              }
              alt="Flip All Cards"
            />
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
