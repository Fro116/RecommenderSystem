import React, { useState, useEffect, useRef, FormEvent, ChangeEvent } from 'react';
import './App.css';

interface Result {
  title: string;
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

const LIMIT = 50;

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

  // IntersectionObserver for infinite scrolling.
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

  const fetchResults = (payload: Payload, offset: number = 0, append: boolean = false) => {
    const extendedPayload = { ...payload, pagination: { offset, limit: LIMIT } };
    return fetch('https://api.recs.moe/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(extendedPayload),
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
    fetchResults(payload, 0, false);
  };

  return (
    <div className="container">
      <header ref={searchRef}>
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
            }}
          />
        </form>
      </header>

      {errorMessage && (
        <div className="error-banner">
          <span>{errorMessage}</span>
          <button onClick={() => setErrorMessage('')}>&times;</button>
        </div>
      )}

      {showButtons && (
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
          <div className="grid-view" ref={gridViewRef}>
            {results.map((item, index) => (
              <div
                key={index}
                className="card"
                onClick={() => window.open(item.url, '_blank', 'noopener,noreferrer')}
                title={`Open ${item.title}`}
              >
                <div className="card-title">
                  <h4>{item.title}</h4>
                </div>
                <div className="card-contents">
                  <div className="card-details">
                    {cardType === 'Anime' ? (
                      <>
                        {item.type && <p><strong>Medium:</strong> {item.type}</p>}
                        {item.season && <p><strong>Season:</strong> {item.season}</p>}
                        {item.episodes != null && <p><strong>Episodes:</strong> {item.episodes}</p>}
                        {item.status && <p><strong>Status:</strong> {item.status}</p>}
                        {item.startdate && <p><strong>Start Date:</strong> {item.startdate}</p>}
                        {item.enddate && <p><strong>End Date:</strong> {item.enddate}</p>}
                        {item.source && <p><strong>Source Material:</strong> {item.source}</p>}
                        {item.duration && <p><strong>Duration:</strong> {item.duration}</p>}
                        {item.studios && <p><strong>Studio:</strong> {item.studios}</p>}
                      </>
                    ) : (
                      <>
                        {item.type && <p><strong>Medium:</strong> {item.type}</p>}
                        {item.status && <p><strong>Status:</strong> {item.status}</p>}
                        {item.startdate && <p><strong>Start Date:</strong> {item.startdate}</p>}
                        {item.enddate && <p><strong>End Date:</strong> {item.enddate}</p>}
                        {item.volumes != null && <p><strong>Volumes:</strong> {item.volumes}</p>}
                        {item.chapters != null && <p><strong>Chapters:</strong> {item.chapters}</p>}
                        {item.source && <p><strong>Source Material:</strong> {item.source}</p>}
                        {item.studios && <p><strong>Studio:</strong> {item.studios}</p>}
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {results.length < totalResults && (
              <div ref={loadMoreRef} style={{ height: '20px' }}></div>
            )}
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
