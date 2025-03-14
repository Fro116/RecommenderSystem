import React, { useState, useEffect, useRef, FormEvent, ChangeEvent } from 'react';
import './App.css';

interface Result {
  title: string;
  source: string;
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

  // IntersectionObserver for infinite scrolling with a threshold of 0.1.
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

  // Build payload for paginated requests.
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

  // Updated fetchResults now resets scroll and, for search actions, resets the media button.
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
          // Reset scroll position when a new view is loaded.
          if (gridViewRef.current) {
            gridViewRef.current.scrollTop = 0;
          }
          // If the action is an "add_user" (i.e. a new search), then reset media type button.
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
    if (!query.trim()) {
      return;
    }
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

    // Trigger new search without immediately resetting the media type.
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
    // Media type change: fetch new results.
    fetchResults(payload, 0, false);
  };

  const sources: SourceType[] = ['MyAnimeList', 'AniList', 'Kitsu', 'Anime-Planet'];
  const placeholders: Record<SourceType, string> = {
    MyAnimeList: 'type a MyAnimeList username',
    AniList: 'type an AniList username',
    Kitsu: 'type a Kitsu username',
    'Anime-Planet': 'type an Anime-Planet username',
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

      {/* Styled error banner */}
      {errorMessage && (
        <div
          style={{
            backgroundColor: '#fdecea',
            color: '#611a15',
            padding: '10px 20px',
            borderRadius: '5px',
            margin: '10px auto',
            maxWidth: '600px',
            border: '1px solid #f5c6cb',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <span>{errorMessage}</span>
          <button
            onClick={() => setErrorMessage('')}
            style={{
              background: 'transparent',
              border: 'none',
              fontSize: '20px',
              cursor: 'pointer',
              color: '#611a15',
            }}
          >
            &times;
          </button>
        </div>
      )}

      {showButtons && (
        <div className="source-buttons" ref={buttonContainerRef}>
          {sources.map((source) => (
            <button
              key={source}
              onClick={() => handleButtonClick(source)}
              style={{
                backgroundColor: activeSource === source ? '#007BFF' : '#e0e0e0',
                color: activeSource === source ? '#fff' : '#000',
              }}
            >
              {source}
            </button>
          ))}
        </div>
      )}

      {results.length > 0 && (
        <>
          <div className="card-toggle">
            <button
              onClick={() => handleMediaTypeChange('Anime')}
              style={{
                backgroundColor: cardType === 'Anime' ? '#007BFF' : '#fff',
                color: cardType === 'Anime' ? '#fff' : '#000',
              }}
            >
              Anime
            </button>
            <button
              onClick={() => handleMediaTypeChange('Manga')}
              style={{
                backgroundColor: cardType === 'Manga' ? '#007BFF' : '#fff',
                color: cardType === 'Manga' ? '#fff' : '#000',
              }}
            >
              Manga
            </button>
          </div>
          <div className="grid-view" ref={gridViewRef}>
            {results.map((item, index) => (
              <div key={index} className="card">
                <h4>{item.title}</h4>
                <p>Source: {item.source}</p>
              </div>
            ))}
            {results.length < totalResults && (
              <div ref={loadMoreRef} style={{ height: '20px' }}></div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default App;
