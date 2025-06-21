// src/HomePage.tsx
import './Header.css';
import './HomePage.css';
import React, { useState, useEffect, useRef, FormEvent, ChangeEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { SourceType, AutocompleteItem, API_BASE, SOURCE_MAP } from './types';

type QueryMode = 'user' | 'item';
type ItemType = 'anime' | 'manga';

const HomePage: React.FC = () => {
    const [query, setQuery] = useState<string>('');

    const getInitialQueryMode = (): QueryMode => {
        const stored = localStorage.getItem('queryMode');
        if (stored === 'user' || stored === 'item') return stored as QueryMode;
        return 'user';
    };
    const [queryMode, setQueryMode] = useState<QueryMode>(getInitialQueryMode());
    useEffect(() => { localStorage.setItem('queryMode', queryMode); }, [queryMode]);

    const getInitialItemType = (): ItemType => {
        const stored = localStorage.getItem('itemType');
        if (stored === 'anime' || stored === 'manga') return stored as ItemType;
        return 'anime';
    };
    const [itemType, setItemType] = useState<ItemType>(getInitialItemType());
    useEffect(() => { localStorage.setItem('itemType', itemType); }, [itemType]);

    const getInitialSource = (): SourceType => {
        const stored = localStorage.getItem('selectedSource');
        if (stored && ['MyAnimeList', 'AniList', 'Kitsu', 'Anime-Planet'].includes(stored)) {
            return stored as SourceType;
        }
        return 'MyAnimeList';
    };
    const [activeSource, setActiveSource] = useState<SourceType>(getInitialSource());
    useEffect(() => { localStorage.setItem('selectedSource', activeSource); }, [activeSource]);

    const [errorMessage, setErrorMessage] = useState<string>('');
    const [autocompleteResults, setAutocompleteResults] = useState<AutocompleteItem[]>([]);
    const [autocompleteDisabled, setAutocompleteDisabled] = useState<boolean>(false);
    const [showButtons, setShowButtons] = useState<boolean>(true);
    const [showModeDropdown, setShowModeDropdown] = useState<boolean>(false);
    const [fetchTrigger, setFetchTrigger] = useState(0);

    const [windowWidth, setWindowWidth] = useState(window.innerWidth);

    const navigate = useNavigate();
    const inputRef = useRef<HTMLInputElement>(null);
    const searchContainerRef = useRef<HTMLDivElement>(null);
    const buttonContainerRef = useRef<HTMLDivElement>(null);
    const modeDropdownRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleResize = () => {
            setWindowWidth(window.innerWidth);
        };
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (showModeDropdown && modeDropdownRef.current && !modeDropdownRef.current.contains(e.target as Node)) {
                setShowModeDropdown(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [showModeDropdown]);

    useEffect(() => {
        setQuery('');
        setAutocompleteResults([]);
    }, [queryMode]);

    const userPlaceholders: Record<SourceType, string> = {
        MyAnimeList: 'Type a MyAnimeList username',
        AniList: 'Type an AniList username',
        Kitsu: 'Type a Kitsu username',
        'Anime-Planet': 'Type an Anime-Planet username',
    };

    const itemPlaceholders: Record<ItemType, string> = {
        anime: 'Type an anime title',
        manga: 'Type a manga title',
    };

    const inputPlaceholder = queryMode === 'user'
        ? userPlaceholders[activeSource]
        : itemPlaceholders[itemType];

    useEffect(() => {
        if (autocompleteDisabled) return;
        if (!query.trim()) {
            setAutocompleteResults([]);
            return;
        }

        const controller = new AbortController();

        const fetchAutocomplete = async () => {
            try {
                const body = JSON.stringify({
                    type: queryMode,
                    medium: queryMode === 'item' ? (itemType === 'anime' ? 'Anime' : 'Manga') : undefined,
                    source: queryMode === 'user' ? SOURCE_MAP[activeSource] : undefined,
                    prefix: query
                });
                const response = await fetch(`${API_BASE}/autocomplete`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body,
                    signal: controller.signal
                });
                if (response.ok) {
                    const data = await response.json();
                    setAutocompleteResults(data?.autocompletes || []);
                } else {
                    setAutocompleteResults([]);
                }
            } catch (error) {
                if ((error as Error).name !== 'AbortError') {
                    setAutocompleteResults([]);
                }
            }
        };

        const timer = setTimeout(fetchAutocomplete, 50);

        return () => {
            clearTimeout(timer);
            controller.abort();
        };
    }, [query, activeSource, itemType, queryMode, autocompleteDisabled, fetchTrigger]);

    const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
        setQuery(e.target.value);
        setAutocompleteDisabled(false);
    };

    const triggerSearch = (value: string) => {
        const trimmed = value.trim(); if (!trimmed) return;
        setErrorMessage('');
        // This function is now only used for user searches.
        const key = SOURCE_MAP[activeSource];
        navigate(`/user/${key}/${trimmed}`);
    };

    const handleSearch = (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
    
        if (queryMode === 'item') {
            // If in item mode and there's at least one suggestion,
            // treat 'Enter' as a click on the first suggestion.
            if (autocompleteResults.length > 0) {
                handleAutocompleteClick(autocompleteResults[0]);
            }
            // If there are no suggestions, do nothing.
            return;
        }
    
        // The rest of the function handles 'user' mode searches.
        if (!query.trim()) return;
        triggerSearch(query);
        setQuery('');
        inputRef.current?.blur();
    };

    const handleAutocompleteClick = (item: AutocompleteItem) => {
        setAutocompleteDisabled(true);
        setAutocompleteResults([]);
        inputRef.current?.blur();
        setQuery('');
    
        if (queryMode === 'user') {
            const val = item.username!;
            triggerSearch(val);
        } else {
            // For item search, navigate to the view page and let it handle the fetch.
            const { source, itemid } = item;
            if (source && itemid) {
                navigate(`/item/${itemType}/${source}/${itemid}`);
            } else {
                console.error("Autocomplete item is missing 'source' or 'itemid'", item);
                setErrorMessage("Selected item is invalid and cannot be used.");
            }
        }
    };

    const handleButtonClick = (value: string) => {
        setAutocompleteResults([]);
        inputRef.current?.focus();
        if (queryMode === 'user') setActiveSource(value as SourceType);
        else setItemType(value as ItemType);
    };

    useEffect(() => {
        const handleOutside = (evt: MouseEvent) => {
            if (searchContainerRef.current && !searchContainerRef.current.contains(evt.target as Node)) {
                setAutocompleteResults([]);
            }
        };
        document.addEventListener('mousedown', handleOutside);
        return () => document.removeEventListener('mousedown', handleOutside);
    }, []);

    const formatDateRange = (start?: string | null, end?: string | null) => {
        const extractYear = (date?: string | null) => date ? date.slice(0, 4) : null;
        const startYear = extractYear(start);
        const endYear = extractYear(end);
        if (startYear && endYear && startYear !== endYear) return `${startYear}–${endYear}`;
        if (startYear) return startYear;
        return 'Unknown';
    };

    let maxSuggestions;
    if (queryMode === 'user') {
        maxSuggestions = 10;
    } else {
        maxSuggestions = windowWidth <= 600 ? 10 : 7;
    }
    const suggestionsToShow = autocompleteResults.slice(0, maxSuggestions);

    return (
        <>
            <header>
                <form onSubmit={handleSearch}>
                    <div ref={searchContainerRef} className="search-container">
                        <input
                            ref={inputRef} type="text" placeholder={inputPlaceholder}
                            value={query} onChange={handleInputChange}
                            onFocus={() => { setShowButtons(true); setErrorMessage(''); if (query.trim()) { setFetchTrigger(c => c + 1); } }}
                            autoComplete="off"
                        />
                        {query.trim() && autocompleteResults.length > 0 && (
                            <div className="autocomplete-container">
                                {suggestionsToShow.map((item, idx) => {
                                    const itemMetadata = queryMode === 'item' ? [
                                        item.mediatype || 'Unknown',
                                        formatDateRange(item.startdate, item.enddate),
                                        item.episodes ? `${item.episodes} ep${item.episodes > 1 ? 's' : ''}` : null,
                                        item.chapters ? `${item.chapters} ch` : null
                                    ].filter(p => p && p !== 'Unknown').join(' | ') : '';
                                    let userMetadata = '';
                                    if (queryMode === 'user') {
                                        const parts = [];
                                        let datePart = '';
                                        if (item.joined) {
                                            datePart = `${item.joined}`;
                                            if (item.last_online) {
                                                datePart += ` - ${item.last_online}`;
                                            }
                                        } else if (item.last_online) {
                                            datePart = `Last online ${item.last_online}`;
                                        }
                                        if (datePart) {
                                            parts.push(datePart);
                                        }
                                        const genderAge = [item.gender, item.age].filter(Boolean).join(', ');
                                        if (genderAge) {
                                            parts.push(genderAge);
                                        }
                                        userMetadata = parts.join(' | ');
                                    }

                                    const imageInfo = item.image?.[0];
                                    let imageStyle: React.CSSProperties = {};
                                    if (imageInfo && imageInfo.width > 0) {
                                        const aspectRatio = imageInfo.height / imageInfo.width;
                                        if (aspectRatio >= 1.3 && aspectRatio <= 1.6) {
                                            imageStyle = { height: '72px', width: 'auto' };
                                        } else {
                                            imageStyle = { height: '72px', width: '50px', objectFit: 'cover' };
                                        }
                                    } else {
                                        // Default style for items with no image info
                                        imageStyle = { height: '72px', width: '50px' };
                                    }

                                    return (
                                        <div key={idx} className="autocomplete-item" onMouseDown={e => e.preventDefault()} onClick={() => handleAutocompleteClick(item)}>
                                            {queryMode === 'user'
                                                ? (item.avatar || item.missing_avatar) && (
                                                    <img className="autocomplete-avatar-user" src={item.avatar || item.missing_avatar!} alt={item.username || ''} onError={e => { if (e.currentTarget.src !== (item.missing_avatar || '')) e.currentTarget.src = item.missing_avatar || ''; }} />
                                                )
                                                : imageInfo?.url && (
                                                    <img className="autocomplete-avatar-item" src={imageInfo.url} alt={item.title || ''} style={imageStyle} onError={e => { e.currentTarget.style.display = 'none'; }}/>
                                                )
                                            }
                                            {queryMode === 'user' ? (
                                                <div className="autocomplete-item-text-wrapper">
                                                    <div className="autocomplete-title">
                                                        {item.username?.split('').map((c, i) => (
                                                            <span key={i} className={item.matched[i] ? 'autocomplete-match' : 'autocomplete-unmatch'}>
                                                                {c}
                                                            </span>
                                                        ))}
                                                    </div>
                                                    <div className="autocomplete-metadata">{userMetadata}</div>
                                                </div>
                                            ) : (
                                                <div className="autocomplete-item-text-wrapper">
                                                    <div className="autocomplete-title">{item.title}</div>
                                                    <div className="autocomplete-metadata">{itemMetadata}</div>
                                                    <div className="autocomplete-matched" data-metadata={itemMetadata}>
                                                        {item.matched_title?.split('').map((c, i) => (
                                                            <span key={i} className={item.matched[i] ? 'autocomplete-match' : 'autocomplete-unmatch'}>
                                                                {c}
                                                            </span>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                </form>

                <div className="mode-dropdown-container">
                    <div ref={modeDropdownRef} style={{ position: 'relative', width: '100%' }}>
                        <button type="button" className="query-mode-button" onClick={() => setShowModeDropdown(prev => !prev)}>
                            <span className="query-mode-button-title">
                                {queryMode === 'user' ? 'Search by User' : <React.Fragment>Search by Title <span className="beta-tag">Beta</span></React.Fragment>}
                            </span>
                            <span className="query-mode-button-description">
                                {queryMode === 'user' ? "Get recommendations based on a user's profile." : 'Discover shows similar to a specific anime or manga.'}
                            </span>
                        </button>
                                {showModeDropdown && (
                                <ul className="query-mode-list">
                                    <li className={`query-mode-item ${queryMode === 'user' ? 'active' : ''}`} onClick={() => { setQueryMode('user'); setShowModeDropdown(false); }}>
                                        <div className="query-mode-title">Search by User</div>
                                        <div className="query-mode-description">Get recommendations based on a user's profile.</div>
                                    </li>
                                    <li className={`query-mode-item ${queryMode === 'item' ? 'active' : ''}`} onClick={() => { setQueryMode('item'); setShowModeDropdown(false); }}>
                                        <div className="query-mode-title">Search by Title <span className="beta-tag">Beta</span></div>
                                        <div className="query-mode-description">Discover shows similar to a specific anime or manga.</div>
                                    </li>
                                </ul>
                                )}
                    </div>
                </div>
            </header>

            {showButtons && (
                <div className="source-buttons" ref={buttonContainerRef} style={{ marginTop: '20px' }}>
                    {queryMode === 'user'
                        ? (['MyAnimeList', 'AniList', 'Kitsu', 'Anime-Planet'] as SourceType[]).map(src => (
                            <button key={src} type="button" className={`source-button ${activeSource === src ? 'selected' : ''}`} onClick={() => handleButtonClick(src)}>
                                {src}
                            </button>
                        ))
                        : (['anime', 'manga'] as ItemType[]).map(type => (
                            <button key={type} type="button" className={`source-button ${itemType === type ? 'selected' : ''}`} onClick={() => handleButtonClick(type)}>
                                {type.charAt(0).toUpperCase() + type.slice(1)}
                            </button>
                        ))
                    }
                </div>
            )}

            {errorMessage && (
                <div className="error-banner" style={{ margin: '40px auto 0', width: '80%', maxWidth: '600px' }}>
                    <span>{errorMessage}</span>
                    <button onClick={() => setErrorMessage('')}>&times;</button>
                </div>
            )}

            <h1 className="home-overlay">Recs☆Moe is a recommender system for anime and manga</h1>
        </>
    );
};

export default HomePage;