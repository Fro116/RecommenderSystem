// src/HomePage.tsx
import React, { useState, useEffect, useRef, FormEvent, ChangeEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { SourceType, AutocompleteItem, API_BASE, SOURCE_MAP } from './types';
import './App.css';

const HomePage: React.FC = () => {
  const [query, setQuery] = useState<string>(''); // State for the input field
  const [activeSource, setActiveSource] = useState<SourceType>('MyAnimeList');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [autocompleteResults, setAutocompleteResults] = useState<AutocompleteItem[]>([]);
  const [autocompleteDisabled, setAutocompleteDisabled] = useState<boolean>(false);
  const [showButtons, setShowButtons] = useState<boolean>(true);

  const navigate = useNavigate();

  const inputRef = useRef<HTMLInputElement>(null);
  const searchContainerRef = useRef<HTMLDivElement>(null);
  const buttonContainerRef = useRef<HTMLDivElement>(null);

  const placeholders: Record<SourceType, string> = {
    MyAnimeList: 'Type a MyAnimeList username',
    AniList: 'Type an AniList username',
    Kitsu: 'Type a Kitsu username',
    'Anime-Planet': 'Type an Anime-Planet username',
  };

  // ... (fetchAutocomplete, useEffects, handleInputChange remain the same) ...
  const fetchAutocomplete = async () => {
     if (query.trim() === '') {
         setAutocompleteResults([]); return;
     }
     try {
       const response = await fetch(`${API_BASE}/autocomplete`, {
         method: "POST", headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ type: "user", source: SOURCE_MAP[activeSource], prefix: query }),
       });
       if (response.ok) {
         const data = await response.json(); setAutocompleteResults(data?.autocompletes || []);
       } else { setAutocompleteResults([]); }
     } catch (e) { console.error("Autocomplete fetch error:", e); setAutocompleteResults([]); }
  };

  useEffect(() => {
     if (autocompleteDisabled) return;
     if (query.trim() === '') { setAutocompleteResults([]); return; }
     const handler = setTimeout(() => { fetchAutocomplete(); }, 50);
     return () => clearTimeout(handler);
  }, [query, activeSource, autocompleteDisabled]);

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value); setAutocompleteDisabled(false);
  };


  // Renamed back from handleSearch in previous steps, this triggers the navigation
  const triggerSearch = (username: string) => {
      const trimmedUsername = username.trim();
      if (!trimmedUsername) return;
      setErrorMessage('');
      const sourceKey = SOURCE_MAP[activeSource];
      console.log(`Navigating to /user/${sourceKey}/${trimmedUsername}`);
      navigate(`/user/${sourceKey}/${trimmedUsername}`);
      // Note: Clearing query state happens in handleSearch now
  }

  // This handles the form submission (e.g., pressing Enter)
  const handleSearch = (e: FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      if (!query.trim()) return; // Don't search if query is empty
      triggerSearch(query); // Navigate using the current query
      setQuery(''); // *** ADDED: Clear the query state after triggering search ***
      if (inputRef.current) inputRef.current.blur(); // Optionally blur input
  };

  // This handles clicking an autocomplete item
  const handleAutocompleteClick = (item: AutocompleteItem) => {
      setAutocompleteDisabled(true); // Prevent fetch firing again
      setQuery(item.username); // Set input briefly to clicked item (will be cleared below)
      setAutocompleteResults([]); // Hide autocomplete
      triggerSearch(item.username); // Navigate using the clicked item's username
      setQuery(''); // *** ADDED: Clear the query state after triggering search ***
      if (inputRef.current) inputRef.current.blur(); // Optionally blur input
  };

  const handleButtonClick = (source: SourceType) => {
      // ... (remains the same)
      setActiveSource(source);
      setAutocompleteResults([]);
      inputRef.current?.focus();
  };

   useEffect(() => {
     // ... (click outside useEffect remains the same)
     const handleClickOutside = (event: MouseEvent) => { if (searchContainerRef.current && !searchContainerRef.current.contains(event.target as Node)) { setAutocompleteResults([]); } };
     document.addEventListener('mousedown', handleClickOutside);
     return () => document.removeEventListener('mousedown', handleClickOutside);
   }, []);


  // --- Render structure ---
  return (
    // ... (rest of the JSX remains the same) ...
    <>
       <header>
         <form onSubmit={handleSearch}>
             <div ref={searchContainerRef} style={{ position: 'relative', width: '80%', margin: '0 auto' }}>
                 <input
                   ref={inputRef} type="text" placeholder={placeholders[activeSource]} value={query}
                   onChange={handleInputChange}
                   onFocus={() => { setShowButtons(true); setErrorMessage(''); if (query.trim()) fetchAutocomplete(); }}
                   autoComplete="off"
                 />
                 {autocompleteResults.length > 0 && (
                   <div className="autocomplete-container">
                       {autocompleteResults.map((item, index) => (
                         <div key={index} className="autocomplete-item" onMouseDown={(e) => e.preventDefault()} onClick={() => handleAutocompleteClick(item)}>
                           {(item.avatar || item.missing_avatar) && ( <img className="autocomplete-avatar" src={item.avatar || item.missing_avatar || ''} alt={item.username} onError={(e) => { if (e.currentTarget.src !== (item.missing_avatar || '')) e.currentTarget.src = item.missing_avatar || ''; }} /> )}
                           <span> {item.username.split('').map((char, idx) => ( <span key={idx} className={item.matched[idx] ? "autocomplete-match" : "autocomplete-unmatch"}>{char}</span> ))} </span>
                         </div>
                       ))}
                   </div>
                 )}
             </div>
         </form>
       </header>

       {showButtons && (
           <div className="source-buttons" ref={buttonContainerRef} style={{ marginTop: '20px' }}>
               {(['MyAnimeList', 'AniList', 'Kitsu', 'Anime-Planet'] as SourceType[]).map(source => (
                 <button type="button" key={source} className={`source-button ${activeSource === source ? 'selected' : ''}`} onClick={() => handleButtonClick(source)}> {source} </button>
               ))}
           </div>
       )}

       {errorMessage && (
         <div className="error-banner" style={{ marginTop: '15px', width: '80%', maxWidth: '600px', marginLeft: 'auto', marginRight: 'auto' }}>
           <span>{errorMessage}</span>
           <button onClick={() => setErrorMessage('')}>&times;</button>
         </div>
       )}

        <div className="home-overlay">
            Recs☆Moe is a recommender system for anime and manga
        </div>
    </>
  );
};

export default HomePage;