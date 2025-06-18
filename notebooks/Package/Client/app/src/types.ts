// src/types.ts

export interface Result {
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

export type SourceType = 'MyAnimeList' | 'AniList' | 'Kitsu' | 'Anime-Planet';
export type CardType = 'Anime' | 'Manga';

export interface AddUserPayload {
  state: string;
  action: {
    type: 'add_user';
    source: string;
    username: string;
  };
}

export interface AddItemPayload {
  state: string;
  action: {
    type: 'add_item';
    medium: CardType;
    source: string;
    itemid: string;
  };
}

export interface MediaTypePayload {
  state: string;
  action: {
    type: 'set_media';
    medium: CardType;
  };
}

export type Payload = AddUserPayload | MediaTypePayload | AddItemPayload;

export interface AutocompleteItem {
  // User properties
  username?: string;
  avatar?: string | null;
  missing_avatar?: string | null;
  last_online?: string | null;
  gender?: string | null;
  age?: number | null;
  joined?: string | null;

  // Item properties
  title?: string;
  matched_title?: string;
  mediatype?: string;
  startdate?: string | null;
  enddate?: string | null;
  image?: any;
  episodes?: number | null;
  chapters?: number | null;
  source?: string;
  itemid?: string;

  // Common properties
  matched: boolean[];
}


// Helper function (can also reside here or in a separate utils file)
export const getBiggestImageUrl = (images: any): string => {
  if (Array.isArray(images) && images.length > 0) {
    return images.reduce((prev: any, curr: any) => {
      return (prev.width * prev.height) >= (curr.width * curr.height) ? prev : curr;
    }).url;
  }
  return images || '';
};

export const API_BASE = 'https://api.recs.moe';
export const UPDATE_URL = `${API_BASE}/update`;

export const SOURCE_MAP: Record<SourceType, string> = {
  MyAnimeList: 'mal',
  AniList: 'anilist',
  Kitsu: 'kitsu',
  'Anime-Planet': 'animeplanet',
};