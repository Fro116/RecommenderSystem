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

export type SourceType = "MyAnimeList" | "AniList" | "Kitsu" | "Anime-Planet";
export type CardType = "Anime" | "Manga";

export interface AddUserPayload {
  state: string;
  action: {
    source: string;
    username: string;
  };
}

export interface AddItemPayload {
  state: string;
  action: {
    medium: CardType;
    source: string;
    itemid: string;
  };
}

export interface MediaTypePayload {
  state: string;
  action: {
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
      return prev.width * prev.height >= curr.width * curr.height ? prev : curr;
    }).url;
  }
  return images || "";
};

export const API_BASE = import.meta.env.VITE_API_BASE || 'https://api2.recs.moe';

export const SOURCE_MAP: Record<SourceType, string> = {
  MyAnimeList: "mal",
  AniList: "anilist",
  Kitsu: "kitsu",
  "Anime-Planet": "animeplanet",
};

interface Hsl {
  h: number;
  s: number;
  l: number;
}

interface Rgb {
  r: number;
  g: number;
  b: number;
}

const scale = (
  x: number,
  inLow: number,
  inHigh: number,
  outLow: number,
  outHigh: number,
): number => ((x - inLow) * (outHigh - outLow)) / (inHigh - inLow) + outLow;

const hslToRgb = ({ h, s, l }: Hsl): Rgb => {
  if (s === 0) {
    const v = Math.round(l * 255);
    return { r: v, g: v, b: v };
  }
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  const [r, g, b] = (
    h < 60
      ? [c, x, 0]
      : h < 120
        ? [x, c, 0]
        : h < 180
          ? [0, c, x]
          : h < 240
            ? [0, x, c]
            : h < 300
              ? [x, 0, c]
              : [c, 0, x]
  ).map((n) => Math.round((n + m) * 255));
  return { r, g, b };
};

const rgbToHsl = ({ r, g, b }: Rgb): Hsl => {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const c = max - min;
  const l = (max + min) / 2;
  if (c === 0) {
    return { h: 0, s: 0, l };
  }
  let h =
    (max === rn
      ? ((gn - bn) / c) % 6
      : max === gn
        ? (bn - rn) / c + 2
        : (rn - gn) / c + 4) * 60;
  if (h < 0) {
    h += 360;
  }
  return { h, s: c / (1 - Math.abs(2 * l - 1)), l };
};

/* #181a1b, Dark Reader's default dark scheme background */
const DARK_SCHEME_BACKGROUND = rgbToHsl({ r: 0x18, g: 0x1a, b: 0x1b });
const MAX_BG_LIGHTNESS = 0.4;

const darkenBackground = ({ h, s, l }: Hsl): Hsl => {
  const pole = DARK_SCHEME_BACKGROUND;
  const isBlue = h > 200 && h < 280;
  const isNeutral = s < 0.12 || (l > 0.8 && isBlue);

  if (l < 0.5) {
    const lx = scale(l, 0, 0.5, 0, MAX_BG_LIGHTNESS);
    return isNeutral ? { h: pole.h, s: pole.s, l: lx } : { h, s, l: lx };
  }

  let lx = scale(l, 0.5, 1, MAX_BG_LIGHTNESS, pole.l);
  if (isNeutral) {
    return { h: pole.h, s: pole.s, l: lx };
  }

  let hx = h;
  if (h > 60 && h < 180) {
    hx = h > 120 ? scale(h, 120, 180, 135, 180) : scale(h, 60, 120, 60, 105);
  }
  if (hx > 40 && hx < 80) {
    lx *= 0.75;
  }
  return { h: hx, s, l: lx };
};

export const stringToHslColor = (str: string, s: number, l: number): string => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  const h = ((hash % 360) + 360) % 360;
  const { r, g, b } = hslToRgb(
    darkenBackground(rgbToHsl(hslToRgb({ h, s: s / 100, l: l / 100 }))),
  );
  return `#${[r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
};
