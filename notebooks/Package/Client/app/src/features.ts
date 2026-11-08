// Build-time feature flags. Vite inlines VITE_* variables when it bundles, so
// a flag that is off ships no reachable UI for that feature.
//
// To turn a flag on locally, add it to .env.development.local (gitignored),
// for example:
//
//   VITE_ENABLE_SEARCH_BY_QUERY=true
//
// then restart `npm run dev`. Production builds use "production" mode, which
// never reads .env.development*, so local flags can't leak into a deploy.

const isOn = (value: string | undefined): boolean => value === "true";

// Search by Query: the /search home mode and /search/:itemType/:query results.
export const SEARCH_BY_QUERY_ENABLED = isOn(
  import.meta.env.VITE_ENABLE_SEARCH_BY_QUERY,
);
