/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
  readonly VITE_ENABLE_SEARCH_BY_QUERY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
