export const SITE_ORIGIN = "https://recs.moe";

export interface PageMeta {
  title: string;
  description: string;
  file: string;
  changefreq: string;
  priority: string;
}

export const PAGE_META: Record<string, PageMeta> = {
  "/": {
    title: "Recs☆Moe | Time for another day of hard work!",
    description:
      "Get recommendations for anime and manga with Recs☆Moe! Search for similar series, or link your profile to get personalized recs.",
    file: "index.html",
    changefreq: "weekly",
    priority: "1.0",
  },
  "/title": {
    title: "Recs☆Moe | Time for another day of hard work!",
    description:
      "Get recommendations for anime and manga with Recs☆Moe! Search for similar series, or link your profile to get personalized recs.",
    file: "title.html",
    changefreq: "weekly",
    priority: "0.8",
  },
  "/about": {
    title: "About | Recs☆Moe",
    description:
      "Learn more about Recs☆Moe. Check status updates and view contact info.",
    file: "about.html",
    changefreq: "weekly",
    priority: "0.5",
  },
};
