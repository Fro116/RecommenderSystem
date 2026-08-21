import "./Header.css";
import "./AboutPage.css";
import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { API_BASE } from "./types";
import Footer from "./Footer";

const CONTACT_EMAIL = "contact@recs.moe";
const SOURCE_URL = "https://github.com/Fro116/RecommenderSystem";
const SOURCE_LABEL = "Fro116/RecommenderSystem";

interface VersionInfo {
  pretrain?: string;
  finetune?: string;
}

type VersionState =
  | { status: "loading" }
  | { status: "ready"; info: VersionInfo }
  | { status: "error" };

const formatBuildDate = (value?: string): string | null => {
  if (!value || !/^\d{8}$/.test(value)) {
    return null;
  }
  const year = Number(value.slice(0, 4));
  const month = Number(value.slice(4, 6));
  const day = Number(value.slice(6, 8));
  const date = new Date(year, month - 1, day);
  if (
    date.getFullYear() !== year ||
    date.getMonth() !== month - 1 ||
    date.getDate() !== day
  ) {
    return null;
  }
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
};

const MailIcon: React.FC = () => (
  <svg
    width="18"
    height="18"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <rect x="2" y="4" width="20" height="16" rx="2" />
    <path d="m2 7 10 6 10-6" />
  </svg>
);

const CodeIcon: React.FC = () => (
  <svg
    width="18"
    height="18"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="m8 6-6 6 6 6" />
    <path d="m16 6 6 6-6 6" />
  </svg>
);

const InlineLink: React.FC<{ href: string; children: React.ReactNode }> = ({
  href,
  children,
}) => (
  <a
    className="about-inline-link"
    href={href}
    target="_blank"
    rel="noopener noreferrer"
  >
    {children}
  </a>
);

const AboutPage: React.FC = () => {
  const [version, setVersion] = useState<VersionState>({ status: "loading" });

  useEffect(() => {
    const siteDefaultTitle = document.title;
    document.title = "About | Recs☆Moe";
    return () => {
      document.title = siteDefaultTitle;
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE}/version`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Version fetch failed (${response.status})`);
        }
        return response.json();
      })
      .then((data: VersionInfo | null) => {
        setVersion({ status: "ready", info: data ?? {} });
      })
      .catch((err: Error) => {
        if (err.name !== "AbortError") {
          console.warn("Could not load version info:", err);
          setVersion({ status: "error" });
        }
      });
    return () => controller.abort();
  }, []);

  const versionCell = (key: keyof VersionInfo) => {
    if (version.status === "loading") {
      return { text: "", pending: true };
    }
    if (version.status === "error") {
      return { text: "Unavailable", pending: true };
    }
    const formatted = formatBuildDate(version.info[key]);
    return formatted
      ? { text: formatted, pending: false }
      : { text: "Unavailable", pending: true };
  };

  const majorVersion = versionCell("pretrain");
  const minorVersion = versionCell("finetune");

  return (
    <>
      <header className="header--about">
        <div className="about-header-bar">
          <h1 className="about-header-title">
            About{" "}
            <Link to="/" className="recsmoe-brand-link">
              Recs☆Moe
            </Link>
          </h1>
        </div>
      </header>

      <main className="about-main">
        <div className="about-card">
          <p className="about-lede">
            Recs☆Moe was created to discover new anime. Make a public profile on{" "}
            <InlineLink href="https://myanimelist.net/">MyAnimeList</InlineLink>
            , <InlineLink href="https://anilist.co/">AniList</InlineLink>,{" "}
            <InlineLink href="https://kitsu.app/explore/anime">
              Kitsu
            </InlineLink>{" "}
            or{" "}
            <InlineLink href="https://www.anime-planet.com/">
              Anime-Planet
            </InlineLink>{" "}
            to get started. Recs☆Moe will analyze your watch history and provide
            curated selections.
          </p>
          <p className="about-lede">
            Don't have a profile? Switch the homepage from "Search by User" to
            "Search by Title" and enter your favorite series to find similar
            shows.
          </p>

          <div className="about-divider" role="presentation">
            <span aria-hidden="true">☆</span>
          </div>

          <h2 className="about-section-title">Get in touch</h2>
          <ul className="about-links">
            <li>
              <a className="about-link-row" href={`mailto:${CONTACT_EMAIL}`}>
                <span className="about-link-label">Email</span>
                <span className="about-link-value">
                  <span className="about-link-icon">
                    <MailIcon />
                  </span>
                  <span className="about-link-text">{CONTACT_EMAIL}</span>
                </span>
              </a>
            </li>
            <li>
              <a
                className="about-link-row"
                href={SOURCE_URL}
                target="_blank"
                rel="noopener noreferrer"
              >
                <span className="about-link-label">GitHub</span>
                <span className="about-link-value">
                  <span className="about-link-icon">
                    <CodeIcon />
                  </span>
                  <span className="about-link-text">{SOURCE_LABEL}</span>
                </span>
              </a>
            </li>
          </ul>

          <div className="about-divider" role="presentation">
            <span aria-hidden="true">☆</span>
          </div>

          <h2 className="about-section-title">Release Info</h2>
          <ul className="about-facts" aria-live="polite">
            <li className="about-fact-row">
              <span className="about-fact-label">Major version</span>
              <span
                className={
                  majorVersion.pending
                    ? "about-fact-value is-pending"
                    : "about-fact-value"
                }
              >
                {majorVersion.text}
              </span>
            </li>
            <li className="about-fact-row">
              <span className="about-fact-label">Minor version</span>
              <span
                className={
                  minorVersion.pending
                    ? "about-fact-value is-pending"
                    : "about-fact-value"
                }
              >
                {minorVersion.text}
              </span>
            </li>
          </ul>
          <p className="about-note">
            Major versions are published once a month. These are significant updates
            that retrain the model to pick up newly released series. Minor versions 
            are published daily and are finetuned on recent activity.
          </p>
        </div>
      </main>

      <Footer />
    </>
  );
};

export default AboutPage;
