// src/Footer.tsx
import "./Footer.css";
import React from "react";
import { Link, useLocation } from "react-router-dom";

interface FooterLink {
  label: string;
  // Internal routes use `to`, external destinations use `href`
  to?: string;
  href?: string;
}

// The single list of footer links. Add entries here and every page picks
// them up.
const FOOTER_LINKS: FooterLink[] = [{ label: "About", to: "/about" }];

interface FooterProps {
  // The homepage pins its footer to the bottom of a non-scrolling viewport.
  // Every other page lets it sit at the end of the document flow.
  fixed?: boolean;
}

const Footer: React.FC<FooterProps> = ({ fixed = false }) => {
  const { pathname } = useLocation();

  const renderLink = (link: FooterLink) => {
    if (link.href) {
      return (
        <a
          className="site-footer-link"
          href={link.href}
          target="_blank"
          rel="noopener noreferrer"
        >
          {link.label}
        </a>
      );
    }
    // The link to the current page stays clickable and looks identical to
    // every other link; aria-current only tells screen readers where they are.
    return (
      <Link
        className="site-footer-link"
        to={link.to!}
        aria-current={link.to === pathname ? "page" : undefined}
      >
        {link.label}
      </Link>
    );
  };

  return (
    <footer
      className={fixed ? "site-footer site-footer--fixed" : "site-footer"}
    >
      <div className="site-footer-inner">
        <span className="site-footer-tagline">
          Recs☆Moe is a recommender system for anime&nbsp;and&nbsp;manga
        </span>
        <nav className="site-footer-nav" aria-label="Site links">
          {FOOTER_LINKS.map((link) => (
            <React.Fragment key={link.label}>{renderLink(link)}</React.Fragment>
          ))}
        </nav>
      </div>
    </footer>
  );
};

export default Footer;
