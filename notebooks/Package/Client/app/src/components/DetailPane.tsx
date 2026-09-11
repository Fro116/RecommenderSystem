import "./DetailPane.css";
import React from "react";
import {
  Result,
  CardType,
  getBiggestImageUrl,
  stringToHslColor,
} from "../types";

const INLINE_TAGS: Record<string, "em" | "strong" | "cite"> = {
  i: "em",
  em: "em",
  b: "strong",
  strong: "strong",
  cite: "cite",
};

const BLOCK_TAGS = new Set(["p", "ul", "li"]);

const TAG_RE = /<\s*(\/?)\s*([a-zA-Z][a-zA-Z0-9]*)([^>]*)>/g;

const safeHref = (attrs: string): string | null => {
  const match = attrs.match(/href\s*=\s*("([^"]*)"|'([^']*)'|([^\s>]+))/i);
  const raw = (match ? (match[2] ?? match[3] ?? match[4] ?? "") : "").trim();
  return /^https?:\/\//i.test(raw) ? raw : null;
};

interface SynopsisFrame {
  tag: string;
  attrs: string;
  children: React.ReactNode[];
}

const wrapFrame = (frame: SynopsisFrame, key: number): React.ReactNode => {
  const { tag, attrs, children } = frame;
  if (tag === "a") {
    const href = safeHref(attrs);
    return href ? (
      <a key={key} href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    ) : (
      <React.Fragment key={key}>{children}</React.Fragment>
    );
  }
  const Tag = INLINE_TAGS[tag] ?? (tag as "p" | "ul" | "li");
  return <Tag key={key}>{children}</Tag>;
};

const renderSynopsis = (raw: string): React.ReactNode[] => {
  const text = raw
    .replace(/\r\n?/g, "\n")
    .replace(/[ \t]*<\s*\/?\s*br\s*\/?\s*>[ \t]*\n?/gi, "\n")
    .replace(/\s*<\s*(\/?)\s*(p|ul|li)\s*([^>]*)>\s*/gi, "<$1$2$3>")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  const root: React.ReactNode[] = [];
  const stack: SynopsisFrame[] = [];
  const push = (node: React.ReactNode) =>
    (stack.length ? stack[stack.length - 1].children : root).push(node);

  let last = 0;
  let key = 0;
  let match: RegExpExecArray | null;
  TAG_RE.lastIndex = 0;
  while ((match = TAG_RE.exec(text)) !== null) {
    const tag = match[2].toLowerCase();
    if (match.index > last) push(text.slice(last, match.index));
    last = TAG_RE.lastIndex;
    if (!(tag in INLINE_TAGS) && !BLOCK_TAGS.has(tag) && tag !== "a") continue;
    if (match[1] !== "/") {
      stack.push({ tag, attrs: match[3], children: [] });
      continue;
    }
    const open = stack.map((frame) => frame.tag).lastIndexOf(tag);
    if (open === -1) continue;
    while (stack.length > open) {
      push(wrapFrame(stack.pop()!, key++));
    }
  }
  if (last < text.length) push(text.slice(last));
  while (stack.length) {
    push(wrapFrame(stack.pop()!, key++));
  }
  return root;
};

interface DetailPaneProps {
  item: Result | null;
  cardType: CardType;
  onClose: () => void;
  isMobile: boolean;
}

const DetailPane: React.FC<DetailPaneProps> = ({
  item,
  cardType,
  isMobile,
  onClose,
}) => {
  const [mouseDownPos, setMouseDownPos] = React.useState<{
    x: number;
    y: number;
  } | null>(null);

  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  if (!item) {
    return null;
  }

  React.useEffect(() => {
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "unset";
    };
  }, []);

  const handleContentMouseDown = (e: React.MouseEvent) => {
    if (!isMobile) return;
    setMouseDownPos({ x: e.clientX, y: e.clientY });
  };

  const handleContentMouseUp = (e: React.MouseEvent) => {
    if (!isMobile || !mouseDownPos) return;
    let target = e.target as HTMLElement;
    while (target && target !== e.currentTarget) {
      if (target.tagName === "A") {
        setMouseDownPos(null);
        return;
      }
      target = target.parentElement as HTMLElement;
    }
    const deltaX = Math.abs(e.clientX - mouseDownPos.x);
    const deltaY = Math.abs(e.clientY - mouseDownPos.y);
    if (deltaX < 5 && deltaY < 5) {
      onClose();
    }
    setMouseDownPos(null);
  };

  const DetailsSection = () => (
    <div className="detail-pane-details-section">
      <h4>Details</h4>
      <div className="details-table">
        {cardType === "Anime" ? (
          <>
            <div>
              <strong>Type</strong>
              <span>{item.type || "-"}</span>
            </div>
            <div>
              <strong>Season</strong>
              <span>{item.season || "-"}</span>
            </div>
            <div>
              <strong>Source</strong>
              <span>{item.source || "-"}</span>
            </div>
            <div>
              <strong>Episodes</strong>
              <span>{item.episodes ?? "-"}</span>
            </div>
            <div>
              <strong>Duration</strong>
              <span>{item.duration || "-"}</span>
            </div>
            <div>
              <strong>Studio</strong>
              <span>{item.studios || "-"}</span>
            </div>
          </>
        ) : (
          <>
            <div>
              <strong>Type</strong>
              <span>{item.type || "-"}</span>
            </div>
            <div>
              <strong>Year</strong>
              <span>
                {item.startdate ? item.startdate.substring(0, 4) : "-"}
              </span>
            </div>
            <div>
              <strong>Status</strong>
              <span>{item.status || "-"}</span>
            </div>
            <div>
              <strong>Volumes</strong>
              <span>{item.volumes ?? "-"}</span>
            </div>
            <div>
              <strong>Chapters</strong>
              <span>{item.chapters ?? "-"}</span>
            </div>
            <div>
              <strong>Magazine</strong>
              <span>{item.studios || "-"}</span>
            </div>
          </>
        )}
      </div>
    </div>
  );

  const TagsSection = () =>
    item.genres ? (
      <div className="detail-pane-tags-section">
        <h4>Tags</h4>
        <div className="tags-container">
          {item.genres.split(", ").map((tag) => {
            const backgroundColor = stringToHslColor(tag, 70, 85);
            return (
              <span key={tag} className="tag" style={{ backgroundColor }}>
                {tag}
              </span>
            );
          })}
        </div>
      </div>
    ) : null;

  const SynopsisSection = () => (
    <div className="detail-pane-synopsis-section">
      <h4>Synopsis</h4>
      {item.synopsis ? (
        <div className="detail-pane-synopsis">
          {renderSynopsis(item.synopsis)}
        </div>
      ) : (
        <p>No synopsis available.</p>
      )}
    </div>
  );

  return (
    <div className="detail-pane-overlay" onClick={onClose}>
      <div
        className="detail-pane-content"
        onClick={(e) => {
          if (!isMobile) {
            e.stopPropagation();
          }
        }}
        onMouseDown={handleContentMouseDown}
        onMouseUp={handleContentMouseUp}
      >
        <button className="detail-pane-close" onClick={onClose}>
          &times;
        </button>

        {isMobile ? (
          <>
            <div className="detail-pane-header-mobile">
              <a
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                className="detail-pane-title-link"
              >
                <h2 className="detail-pane-title">{item.title}</h2>
              </a>
              {item.english_title && (
                <h3 className="detail-pane-english-title">
                  {item.english_title}
                </h3>
              )}
            </div>
            <DetailsSection />
            <TagsSection />
            <SynopsisSection />
          </>
        ) : (
          <>
            <div className="detail-pane-grid">
              <div className="detail-pane-left">
                <img
                  src={
                    getBiggestImageUrl(item.image) ||
                    getBiggestImageUrl(item.missing_image) ||
                    ""
                  }
                  alt={item.title}
                  className="detail-pane-image"
                />
              </div>
              <div className="detail-pane-right">
                <a
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="detail-pane-title-link"
                >
                  <h2 className="detail-pane-title">{item.title}</h2>
                </a>
                {item.english_title && (
                  <h3 className="detail-pane-english-title">
                    {item.english_title}
                  </h3>
                )}
                <DetailsSection />
                <TagsSection />
              </div>
            </div>
            <SynopsisSection />
          </>
        )}
      </div>
    </div>
  );
};

export default DetailPane;
