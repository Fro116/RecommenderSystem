// src/ViewPage.tsx
import "./Header.css";
import "./ViewPage.css";
import "./components/DetailPane.css";
import React, { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate, useParams, Link } from "react-router-dom";
import pako from "pako";
import {
  Result,
  CardType,
  MediaTypePayload,
  Payload,
  getBiggestImageUrl,
  API_BASE,
  stringToHslColor,
} from "./types";
import CardImage from "./components/CardImage";
import ManualScrollDiv from "./components/ManualScrollDiv";
import DetailPane from "./components/DetailPane";

interface ViewPageProps {
  isMobile: boolean;
}

const loadingMessages = [
  "Dusting off some hidden gems...",
  "Consulting the head maid...",
  "Polishing your recommendations...",
  "Accessing your profile...",
  "Analyzing your watch history...",
  "Tidying up the final selections...",
  "Please wait a moment...",
  "Scrubbing away fanservice...",
  "Loading recommendations...",
];

const ViewPage: React.FC<ViewPageProps> = ({ isMobile }) => {
  // Hooks and State
  const navigate = useNavigate();
  const { source, username, itemType, itemid } = useParams<{
    source: string;
    username?: string;
    itemType?: "anime" | "manga";
    itemid?: string;
  }>();
  const [results, setResults] = useState<Result[]>([]);
  const [apiState, setApiState] = useState<string>("");
  const [titleName, setTitleName] = useState<string>("");
  const [titleUrl, setTitleUrl] = useState<string>("");
  const [totalResults, setTotalResults] = useState<number>(0);
  const [cardType, setCardType] = useState<CardType>("Anime");
  const [displayCardType, setDisplayCardType] = useState<CardType>("Anime");
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [loadingText, setLoadingText] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [loadingMore, setLoadingMore] = useState<boolean>(false);
  const [selectedItem, setSelectedItem] = useState<Result | null>(null);
  const [cardRotation, setCardRotation] = useState<{ [index: number]: number }>(
    {},
  );
  const [pinnedCardIndex, setPinnedCardIndex] = useState<number | null>(null);

  const getInitialViewType = useCallback((): "grid" | "list" => {
    const stored = localStorage.getItem("viewType");
    if (stored === "grid" || stored === "list") {
      return stored as "grid" | "list";
    }
    return isMobile ? "list" : "grid";
  }, [isMobile]);

  const [viewType, setViewType] = useState<"grid" | "list">(
    getInitialViewType(),
  );

  useEffect(() => {
    localStorage.setItem("viewType", viewType);
  }, [viewType]);

  // Refs
  const gridViewRef = useRef<HTMLDivElement>(null);
  const loadMoreRef = useRef<HTMLDivElement>(null);
  const initialFetchStartedRef = useRef<boolean>(false);

  const isUserMode = !!(source && username);
  const isItemMode = !!(source && itemType && itemid);

  useEffect(() => {
    const siteDefaultTitle = document.title;
    if (titleName) {
      document.title = `${titleName} | Recs☆Moe`;
    }
    return () => {
      document.title = siteDefaultTitle;
    };
  }, [titleName]);

  // --- Functions ---
  const fetchResults = useCallback(
    (
      endpoint: string,
      payload: Payload,
      offset: number = 0,
      append: boolean = false,
      isInitialLoad: boolean = false,
    ) => {
      if (isInitialLoad) {
        setIsLoading(true);
        setError("");
        setResults([]);
      } else if (append) {
        setLoadingMore(true);
        setError("");
      } else {
        setIsLoading(true);
        setError("");
      }

      const limit = isMobile ? 8 : 16;
      const extendedPayload = { ...payload, pagination: { offset, limit } };
      const payloadString = JSON.stringify(extendedPayload);
      const compressedPayload = pako.gzip(payloadString);

      fetch(API_BASE + endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Encoding": "gzip",
        },
        body: compressedPayload,
      })
        .then((response) => {
          if (response.status === 404) {
            navigate("/404", { replace: true });
            return null;
          }
          if (!response.ok) {
            return response.text().then((text) => {
              throw new Error(
                `Workspace failed: ${response.statusText} (${response.status}) ${text || ""}`,
              );
            });
          }
          return response.json();
        })
        .then((data) => {
          if (!data) return;
          if (
            !data ||
            data.view === undefined ||
            data.state === undefined ||
            data.total === undefined
          ) {
            throw new Error("Invalid data structure received from server.");
          }
          const newResults = data.view;

          setApiState(data.state);
          setTotalResults(data.total);
          setResults(append ? (prev) => [...prev, ...newResults] : newResults);
          setCardType(data.medium);
          setDisplayCardType(data.medium); // Sync display type with the new data
          if (data.titlename) setTitleName(data.titlename);
          if (data.titleurl) setTitleUrl(data.titleurl);
          setError("");

          if (isInitialLoad) setIsLoading(false);
          if (append) setLoadingMore(false);
          if (!isInitialLoad && !append) setIsLoading(false);

          if (isInitialLoad && data.followup_action) {
            const followupPayload = {
              state: data.state,
              action: data.followup_action,
              pagination: extendedPayload.pagination,
            };
            fetch(API_BASE + data.followup_action.endpoint, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
              },
              body: pako.gzip(JSON.stringify(followupPayload)),
            })
              .then((res) => {
                if (!res.ok) {
                  res
                    .text()
                    .then((text) =>
                      console.warn(
                        `Async Followup fetch failed (${res.status}): ${text}`,
                      ),
                    );
                  return null;
                }
                return res.json();
              })
              .then((followupData) => {
                if (
                  followupData &&
                  followupData.view !== undefined &&
                  followupData.state !== undefined &&
                  followupData.total !== undefined
                ) {
                  setApiState(followupData.state);
                  setTotalResults(followupData.total);
                  setResults(followupData.view);
                  setCardType(followupData.medium);
                  setDisplayCardType(followupData.medium); // Also sync display type on followup
                  if (followupData.titlename)
                    setTitleName(followupData.titlename);
                  if (followupData.titleurl) setTitleUrl(followupData.titleurl);
                  gridViewRef.current?.scrollTo(0, 0);
                }
              })
              .catch((followupError) => {
                console.error(
                  "ERROR during background followup fetch:",
                  followupError,
                );
              });
          }

          if (!append) {
            gridViewRef.current?.scrollTo(0, 0);
          }
        })
        .catch((err) => {
          console.error("Error in main fetchResults promise chain:", err);
          setError(`Failed to load results: ${err.message}. Please try again.`);
          if (isInitialLoad) setResults([]);
          setIsLoading(false);
          setLoadingMore(false);
        });
    },
    [isMobile, navigate],
  );

  // --- Other useEffects ---
  useEffect(() => {
    if (initialFetchStartedRef.current) return;

    let endpoint: string;
    let payload: Payload;

    if (isUserMode) {
      endpoint = "/add_user";
      payload = { state: "", action: { source: source!, username: username! } };
    } else if (isItemMode) {
      endpoint = "/add_item";
      payload = {
        state: "",
        action: {
          medium: itemType === "anime" ? "Anime" : "Manga",
          source: source!,
          itemid: itemid!,
        },
      };
    } else{
      console.warn("Missing or invalid URL parameters. Redirecting home.");
      setError("Invalid URL. Please start a search from the homepage.");
      setIsLoading(false);
      const timer = setTimeout(() => navigate("/"), 3000);
      return () => clearTimeout(timer);
    }

    const randomTextIndex = Math.floor(
      Math.random() * loadingMessages.length,
    );
    setLoadingText(loadingMessages[randomTextIndex]);
    initialFetchStartedRef.current = true;
    fetchResults(endpoint, payload, 0, false, true);
  }, [
    fetchResults,
    navigate,
    source,
    username,
    itemType,
    itemid,
    isUserMode,
    isItemMode,
  ]);

  useEffect(() => {
    const currentLoadMoreRef = loadMoreRef.current;
    if (
      !currentLoadMoreRef ||
      isLoading ||
      loadingMore ||
      !apiState ||
      results.length === 0
    )
      return;
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (
            entry.isIntersecting &&
            !loadingMore &&
            results.length < totalResults
          ) {
            const offset = results.length;
            const payload: MediaTypePayload = {
              state: apiState,
              action: { medium: cardType },
            };
            fetchResults("/set_media", payload, offset, true, false);
          }
        });
      },
      { root: gridViewRef.current, rootMargin: "400px", threshold: 0.01 },
    );
    observer.observe(currentLoadMoreRef);
    return () => {
      if (currentLoadMoreRef) observer.unobserve(currentLoadMoreRef);
    };
  }, [
    results.length,
    totalResults,
    loadingMore,
    apiState,
    cardType,
    isLoading,
    fetchResults,
  ]);

  const toggleMediaType = () => {
    // Guard against toggling while a fetch is in progress.
    if (isLoading || loadingMore || !apiState) return;
    const newType = cardType === "Anime" ? "Manga" : "Anime";
    setCardType(newType);
    fetchResults(
      "/set_media",
      { state: apiState, action: { medium: newType } },
      0,
      false,
      false,
    );
  };

  const handleCardClick = (item: Result, index: number) => {
    setSelectedItem(item);
    if (!isMobile) {
      setPinnedCardIndex(index);
      setCardRotation((prev) => ({ ...prev, [index]: 180 }));
    }
  };

  const closeDetailPane = () => {
    setSelectedItem(null);
    if (pinnedCardIndex !== null) {
      setCardRotation((prev) => ({ ...prev, [pinnedCardIndex]: 0 }));
    }
    setPinnedCardIndex(null);
  };

  // Desktop hover handlers for grid view
  const handleMouseEnter = (index: number) => {
    if (!isMobile) {
      setCardRotation((prev) => ({ ...prev, [index]: 180 }));
    }
  };

  const handleMouseLeave = (index: number) => {
    if (!isMobile && index !== pinnedCardIndex) {
      setCardRotation((prev) => ({ ...prev, [index]: 0 }));
    }
  };

  // --- Render Logic ---
  if (isLoading && results.length === 0) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>{loadingText}</p>
      </div>
    );
  }

  if (error && results.length === 0) {
    return (
      <div className="error-banner" style={{ margin: "20px auto" }}>
        <span>{error}</span>
      </div>
    );
  }

  const listViewHeader = (
    <div className="list-view-header">
      <div className="header-cell image-cell"></div>
      <div className="header-cell title-cell">Title</div>
      {displayCardType === "Anime" ? (
        <>
          <div className="header-cell detail-cell">Season</div>
          <div className="header-cell detail-cell type-cell">Type</div>
          <div className="header-cell detail-cell">Episodes</div>
          <div className="header-cell detail-cell">Duration</div>
          <div className="header-cell detail-cell tags-cell">Tags</div>
        </>
      ) : (
        <>
          <div className="header-cell detail-cell">Year</div>
          <div className="header-cell detail-cell type-cell">Type</div>
          <div className="header-cell detail-cell">Volumes</div>
          <div className="header-cell detail-cell">Chapters</div>
          <div className="header-cell detail-cell tags-cell">Tags</div>
        </>
      )}
    </div>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Header */}
      <header className="header--toggle">
        <div className="header-toggle">
          <div className="recommendations-title">
            <h2>
              <Link to="/" className="recsmoe-brand-link">
                Recs☆Moe
              </Link>
              's picks
              {titleName && (
                <>
                  {" for "}
                  <a
                    href={titleUrl || "#"}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="profile-link"
                  >
                    {titleName}
                  </a>
                </>
              )}
            </h2>
          </div>
          <div className="media-toggle" onClick={toggleMediaType}>
            <div
              className={`toggle-option ${cardType === "Anime" ? "active" : ""}`}
            >
              Anime
            </div>
            <div
              className={`toggle-option ${cardType === "Manga" ? "active" : ""}`}
            >
              Manga
            </div>
            <div className={`toggle-slider ${cardType}`}></div>
          </div>
          <div
            className="view-type-toggle"
            onClick={() => setViewType(viewType === "grid" ? "list" : "grid")}
          >
            <div
              className={`toggle-option ${viewType === "grid" ? "active" : ""}`}
            >
              Grid
            </div>
            <div
              className={`toggle-option ${viewType === "list" ? "active" : ""}`}
            >
              List
            </div>
            <div className={`toggle-slider ${viewType}`}></div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && results.length > 0 && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError("")}>&times;</button>
        </div>
      )}

      {/* Results Grid/List */}
      {results.length > 0 ? (
        <div className="grid-container" ref={gridViewRef}>
          <div className={viewType === "grid" ? "grid-view" : "list-view"}>
            {viewType === "list" && listViewHeader}
            {results.map((item, index) => {
              if (viewType === "grid") {
                return (
                  <div
                    key={`${apiState}-${item.url}-${index}`}
                    className="card"
                    onClick={() => handleCardClick(item, index)}
                    onMouseEnter={() => handleMouseEnter(index)}
                    onMouseLeave={() => handleMouseLeave(index)}
                  >
                    <div
                      className="card-inner"
                      style={{
                        transform: `rotateY(${cardRotation[index] || 0}deg)`,
                      }}
                    >
                      <div className="card-front">
                        <CardImage item={item} onClick={() => {}} />
                      </div>
                      <div className="card-back">
                        <div
                          className="card-back-bg"
                          style={{
                            backgroundImage: `url(${getBiggestImageUrl(item.image) || getBiggestImageUrl(item.missing_image)})`,
                          }}
                        ></div>
                        <div className="card-back-container">
                          <div
                            className="card-back-header"
                            onClick={(e) => {
                              e.stopPropagation();
                              window.open(
                                item.url,
                                "_blank",
                                "noopener,noreferrer",
                              );
                            }}
                          >
                            <h2 className="card-back-title">{item.title}</h2>
                            {item.english_title && (
                              <h3 className="card-back-english-title">
                                {item.english_title}
                              </h3>
                            )}
                          </div>
                          <ManualScrollDiv className="card-back-body">
                            <div className="card-back-details-section">
                              <h4>Details</h4>
                              <div className="card-back-details-table">
                                {displayCardType === "Anime" ? (
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
                                        {item.startdate
                                          ? item.startdate.substring(0, 4)
                                          : "-"}
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
                            {item.genres && (
                              <div className="card-back-tags-section">
                                <h4>Tags</h4>
                                <div className="card-back-tags-container">
                                  {item.genres.split(", ").map((tag) => {
                                    const backgroundColor = stringToHslColor(
                                      tag,
                                      50,
                                      40,
                                    );
                                    return (
                                      <span
                                        key={tag}
                                        className="tag"
                                        style={{ backgroundColor }}
                                      >
                                        {tag}
                                      </span>
                                    );
                                  })}
                                </div>
                              </div>
                            )}
                          </ManualScrollDiv>
                          <div className="card-back-cta">
                            Click for more information
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              } else {
                // LIST VIEW ITEM
                return (
                  <div
                    key={`${apiState}-${item.url}-${index}`}
                    className="list-item"
                    onClick={() => setSelectedItem(item)}
                  >
                    <div className="item-cell image-cell">
                      <img
                        src={
                          getBiggestImageUrl(item.image) ||
                          getBiggestImageUrl(item.missing_image) ||
                          ""
                        }
                        alt={item.title}
                        className="list-item-image"
                        loading="lazy"
                      />
                    </div>
                    <div className="item-cell title-cell">
                      {isMobile ? (
                        <>
                          <div className="list-item-title">{item.title}</div>
                          {item.english_title && (
                            <div className="list-item-english-title">
                              {item.english_title}
                            </div>
                          )}
                        </>
                      ) : (
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="list-item-title-link"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <div className="list-item-title">{item.title}</div>
                          {item.english_title && (
                            <div className="list-item-english-title">
                              {item.english_title}
                            </div>
                          )}
                        </a>
                      )}
                      <div className="list-item-mobile-details">
                        {displayCardType === "Anime"
                          ? [
                              item.season,
                              item.type,
                              item.episodes && `${item.episodes} ep`,
                              item.duration,
                            ]
                              .filter(Boolean)
                              .join(" • ")
                          : [
                              item.startdate
                                ? item.startdate.substring(0, 4)
                                : null,
                              item.type,
                              item.volumes && `${item.volumes} vol`,
                              item.chapters && `${item.chapters} ch`,
                            ]
                              .filter(Boolean)
                              .join(" • ")}
                      </div>
                    </div>
                    {displayCardType === "Anime" ? (
                      <>
                        <div className="item-cell detail-cell">
                          {item.season || "-"}
                        </div>
                        <div className="item-cell detail-cell type-cell">
                          {item.type || "-"}
                        </div>
                        <div className="item-cell detail-cell">
                          {item.episodes ?? "-"}
                        </div>
                        <div className="item-cell detail-cell">
                          {item.duration || "-"}
                        </div>
                        <div className="item-cell detail-cell tags-cell">
                          {item.genres || "-"}
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="item-cell detail-cell">
                          {item.startdate
                            ? item.startdate.substring(0, 4)
                            : "-"}
                        </div>
                        <div className="item-cell detail-cell type-cell">
                          {item.type || "-"}
                        </div>
                        <div className="item-cell detail-cell">
                          {item.volumes ?? "-"}
                        </div>
                        <div className="item-cell detail-cell">
                          {item.chapters ?? "-"}
                        </div>
                        <div className="item-cell detail-cell tags-cell">
                          {item.genres || "-"}
                        </div>
                      </>
                    )}
                  </div>
                );
              }
            })}
            <div
              ref={loadMoreRef}
              style={{
                height: "10px",
                gridColumn: "1 / -1",
                visibility: "hidden",
              }}
            ></div>
            {loadingMore && (
              <div
                style={{
                  textAlign: "center",
                  gridColumn: "1 / -1",
                  padding: "20px",
                }}
              >
                Loading More...
              </div>
            )}
            {!loadingMore &&
              results.length > 0 &&
              results.length >= totalResults && (
                <div
                  style={{
                    textAlign: "center",
                    gridColumn: "1 / -1",
                    padding: "20px",
                    color: "#777",
                  }}
                >
                  End of results.
                </div>
              )}
          </div>
        </div>
      ) : (
        !isLoading &&
        !error && (
          <div style={{ textAlign: "center", padding: "40px", flexGrow: 1 }}>
            No recommendations found.
          </div>
        )
      )}

      {/* Detail Pane */}
      {selectedItem && (
        <DetailPane
          item={selectedItem}
          cardType={displayCardType}
          onClose={closeDetailPane}
          isMobile={isMobile}
        />
      )}
    </div>
  );
};

export default ViewPage;
