import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(root, "dist");
const serverEntry = path.join(root, ".ssr", "entry-server.js");

if (!fs.existsSync(serverEntry)) {
  throw new Error(`Missing SSR bundle at ${serverEntry}. Run "npm run build:server" first.`);
}

const { render, PAGE_META, SITE_ORIGIN } = await import(pathToFileURL(serverEntry).href);

const shell = fs.readFileSync(path.join(outDir, "index.html"), "utf8");

if (!shell.includes('<div id="root"></div>')) {
  throw new Error('Root element not found in dist/index.html; cannot inject markup.');
}

const escape = (value) =>
  String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

fs.writeFileSync(path.join(outDir, "app.html"), shell);

for (const [route, meta] of Object.entries(PAGE_META)) {
  const canonical = SITE_ORIGIN + route;
  const markup = render(route);

  const head = [
    `<link rel="canonical" href="${escape(canonical)}">`,
    `<meta property="og:type" content="website">`,
    `<meta property="og:site_name" content="Recs☆Moe">`,
    `<meta property="og:url" content="${escape(canonical)}">`,
    `<meta property="og:title" content="${escape(meta.title)}">`,
    `<meta property="og:description" content="${escape(meta.description)}">`,
    `<meta name="twitter:card" content="summary">`,
  ]
    .map((tag) => `    ${tag}`)
    .join("\n");

  const html = shell
    .replace(/<title>[\s\S]*?<\/title>/, () => `<title>${escape(meta.title)}</title>`)
    .replace(
      /<meta\s+name="description"\s+content="[^"]*"\s*\/?>/,
      () => `<meta name="description" content="${escape(meta.description)}">`,
    )
    .replace("</head>", () => `${head}\n</head>`)
    .replace('<div id="root"></div>', () => `<div id="root">${markup}</div>`);

  fs.writeFileSync(path.join(outDir, meta.file), html);
  console.log(`prerendered ${route.padEnd(8)} -> dist/${meta.file} (${(markup.length / 1024).toFixed(1)} kB of markup)`);
}

const sitemap = [
  `<?xml version="1.0" encoding="UTF-8"?>`,
  `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">`,
  ...Object.entries(PAGE_META).map(([route, meta]) =>
    [
      `  <url>`,
      `    <loc>${escape(SITE_ORIGIN + route)}</loc>`,
      `    <changefreq>${meta.changefreq}</changefreq>`,
      `    <priority>${meta.priority}</priority>`,
      `  </url>`,
    ].join("\n"),
  ),
  `</urlset>`,
  ``,
].join("\n");

fs.writeFileSync(path.join(outDir, "sitemap.xml"), sitemap);
console.log(`wrote dist/sitemap.xml (${Object.keys(PAGE_META).length} urls)`);
