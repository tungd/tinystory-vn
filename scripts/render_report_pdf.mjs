#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const nodeModules =
  process.env.CODEX_NODE_MODULES ||
  "/Users/tung/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const { marked } = require(path.join(nodeModules, "marked"));
const { chromium } = require(path.join(nodeModules, "playwright"));

const root = process.cwd();
const input = path.resolve(root, process.argv[2] || "report.md");
const output = path.resolve(
  root,
  process.argv[3] || "output/pdf/tinystory-vn-team-report.pdf",
);
const tempDir = path.resolve(root, "tmp/pdfs/team-report");
const htmlPath = path.join(tempDir, "report.html");

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function slugify(value) {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function normalizeText(value) {
  return value
    .normalize("NFC")
    .replace(/\s+/g, " ")
    .trim();
}

const source = fs.readFileSync(input, "utf8");
const frontMatterMatch = source.match(/^---\n([\s\S]*?)\n---\n/);
if (!frontMatterMatch) {
  throw new Error(`Missing YAML front matter in ${input}`);
}

const metadata = Object.fromEntries(
  frontMatterMatch[1]
    .split("\n")
    .map((line) => line.match(/^([^:]+):\s*"?(.+?)"?$/))
    .filter(Boolean)
    .map((match) => [match[1].trim(), match[2].trim().replace(/^"|"$/g, "")]),
);

let figureNumber = 0;
const markdown = source
  .slice(frontMatterMatch[0].length)
  .replace(/^\\newpage\s*$/gm, "")
  .replace(/\u2011/g, "-")
  .replace(
    /!\[([^\]]*)\]\(([^)]+)\)(?:\{width=(\d+)%\})?/g,
    (_, alt, src, width) => {
      figureNumber += 1;
      const renderedWidth = width || "72";
      return [
        `<figure style="width:${renderedWidth}%">`,
        `<img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}">`,
        `<figcaption>Hình ${figureNumber}: ${escapeHtml(alt)}</figcaption>`,
        "</figure>",
      ].join("");
    },
  );

const usedIds = new Map();
const headings = [];
const renderer = new marked.Renderer();
renderer.heading = ({ tokens, depth }) => {
  const text = renderer.parser.parseInline(tokens);
  const plain = text.replace(/<[^>]+>/g, "");
  const base = slugify(plain) || `section-${headings.length + 1}`;
  const count = usedIds.get(base) || 0;
  usedIds.set(base, count + 1);
  const id = count ? `${base}-${count + 1}` : base;
  if (depth === 2 || depth === 3) headings.push({ depth, plain, id });
  return `<h${depth} id="${id}">${text}</h${depth}>\n`;
};

marked.setOptions({ gfm: true, breaks: false, renderer });
const body = marked.parse(markdown);
const tocHeadings = headings.filter(({ depth }) => depth === 2 || depth === 3);
const baseHref = `${pathToFileURL(`${root}${path.sep}`).href}`;

function buildHtml(tocPages = {}) {
  const authorHtml = metadata.author
    .split("|")
    .map((line) => `<span>${escapeHtml(line.trim())}</span>`)
    .join("");
  const toc = tocHeadings
    .map(
      ({ depth, plain, id }) => `
        <li class="toc-level-${depth}">
          <a href="#${id}">${escapeHtml(plain)}</a>
          <span class="toc-dots"></span>
          <span class="toc-page">${tocPages[id] || ""}</span>
        </li>`,
    )
    .join("\n");

  return `<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <base href="${baseHref}">
  <title>${escapeHtml(metadata.title)}</title>
  <style>
    @page {
      size: Letter;
      margin: 22mm;
      @bottom-center {
        content: counter(page);
        color: #000;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        font-size: 10pt;
      }
    }
    * { box-sizing: border-box; }
    html { font-size: 11pt; }
    body {
      margin: 0;
      color: #000;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      line-height: 1.18;
      text-rendering: optimizeLegibility;
    }
    .title-page {
      min-height: 235.4mm;
      page-break-after: always;
    }
    .title-block {
      padding-top: 19mm;
      text-align: center;
    }
    .title-block h1 {
      max-width: 171mm;
      margin: 0 auto 2mm;
      font-size: 17.3pt;
      font-weight: 400;
      line-height: 1.16;
    }
    .subtitle {
      max-width: 160mm;
      margin: 0 auto;
      font-size: 10pt;
      line-height: 1.2;
      text-align: center;
    }
    .author {
      margin: 10mm 0 0;
      font-size: 11pt;
      line-height: 1.35;
      text-align: center;
    }
    .author span { display: block; }
    .author span:first-child {
      margin-bottom: 1.5mm;
      font-weight: 600;
    }
    .date {
      margin: 6mm 0 0;
      font-size: 11pt;
      text-align: center;
    }
    .toc {
      break-before: page;
      margin-top: 0;
      padding-top: 0;
    }
    .toc h2 {
      margin: 0 0 3mm;
      font-size: 12pt;
      font-weight: 700;
    }
    .toc ol {
      margin: 0 0 0 4mm;
      padding: 0;
      list-style: none;
    }
    .toc li {
      display: flex;
      align-items: baseline;
      margin: 0 0 .45mm;
      color: #a00000;
      font-size: 8.8pt;
      line-height: 1.08;
    }
    .toc li.toc-level-2 {
      margin-top: .8mm;
      font-weight: 600;
    }
    .toc li.toc-level-3 {
      padding-left: 5mm;
      font-weight: 400;
    }
    .toc a {
      flex: 0 1 auto;
      color: #a00000;
      text-decoration: none;
    }
    .toc-dots {
      flex: 1 1 auto;
      min-width: 8mm;
      margin: 0 1.5mm;
      border-bottom: .35pt dotted #a00000;
      transform: translateY(-.7mm);
    }
    .toc-page {
      flex: 0 0 7mm;
      text-align: right;
    }
    h2, h3 {
      color: #000;
      break-after: avoid-page;
    }
    h2 {
      margin: 1.45em 0 .72em;
      font-size: 12pt;
      font-weight: 700;
      line-height: 1.15;
    }
    h3 {
      margin: 1.25em 0 .55em;
      font-size: 11pt;
      font-weight: 700;
      line-height: 1.15;
    }
    p {
      margin: 0 0 .72em;
      text-align: justify;
      orphans: 2;
      widows: 2;
    }
    ul, ol {
      margin: 0 0 .75em;
      padding-left: 6.5mm;
    }
    li { margin-bottom: .18em; }
    strong { color: #000; }
    code {
      padding: 0;
      color: #000;
      background: none;
      font-family: "Latin Modern Mono", "SFMono-Regular", Menlo, Consolas, monospace;
      font-size: .88em;
      overflow-wrap: anywhere;
    }
    pre {
      margin: .8em 0;
      padding: .65em;
      border: .4pt solid #bbb;
      color: #000;
      background: #f6f6f6;
      font-size: 8.5pt;
      line-height: 1.2;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      break-inside: avoid-page;
    }
    pre code {
      padding: 0;
      color: inherit;
      background: none;
      font-size: inherit;
    }
    blockquote {
      margin: .8em 5%;
      padding: 0;
      border: 0;
      color: #000;
      background: none;
      font-size: 9.5pt;
    }
    blockquote p:last-child { margin-bottom: 0; }
    table {
      width: 100%;
      margin: .8em 0 1em;
      border-collapse: collapse;
      table-layout: auto;
      border-top: .75pt solid #000;
      border-bottom: .75pt solid #000;
      font-size: 8.2pt;
      line-height: 1.12;
    }
    thead { display: table-header-group; }
    tr { break-inside: avoid; }
    th, td {
      padding: 1.1mm 1.5mm;
      border: 0;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th:first-child, td:first-child {
      overflow-wrap: normal;
    }
    th {
      color: #000;
      border-bottom: .45pt solid #000;
      background: none;
      font-weight: 400;
      text-align: left;
    }
    figure {
      margin: 1em auto .9em;
      text-align: center;
      break-inside: avoid-page;
    }
    img {
      display: block;
      max-width: 100%;
      max-height: 190mm;
      width: 100%;
      height: auto;
      margin: 0 auto;
      object-fit: contain;
    }
    figcaption {
      margin-top: .55em;
      font-size: 9.5pt;
      line-height: 1.2;
      text-align: center;
    }
    a { color: #a00000; }
    hr {
      margin: 1em 0;
      border: 0;
      border-top: .4pt solid #000;
    }
  </style>
</head>
<body>
  <section class="title-page">
    <div class="title-block">
      <h1>${escapeHtml(metadata.title)}</h1>
      <p class="subtitle">${escapeHtml(metadata.subtitle)}</p>
      <p class="author">${authorHtml}</p>
      <p class="date">${escapeHtml(metadata.date)}</p>
    </div>
    <nav class="toc">
      <h2>Mục lục</h2>
      <ol>${toc}</ol>
    </nav>
  </section>
  <main>${body}</main>
</body>
</html>`;
}

fs.mkdirSync(path.dirname(output), { recursive: true });
fs.mkdirSync(tempDir, { recursive: true });

const systemChrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const browser = await chromium.launch({
  headless: true,
  ...(fs.existsSync(systemChrome) ? { executablePath: systemChrome } : {}),
});

async function renderPdf(tocPages = {}) {
  fs.writeFileSync(htmlPath, buildHtml(tocPages));
  const page = await browser.newPage({ viewport: { width: 1020, height: 1320 } });
  await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "networkidle" });
  await page.emulateMedia({ media: "print" });
  await page.pdf({
    path: output,
    format: "Letter",
    printBackground: true,
    preferCSSPageSize: true,
    displayHeaderFooter: false,
    tagged: true,
    outline: true,
  });
  await page.close();
}

async function extractTocPages() {
  const pdfjs = await import(
    pathToFileURL(
      path.join(nodeModules, "pdfjs-dist", "legacy", "build", "pdf.mjs"),
    ).href
  );
  const document = await pdfjs.getDocument({
    data: new Uint8Array(fs.readFileSync(output)),
    disableWorker: true,
  }).promise;
  const pageTexts = [];
  for (let index = 1; index <= document.numPages; index += 1) {
    const page = await document.getPage(index);
    const content = await page.getTextContent();
    pageTexts.push(normalizeText(content.items.map(({ str }) => str).join(" ")));
  }

  const tocPages = {};
  for (const { plain, id } of tocHeadings) {
    const needle = normalizeText(plain);
    const pageIndex = pageTexts.findLastIndex((text) => text.includes(needle));
    if (pageIndex >= 0) tocPages[id] = pageIndex + 1;
  }
  return tocPages;
}

await renderPdf();
const tocPages = await extractTocPages();
await renderPdf(tocPages);
await browser.close();

console.log(output);
