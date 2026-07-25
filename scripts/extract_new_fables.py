#!/usr/bin/env python3
"""
Extract prose fables from three Vietnamese/Aesop sources:
1. Truyen Ngu Ngon Aesop - Aesop.epub  (Calibre epub, bold-span titles)
2. 108 Truyen Ngu Ngon Hay Nhat - Nhieu tac gia.mobi  (per-xhtml-file stories)
3. Truyen Ngu Ngon Viet Nam Chon Loc - Nguyen Cu.mobi  (single book.html, anchor-split)

Output:
  data/raw/fables_new.jsonl   — stories from these 3 sources only
  data/raw/fables_all.jsonl   — merged with existing fables.jsonl, deduplicated

Schema: {"topic", "moral", "age_range", "story", "source"}
"""

import os
import re
import json
import html
import zipfile
import unicodedata

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASOURCES = os.path.join(BASE_DIR, "datasources")
DATA_RAW    = os.path.join(BASE_DIR, "data", "raw")

AESOP_EPUB  = os.path.join(DATASOURCES, "Truyen Ngu Ngon Aesop - Aesop.epub")
MOBI_108    = os.path.join(DATASOURCES, "108 Truyen Ngu Ngon Hay Nhat - Nhieu tac gia.mobi")
MOBI_VN     = os.path.join(DATASOURCES, "Truyen Ngu Ngon Viet Nam Chon Loc - Nguyen Cu.mobi")

OUT_NEW     = os.path.join(DATA_RAW, "fables_new.jsonl")
OUT_ALL     = os.path.join(DATA_RAW, "fables_all.jsonl")
EXISTING    = os.path.join(DATA_RAW, "fables.jsonl")

MIN_CHARS   = 100
MAX_CHARS   = 6000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Strip HTML entities, collapse whitespace, normalize unicode."""
    text = html.unescape(text)
    # Remove zero-width / non-breaking spaces
    text = text.replace("​", "").replace("‌", "").replace("‍", "")
    text = text.replace("\xa0", " ")
    # Collapse multiple spaces/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_tags(html_fragment: str) -> str:
    """Remove HTML tags and clean the resulting text."""
    text = re.sub(r"<[^>]+>", " ", html_fragment)
    return clean_text(text)


def extract_moral(story_text: str) -> str:
    """
    Try to extract the moral sentence.
    Vietnamese moral markers: last sentence that looks like a standalone lesson.
    Common patterns:
      - Bài học: ...
      - Ngụ ý: ...
      - A standalone last sentence (italic or after bullet)
    """
    # Try: last paragraph is often the moral (single sentence, no dialog marker)
    paragraphs = [p.strip() for p in story_text.split("\n") if p.strip()]
    if not paragraphs:
        return ""
    last = paragraphs[-1]
    # If the last paragraph is short (< 200 chars) and doesn't start with dialog
    # and isn't a continuation (no ending punctuation in previous paragraph)
    if len(last) < 250 and not last.startswith(("-", "–", "•", '"', "«")):
        # Check that it reads like a lesson (imperative, general statement)
        return last
    return ""


def make_record(topic: str, story: str, source: str) -> dict:
    story = clean_text(story)
    # Normalise topic
    topic = clean_text(re.sub(r"^\d+[-–.\s]+", "", topic))
    moral = extract_moral(story)
    return {
        "topic":     topic,
        "moral":     moral,
        "age_range": "",
        "story":     story,
        "source":    source,
    }


def is_prose(story_text: str) -> bool:
    """
    Returns True if the story is predominantly prose.
    Pure poetry = many very short lines (< 50 chars) with no long prose paragraphs.
    We allow embedded verse in prose fables.
    """
    paragraphs = [p.strip() for p in story_text.split("\n") if p.strip()]
    if not paragraphs:
        return False
    long_paras = [p for p in paragraphs if len(p) >= 80]
    # Prose requires at least one proper paragraph OR more than half are >= 50 chars
    medium_paras = [p for p in paragraphs if len(p) >= 50]
    return len(long_paras) >= 1 or (len(medium_paras) / len(paragraphs) > 0.35)


def deduplicate(records: list) -> list:
    """Deduplicate by normalized story content."""
    seen = set()
    result = []
    for rec in records:
        # Normalize for comparison
        key = re.sub(r"\s+", "", unicodedata.normalize("NFC", rec["story"])).lower()[:500]
        if key not in seen:
            seen.add(key)
            result.append(rec)
    return result


# ---------------------------------------------------------------------------
# Source 1: Aesop EPUB
# ---------------------------------------------------------------------------
# Structure: 4 HTML files inside zip.
# index_split_000.html and index_split_001.html contain stories.
# index_split_002.html = Table of Contents (skip).
# All paragraphs are <p class="calibre_3">.
# Story titles: calibre_3 paragraph whose stripped text matches
#   a numbered pattern like "1- TITLE", "1-TITLE", "1 – TITLE", etc.
# Non-title calibre_3 paragraphs are story body text.
# Skip preamble until we hit the first numbered title.
# ---------------------------------------------------------------------------

# Title detection pattern: starts with number + separator
TITLE_RE = re.compile(r"^\d+\s*[-–.]\s*.+", re.UNICODE)
CHAPTER_RE = re.compile(r"^CHƯƠNG\s+\d+", re.IGNORECASE | re.UNICODE)
# Section headers to skip (not story titles)
SKIP_HEADERS = {
    "Đôi lời cùng bạn đọc", "AESOP", "Aesop", "Æsop", "Tiểu Sử",
    "Thân Thế & Sự Nghiệp", "Truyện Ngụ Ngôn Aesop", "Truyện ngụ ngôn Aesop",
    "Table of Contents",
}


def parse_aesop_html(html_content: str) -> list:
    """
    Parse one HTML file from Aesop epub.
    Returns list of (title_raw, story_text) tuples.
    """
    para_re = re.compile(r"<p[^>]*class=\"calibre_3[^\"]*\"[^>]*>(.*?)</p>", re.DOTALL)
    paragraphs = para_re.findall(html_content)

    stories = []
    current_title = None
    current_paras = []

    for para_html in paragraphs:
        text = strip_tags(para_html)
        if not text:
            continue

        is_chapter = CHAPTER_RE.match(text)
        is_story_title = TITLE_RE.match(text) and not is_chapter

        if is_chapter:
            # Chapter header — flush current story, reset
            if current_title and current_paras:
                stories.append((current_title, "\n".join(current_paras)))
            current_title = None
            current_paras = []
            continue

        if is_story_title:
            # New story title — flush previous
            if current_title and current_paras:
                stories.append((current_title, "\n".join(current_paras)))
            current_title = text
            current_paras = []
            continue

        # Body paragraph
        if current_title:
            current_paras.append(text)

    # Flush last story
    if current_title and current_paras:
        stories.append((current_title, "\n".join(current_paras)))

    return stories


def extract_aesop_epub(path: str) -> list:
    """Extract fables from Aesop epub."""
    source_name = os.path.basename(path)
    records = []

    story_files = ["index_split_000.html", "index_split_001.html"]

    with zipfile.ZipFile(path, "r") as zf:
        for fname in story_files:
            try:
                data = zf.read(fname).decode("utf-8", errors="replace")
            except KeyError:
                print(f"  [aesop] WARNING: {fname} not found in epub")
                continue
            stories = parse_aesop_html(data)
            for title_raw, story_text in stories:
                rec = make_record(title_raw, story_text, source_name)
                records.append(rec)

    print(f"  [aesop] raw extracted: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Source 2: 108 Stories MOBI (Nhieu tac gia)
# ---------------------------------------------------------------------------
# mobi.extract() gives us a mobi8/ epub directory.
# content.opf defines reading order in <spine>.
# Each part00XX.xhtml is one story (mostly).
# Story title: <span class="calibre10"><span class="bold"><span>TITLE</span></span></span>
# Body paragraphs: <p class="calibre_6">...</p> or similar calibre classes
#
# Skip:
#   cover_page.xhtml
#   part0000.xhtml (Table of Contents)
#   part0001.xhtml (intro/credits)
# ---------------------------------------------------------------------------

def extract_mobi8_story(xhtml_content: str) -> tuple:
    """
    Extract (title, story_text) from a single story xhtml.
    Returns (None, None) if no title found.
    """
    # Title: inside nested bold > span pattern
    title_match = re.search(
        r'class="bold"[^>]*>.*?<span>(.*?)</span>',
        xhtml_content, re.DOTALL
    )
    if not title_match:
        # Fallback: any bold text
        title_match = re.search(r'class="bold"[^>]*>(.*?)</span>', xhtml_content, re.DOTALL)
    if not title_match:
        return None, None

    title = strip_tags(title_match.group(1))
    if not title or len(title) < 2:
        return None, None

    # Body: all <p> tags after the title, excluding title paragraph itself
    # Collect all <p class="calibre_..."> paragraphs
    para_re = re.compile(r"<p[^>]*class=\"calibre[^\"]*\"[^>]*>(.*?)</p>", re.DOTALL)
    all_paras = para_re.findall(xhtml_content)

    body_paras = []
    past_title = False
    for para_html in all_paras:
        text = strip_tags(para_html)
        if not text:
            continue
        if not past_title:
            # Check if this paragraph contains the title
            if title in text or text in title:
                past_title = True
            continue
        body_paras.append(text)

    story_text = "\n".join(body_paras)
    return title, story_text


def extract_108_mobi(path: str) -> list:
    """Extract fables from 108 stories mobi."""
    import mobi  # standard per task requirements
    source_name = os.path.basename(path)

    tmpdir, fpath = mobi.extract(path)

    # Find the epub directory (mobi8)
    epub_dir = None
    for d in os.listdir(tmpdir):
        candidate = os.path.join(tmpdir, d)
        if os.path.isdir(candidate) and d.startswith("mobi8"):
            epub_dir = candidate
            break

    if not epub_dir:
        print(f"  [108] ERROR: No mobi8 dir found in {tmpdir}")
        return []

    # Parse content.opf for spine order.
    # We use regex instead of xml.etree.ElementTree to avoid XXE / billion-laughs
    # vulnerabilities in stdlib XML parsers (no defusedxml available per constraints).
    # The OPF file is small, local, and has a regular structure — regex is safe here.
    opf_path = os.path.join(epub_dir, "OEBPS", "content.opf")
    with open(opf_path, "r", encoding="utf-8", errors="replace") as f:
        opf_text = f.read()

    # Extract spine idrefs (ordered)
    idrefs = re.findall(r'<itemref\s+idref="([^"]+)"', opf_text)

    # Extract manifest id -> href mapping
    id_to_href = dict(re.findall(r'<item\s[^>]*\bid="([^"]+)"[^>]*\bhref="([^"]+)"', opf_text))
    # Also handle href before id attribute ordering
    for m in re.finditer(r'<item\s[^>]*\bhref="([^"]+)"[^>]*\bid="([^"]+)"', opf_text):
        id_to_href.setdefault(m.group(2), m.group(1))

    text_dir = os.path.join(epub_dir, "OEBPS")

    # Skip files
    skip_files = {"cover_page.xhtml", "part0000.xhtml", "part0001.xhtml"}

    records = []
    for idref in idrefs:
        href = id_to_href.get(idref, "")
        fname = os.path.basename(href)
        if fname in skip_files:
            continue

        fpath_full = os.path.join(text_dir, href)
        if not os.path.exists(fpath_full):
            continue

        with open(fpath_full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        title, story_text = extract_mobi8_story(content)
        if not title or not story_text:
            continue

        rec = make_record(title, story_text, source_name)
        records.append(rec)

    print(f"  [108] raw extracted: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Source 3: Vietnamese Fables MOBI (Nguyen Cu)
# ---------------------------------------------------------------------------
# mobi.extract() gives mobi7/book.html (single large HTML file).
# Stories are separated by the pattern:
#   <a id="fileposXXX" /><p height="0pt"...><font...>TRUYỆN NGỤ NGÔN VIỆT NAM...</font></p>
# Each section starts with the book title + author header, then:
#   <p ...><b><font color="#666633">STORY TITLE</font></b></p>
# Story paragraphs:
#   <p height="1em" width="2em">TEXT</p>
# ---------------------------------------------------------------------------

# Separator: anchor followed by book-title paragraph
VN_SPLIT_RE = re.compile(
    r'<a id="filepos\d+" /><p[^>]*height="0pt"[^>]*>.*?TRUYỆN',
    re.DOTALL
)

VN_TITLE_RE = re.compile(r'<b><font color="#666633">(.*?)</font></b>')
VN_PARA_RE  = re.compile(r'<p height="1em" width="2em">(.*?)</p>', re.DOTALL)


def extract_vn_mobi(path: str) -> list:
    """Extract fables from Vietnamese fables mobi (Nguyen Cu)."""
    import mobi  # standard per task requirements
    source_name = os.path.basename(path)

    tmpdir, fpath = mobi.extract(path)

    # This file uses mobi7 format — single book.html
    book_html = os.path.join(tmpdir, "mobi7", "book.html")
    if not os.path.exists(book_html):
        # Fallback: find any book.html
        for root_d, dirs, files in os.walk(tmpdir):
            for f in files:
                if f == "book.html":
                    book_html = os.path.join(root_d, f)
                    break

    with open(book_html, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Split into story sections
    # Each split after first contains: header remnant + title + story paras
    splits = VN_SPLIT_RE.split(content)
    # splits[0] = everything before first story (TOC)
    # splits[1..] = story sections (after the TRUYỆN... header was cut)

    records = []
    for section in splits[1:]:
        # Find story title
        title_m = VN_TITLE_RE.search(section)
        if not title_m:
            continue
        title = strip_tags(title_m.group(1))
        if not title or len(title) < 2:
            continue

        # Collect story paragraphs
        paras = VN_PARA_RE.findall(section)
        body_paras = []
        for p_html in paras:
            text = strip_tags(p_html)
            if text:
                body_paras.append(text)

        story_text = "\n".join(body_paras)
        rec = make_record(title, story_text, source_name)
        records.append(rec)

    print(f"  [vn] raw extracted: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Filter, deduplicate, write
# ---------------------------------------------------------------------------

def filter_records(records: list, source_label: str) -> list:
    """Apply length filter and prose check."""
    kept = []
    skipped_short = 0
    skipped_long  = 0
    skipped_poetry = 0

    for rec in records:
        story_len = len(rec["story"])
        if story_len < MIN_CHARS:
            skipped_short += 1
            continue
        if story_len > MAX_CHARS:
            skipped_long += 1
            continue
        if not is_prose(rec["story"]):
            skipped_poetry += 1
            continue
        kept.append(rec)

    print(
        f"  [{source_label}] kept: {len(kept)} "
        f"(dropped: {skipped_short} too short, {skipped_long} too long, "
        f"{skipped_poetry} poetry)"
    )
    return kept


def write_jsonl(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records -> {path}")


def read_jsonl(path: str) -> list:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Extracting Aesop EPUB ===")
    aesop_raw = extract_aesop_epub(AESOP_EPUB)
    aesop_filtered = filter_records(aesop_raw, "aesop")

    print("\n=== Extracting 108 Stories MOBI ===")
    mobi108_raw = extract_108_mobi(MOBI_108)
    mobi108_filtered = filter_records(mobi108_raw, "108")

    print("\n=== Extracting Vietnamese Fables MOBI ===")
    vnfable_raw = extract_vn_mobi(MOBI_VN)
    vnfable_filtered = filter_records(vnfable_raw, "vn")

    # Combine new sources
    new_records = aesop_filtered + mobi108_filtered + vnfable_filtered
    new_records = deduplicate(new_records)
    print(f"\nTotal new (after dedup): {len(new_records)}")

    # Write fables_new.jsonl
    write_jsonl(new_records, OUT_NEW)

    # Merge with existing
    existing = read_jsonl(EXISTING)
    print(f"Existing fables: {len(existing)}")
    all_records = existing + new_records
    all_records = deduplicate(all_records)
    print(f"Total all (after dedup): {len(all_records)}")

    write_jsonl(all_records, OUT_ALL)

    # Summary stats
    lengths = [len(r["story"]) for r in all_records]
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    median = lengths_sorted[n // 2]
    max_len = max(lengths_sorted)
    print(f"\n=== fables_all.jsonl stats ===")
    print(f"  Total: {n}")
    print(f"  Median length: {median} chars")
    print(f"  Max length:    {max_len} chars")


if __name__ == "__main__":
    main()
