"""Fetch and split public-domain Project Gutenberg story collections."""

from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path


BLOCK_TAGS = {"p", "blockquote", "li", "div", "pre"}
IGNORED_TAGS = {"script", "style", "nav", "header", "footer"}
UNDERSTANDING_FABLES_URL = (
    "https://huggingface.co/datasets/demelin/understanding_fables/resolve/main/test.jsonl"
)


def normalize_text(value: str) -> str:
    value = html.unescape(value).replace("\u00ad", "").replace("\xa0", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    return value.strip()


def normalize_heading(value: str) -> str:
    value = normalize_text(value).replace("’", "'").replace("‘", "'")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .").upper()


def clean_public_story(value: str) -> str:
    """Remove Gutenberg navigation/page artifacts without rewriting prose."""
    value = normalize_text(value)
    value = re.split(r"\n\n\[Contents\]", value, maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.sub(r"\[\d+\]", "", value)
    value = re.sub(r"(?m)^\s*\d+\s*$", "", value)
    value = re.sub(r"(?is)\[Illustration(?::[^\]]*)?\]", "", value)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", value) if part.strip()]
    if paragraphs and paragraphs[0].casefold() == "persons":
        note_index = next(
            (index for index, paragraph in enumerate(paragraphs) if paragraph == "NOTE"),
            None,
        )
        search_from = note_index + 2 if note_index is not None else 1
        start = next(
            (
                index for index, paragraph in enumerate(paragraphs[search_from:], search_from)
                if len(paragraph.split()) >= 8
            ),
            None,
        )
        if start is not None:
            paragraphs = paragraphs[start:]
    paragraphs = [
        paragraph for paragraph in paragraphs
        if not (
            len(paragraph.split()) <= 16
            and re.search(r"[A-Z]", paragraph)
            and paragraph == paragraph.upper()
        )
    ]
    value = normalize_text("\n\n".join(paragraphs))
    dropcap_repairs = {
        r"^he\b": "The",
        r"^t the\b": "At the",
        r"^nce\b": "Once",
        r"^ne day\b": "One day",
        r"^very\b": "A very",
        r"^here\b": "There",
    }
    for pattern, replacement in dropcap_repairs.items():
        if re.match(pattern, value):
            value = re.sub(pattern, replacement, value, count=1)
            break
    return value


@dataclass
class Section:
    title: str
    story: str


@dataclass
class _Heading:
    level: int
    attrs: dict[str, str]
    ids: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)


class GutenbergSectionParser(HTMLParser):
    def __init__(self, config: dict):
        super().__init__(convert_charrefs=True)
        self.config = config
        self.sections: list[Section] = []
        self.heading: _Heading | None = None
        self.current_title = ""
        self.paragraph_text: list[str] = []
        self.body_parts: list[str] = []
        self.ignored_depth = 0
        self.active = not config.get("start_marker")
        self.stopped = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        if tag in IGNORED_TAGS:
            self.ignored_depth += 1
            return
        if self.ignored_depth:
            return
        if re.fullmatch(r"h[1-4]", tag):
            self._flush_paragraph()
            self.heading = _Heading(int(tag[1]), attrs_dict)
            if attrs_dict.get("id"):
                self.heading.ids.append(attrs_dict["id"])
            return
        if self.heading is not None and attrs_dict.get("id"):
            self.heading.ids.append(attrs_dict["id"])
        if tag == "br":
            target = self.heading.text if self.heading is not None else self.paragraph_text
            target.append(" ")
        if tag == "img" and len(attrs_dict.get("alt", "")) == 1:
            target = self.heading.text if self.heading is not None else self.paragraph_text
            target.append(attrs_dict["alt"])

    def handle_endtag(self, tag: str) -> None:
        if tag in IGNORED_TAGS:
            self.ignored_depth = max(0, self.ignored_depth - 1)
            return
        if self.ignored_depth:
            return
        if self.heading is not None and tag == f"h{self.heading.level}":
            self._finish_heading()
            return
        if tag in BLOCK_TAGS:
            self._flush_paragraph()

    def handle_data(self, data: str) -> None:
        if self.ignored_depth or self.stopped:
            return
        if self.heading is not None:
            self.heading.text.append(data)
        elif self.active and self.current_title:
            self.paragraph_text.append(data)

    def close(self) -> None:
        super().close()
        self._flush_section()

    def _heading_matches(self, heading: _Heading) -> bool:
        if heading.level != self.config["heading_level"]:
            return False
        class_name = self.config.get("heading_class")
        if class_name and class_name not in heading.attrs.get("class", "").split():
            return False
        if self.config.get("require_heading_id") and not heading.ids:
            return False
        prefixes = self.config.get("heading_id_prefixes", [])
        if prefixes and not any(
            ident.startswith(prefix) for ident in heading.ids for prefix in prefixes
        ):
            return False
        return True

    def _finish_heading(self) -> None:
        assert self.heading is not None
        heading = self.heading
        self.heading = None
        title = normalize_text("".join(heading.text))
        normalized = normalize_heading(title)

        start = self.config.get("start_marker")
        if not self.active and start and normalize_heading(start) == normalized:
            self.active = True
            if self._heading_matches(heading):
                self._flush_section()
                self.current_title = title
            return
        stop = self.config.get("stop_marker")
        if self.active and stop and normalize_heading(stop) == normalized:
            self._flush_section()
            self.stopped = True
            self.active = False
            return
        if not self.active:
            return
        if self._heading_matches(heading):
            self._flush_section()
            self.current_title = title
        elif self.current_title and title:
            self._flush_paragraph()
            self.body_parts.append(title)

    def _flush_paragraph(self) -> None:
        paragraph = normalize_text("".join(self.paragraph_text))
        self.paragraph_text.clear()
        if paragraph and self.active and self.current_title:
            self.body_parts.append(paragraph)

    def _flush_section(self) -> None:
        self._flush_paragraph()
        story = clean_public_story("\n\n".join(self.body_parts))
        if self.current_title and story:
            self.sections.append(Section(self.current_title, story))
        self.current_title = ""
        self.body_parts.clear()


def extract_sections(document: str, config: dict) -> list[Section]:
    parser = GutenbergSectionParser(config)
    parser.feed(document)
    parser.close()
    return parser.sections


def fetch_html(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "tinystory-vn-v5/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def collect_understanding_fables(
    document: str, *, seed: int = 42, holdout_fraction: float = 0.2
) -> list[dict]:
    rows = []
    for index, line in enumerate(document.splitlines()):
        if not line.strip():
            continue
        source = json.loads(line)
        story = re.sub(
            r"\s*What is the moral of this story\?\s*$", "", source["story"], flags=re.I
        ).strip()
        moral = source[f"answer{source['label']}"]
        source_id = f"understanding-fables:{index}"
        value = int.from_bytes(
            hashlib.sha256(f"{seed}:{source_id}".encode()).digest()[:8], "big"
        ) / 2**64
        rows.append({
            "source": source_id,
            "collection": "Understanding Fables",
            "url": UNDERSTANDING_FABLES_URL,
            "title": source_id,
            "story": story,
            "word_count": len(story.split()),
            "story_sha256": hashlib.sha256(story.encode()).hexdigest(),
            "license": "MIT",
            "provided_moral": moral,
            "source_split": "external_holdout" if value < holdout_fraction else "train",
        })
    return rows


def load_manifest(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["sources"]


def collect_candidates(
    sources: list[dict], *, min_words: int = 70, max_words: int = 380
) -> tuple[list[dict], dict]:
    candidates: list[dict] = []
    seen: set[str] = set()
    per_source: dict[str, dict] = {}
    for source in sources:
        document = fetch_html(source["url"])
        sections = extract_sections(document, source)
        accepted = 0
        for index, section in enumerate(sections):
            words = len(section.story.split())
            if not min_words <= words <= max_words:
                continue
            digest = hashlib.sha256(
                re.sub(r"\W+", "", section.story.casefold()).encode()
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            candidates.append({
                "source": f"gutenberg:{source['gutenberg_id']}:{index}",
                "gutenberg_id": source["gutenberg_id"],
                "collection": source["title"],
                "url": source["url"],
                "title": section.title,
                "story": section.story,
                "word_count": words,
                "story_sha256": digest,
                "license": "Project Gutenberg public domain in the USA",
            })
            accepted += 1
        per_source[str(source["gutenberg_id"])] = {
            "title": source["title"],
            "sections": len(sections),
            "length_accepted": accepted,
        }
    return candidates, {"candidates": len(candidates), "sources": per_source}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="runs/v5/sources.json")
    parser.add_argument("--out", default="runs/v5/data/candidates.jsonl")
    parser.add_argument("--meta", default="runs/v5/data/source_meta.json")
    parser.add_argument("--min-words", type=int, default=70)
    parser.add_argument("--max-words", type=int, default=380)
    parser.add_argument("--without-modern", action="store_true")
    args = parser.parse_args()

    candidates, meta = collect_candidates(
        load_manifest(args.manifest), min_words=args.min_words, max_words=args.max_words
    )
    if not args.without_modern:
        modern = collect_understanding_fables(fetch_html(UNDERSTANDING_FABLES_URL))
        candidates.extend(modern)
        meta["modern_candidates"] = len(modern)
        meta["modern_train"] = sum(row["source_split"] == "train" for row in modern)
        meta["modern_external_holdout"] = sum(
            row["source_split"] == "external_holdout" for row in modern
        )
        meta["candidates"] = len(candidates)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or Path(args.meta).exists():
        raise FileExistsError(f"Refusing existing v5 source output: {output}")
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates),
        encoding="utf-8",
    )
    Path(args.meta).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
