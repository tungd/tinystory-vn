"""Trích truyện từ các file .epub thành data/raw/stories.jsonl cho fine-tune.

Hỗ trợ 2 định dạng markup:
  * vnthuquan (Aesop / Grimm / La Fontaine): tiêu đề ở <p class="style32">,
    thân truyện trong <div id="fontchu">; nhiều truyện/file ngăn bằng dòng "* Tiêu đề".
  * Calibre (Andersen): tiêu đề ở <h2>, các đoạn ở <p class="para">.

Metadata trích bằng heuristic (ưu tiên độ chính xác — thà để trống còn hơn sai):
  * topic     = tiêu đề truyện (chương số La Mã được gộp về tên truyện cha).
  * moral     = câu mang dấu hiệu bài học rõ ràng ("Câu chuyện muốn nói...",
                "ngụ ý...", "ca ngợi...", "khuyên..."). Riêng ngụ ngôn Aesop
                (văn xuôi, kết bằng câu châm ngôn) cho phép lấy câu chốt cuối.
  * age_range = tham số --default-age (mặc định rỗng để review/điền sau).

Chỉ dùng thư viện chuẩn — không thêm dependency.
Parse OPF/container bằng regex thay vì xml.etree để tránh rủi ro XXE/billion-laughs.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
import zipfile
from pathlib import Path


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)

# ---------------------------------------------------------------------------
# Đọc cấu trúc epub
# ---------------------------------------------------------------------------

def _opf_path(zf: zipfile.ZipFile) -> str:
    container = zf.read("META-INF/container.xml").decode("utf-8", errors="replace")
    m = re.search(r'full-path="([^"]+)"', container)
    if not m:
        raise ValueError("Không tìm thấy rootfile trong container.xml")
    return m.group(1)


def _spine_files(zf: zipfile.ZipFile) -> list[str]:
    """Danh sách path XHTML theo đúng thứ tự đọc (spine)."""
    opf = _opf_path(zf)
    base = opf.rsplit("/", 1)[0] if "/" in opf else ""
    content = zf.read(opf).decode("utf-8", errors="replace")

    manifest: dict[str, str] = {}
    for item in re.finditer(r"<item\b[^>]*>", content, re.IGNORECASE):
        tag = item.group(0)
        mid = re.search(r'\bid="([^"]+)"', tag)
        href = re.search(r'\bhref="([^"]+)"', tag)
        if mid and href:
            manifest[mid.group(1)] = href.group(1)

    files: list[str] = []
    spine_m = re.search(r"<spine\b[^>]*>(.*?)</spine>", content, re.IGNORECASE | re.DOTALL)
    spine_block = spine_m.group(1) if spine_m else content
    for ref in re.finditer(r'<itemref\b[^>]*\bidref="([^"]+)"', spine_block, re.IGNORECASE):
        href = manifest.get(ref.group(1))
        if href:
            files.append(f"{base}/{href}" if base else href)
    return files


# ---------------------------------------------------------------------------
# Làm sạch HTML -> text
# ---------------------------------------------------------------------------

def _strip_html(fragment: str) -> str:
    fragment = re.sub(r"<br[^>]*>", "\n", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"</?(p|div|h\d|li)[^>]*>", "\n", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    fragment = html.unescape(fragment)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in fragment.split("\n")]
    return "\n".join(ln for ln in lines if ln)


# Footer / dòng ghi nguồn hay gặp ở ebook
_FOOTER_RE = re.compile(
    r"\(?\s*Tr[íi]ch|NXB|First News|Dịch[\s-]?giả|Nguồn:|Tạo bởi|vnthuquan|nguyenkimvy",
    re.IGNORECASE,
)
# Ký tự có dấu kiểu La-tinh mở rộng (tiếng Việt). Dùng để nhận diện dòng
# tiếng Pháp/ghi chú không dấu lọt ở cuối truyện (vd "Le Loup & La Cigogne").
_VN_ACCENT = re.compile(r"[À-ɏḀ-ỿ]")
_LATIN = re.compile(r"[A-Za-z]")


def _clean_body(text: str) -> str:
    lines = text.split("\n")
    # Cắt từ dòng footer trở đi
    cut = len(lines)
    for i, ln in enumerate(lines):
        if _FOOTER_RE.search(ln):
            cut = i
            break
    lines = lines[:cut]
    # Bỏ các dòng nhiễu ở cuối: không có dấu tiếng Việt nhưng có chữ La-tinh
    while lines:
        last = lines[-1]
        if _LATIN.search(last) and not _VN_ACCENT.search(last):
            lines.pop()
        else:
            break
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Heuristic trích bài học (moral)
# ---------------------------------------------------------------------------

_MORAL_MARKERS = re.compile(
    r"(câu chuyện.{0,20}(ca ngợi|phê phán|khuyên|muốn nói|nói lên|dạy|nhắc|ngụ ý)"
    r"|qua (câu chuyện|đó).{0,30}(khuyên|dạy|nói|ca ngợi|rằng)"
    r"|truyện (này |trên )?(muốn nói|khuyên|dạy|ca ngợi)"
    r"|ngụ ý (?:rằng|là|muốn)|hàm ý"
    r"|bài học (ở đây|rút ra|hôm nay)"
    r"|muốn nói (?:lên )?rằng)",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def _is_dialogue(sentence: str) -> bool:
    return sentence.startswith("-") or bool(re.search(r":\s*-", sentence))


# Từ nối mở đầu câu tự sự (kết truyện) — không phải châm ngôn/bài học
_NARRATIVE_OPENER = re.compile(
    r"^(rồi|thế là|sau đó|sau cùng|cuối cùng|bỗng|liền|từ đó|và rồi|kể từ)\b",
    re.IGNORECASE,
)


def _extract_moral(body: str, *, fable: bool) -> str:
    flat = " ".join(body.split("\n"))
    # 1) Câu mang dấu hiệu bài học rõ ràng (lấy trọn câu, không cắt cụt)
    m = _MORAL_MARKERS.search(flat)
    if m:
        sents = _split_sentences(flat[m.start():])
        if sents:
            return sents[0]
    if not fable:
        return ""
    # 2) Ngụ ngôn văn xuôi (Aesop): câu chốt cuối, nếu không phải lời thoại
    sents = _split_sentences(body)
    if not sents:
        return ""
    last = sents[-1]
    if (
        _is_dialogue(last)
        or _NARRATIVE_OPENER.match(last)
        or len(last) > 220
        or not last.rstrip().endswith((".", "!", "…", "?"))
    ):
        return ""
    return last


# ---------------------------------------------------------------------------
# Parser theo định dạng -> list[dict(topic, body)]  (chưa lọc độ dài)
# ---------------------------------------------------------------------------

def _parse_vnthuquan(raw: str) -> list[dict]:
    m_div = re.search(r'<div[^>]*id="fontchu"[^>]*>(.*?)</div>', raw, re.IGNORECASE | re.DOTALL)
    if not m_div:
        return []
    m_title = re.search(
        r'<p[^>]*class="[^"]*style32[^"]*"[^>]*>\s*<strong>(.*?)</strong>',
        raw, re.IGNORECASE | re.DOTALL,
    )
    first_title = _strip_html(m_title.group(1)) if m_title else ""

    lines = _strip_html(m_div.group(1)).split("\n")
    stories: list[tuple[str, list[str]]] = []
    cur_title, cur_body = first_title, []
    for ln in lines:
        marker = re.match(r"^\*\s*(.+)$", ln)
        if marker:
            if cur_body:
                stories.append((cur_title, cur_body))
            cur_title, cur_body = marker.group(1).strip(), []
            continue
        if ln == cur_title:  # tiêu đề thường lặp lại ở đầu thân truyện
            continue
        cur_body.append(ln)
    if cur_body:
        stories.append((cur_title, cur_body))

    return [{"topic": t, "body": _clean_body("\n".join(b))} for t, b in stories]


def _parse_calibre(raw: str) -> list[dict]:
    body_m = re.search(r"<body[^>]*>(.*)</body>", raw, re.IGNORECASE | re.DOTALL)
    if not body_m:
        return []
    body = body_m.group(1)
    headings = list(re.finditer(r"<h2[^>]*>(.*?)</h2>", body, re.IGNORECASE | re.DOTALL))
    if not headings:
        return []
    out = []
    for i, h in enumerate(headings):
        title = _strip_html(h.group(1))
        start = h.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        paras = re.findall(
            r'<p[^>]*class="[^"]*para[^"]*"[^>]*>(.*?)</p>',
            body[start:end], re.IGNORECASE | re.DOTALL,
        )
        text = _clean_body("\n".join(_strip_html(p) for p in paras))
        out.append({"topic": title, "body": text})
    return out


# ---------------------------------------------------------------------------
# Chuẩn hoá tiêu đề
# ---------------------------------------------------------------------------

_ROMAN = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_SKIP_TOPIC = re.compile(
    r"(lời giới thiệu|giới thiệu|mục lục|lời nói đầu|lời tựa|tiểu sử|người kể chuyện)",
    re.IGNORECASE,
)


def _clean_topic(topic: str) -> str:
    # Bỏ tiền tố số La Mã: "I - EM BÉ RUDY" -> "EM BÉ RUDY"
    return re.sub(r"^[IVXLCDM]+\s*[-.–]\s*", "", topic).strip()


def _resolve_titles(records: list[dict]) -> None:
    """Gộp chương số La Mã ('I', 'II', 'KẾT') về tên truyện cha gần nhất."""
    last_real = ""
    for r in records:
        topic = _clean_topic(r["topic"])
        if _ROMAN.match(topic) or len(topic) < 4 or topic.upper() in {"KẾT", "PHẦN"}:
            r["topic"] = last_real or topic or "(chưa rõ)"
        else:
            r["topic"] = topic
            last_real = topic


def parse_epub(path: Path) -> list[dict]:
    fable = bool(re.search(r"Aesop", path.name, re.IGNORECASE))
    records: list[dict] = []
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for spine in _spine_files(zf):
            if spine not in names:
                continue
            raw = zf.read(spine).decode("utf-8", errors="replace")
            if 'id="fontchu"' in raw:
                records.extend(_parse_vnthuquan(raw))
            elif "<h2" in raw.lower():
                records.extend(_parse_calibre(raw))
    _resolve_titles(records)
    for r in records:
        r["fable"] = fable
        r["source"] = path.name
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasources", help="Thư mục chứa .epub")
    ap.add_argument("--out", default="data/raw/stories.jsonl")
    ap.add_argument("--default-age", default="", help="Giá trị age_range mặc định")
    ap.add_argument("--min-len", type=int, default=120)
    args = ap.parse_args()

    src = Path(args.src)
    epubs = sorted(src.glob("*.epub"))
    if not epubs:
        raise SystemExit(f"Không tìm thấy .epub trong {src}")

    records: list[dict] = []
    seen: set[str] = set()
    per_book: dict[str, int] = {}
    for epub in epubs:
        kept = 0
        for r in parse_epub(epub):
            body = r["body"].strip()
            if len(body) < args.min_len or body in seen:
                continue
            if _SKIP_TOPIC.search(r["topic"]):
                continue
            seen.add(body)
            records.append({
                "story": _nfc(body),
                "topic": _nfc(r["topic"] or "(chưa rõ)"),
                "moral": _nfc(_extract_moral(body, fable=r["fable"])),
                "age_range": args.default_age,
                "source": _nfc(epub.name),
            })
            kept += 1
        per_book[epub.name] = kept

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Đã ghi {len(records)} truyện -> {out}")
    for name, n in per_book.items():
        print(f"  - {name}: {n}")
    with_moral = sum(1 for r in records if r["moral"])
    print(f"Có trích được bài học (moral): {with_moral}/{len(records)}")


if __name__ == "__main__":
    main()
