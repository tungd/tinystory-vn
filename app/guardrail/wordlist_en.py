"""English wordlist: banned words + out-of-scope patterns for fable generation."""

BANNED_WORDS_EN = frozenset({
    "fuck", "fucking", "shit", "bitch", "bastard", "asshole", "dick", "cunt",
    # bổ sung khi thực thi
})

OUT_OF_SCOPE_PATTERNS_EN = (
    # Prompt injection / off-task requests
    r"ignore (the )?(previous )?(instructions|prompt)",
    r"write (me )?(some )?(malware|code|a program|an essay|an email)",
    r"(hack|exploit|virus)",
    # Sexual / explicit
    r"(sexual|porn|explicit|erotic|erotica|nude|nudity)",
    # Age-restricted / adult / mature content (not for a children's fable).
    # Note: bare "adult" is allowed (e.g. "an adult lion"); only age markers
    # and adult/mature-*content* phrasing are blocked to avoid false positives.
    r"\b(18|21)\s*\+",                       # 18+, 21+
    r"\bnsfw\b",
    r"\b[xr]-?\s*rated\b",                   # x-rated, r rated
    r"\bfor\s+adults?\s+only\b",
    r"\badults?[- ]only\b",
    r"\bage[- ]restricted\b",
    r"\b(adult|mature|grown[- ]?up)\s+(content|theme|themes|material|humou?r|joke|jokes|audience|audiences|reader|readers|lesson|lessons)\b",
    # Graphic violence / gore
    r"\b(gore|gory|graphic\s+violence)\b",
)
