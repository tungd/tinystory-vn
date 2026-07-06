"""English wordlist: banned words + out-of-scope patterns for fable generation."""

BANNED_WORDS_EN = frozenset({
    "fuck", "fucking", "shit", "bitch", "bastard", "asshole", "dick", "cunt",
    # bổ sung khi thực thi
})

OUT_OF_SCOPE_PATTERNS_EN = (
    r"ignore (the )?(previous )?(instructions|prompt)",
    r"write (me )?(some )?(malware|code|a program|an essay|an email)",
    r"(hack|exploit|virus)",
    r"(sexual|porn|explicit)",
)
