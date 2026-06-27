"""Danh sách từ cấm + mẫu intent ngoài phạm vi. Mở rộng khi thực thi."""

BANNED_WORDS = frozenset({
    "đụ", "địt", "lồn", "cặc", "đm", "đmm", "vcl", "đụ má",
    # ... bổ sung từ tục/bậy khác trong giai đoạn thực thi
})

OUT_OF_SCOPE_PATTERNS = (
    r"bỏ qua (hướng dẫn|chỉ thị)",
    r"viết (mã|code|phần mềm)",
    r"mã độc|virus|hack",
    r"(làm|viết) (bài tập|luận văn|email)",
    r"đóng vai(?! .*truyện)",  # yêu cầu role-play ngoài kể truyện
)
