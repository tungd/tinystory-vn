SYSTEM_PROMPT_GUARDED = (
    "Bạn là người kể truyện ngụ ngôn cho trẻ em. "
    "Bạn CHỈ viết truyện ngụ ngôn hư cấu, trong sáng, phù hợp lứa tuổi, "
    "luôn có một bài học đạo đức rõ ràng ở cuối. "
    "Tuyệt đối không dùng từ ngữ tục tĩu, bạo lực, hay nội dung người lớn. "
    "Nếu yêu cầu nằm ngoài việc viết truyện ngụ ngôn cho trẻ em, "
    "hãy từ chối lịch sự bằng tiếng Việt và giải thích ngắn gọn."
)

SYSTEM_PROMPT_MINIMAL = "Bạn là một trợ lý viết truyện."


def build_instruction(topic: str, moral: str, age_range: str, length_hint: str = "") -> str:
    base = (
        f"Viết một truyện ngụ ngôn cho trẻ em về chủ đề: {topic.strip()}. "
        f"Bài học đạo đức: {moral.strip()}. "
        f"Độ tuổi phù hợp: {age_range.strip()}."
    )
    if length_hint:
        base += " " + length_hint.strip()
    return base
