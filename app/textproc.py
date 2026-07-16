"""Text post-processing for generated stories."""
import re

# Kết câu = dấu . ! ? có thể theo sau bởi một dấu đóng ngoặc kép/nháy.
# Tìm vị trí kết câu HOÀN CHỈNH cuối cùng để cắt phần đuôi dở khi truyện bị
# cắt cứng (done_reason == "length"), tránh cụt giữa từ/câu.
_TERMINATOR = re.compile(r'[.!?]["”’\']?')


def trim_to_last_sentence(text: str) -> str:
    """Cắt text về sau dấu kết câu hoàn chỉnh cuối cùng.

    - Nếu text đã kết bằng dấu câu hoàn chỉnh -> giữ nguyên.
    - Nếu bị cụt giữa câu -> bỏ phần sau dấu kết câu cuối.
    - Nếu không tìm thấy dấu kết câu nào -> giữ nguyên (không bịa).
    """
    if not text:
        return text
    matches = list(_TERMINATOR.finditer(text))
    if not matches:
        return text
    end = matches[-1].end()
    if end >= len(text.rstrip()):
        return text          # đã kết trọn câu -> giữ nguyên
    return text[:end]
