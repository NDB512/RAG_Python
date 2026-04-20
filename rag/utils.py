import re
from enum import Enum


class QueryIntent(str, Enum):
    ARTICLE_LOOKUP  = "article_lookup"   # hỏi Điều X
    COMPARISON      = "comparison"       # so sánh với luật
    SUMMARY         = "summary"          # tóm tắt
    PARTY_INFO      = "party_info"       # thông tin bên ký
    OBLIGATION      = "obligation"       # nghĩa vụ / quyền lợi
    DATE_TERM       = "date_term"        # thời hạn, ngày tháng
    PENALTY         = "penalty"          # phạt vi phạm
    GENERAL         = "general"          # câu hỏi chung


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def detect_article_query(query: str) -> str | None:
    """Trả về số điều nếu query hỏi về Điều X."""
    match = re.search(r"điều\s*(\d+)", query.lower())
    return match.group(1) if match else None


def classify_intent(query: str) -> QueryIntent:
    """Phân loại ý định câu hỏi để chọn prompt phù hợp."""
    q = query.lower()

    if detect_article_query(query):
        return QueryIntent.ARTICLE_LOOKUP

    compare_kws = [
        "hợp pháp không", "trái luật", "đúng luật", "vi phạm",
        "đối chiếu", "so sánh", "có phù hợp", "có đúng quy định",
        "vượt trần", "vượt mức", "theo luật",
    ]
    if any(k in q for k in compare_kws):
        return QueryIntent.COMPARISON

    summary_kws = ["tóm tắt", "nội dung chính", "tổng quan", "giải thích", "nói về gì"]
    if any(k in q for k in summary_kws):
        return QueryIntent.SUMMARY

    party_kws = ["bên a", "bên b", "bên vay", "bên cho vay", "ai ký", "ai là", "thông tin"]
    if any(k in q for k in party_kws):
        return QueryIntent.PARTY_INFO

    obligation_kws = ["nghĩa vụ", "trách nhiệm", "quyền lợi", "quyền của", "phải làm"]
    if any(k in q for k in obligation_kws):
        return QueryIntent.OBLIGATION

    date_kws = ["thời hạn", "bao lâu", "ngày nào", "khi nào", "hết hạn", "hiệu lực"]
    if any(k in q for k in date_kws):
        return QueryIntent.DATE_TERM

    penalty_kws = ["phạt", "bồi thường", "vi phạm", "chế tài", "xử lý"]
    if any(k in q for k in penalty_kws):
        return QueryIntent.PENALTY

    return QueryIntent.GENERAL


# Các hàm tiện ích khác
def is_compare_query(query: str) -> bool:
    return classify_intent(query) == QueryIntent.COMPARISON

def is_summary_query(query: str) -> bool:
    return classify_intent(query) == QueryIntent.SUMMARY

def is_article_query(query: str) -> bool:
    return classify_intent(query) == QueryIntent.ARTICLE_LOOKUP