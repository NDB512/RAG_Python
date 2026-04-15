import re

def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()

def detect_article_query(query: str):
    """
    Trả về số điều nếu query kiểu: 'điều 3 là gì'
    """
    match = re.search(r"điều\s*(\d+)", query.lower())
    return match.group(1) if match else None

def is_compare_query(query: str):
    keywords = [
        "hợp pháp không",
        "trái luật không",
        "đúng luật không",
        "có vi phạm không",
        "đối chiếu",
        "so sánh"
    ]
    q = query.lower()
    return any(k in q for k in keywords)

def is_summary_query(query: str):
    keywords = [
        "tóm tắt",
        "nội dung chính",
        "tổng quan",
        "giải thích"
    ]
    q = query.lower()
    return any(k in q for k in keywords)