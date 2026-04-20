import os
import re
import functools

from groq import Groq

from rag.retriever import retrieve_docs
from rag.utils import normalize_text, detect_article_query, classify_intent, QueryIntent

USER_STORE  = "data/vector_store/user_docs"
LEGAL_STORE = "data/vector_store/legal_docs"

GROQ_MODEL   = "llama-3.1-8b-instant"


@functools.lru_cache(maxsize=1)
def load_llm() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Chưa set GROQ_API_KEY trong file .env")
    return Groq(api_key=api_key)


def _call_llm(prompt: str) -> str:
    client = load_llm()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý phân tích hợp đồng và pháp lý chuyên nghiệp. "
                    "Trả lời bằng tiếng Việt, ngắn gọn, chính xác. "
                    "Chỉ dựa vào nội dung được cung cấp, không tự bịa thêm."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def _format_context(docs, max_chars: int = 4000) -> str:
    parts, total = [], 0
    for doc in docs:
        article = doc.metadata.get("article") or ""
        title   = doc.metadata.get("title")   or ""
        label   = f"[{article} – {title}]\n" if article else ""
        block   = label + doc.page_content.strip()
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(block[:remaining] + "…")
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def _find_article_docs(article_num: str, docs: list) -> list:
    target  = f"điều {article_num}"
    matched = [
        d for d in docs
        if str(d.metadata.get("article", "")).lower() == target
    ]
    if not matched:
        matched = [
            d for d in docs
            if re.search(rf"\bĐiều\s*{article_num}\b", d.page_content, re.IGNORECASE)
        ]
    return matched


_PROMPTS = {
    QueryIntent.COMPARISON: """\
Đối chiếu điều khoản hợp đồng với quy định pháp luật bên dưới.
Kết luận rõ: hợp pháp / không hợp pháp / cần xem xét thêm.
Nếu không đủ căn cứ, nói rõ "Không đủ căn cứ để kết luận."

[HỢP ĐỒNG]
{context_user}

[QUY ĐỊNH PHÁP LUẬT]
{context_legal}

[CÂU HỎI]
{query}
""",

    QueryIntent.SUMMARY: """\
Tóm tắt nội dung hợp đồng liên quan đến câu hỏi.
Trình bày rõ ràng, có cấu trúc, dễ hiểu.

[HỢP ĐỒNG]
{context_user}

[CÂU HỎI]
{query}
""",

    QueryIntent.PARTY_INFO: """\
Trích xuất chính xác thông tin được hỏi từ hợp đồng.
Nếu không có, nói "Không tìm thấy thông tin này trong hợp đồng."

[HỢP ĐỒNG]
{context_user}

[CÂU HỎI]
{query}
""",

    QueryIntent.OBLIGATION: """\
Liệt kê đầy đủ nghĩa vụ và quyền lợi liên quan đến câu hỏi.
Phân biệt rõ: nghĩa vụ của Bên A / Bên B nếu có.

[HỢP ĐỒNG]
{context_user}

[QUY ĐỊNH PHÁP LUẬT]
{context_legal}

[CÂU HỎI]
{query}
""",

    QueryIntent.DATE_TERM: """\
Trả lời ngắn gọn về thời hạn, trích dẫn điều khoản cụ thể nếu có.

[HỢP ĐỒNG]
{context_user}

[CÂU HỎI]
{query}
""",

    QueryIntent.PENALTY: """\
Nêu rõ chế tài vi phạm: mức phạt, điều kiện áp dụng, căn cứ pháp lý.

[HỢP ĐỒNG]
{context_user}

[QUY ĐỊNH PHÁP LUẬT]
{context_legal}

[CÂU HỎI]
{query}
""",

    QueryIntent.GENERAL: """\
Trả lời dựa trên nội dung tài liệu bên dưới.
Nếu không tìm thấy, nói: "Không tìm thấy trong tài liệu."

[HỢP ĐỒNG]
{context_user}

[QUY ĐỊNH PHÁP LUẬT]
{context_legal}

[CÂU HỎI]
{query}
""",
}


def answer_question(query: str) -> tuple[str, list, list]:
    all_user_docs  = retrieve_docs(USER_STORE,  query, k=5) if os.path.exists(USER_STORE)  else []
    all_legal_docs = retrieve_docs(LEGAL_STORE, query, k=5) if os.path.exists(LEGAL_STORE) else []

    intent = classify_intent(query)

    # Tra cứu Điều X trực tiếp — không cần LLM phân tích, tránh nhầm lẫn khi điều khoản dài hoặc phức tạp.
    if intent == QueryIntent.ARTICLE_LOOKUP:
        article_num = detect_article_query(query)
        for source, u, l in [
            (all_user_docs,  all_user_docs, []),
            (all_legal_docs, [],            all_legal_docs),
        ]:
            matched = _find_article_docs(article_num, source)
            if matched:
                answer = normalize_text("\n\n".join(d.page_content for d in matched))
                return answer, u, l
        return "Không tìm thấy điều khoản này trong tài liệu.", [], []

    # Các intent cần LLM phân tích
    template = _PROMPTS.get(intent, _PROMPTS[QueryIntent.GENERAL])
    prompt   = template.format(
        context_user  = _format_context(all_user_docs),
        context_legal = _format_context(all_legal_docs),
        query         = query,
    )

    try:
        answer = _call_llm(prompt)
    except Exception as e:
        answer = f"Lỗi khi gọi LLM: {e}"

    return normalize_text(answer), all_user_docs, all_legal_docs