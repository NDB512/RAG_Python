import os
import re
import functools

from groq import Groq

from rag.retriever import retrieve_docs
from rag.utils import normalize_text, detect_article_query, classify_intent, QueryIntent

USER_STORE  = "data/vector_store/user_docs"
LEGAL_STORE = "data/vector_store/legal_docs"

GROQ_MODEL = "llama-3.1-8b-instant"


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


def _format_context(docs, max_chars: int = 6000) -> str:
    """Tăng max_chars và deduplicate trước khi format."""
    # Deduplicate theo page_content
    seen, unique = set(), []
    for doc in docs:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    parts, total = [], 0
    for doc in unique:
        article = doc.metadata.get("article") or ""
        title   = doc.metadata.get("title")   or ""

        if article and title:
            label = f"[{article.upper()} – {title}]\n"
        elif article:
            label = f"[{article.upper()}]\n"
        elif title:
            label = f"[{title}]\n"
        else:
            label = ""

        block = label + doc.page_content.strip()
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(block[:remaining] + "…")
            break
        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts) if parts else "Không có dữ liệu."


def _find_article_docs(article_num: str, docs: list) -> list:
    """
    Tìm chunk theo số điều. Nếu điều quá ngắn, gộp thêm chunk liền kề.
    """
    target = f"điều {article_num}"

    # 1. Exact match metadata
    matched = [
        d for d in docs
        if str(d.metadata.get("article", "")).lower().strip() == target
    ]

    # 2. Regex trong content
    if not matched:
        pattern = re.compile(rf"\bĐiều\s*{re.escape(article_num)}\b", re.IGNORECASE)
        matched = [d for d in docs if pattern.search(d.page_content)]

    # 3. Prefix match
    if not matched:
        matched = [
            d for d in docs
            if str(d.metadata.get("article", "")).lower().startswith(target)
        ]

    if not matched:
        return []

    # Nếu chunk quá ngắn: tìm trong raw text xung quanh
    result = []
    seen = set()
    for doc in matched:
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            result.append(doc)

        # Nếu nội dung < 100 ký tự, tìm thêm từ chunk chứa Điều này trong content
        if len(doc.page_content.strip()) < 150:
            art_pattern = re.compile(
                rf"(?m)^(Điều\s*{re.escape(article_num)}\b.+?)(?=^Điều\s*\d+\b|$\Z)",
                re.IGNORECASE | re.DOTALL | re.MULTILINE
            )
            for d in docs:
                m = art_pattern.search(d.page_content)
                if m and len(m.group(0)) > len(doc.page_content):
                    k2 = d.page_content[:80]
                    if k2 not in seen:
                        seen.add(k2)
                        result.append(d)

    return result

#  PROMPT TEMPLATES 

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

    QueryIntent.ARTICLE_LOOKUP: """\
Trích dẫn và giải thích nội dung điều khoản được hỏi từ tài liệu.
Nếu có nhiều khoản con, liệt kê rõ từng khoản.

[HỢP ĐỒNG]
{context_user}

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

def _get_all_docs_from_store(store_path: str) -> list:
    """Load toàn bộ documents từ FAISS store (không dùng similarity search)."""
    from rag.retriever import load_store
    index_file = os.path.join(store_path, "index.faiss")
    if not os.path.exists(index_file):
        return []
    try:
        store = load_store(store_path)
        # Lấy tất cả docs từ docstore
        all_docs = list(store.docstore._dict.values())
        return all_docs
    except Exception as e:
        print(f"[pipeline] Lỗi load all docs: {e}")
        return []


def answer_question(query: str) -> tuple[str, list, list]:
    intent = classify_intent(query)
    print(f"[pipeline] intent={intent}, query={query!r}")

    # ── ARTICLE_LOOKUP: scan toàn bộ store, không dùng similarity ───────────
    if intent == QueryIntent.ARTICLE_LOOKUP:
        article_num = detect_article_query(query)
        print(f"[pipeline] article_num={article_num!r}")

        if article_num:
            all_user_all  = _get_all_docs_from_store(USER_STORE)
            all_legal_all = _get_all_docs_from_store(LEGAL_STORE)

            print(f"[pipeline] total docs in store: user={len(all_user_all)}, legal={len(all_legal_all)}")
            # Debug: in toàn bộ article metadata
            for d in all_user_all:
                print(f"  article={d.metadata.get('article')!r}")

            matched_user  = _find_article_docs(article_num, all_user_all)
            matched_legal = _find_article_docs(article_num, all_legal_all)
            print(f"[pipeline] matched: user={len(matched_user)}, legal={len(matched_legal)}")

            if matched_user or matched_legal:
                all_matched = matched_user + matched_legal

                # Extract chính xác đoạn từ "Điều X" đến "Điều X+1"
                extracted_parts = []
                art_pattern = re.compile(
                    rf"(?m)^(Điều\s*{re.escape(article_num)}\b.+?)(?=^Điều\s*\d+\b|$\Z)",
                    re.IGNORECASE | re.DOTALL | re.MULTILINE
                )
                for doc in all_matched:
                    m = art_pattern.search(doc.page_content)
                    if m:
                        extracted_parts.append(m.group(1).strip())
                    else:
                        extracted_parts.append(doc.page_content.strip())

                # Deduplicate
                seen_text, unique_parts = set(), []
                for part in extracted_parts:
                    key = part[:100]
                    if key not in seen_text:
                        seen_text.add(key)
                        unique_parts.append(part)

                raw = "\n\n".join(unique_parts)
                return normalize_text(raw), matched_user, matched_legal

        # Không tìm thấy qua metadata → thử full-text search trong store
        all_user_all = _get_all_docs_from_store(USER_STORE)
        if article_num:
            pattern = re.compile(rf"\bĐiều\s*{re.escape(article_num)}\b", re.IGNORECASE)
            matched = [d for d in all_user_all if pattern.search(d.page_content)]
            if matched:
                raw = "\n\n".join(d.page_content for d in matched)
                return normalize_text(raw), matched, []

        return "Không tìm thấy điều khoản này trong tài liệu.", [], []

    all_user_docs  = retrieve_docs(USER_STORE,  query, k=6) if os.path.exists(USER_STORE)  else []
    all_legal_docs = retrieve_docs(LEGAL_STORE, query, k=5) if os.path.exists(LEGAL_STORE) else []

    context_user  = _format_context(all_user_docs)
    context_legal = _format_context(all_legal_docs)

    template = _PROMPTS.get(intent, _PROMPTS[QueryIntent.GENERAL])
    try:
        prompt = template.format(
            context_user=context_user,
            context_legal=context_legal,
            query=query,
        )
    except KeyError:
        prompt = template.format(
            context_user=context_user,
            query=query,
        )

    print(f"[pipeline] prompt_len={len(prompt)}")
    try:
        answer = _call_llm(prompt)
    except Exception as e:
        answer = f"Lỗi khi gọi LLM: {e}"

    return normalize_text(answer), all_user_docs, all_legal_docs