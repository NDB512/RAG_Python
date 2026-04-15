import os
import re

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline

from rag.retriever import retrieve_docs
from rag.utils import normalize_text, detect_article_query, is_compare_query, is_summary_query

USER_STORE = "data/vector_store/user_docs"
LEGAL_STORE = "data/vector_store/legal_docs"


def load_llm():
    model_name = "distilgpt2"  # đổi TinyLlama nếu máy khỏe

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=180,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=pipe)


def extract_article_from_docs(article_num: str, docs):
    target_article = f"Điều {article_num}".lower()

    # Sắp xếp theo page trước
    docs_sorted = sorted(docs, key=lambda d: d.metadata.get("page", 0))

    collected = []
    capture = False

    for doc in docs_sorted:
        article_meta = str(doc.metadata.get("article", "")).strip().lower()
        text = doc.page_content.strip()

        # Nếu đúng Điều cần tìm -> bắt đầu thu
        if article_meta == target_article:
            collected.append(text)
            capture = True
            continue

        # Nếu đang thu Điều 3 mà doc hiện tại KHÔNG có article
        # thì coi như phần tiếp nối cùng điều
        if capture and not article_meta:
            collected.append(text)
            continue

        # Nếu đang thu mà gặp điều khác -> dừng
        if capture and article_meta and article_meta != target_article:
            break

    if collected:
        return normalize_text("\n".join(collected))

    # fallback regex từng doc riêng
    for doc in docs_sorted:
        text = doc.page_content
        pattern = rf"(Điều\s*{article_num}\s*[:.]?.*)"
        found = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if found:
            return normalize_text(found.group(1))

    return None


def format_context(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def answer_question(query: str):
    user_docs = []
    legal_docs = []

    if os.path.exists(USER_STORE):
        user_docs = retrieve_docs(USER_STORE, query, k=5)

    if os.path.exists(LEGAL_STORE):
        legal_docs = retrieve_docs(LEGAL_STORE, query, k=5)

    # 1) Nếu hỏi Điều X -> ưu tiên user_docs, fallback legal_docs
    article_num = detect_article_query(query)
    if article_num:
        direct_user = extract_article_from_docs(article_num, user_docs)
        if direct_user:
            return direct_user, user_docs, legal_docs

        direct_legal = extract_article_from_docs(article_num, legal_docs)
        if direct_legal:
            return direct_legal, user_docs, legal_docs

    # 2) Nếu là câu hỏi so sánh / hợp pháp
    if is_compare_query(query):
        context_user = format_context(user_docs)
        context_legal = format_context(legal_docs)

        llm = load_llm()
        prompt = f"""
Bạn là trợ lý phân tích tài liệu pháp lý.

Dựa trên tài liệu người dùng và căn cứ pháp lý dưới đây, hãy trả lời NGẮN GỌN, RÕ RÀNG.
Không được bịa nếu không có căn cứ.

[TÀI LIỆU NGƯỜI DÙNG]
{context_user}

[CĂN CỨ PHÁP LÝ]
{context_legal}

[CÂU HỎI]
{query}

[TRẢ LỜI]
"""
        answer = llm.invoke(prompt).strip()
        return normalize_text(answer), user_docs, legal_docs

    # 3) Nếu là câu hỏi tóm tắt / giải thích
    if is_summary_query(query):
        context_user = format_context(user_docs)
        context_legal = format_context(legal_docs)

        llm = load_llm()
        prompt = f"""
Bạn là trợ lý tóm tắt tài liệu.

Dựa trên ngữ cảnh sau, hãy trả lời bằng tiếng Việt, ngắn gọn, dễ hiểu.

[TÀI LIỆU NGƯỜI DÙNG]
{context_user}

[CĂN CỨ PHÁP LÝ]
{context_legal}

[CÂU HỎI]
{query}

[TRẢ LỜI]
"""
        answer = llm.invoke(prompt).strip()
        return normalize_text(answer), user_docs, legal_docs

    # 4) Mặc định: retrieve + answer từ cả 2 nguồn
    context_user = format_context(user_docs)
    context_legal = format_context(legal_docs)

    llm = load_llm()
    prompt = f"""
Bạn là trợ lý hỏi đáp tài liệu.

Chỉ trả lời dựa trên nội dung dưới đây.
Nếu không tìm thấy thì nói:
"Không tìm thấy trong tài liệu."

[TÀI LIỆU NGƯỜI DÙNG]
{context_user}

[CĂN CỨ PHÁP LÝ]
{context_legal}

[CÂU HỎI]
{query}

[TRẢ LỜI]
"""
    answer = llm.invoke(prompt).strip()
    return normalize_text(answer), user_docs, legal_docs