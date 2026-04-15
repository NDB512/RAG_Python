import re
from langchain_core.documents import Document


def split_legal_text(text: str):
    """
    Tách văn bản pháp lý theo đầu dòng Điều X.
    Chỉ tách khi 'Điều X' nằm ở đầu dòng / đầu đoạn.
    Không tách nếu 'Điều X' xuất hiện giữa câu.
    """
    text = text.replace("\r", "\n")

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text).strip()

    # Regex: chỉ match Điều ở đầu dòng
    article_pattern = r"(?m)^(Điều\s+\d+\s*[:.]?.*)"

    matches = list(re.finditer(article_pattern, text, flags=re.IGNORECASE))

    chunks = []

    if matches:
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[start:end].strip()

            if len(chunk) > 30:
                chunks.append(chunk)
    else:
        # fallback: chia theo đoạn
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = [p for p in paras if len(p) > 30]

    return chunks


def extract_article_title(chunk: str):
    """
    Lấy:
    - article: Điều 3
    - title: Lãi suất cho mượn và hình thức trả nợ
    """
    first_line = chunk.split("\n")[0].strip()

    # Match kiểu:
    # Điều 3: Lãi suất cho mượn...
    # Điều 3. Lãi suất...
    # Điều 3 Lãi suất...
    m = re.match(
        r"^(Điều\s+\d+)\s*[:.]?\s*(.*)$",
        first_line,
        flags=re.IGNORECASE
    )

    if m:
        article = m.group(1).strip()
        title = m.group(2).strip() if m.group(2) else ""
        return article, title

    return None, None


def build_documents_from_pages(docs, doc_type="user"):
    """
    docs: output từ PyPDFLoader.load()
    Tách từng page thành legal chunks sạch hơn.
    """
    final_docs = []

    for doc in docs:
        page_text = doc.page_content
        page_meta = doc.metadata.copy()

        chunks = split_legal_text(page_text)

        for chunk in chunks:
            article, title = extract_article_title(chunk)

            meta = {
                **page_meta,
                "doc_type": doc_type,
                "article": article,
                "title": title,
            }

            final_docs.append(
                Document(
                    page_content=chunk.strip(),
                    metadata=meta
                )
            )

    return final_docs