import re
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================== PATTERNS MỞ RỘNG ==================
ARTICLE_PATTERN = re.compile(r"(?m)^(Điều\s+\d+[^\n]*)", re.IGNORECASE)
CHAPTER_PATTERN = re.compile(r"(?m)^(CHƯƠNG\s+[IVX\d]+|PHẦN\s+[IVX\d]+|MỤC\s+[IVX\d]+)", re.IGNORECASE)
ROMAN_PATTERN   = re.compile(r"(?m)^([IVX]{1,5})\.\s+", re.IGNORECASE)

HEADER_KEYWORDS = [
    "cộng hòa xã hội", "độc lập - tự do", "hạnh phúc",
    "hợp đồng", "biên bản", "quy chế", "quy định", "nội bộ", "điều lệ",
    "căn cứ", "bên a", "bên b", "bên cho thuê", "bên thuê", "bên cung cấp", "bên nhận",
    "đại diện", "hôm nay, ngày", "chúng tôi gồm", "sau khi thỏa thuận",
    "phạm vi điều chỉnh", "đối tượng áp dụng", "quy định chung", "hiệu lực thi hành"
]

def _clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _is_header_block(text: str) -> bool:
    if len(text) < 50:
        return False
    low = text.lower()
    hits = sum(1 for kw in HEADER_KEYWORDS if kw in low)
    return hits >= 2   # Có thể giảm xuống 1 nếu vẫn miss header

def _extract_title(chunk: str) -> tuple[str | None, str | None]:
    first_line = chunk.split("\n")[0].strip()
    
    # Bắt Điều
    m = re.match(r"^(Điều\s+\d+[^\n]*)", first_line, re.IGNORECASE)
    if m:
        title_part = first_line[len(m.group(1)):].strip(" :.")
        return m.group(1).strip(), title_part or None
    
    # Bắt Chương, Phần, Mục
    m = re.match(r"^(CHƯƠNG|PHẦN|MỤC)\s+[IVX\d]+[^\n]*", first_line, re.IGNORECASE)
    if m:
        return None, m.group(0).strip()
    
    return None, None

def _split_long_article(chunk: str, article: str | None, title: str | None) -> list[dict]:
    sub_pattern = re.compile(r"(?m)^(\d+\.\s+|\d+\.\d+\.?\s+|[a-z]\)\s+)", re.IGNORECASE)
    matches = list(sub_pattern.finditer(chunk))

    if len(matches) < 2:
        return [{"text": chunk, "chunk_type": "article", "article": article, "title": title}]

    sub_chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chunk)
        sub = chunk[start:end].strip()
        if len(sub) > 25:
            sub_chunks.append({
                "text": sub,
                "chunk_type": "clause",
                "article": article,
                "title": title
            })
    return sub_chunks or [{"text": chunk, "chunk_type": "article", "article": article, "title": title}]

def split_legal_text(text: str) -> list[dict]:
    text = _clean_text(text)
    if not text:
        return []

    # Mở rộng pattern để bắt I. II. III. và 1. 2. 
    article_matches = list(ARTICLE_PATTERN.finditer(text))
    chapter_matches = list(CHAPTER_PATTERN.finditer(text))
    roman_matches   = list(ROMAN_PATTERN.finditer(text))

    if article_matches:
        primary_matches = article_matches
        chunk_type = "article"
    elif chapter_matches:
        primary_matches = chapter_matches
        chunk_type = "chapter"
    elif roman_matches:
        primary_matches = roman_matches
        chunk_type = "roman_section"
    else:
        primary_matches = []
        chunk_type = "paragraph"

    chunks = []

    # ================== HEADER ==================
    header_end = primary_matches[0].start() if primary_matches else min(4500, len(text)//2)
    header_text = text[:header_end].strip()

    if header_text and _is_header_block(header_text):
        chunks.append({
            "text": header_text,
            "chunk_type": "header",
            "article": None,
            "title": "Thông tin chung / Căn cứ / Các bên"
        })

    # ================== Tách theo cấu trúc ==================
    if primary_matches and len(primary_matches) >= 1:   # giảm xuống >=1 để linh hoạt hơn
        for i, match in enumerate(primary_matches):
            start = match.start()
            end = primary_matches[i + 1].start() if i + 1 < len(primary_matches) else len(text)
            chunk = text[start:end].strip()

            if len(chunk) < 40:
                continue

            article, title = _extract_title(chunk)

            if len(chunk) > 1800:
                sub_chunks = _split_long_article(chunk, article, title)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "text": chunk,
                    "chunk_type": chunk_type,
                    "article": article,
                    "title": title
                })
    else:
        # Fallback Recursive (rất quan trọng cho hợp đồng này)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1100,
            chunk_overlap=120,
            separators=[
                "\n\n", 
                "\nĐiều ", "\nCHƯƠNG ", "\nPHẦN ", "\nMỤC ",
                "\nI. ", "\nII. ", "\nIII. ", "\nIV. ", "\nV. ", "\nVI. ",
                "\n1. ", "\n2. ", "\n3. ", "\n4. ", "\na) ", "\nb) ", "\nc) "
            ],
            keep_separator=True
        )
        
        raw_chunks = splitter.split_text(text)
        current_article = None
        current_title = None

        for raw in raw_chunks:
            raw = raw.strip()
            if len(raw) < 40:
                continue

            art_match = ARTICLE_PATTERN.search(raw)
            if art_match:
                current_article = art_match.group(1).strip()
                current_title = raw.split("\n")[0].strip()

            chunks.append({
                "text": raw,
                "chunk_type": "recursive" if not current_article else "article",
                "article": current_article,
                "title": current_title
            })

    return chunks


def build_documents_from_pages(docs, doc_type: str = "user") -> list[Document]:
    if not docs:
        return []

    full_text = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())
    base_meta = docs[0].metadata.copy() if docs else {}
    base_meta["file_name"] = os.path.basename(base_meta.get("source", "unknown.pdf"))

    chunks = split_legal_text(full_text)

    final_docs = []
    for i, chunk in enumerate(chunks):
        meta = {
            **base_meta,
            "doc_type": doc_type,
            "chunk_index": i,
            "chunk_type": chunk["chunk_type"],
            "article": chunk["article"],
            "title": chunk["title"],
        }
        final_docs.append(Document(page_content=chunk["text"], metadata=meta))

    return final_docs