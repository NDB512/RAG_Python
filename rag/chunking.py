import re
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ARTICLE_PATTERN chỉ match khi "Điều X" đứng đầu dòng VÀ
# KHÔNG theo sau bởi từ nối (của, trong, theo, này, và, hoặc...)
ARTICLE_PATTERN = re.compile(
    r"(?m)^[\s]*(Điều\s+\d+[A-Za-z]?[\.\s:])",
    re.IGNORECASE | re.MULTILINE
)

# Pattern để nhận diện "Điều X" giả (trong giữa câu)
_INLINE_DIEU_PATTERN = re.compile(
    r"\bĐiều\s+\d+[A-Za-z]?\s+(?:của|trong|theo|này|và|hoặc|với|nêu|nói|quy định)",
    re.IGNORECASE
)

CHAPTER_PATTERN = re.compile(
    r"(?m)^(CHƯƠNG\s+[IVX\d]+|PHẦN\s+[IVX\d]+|MỤC\s+[IVX\d]+)", re.IGNORECASE
)
ROMAN_PATTERN = re.compile(r"(?m)^([IVX]{1,5})\.\s+[^\n]{3,}", re.IGNORECASE)

HEADER_KEYWORDS = [
    "cộng hòa xã hội", "độc lập", "hạnh phúc",
    "hợp đồng", "biên bản", "quy chế", "quy định", "nội bộ", "điều lệ",
    "căn cứ", "bên a", "bên b", "bên cho thuê", "bên thuê",
    "bên cung cấp", "bên nhận", "bên vay", "bên cho vay",
    "đại diện", "hôm nay, ngày", "chúng tôi gồm", "sau khi thỏa thuận",
    "phạm vi điều chỉnh", "đối tượng áp dụng", "quy định chung",
]

def detect_contract_type(filename: str) -> str:
    """
    Nhận diện loại hợp đồng dựa trên tên file.
    Giúp hệ thống xử lý thông minh hơn với nhiều loại hợp đồng khác nhau.
    """
    if not filename:
        return "hop_dong_khac"
    
    name = filename.lower().replace("_", " ").replace("-", " ")
    
    # Giấy mượn tiền / vay nợ
    if any(keyword in name for keyword in [
        "muon tien", "giay muon", "vay tien", "lãi suất", "tra no", "cho muon"
    ]):
        return "giay_muon_tien"
    
    # Hợp đồng thuê đất
    if any(keyword in name for keyword in [
        "thue dat", "hop dong thue", "cho thue dat", "thue dat"
    ]):
        return "hop_dong_thue_dat"
    
    # Hợp đồng mua bán
    if any(keyword in name for keyword in [
        "mua ban", "chuyen nhuong", "ban dat", "hop dong mua", "chuyen nhuong"
    ]):
        return "hop_dong_mua_ban"
    
    # Hợp đồng lao động
    if any(keyword in name for keyword in [
        "lao dong", "hop dong lao", "nhan cong", "tuyen dung"
    ]):
        return "hop_dong_lao_dong"
    
    # Có thể thêm nhiều loại khác sau này
    return "hop_dong_khac"

def _fix_paragraph_breaks(text: str) -> str:
    """Cải tiến nối nội dung giữa trang và format"""
    # Nối điều khoản bị cắt trang
    text = re.sub(r"(?<=[^\n])(\s)(Điều\s+\d+[A-Za-z]?[\.\s:])", r"\n\2", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=[^\n])\s+(Điều\s+\d+[A-Za-z]?[\.\s:])", r"\n\1", text, flags=re.IGNORECASE)
    
    # Nối dấu gạch đầu dòng bị xuống dòng
    text = re.sub(r"([^\.\:\;\?\!\n])\n\s*([\-•–])\s*", r"\1 \2 ", text)
    
    # Thêm newline trước số khoản và chữ cái
    text = re.sub(r"(?<=[^\n])\s+(\d+\.\s)", r"\n\1", text)
    text = re.sub(r"(?<=[^\n])\s+([a-zđ]\)\s)", r"\n\1", text)
    
    return text

def _clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = _fix_paragraph_breaks(text)

    # Nối nội dung điều khoản bị cắt giữa trang
    text = re.sub(
        r"([^\.\:\;\?\!\n])\n\s*([\-•–])\s*", 
        r"\1 \2 ", text, flags=re.IGNORECASE
    )

    # Nối nếu trang sau bắt đầu bằng dấu gạch đầu dòng mà không có "Điều X"
    text = re.sub(
        r"(Điều\s+\d+[^\n]+)\n\s*(?=\-|\d+\.)", 
        r"\1\n", text, flags=re.IGNORECASE
    )

    # Trường hợp: dòng trước chưa kết thúc (không có dấu . : ; ?)
    # và dòng tiếp theo bắt đầu bằng "Điều X của/trong/theo/này..."
    # Thì đây là tham chiếu nội tuyến, không phải đầu điều khoản mới
    text = re.sub(
        r"([^\.\:\;\?\!\n])\n(Điều\s+\d+[A-Za-z]?\s+(?:của|trong|theo|này|và|hoặc|với|nêu|được\s+quy\s+định))",
        r"\1 \2",
        text,
        flags=re.IGNORECASE,
    )

    # Nối dòng bị wrap giữa chừng: dòng trên chưa kết thúc câu, dòng dưới bắt đầu bằng chữ thường hoặc chữ số tiếp nối
    # (tránh nối khi dòng dưới là đầu điều khoản thực sự)
    text = re.sub(
        r"([a-zđàáâãèéêìíòóôõùúýăắặẳẵẻẽếềệổỗộờớợủữựẫẩảạ,])\n"
        r"(?!Điều\s+\d|CHƯƠNG|PHẦN|MỤC|[IVX]{1,5}\.\s)"
        r"([a-zđàáâãèéêìíòóôõùúýăắặẳẵẻẽếềệổỗộờớợủữựẫẩảạ])",
        r"\1 \2",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _is_header_block(text: str) -> bool:
    """Nhận diện phần đầu hợp đồng/văn bản pháp lý."""
    if len(text) < 30:
        return False
    low = text.lower()
    hits = sum(1 for kw in HEADER_KEYWORDS if kw in low)
    return hits >= 1  # Giảm xuống 1 để bắt được nhiều trường hợp hơn

def _normalize_article_key(raw: str) -> str:
    """
    Chuẩn hóa metadata article về dạng 'Điều N' để tra cứu nhất quán.
    'Điều 3. Việc sử dụng đất...' -> 'điều 3'
    """
    m = re.match(r"(Điều\s+\d+[A-Za-z]?)", raw.strip(), re.IGNORECASE)
    if m:
        # Chuẩn hóa khoảng trắng, lowercase
        return re.sub(r"\s+", " ", m.group(1).strip()).lower()
    return raw.strip().lower()

def _extract_title(chunk: str) -> tuple[str | None, str | None]:
    """Trích xuất article key và title từ chunk text."""
    first_line = chunk.split("\n")[0].strip()

    # Điều X
    m = re.match(r"^(Điều\s+\d+[A-Za-z]?)([\.\s:]+)(.*)?$", first_line, re.IGNORECASE)
    if m:
        article_key = _normalize_article_key(m.group(1))
        title = m.group(3).strip() if m.group(3) else None
        return article_key, title or first_line

    # Chương, Phần, Mục
    m = re.match(r"^(CHƯƠNG|PHẦN|MỤC)\s+[IVX\d]+[^\n]*", first_line, re.IGNORECASE)
    if m:
        return None, m.group(0).strip()

    return None, None

def _is_real_article_match(match, text: str) -> bool:
    """
    Kiểm tra xem match có phải đầu điều khoản thực sự không,
    hay chỉ là tham chiếu nội tuyến bị xuống dòng.
    """
    start = match.start()

    # Phải đứng đầu dòng (ký tự trước là \n hoặc start of string)
    if start > 0 and text[start - 1] != "\n":
        return False

    matched_text = match.group(0)

    # Nếu sau "Điều X" là từ nối → tham chiếu nội tuyến, không phải điều khoản
    after_num = re.match(
        r"Điều\s+\d+[A-Za-z]?\s*([\.\s])\s*(.{0,40})",
        matched_text,
        re.IGNORECASE,
    )
    if after_num:
        separator = after_num.group(1).strip()
        rest = after_num.group(2).strip().lower()
        # Nếu không có dấu chấm sau số điều và bắt đầu bằng từ nối → giả
        if not separator and re.match(
            r"^(của|trong|theo|này|và|hoặc|với|nêu|được)", rest
        ):
            return False

    return True

def _split_long_article(chunk: str, article: str | None, title: str | None) -> list[dict]:
    """Chia điều khoản dài thành các clause nhỏ hơn."""
    sub_pattern = re.compile(
        r"(?m)^(\d+\.\s+|\d+\.\d+\.?\s+|[a-zđ]\)\s+)", re.IGNORECASE
    )
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
                "title": title,
            })

    return sub_chunks or [{"text": chunk, "chunk_type": "article", "article": article, "title": title}]

# Hàm chính tách văn bản pháp lý thành các chunk có cấu trúc
def split_legal_text(text: str) -> list[dict]:
    """
    Tách văn bản pháp lý thành các chunk có cấu trúc.
    Ưu tiên: Điều X > Chương/Phần > Roman > Fallback recursive
    """
    text = _clean_text(text)
    if not text:
        return []

    # Lọc chỉ giữ các match thực sự là đầu điều khoản
    all_article_matches = list(ARTICLE_PATTERN.finditer(text))
    article_matches = [
        m for m in all_article_matches
        if _is_real_article_match(m, text)
    ]

    chapter_matches = list(CHAPTER_PATTERN.finditer(text))
    roman_matches = list(ROMAN_PATTERN.finditer(text))

    if article_matches:
        primary_matches = article_matches
        chunk_type_default = "article"
    elif chapter_matches:
        primary_matches = chapter_matches
        chunk_type_default = "chapter"
    elif roman_matches:
        primary_matches = roman_matches
        chunk_type_default = "roman_section"
    else:
        primary_matches = []
        chunk_type_default = "paragraph"

    chunks = []

    # HEADER 
    header_end = primary_matches[0].start() if primary_matches else min(4500, len(text) // 2)
    header_text = text[:header_end].strip()

    if header_text and _is_header_block(header_text):
        chunks.append({
            "text": header_text,
            "chunk_type": "header",
            "article": None,
            "title": "Thông tin chung / Căn cứ / Các bên",
        })
    elif header_text and len(header_text) > 50:
        # Vẫn lưu header dù không nhận diện được, để không mất thông tin
        chunks.append({
            "text": header_text,
            "chunk_type": "preamble",
            "article": None,
            "title": "Phần mở đầu",
        })

    # Tách theo cấu trúc chính 
    if primary_matches:
        for i, match in enumerate(primary_matches):
            start = match.start()
            end = primary_matches[i + 1].start() if i + 1 < len(primary_matches) else len(text)
            chunk = text[start:end].strip()

            if len(chunk) < 30:
                continue

            article, title = _extract_title(chunk)

            # Tăng ngưỡng lên 2500 cho văn bản tiếng Việt dày
            if len(chunk) > 2500:
                sub_chunks = _split_long_article(chunk, article, title)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "text": chunk,
                    "chunk_type": chunk_type_default,
                    "article": article,
                    "title": title,
                })

    else:
        #  Fallback: RecursiveCharacterTextSplitter ─
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1100,
            chunk_overlap=150,
            separators=[
                "\n\n",
                "\nĐiều ", "\nCHƯƠNG ", "\nPHẦN ", "\nMỤC ",
                "\nI. ", "\nII. ", "\nIII. ", "\nIV. ", "\nV. ",
                "\n1. ", "\n2. ", "\n3. ", "\n4. ",
                "\na) ", "\nb) ", "\nc) ", "\nđ) ",
            ],
            keep_separator=True,
        )

        raw_chunks = splitter.split_text(text)
        current_article = None
        current_title = None

        for raw in raw_chunks:
            raw = raw.strip()
            if len(raw) < 30:
                continue

            art_match = ARTICLE_PATTERN.search(raw)
            if art_match and _is_real_article_match(art_match, raw + "\n"):
                current_article, current_title = _extract_title(art_match.group(1) + "\n")

            chunks.append({
                "text": raw,
                "chunk_type": "article" if current_article else "recursive",
                "article": current_article,
                "title": current_title,
            })

    return chunks

# Hàm tiện ích để build Document list từ text đã tách chunk
def build_documents_from_pages(docs, doc_type: str = "user") -> list[Document]:
    """Xây dựng danh sách Document từ các trang PDF"""
    if not docs:
        return []

    # Join text từ tất cả các trang
    page_boundaries = []  # (char_start, page_label)
    parts = []
    pos = 0
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue
        page_boundaries.append((pos, doc.metadata.get("page_label", "?")))
        parts.append(content)
        pos += len(content) + 2

    full_text = "\n\n".join(parts)

    # METADATA chung cho tất cả chunk, lấy từ trang đầu tiên (thường chứa header/căn cứ)
    base_meta = docs[0].metadata.copy()
    file_name = os.path.basename(base_meta.get("source", "unknown.pdf"))
    
    base_meta.update({
        "file_name": file_name,
        "document_title": file_name.replace(".pdf", "").replace("_", " "),
        "contract_type": detect_contract_type(file_name),
        "doc_type": doc_type,
    })

    # Tách chunk
    chunks = split_legal_text(full_text)

    # Hàm lấy số trang (nested function - closure)
    def _get_page_label(chunk_text: str) -> str:
        """Tìm trang chứa đoạn text này"""
        if not chunk_text:
            return "?"
        pos = full_text.find(chunk_text[:100])   # tăng lên 100 cho chắc ăn
        if pos == -1:
            return "?"
        
        label = "1"
        for boundary_pos, page_label in page_boundaries:
            if boundary_pos <= pos:
                label = page_label
            else:
                break
        return label

    final_docs = []
    for i, chunk in enumerate(chunks):
        page_label = _get_page_label(chunk["text"])        # <-- Chỉ truyền 1 tham số
        
        meta = {
            **base_meta,
            "chunk_index": i,
            "chunk_type": chunk["chunk_type"],
            "article": chunk["article"],
            "title": chunk["title"],
            "page_label": page_label,
        }
        final_docs.append(Document(page_content=chunk["text"], metadata=meta))

    return final_docs