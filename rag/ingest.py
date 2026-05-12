import os
import re
import pdfplumber
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.chunking import build_documents_from_pages

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def _fix_paragraph_breaks(text: str) -> str:
    """
    Thêm xuống dòng trước các khoản con và điều khoản bị dính liền.
    PDF thường không có newline giữa "...câu trước. 1. Khoản tiếp theo"
    """
    # Thêm newline trước "1. " "2. " ... khi bị dính vào câu trước
    text = re.sub(r"(?<=[^\n])(\s)(\d{1,2})\.\s+(?=[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ])", r"\n\2. ", text)

    # Thêm newline trước "a) " "b) " ... khi bị dính
    text = re.sub(r"(?<=[^\n])(\s)([a-zđ])\)\s+", r"\n\2) ", text)

    # Thêm newline trước "Điều X." khi bị dính vào câu trước
    # (chỉ khi là đầu điều khoản thực sự, không phải tham chiếu)
    text = re.sub(
        r"(?<=[^\n])\s+(Điều\s+\d+[A-Za-z]?\.)\s+",
        r"\n\1 ",
        text,
        flags=re.IGNORECASE,
    )

    return text

def _load_pdf_with_plumber(file_path: str) -> list[Document]:
    docs = []
    previous_end = ""   # Lưu dòng cuối trang trước để nối

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            if not text.strip():
                continue

            text = previous_end + "\n" + text if previous_end else text
            text = _fix_paragraph_breaks(text)
            
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "page": i,
                    "page_label": str(i + 1),
                }
            ))

            # Lưu lại đoạn cuối trang để nối với trang sau (nếu đang giữa điều khoản)
            lines = text.strip().split("\n")
            previous_end = ""
            if lines:
                last_line = lines[-1].strip()
                # Nếu dòng cuối không kết thúc bằng dấu chấm hoặc dấu hai chấm → có khả năng bị cắt
                if not re.search(r'[。.!?；:]$', last_line) and len(last_line) > 30:
                    previous_end = last_line + "\n"

    return docs

def ingest_document(file_path: str, store_path: str, doc_type: str = "user"):
    docs = _load_pdf_with_plumber(file_path)

    if not docs:
        raise ValueError(f"Không đọc được nội dung từ file: {file_path}")

    chunks = build_documents_from_pages(docs, doc_type=doc_type)

    if not chunks:
        raise ValueError(f"Không tách được chunk từ file: {file_path}")

    embeddings = get_embeddings()

    # Merge vào store hiện có (nếu có) 
    index_file = os.path.join(store_path, "index.faiss")
    if os.path.exists(index_file):
        try:
            existing = FAISS.load_local(
                store_path, embeddings, allow_dangerous_deserialization=True
            )
            new_store = FAISS.from_documents(chunks, embeddings)
            existing.merge_from(new_store)
            existing.save_local(store_path)
            print(f"[ingest] Merged {len(chunks)} chunks vào store -> {store_path}")
        except Exception as e:
            print(f"[ingest] Không merge được ({e}), tạo store mới.")
            _save_new(chunks, embeddings, store_path)
    else:
        _save_new(chunks, embeddings, store_path)

    return store_path

def _save_new(chunks, embeddings, store_path):
    os.makedirs(store_path, exist_ok=True)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(store_path)
    print(f"[ingest] Created store with {len(chunks)} chunks -> {store_path}")

def ingest_folder(folder_path: str, store_path: str, doc_type: str = "legal"):
    """
    Ingest toàn bộ PDF trong một folder vào vector store.
    Ghi đè store cũ (dùng cho legal_docs load batch).
    """
    embeddings = get_embeddings()
    all_docs = []

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise ValueError(f"Không tìm thấy PDF trong folder: {folder_path}")

    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)
        try:
            docs = _load_pdf_with_plumber(file_path)
            chunks = build_documents_from_pages(docs, doc_type=doc_type)
            all_docs.extend(chunks)
            print(f"[ingest_folder] {filename}: {len(chunks)} chunks")
        except Exception as e:
            print(f"[ingest_folder] Bỏ qua {filename}: {e}")

    if not all_docs:
        raise ValueError("Không có chunk nào được tạo từ folder.")

    os.makedirs(store_path, exist_ok=True)
    vector_store = FAISS.from_documents(all_docs, embeddings)
    vector_store.save_local(store_path)
    print(f"[ingest_folder] Total {len(all_docs)} chunks -> {store_path}")
    return store_path