# app.py
from dotenv import load_dotenv
load_dotenv()

import os
import re
import shutil
import streamlit as st
from rag.ingest import ingest_document, ingest_folder
from rag.pipeline import answer_question
import warnings
import logging

# Tắt warning __path__ từ transformers
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="Legal RAG Mini", layout="wide")
st.title("⚖️ Legal RAG Mini – Tra cứu hợp đồng & pháp lý")

USER_STORE     = "data/vector_store/user_docs"
LEGAL_STORE    = "data/vector_store/legal_docs"
UPLOAD_DIR     = "data/uploads"
LEGAL_DOCS_DIR = "legal_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("data/vector_store", exist_ok=True)

#  Dọn file upload cũ khi app khởi động ─
# Tránh tích lũy file qua nhiều session
if os.path.exists(UPLOAD_DIR):
    for old_file in os.listdir(UPLOAD_DIR):
        try:
            os.remove(os.path.join(UPLOAD_DIR, old_file))
        except Exception:
            pass


#  Helpers 
def _render_source_card(doc):
    meta    = doc.metadata
    article = meta.get("article") or ""
    title   = meta.get("title")   or ""
    source  = os.path.basename(meta.get("source", ""))
    page    = meta.get("page_label") or meta.get("page", "")

    if article:
        label = f"**{article}** – {title}" if title else f"**{article}**"
    else:
        label = title or source or "Đoạn văn"

    sub = []
    if source: sub.append(f"📁 {source}")
    if page:   sub.append(f"trang {page}")

    with st.expander(label, expanded=True):
        if sub:
            st.caption(" · ".join(sub))
        content = doc.page_content.strip()
        st.write(content[:600] + "…" if len(content) > 600 else content)


def _filter_relevant_docs(docs: list, query: str) -> list:
    q = query.lower()

    is_asking_party  = any(k in q for k in [
        "bên a", "bên b", "thông tin", "ai ký", "ông", "bà", "cmnd", "địa chỉ"
    ])
    is_asking_amount = any(k in q for k in [
        "số tiền", "bao nhiêu", "tiền", "vnđ", "điều 1"
    ])

    filtered = []
    for doc in docs:
        chunk_type = doc.metadata.get("chunk_type", "")
        article    = str(doc.metadata.get("article") or "").lower()

        if chunk_type == "header" and not is_asking_party:
            continue
        if article == "điều 1" and not is_asking_amount:
            continue

        filtered.append(doc)

    def sort_key(doc):
        if doc.metadata.get("chunk_type") == "header":
            return 0
        match = re.search(r"\d+", doc.metadata.get("article") or "")
        return int(match.group()) if match else 999

    return sorted(filtered, key=sort_key)


#  Session state init
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()
if "answer"     not in st.session_state: st.session_state.answer     = None
if "user_docs"  not in st.session_state: st.session_state.user_docs  = []
if "legal_docs" not in st.session_state: st.session_state.legal_docs = []
if "last_query" not in st.session_state: st.session_state.last_query = ""


#  Sidebar 
st.sidebar.header("⚙️ Quản lý dữ liệu")

if st.sidebar.button("📚 Ingest kho luật / quy định"):
    if not os.path.exists(LEGAL_DOCS_DIR):
        st.sidebar.error("❌ Chưa có folder legal_docs/")
    else:
        with st.spinner("Đang ingest kho luật..."):
            ingest_folder(LEGAL_DOCS_DIR, LEGAL_STORE, doc_type="legal")
        st.sidebar.success("✅ Đã ingest legal docs")

if st.sidebar.button("🗑️ Xóa toàn bộ vector store"):
    if os.path.exists("data/vector_store"):
        shutil.rmtree("data/vector_store")
    os.makedirs("data/vector_store", exist_ok=True)
    st.session_state.ingested_files = set()
    st.session_state.answer         = None
    st.session_state.user_docs      = []
    st.session_state.legal_docs     = []
    st.session_state.last_query     = ""
    st.sidebar.success("✅ Đã xóa vector store")


# Upload 
st.subheader("1) Upload tài liệu")
uploaded_file = st.file_uploader("Upload PDF hợp đồng / giấy tờ", type="pdf")

if uploaded_file is not None:
    file_name = uploaded_file.name

    if file_name not in st.session_state.ingested_files:
        file_path = os.path.join(UPLOAD_DIR, file_name)

        # Ghi file tạm để đọc
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with st.spinner(f"Đang xử lý **{file_name}**..."):
                ingest_document(file_path, USER_STORE, doc_type="user")
            st.session_state.ingested_files.add(file_name)
            st.success(f"✅ Đã ingest: {file_name}")
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý file: {e}")
        finally:
            # Xóa file tạm ngay sau khi ingest xong dù thành công hay lỗi
            if os.path.exists(file_path):
                os.remove(file_path)

    else:
        st.info(f"📎 Đã tải lên trước đó: **{file_name}**")

if st.session_state.ingested_files:
    st.caption("📂 Tài liệu đang dùng: " + ", ".join(st.session_state.ingested_files))


# Query
st.subheader("2) Hỏi đáp")
query = st.text_input(
    "Nhập câu hỏi",
    placeholder="Ví dụ: Điều 3 là gì? / Lãi suất có hợp pháp không? / Tóm tắt nghĩa vụ bên B",
)

if st.button("🔍 Tra cứu") and query:
    with st.spinner("Đang phân tích..."):
        answer, user_docs, legal_docs = answer_question(query)

    st.session_state.answer     = answer
    st.session_state.user_docs  = user_docs
    st.session_state.legal_docs = legal_docs
    st.session_state.last_query = query


# Hiển thị kết quả và nguồn tham chiếu
if st.session_state.answer:
    st.markdown("## 📌 Kết quả")
    st.write(st.session_state.answer)

    user_docs  = st.session_state.user_docs
    legal_docs = st.session_state.legal_docs
    last_query = st.session_state.last_query

    if user_docs or legal_docs:
        st.markdown("---")
        st.markdown("### 📎 Nguồn tham chiếu")

        col1, col2 = st.columns(2)

        with col1:
            if user_docs:
                filtered_user = _filter_relevant_docs(user_docs, last_query)
                if filtered_user:
                    st.markdown("**📄 Từ hợp đồng / tài liệu người dùng**")
                    for doc in filtered_user:
                        _render_source_card(doc)

        with col2:
            if legal_docs:
                filtered_legal = _filter_relevant_docs(legal_docs, last_query)
                if filtered_legal:
                    st.markdown("**⚖️ Từ kho luật / quy định**")
                    for doc in filtered_legal:
                        _render_source_card(doc)