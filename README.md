# Legal RAG Mini

Hệ thống Retrieval-Augmented Generation (RAG) đơn giản phục vụ phân tích hợp đồng và văn bản pháp lý tiếng Việt.

---

## Giới thiệu

Dự án xây dựng một pipeline RAG chuyên biệt cho tài liệu pháp lý, cho phép:

* Nạp và xử lý file PDF
* Tách văn bản theo cấu trúc pháp lý (Điều, Chương, Mục)
* Sinh embedding và lưu trữ vector bằng FAISS
* Truy vấn ngữ nghĩa trên nhiều tài liệu
* Trả lời câu hỏi bằng mô hình ngôn ngữ (LLM)
* Hiển thị kết quả và nguồn tham chiếu qua giao diện Streamlit

---

## Điểm nổi bật

### 1. Tách văn bản có cấu trúc

Hệ thống không chia text theo cách thông thường mà nhận diện cấu trúc pháp lý:

* Điều X (Điều 1, Điều 2,...)
* Chương / Phần / Mục
* Header (thông tin hợp đồng)
* Các khoản nhỏ (1., a), ...)

Ngoài ra:

* Điều dài sẽ được tách thành nhiều clause nhỏ
* Metadata được giữ lại:

  * article (ví dụ: "điều 3")
  * title
  * chunk_type (header, article, clause,...)

---

### 2. Tìm kiếm và lưu trữ

* Sử dụng Sentence Transformers để tạo embedding
* Lưu vector bằng FAISS
* Hỗ trợ:

  * nhiều file người dùng (user_docs)
  * kho văn bản pháp lý (legal_docs)

---

### 3. Xử lý truy vấn thông minh

Hệ thống tự phân loại câu hỏi:

* Hỏi điều khoản (Điều X)
* Tóm tắt nội dung
* Nghĩa vụ / quyền lợi
* Thời hạn
* Chế tài
* So sánh với quy định pháp luật
* Câu hỏi chung

Trường hợp đặc biệt:

* Nếu hỏi "Điều X" → không dùng semantic search
* Thay vào đó scan toàn bộ dữ liệu để đảm bảo chính xác

---

### 4. Sinh câu trả lời

* Sử dụng Groq API với model:

  * llama-3.1-8b-instant
* Prompt được chọn theo loại câu hỏi
* Trả lời:

  * ngắn gọn
  * đúng ngữ cảnh
  * không suy diễn ngoài dữ liệu

---

### 5. Giao diện

* Xây dựng bằng Streamlit
* Cho phép:

  * upload file PDF
  * ingest dữ liệu
  * nhập câu hỏi
  * xem câu trả lời và nguồn trích dẫn

---

## Luồng xử lý

1. Ingest

   * PDF → text → chunk theo cấu trúc → embedding → lưu FAISS

2. Query

   * Phân loại câu hỏi
   * Truy xuất dữ liệu liên quan

3. LLM

   * Tạo prompt
   * Sinh câu trả lời

4. Hiển thị

   * Kết quả + nguồn tham chiếu

---

## Công nghệ sử dụng

* LangChain
* FAISS
* Sentence Transformers
* Groq API
* Streamlit

---

## Yêu cầu

* Python >= 3.10
* Tạo file `.env`:

```env
GROQ_API_KEY=your_api_key
```

---

## Chạy ứng dụng

```bash
streamlit run app.py
```
