# 🛒 Bài Tập Nhóm — E-commerce Support RAG Chatbot

## 📌 Tổng Quan Dự Án
Hệ thống **RAG Chatbot hỗ trợ e-commerce** giúp tự động tìm kiếm và trả lời các thắc mắc về chính sách thương mại điện tử (đổi trả, thanh toán, giao hàng, bảo mật, quy định người bán) dựa trên nguồn tài liệu chuẩn hóa, có kèm trích dẫn nguồn (Citation) minh bạch.

---

## 🏗️ Kiến Trúc Hệ Thống Hiện Tại

```
                  ┌──────────────────────────────────────────┐
                  │ 1. Dữ Liệu Đầu Vào (Legal PDFs + JSON)   │
                  └────────────────────┬─────────────────────┘
                                       │ MarkItDown
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │ 2. Chuẩn hóa Markdown (data/standardized)│
                  └────────────────────┬─────────────────────┘
                                       │ RecursiveCharacterTextSplitter (500/50)
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │ 3. Chunking & Indexing (baai/bge-m3)     │
                  │    Lưu trữ persistent tại ChromaDB       │
                  └────────────────────┬─────────────────────┘
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
        ┌─────────────────────────────┐ ┌─────────────────────────────┐
        │ 4a. Dense Semantic Search   │ │ 4b. Sparse Lexical Search   │
        │     (OpenRouter / Vector)   │ │     (BM25Okapi + Term Exp)  │
        └──────────────┬──────────────┘ └──────────────┬──────────────┘
                       │                               │
                       └───────────────┬───────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │ 5. Hybrid Fusion & Reranking (RRF)       │
                  └────────────────────┬─────────────────────┘
                                       │ (Best score < 0.3?)
                       ┌───────────────┴───────────────┐
             Không     ▼                               ▼ Có
        ┌─────────────────────────────┐ ┌─────────────────────────────┐
        │ 6a. Context Reordering      │ │ 6b. PageIndex Fallback      │
        │     (Anti "Lost in Middle") │ │     (Vectorless Search)     │
        └──────────────┬──────────────┘ └──────────────┬──────────────┘
                       │                               │
                       └───────────────┬───────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │ 7. Generation (OpenRouter / OpenAI LLM)  │
                  │    Trả lời tự động kèm Citation          │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │ 8. Giao diện người dùng Streamlit UI     │
                  └──────────────────────────────────────────┘
```

---

## 🛠️ Công Nghệ & Thành Phần Triển Khai

| Thành phần | Công nghệ / Thư viện | Mô tả chi tiết |
|------------|──────────────────────|────────────────|
| **Data Crawling & Parsing** | `crawl4ai`, `markitdown` | Crawl bài viết trợ giúp Shopee & chuyển PDF/DOCX sang Markdown |
| **Chunking Strategy** | `RecursiveCharacterTextSplitter` | `CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 50` |
| **Embedding Model** | `baai/bge-m3` | Chạy qua OpenRouter API (vector 1024 dimensions) |
| **Vector Database** | `ChromaDB` | Lưu trữ vector local persistent (`chroma_db/`) |
| **Lexical Search** | `rank-bm25` | Giải thuật BM25Okapi có mở rộng từ khóa Anh-Việt |
| **Hybrid Reranker** | `Reciprocal Rank Fusion (RRF)` | Gộp kết quả Dense & Sparse bằng công thức RRF $1/(60 + rank)$ |
| **Fallback Retrieval** | `PageIndex` | Vectorless retrieval fallback khi điểm Cosine gốc < 0.3 |
| **LLM Generation** | OpenRouter / OpenAI API | Xử lý chống *"Lost in the Middle"* & sinh câu trả lời có Citation |
| **User Interface** | `Streamlit` | Giao diện Chatbot web tương tác trực quan (`app.py`) |

---

## 🚀 Hướng Dẫn Kích Hoạt & Chạy Ứng Dụng

### 1. Cài đặt môi trường
```powershell
# Kích hoạt môi trường ảo (.venv)
.\.venv\Scripts\Activate.ps1

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Cấu hình file `.env`
Tạo file `.env` tại thư mục gốc dự án:
```ini
OPENROUTER_API_KEY=sk-or-v1-...
# Hoặc OPENAI_API_KEY=sk-proj-...
```

### 3. Thực thi Indexing dữ liệu
```powershell
python .\src\task4_chunking_indexing.py
```

### 4. Khởi chạy Giao diện Chatbot Streamlit
```powershell
python -m streamlit run app.py
```
👉 Sau đó truy cập giao diện tại **`http://localhost:8501`**.

---

## 👥 Phân Công Công Việc Nhóm

| Thành viên | Nhiệm vụ | Trạng thái |
|-----------|----------|------------|
| Thành viên 1 | Data Crawling & Standardizing Markdown (Task 1-3) |  Hoàn thành |
| Thành viên 2 | Chunking, Embedding & ChromaDB Indexing (Task 4-5) |  Hoàn thành |
| Thành viên 3 | BM25 Lexical Search & RRF Reranking (Task 6-7) |  Hoàn thành |
| Thành viên 4 | Retrieval Pipeline & PageIndex Fallback (Task 8-9) |  Hoàn thành |
| Thành viên 5 | LLM Generation & Giao diện Streamlit App (Task 10 & UI) |  Hoàn thành |

---

## 📊 Hướng Dẫn Chạy Evaluation (RAGAS / DeepEval)
Vui lòng tham khảo thư mục `group_project/evaluation/` để kiểm thử bộ **Golden Dataset (15+ Q&A)** và đo lường các chỉ số *Faithfulness, Answer Relevance, Context Recall, Context Precision*.
