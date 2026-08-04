"""
RAG Chatbot — E-commerce Support (Glassmorphism Edition)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="E-commerce Support RAG Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS — Glassmorphism, tự động theo Light/Dark hệ thống
# =============================================================================

st.markdown(
    """
    <style>
    :root {
        --bg-gradient: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --sidebar-bg: rgba(15, 23, 42, 0.75);
        --card-bg: rgba(30, 41, 59, 0.6);
        --card-border: rgba(255, 255, 255, 0.08);
        --metric-value-color: #38bdf8;
        --hero-bg: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
        --hero-border: rgba(255, 255, 255, 0.1);
        --hero-title-grad: linear-gradient(90deg, #818cf8 0%, #34d399 100%);
        --button-bg: rgba(30, 41, 59, 0.7);
        --button-text: #e2e8f0;
        --button-border: rgba(255, 255, 255, 0.1);
        --button-hover-bg: rgba(99, 102, 241, 0.3);
        --button-hover-border: rgba(129, 140, 248, 0.5);
        --chat-bg: rgba(30, 41, 59, 0.5);
        --shadow-color: rgba(0, 0, 0, 0.37);
        --tag-legal-bg: rgba(139, 92, 246, 0.25);
        --tag-legal-text: #c084fc;
        --tag-legal-border: rgba(192, 132, 252, 0.3);
        --tag-news-bg: rgba(16, 185, 129, 0.25);
        --tag-news-text: #34d399;
        --tag-news-border: rgba(52, 211, 153, 0.3);
    }

    @media (prefers-color-scheme: light) {
        :root {
            --bg-gradient: linear-gradient(135deg, #f8fafc 0%, #eef2ff 50%, #f1f5f9 100%);
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --sidebar-bg: rgba(255, 255, 255, 0.8);
            --card-bg: rgba(255, 255, 255, 0.7);
            --card-border: rgba(15, 23, 42, 0.08);
            --metric-value-color: #0284c7;
            --hero-bg: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(16, 185, 129, 0.12) 100%);
            --hero-border: rgba(15, 23, 42, 0.08);
            --hero-title-grad: linear-gradient(90deg, #4f46e5 0%, #059669 100%);
            --button-bg: rgba(255, 255, 255, 0.85);
            --button-text: #1e293b;
            --button-border: rgba(15, 23, 42, 0.1);
            --button-hover-bg: rgba(99, 102, 241, 0.12);
            --button-hover-border: rgba(79, 70, 229, 0.4);
            --chat-bg: rgba(255, 255, 255, 0.7);
            --shadow-color: rgba(15, 23, 42, 0.08);
            --tag-legal-bg: rgba(139, 92, 246, 0.12);
            --tag-legal-text: #7c3aed;
            --tag-legal-border: rgba(124, 58, 237, 0.25);
            --tag-news-bg: rgba(16, 185, 129, 0.12);
            --tag-news-text: #059669;
            --tag-news-border: rgba(5, 150, 105, 0.25);
        }
    }

    .stApp {
        background: var(--bg-gradient);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid var(--card-border);
    }

    .hero-header {
        background: var(--hero-bg);
        border: 1px solid var(--hero-border);
        backdrop-filter: blur(12px);
        padding: 24px 32px;
        border-radius: 20px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 var(--shadow-color);
    }
    .hero-title {
        background: var(--hero-title-grad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1.05rem;
    }

    .metric-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 14px 18px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--metric-value-color);
    }
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .source-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .source-tag-legal {
        background: var(--tag-legal-bg);
        color: var(--tag-legal-text);
        border: 1px solid var(--tag-legal-border);
    }
    .source-tag-news {
        background: var(--tag-news-bg);
        color: var(--tag-news-text);
        border: 1px solid var(--tag-news-border);
    }

    .stButton>button {
        background: var(--button-bg) !important;
        color: var(--button-text) !important;
        border: 1px solid var(--button-border) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        text-align: left !important;
    }
    .stButton>button:hover {
        background: var(--button-hover-bg) !important;
        border-color: var(--button-hover-border) !important;
        box-shadow: 0 0 15px var(--button-hover-bg) !important;
        transform: translateY(-2px);
    }

    [data-testid="stChatMessage"] {
        background: var(--chat-bg) !important;
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
    }

    [data-testid="stSidebar"] .stCaption, .stCaption {
        color: var(--text-secondary) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# SIDEBAR — INFO, CONFIG & GỢI Ý
# =============================================================================

with st.sidebar:
    st.markdown("### 🛒 E-commerce Support")
    st.caption("Trợ lý hỏi đáp về chính sách thương mại điện tử và hỗ trợ khách hàng (đổi trả, thanh toán, bảo mật, người bán)")

    st.divider()

    st.subheader("⚙️ Cấu hình Pipeline")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)
    model_name = st.selectbox(
        "Mô hình LLM (OpenRouter)",
        ["openai/gpt-4o-mini", "google/gemini-flash-1.5", "meta-llama/llama-3-70b-instruct:free"],
        index=0,
        help="Chỉ áp dụng khi dùng OPENROUTER_API_KEY trong .env",
    )
    os.environ["LLM_MODEL"] = model_name

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    category = st.radio("Chủ đề câu hỏi:", ["💳 Thanh toán & Đổi trả", "🏪 Người bán & Vận chuyển"], index=0)

    if "Thanh toán" in category:
        suggestions = [
            "Shopee hỗ trợ những phương thức thanh toán nào?",
            "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
            "Làm sao để đổi phương thức thanh toán đơn hàng?",
        ]
    else:
        suggestions = [
            "Quy định về đăng bán sản phẩm cho người bán?",
            "Cách mua hàng trên Shopee của quốc gia khác?",
            "Cần chuẩn bị bằng chứng gì khi yêu cầu hoàn tiền?",
        ]

    for s in suggestions:
        if st.button(f"📌 {s}", use_container_width=True, key=f"sug_{hash(s)}"):
            st.session_state["pending_query"] = s

    st.divider()

    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Kiến trúc hệ thống:**")
    st.caption("Hybrid Retrieval (Semantic + BM25) → RRF Rerank → PageIndex Fallback → LLM Generation có Citation")

# =============================================================================
# MAIN UI — HERO HEADER & METRICS
# =============================================================================

st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">🛒 E-Commerce Support RAG Chatbot</div>
        <div class="hero-subtitle">Trợ lý trí tuệ nhân tạo hỗ trợ tra cứu chính sách Shopee, đổi trả, thanh toán & quy định người bán (kèm trích dẫn nguồn)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-value">{len(st.session_state.messages)}</div><div class="metric-label">Tin nhắn</div></div>""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-value">{top_k}</div><div class="metric-label">Top-K Chunks</div></div>""",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """<div class="metric-card"><div class="metric-value">Hybrid</div><div class="metric-label">Search Mode</div></div>""",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        """<div class="metric-card"><div class="metric-value">Active</div><div class="metric-label">PageIndex Fallback</div></div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# HELPERS — hiển thị nguồn trích dẫn
# =============================================================================

def render_sources(sources: list[dict]) -> None:
    with st.expander(f"📚 Nguồn tham khảo trích dẫn ({len(sources)} chunks)"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            source_name = meta.get("source", "Unknown Source")
            doc_type = meta.get("type", "legal")
            score = src.get("score", 0)

            tag_class = "source-tag-legal" if "legal" in doc_type else "source-tag-news"
            st.markdown(
                f"""<span class="source-tag {tag_class}">[{i}] {doc_type.upper()}</span> **{source_name}** | score: `{score:.4f}`""",
                unsafe_allow_html=True,
            )
            st.text(src.get("content", "")[:350] + "...")
            st.divider()


# =============================================================================
# CHAT HISTORY DISPLAY
# =============================================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# =============================================================================
# QUERY HANDLING & GENERATION
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi về chính sách Shopee / hỗ trợ khách hàng...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang thực thi Hybrid Search & tổng hợp câu trả lời có citation..."):
            start_time = time.time()
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(query, top_k=top_k)
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])
                elapsed = time.time() - start_time

            except NotImplementedError:
                answer = "⚠️ **Task 9/10 chưa được implement.** Hãy hoàn thành `src/task9_retrieval_pipeline.py` và `src/task10_generation.py` để kết nối pipeline vào UI!"
                sources = []
                elapsed = 0
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources = []
                elapsed = 0

            st.markdown(answer)
            if elapsed > 0:
                st.caption(f"⚡ *Thời gian phản hồi: {elapsed:.2f} giây*")

            if sources:
                render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
