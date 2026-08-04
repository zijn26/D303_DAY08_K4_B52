"""
RAG Chatbot — E-commerce Support (Ultra-Premium Glassmorphism Edition)
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
    page_title="Shopee Support RAG AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS — ULTRA-PREMIUM GLASSMORPHISM DESIGN SYSTEM
# =============================================================================

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

    <style>
    /* Theme Tokens */
    :root {
        --font-main: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
        
        --bg-main: #090d16;
        --bg-card: rgba(18, 26, 43, 0.65);
        --bg-card-hover: rgba(26, 38, 63, 0.75);
        --bg-sidebar: rgba(11, 15, 25, 0.85);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(99, 102, 241, 0.35);
        
        --text-heading: #f8fafc;
        --text-body: #cbd5e1;
        --text-muted: #64748b;
        
        --accent-indigo: #6366f1;
        --accent-violet: #8b5cf6;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
        
        --grad-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        --grad-hero: linear-gradient(135deg, rgba(99, 102, 241, 0.18) 0%, rgba(139, 92, 246, 0.12) 50%, rgba(6, 182, 212, 0.15) 100%);
        --grad-accent-border: linear-gradient(90deg, #6366f1, #06b6d4, #10b981);
        --shadow-glow: 0 8px 32px 0 rgba(99, 102, 241, 0.2);
    }

    /* Base Reset & Typography */
    .stApp {
        background-color: var(--bg-main) !important;
        background-image: 
            radial-gradient(at 10% 10%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
            radial-gradient(at 90% 90%, rgba(6, 182, 212, 0.1) 0px, transparent 50%),
            radial-gradient(at 50% 50%, rgba(139, 92, 246, 0.08) 0px, transparent 50%) !important;
        color: var(--text-body) !important;
        font-family: var(--font-main) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-main) !important;
        color: var(--text-heading) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid var(--border-glass) !important;
    }

    /* Hero Header Banner */
    .hero-container {
        position: relative;
        background: var(--grad-hero);
        border: 1px solid var(--border-glass);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 28px 36px;
        margin-bottom: 24px;
        overflow: hidden;
        box-shadow: var(--shadow-glow);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--grad-accent-border);
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #818cf8;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: var(--accent-emerald);
        border-radius: 50%;
        box-shadow: 0 0 10px var(--accent-emerald);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .hero-title-text {
        font-size: 2.2rem;
        font-weight: 800;
        background: var(--grad-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        line-height: 1.2;
    }

    .hero-desc {
        color: var(--text-body);
        font-size: 1.05rem;
        margin: 0;
        max-width: 800px;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }

    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 16px 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stat-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-glow);
        transform: translateY(-3px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text-heading);
        font-family: var(--font-mono);
        margin-bottom: 4px;
    }
    .stat-label {
        font-size: 0.78rem;
        color: var(--text-muted);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* Interactive Buttons */
    .stButton>button {
        background: var(--bg-card) !important;
        color: var(--text-body) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        font-family: var(--font-main) !important;
        transition: all 0.25s ease !important;
        text-align: left !important;
    }

    .stButton>button:hover {
        background: rgba(99, 102, 241, 0.2) !important;
        border-color: var(--accent-indigo) !important;
        color: var(--text-heading) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25) !important;
    }

    /* Chat Messages styling */
    [data-testid="stChatMessage"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-glass) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 18px !important;
        padding: 20px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
    }

    /* Source Citation Badges */
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: var(--font-mono);
    }
    .chip-legal {
        background: rgba(139, 92, 246, 0.2);
        color: #a78bfa;
        border: 1px solid rgba(139, 92, 246, 0.35);
    }
    .chip-news {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    /* Score Metric Badges */
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 8px;
        font-family: var(--font-mono);
        font-size: 0.82rem;
        font-weight: 600;
        backdrop-filter: blur(8px);
    }
    .badge-rrf {
        background: rgba(6, 182, 212, 0.15);
        color: #22d3ee;
        border: 1px solid rgba(6, 182, 212, 0.4);
    }
    .badge-dense {
        background: rgba(139, 92, 246, 0.15);
        color: #c084fc;
        border: 1px solid rgba(139, 92, 246, 0.4);
    }
    .badge-bm25 {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    /* Expander Source Styling */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-glass) !important;
        color: var(--text-heading) !important;
        font-weight: 600 !important;
    }

    /* Sliders & Inputs */
    .stSlider > div {
        color: var(--accent-indigo) !important;
    }
    
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border-color: var(--border-glass) !important;
        color: var(--text-heading) !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# SIDEBAR — CONTROLS & SUGGESTIONS
# =============================================================================

with st.sidebar:
    st.markdown("### 🤖 Shopee RAG Agent")
    st.caption("Hệ thống hỏi đáp thông minh kết hợp Hybrid Search, RRF Reranking và LLM Generation có trích dẫn.")

    st.divider()

    st.markdown("#### ⚙️ Cấu Hình RAG Pipeline")
    top_k = st.slider("Số lượng Chunks (top_k)", 3, 10, 5, help="Số đoạn tài liệu liên quan nhất được trích xuất cho LLM")
    
    model_name = st.selectbox(
        "Mô hình LLM (OpenRouter)",
        ["openai/gpt-4o-mini", "google/gemini-2.0-flash-exp:free", "meta-llama/llama-3.3-70b-instruct:free"],
        index=0,
        help="Lựa chọn LLM model thực thi sinh câu trả lời",
    )
    os.environ["LLM_MODEL"] = model_name

    st.divider()

    st.markdown("#### 💡 Câu Hỏi Mẫu")
    category = st.radio("Chủ đề tra cứu:", ["💳 Đổi Trả & Thanh Toán", "📦 Vận Chuyển & Người Bán"], index=0)

    if "Đổi Trả" in category:
        suggestions = [
            "Thời hạn tối đa để yêu cầu Trả hàng/Hoàn tiền Shopee là bao lâu?",
            "Shopee hiện hỗ trợ những phương thức thanh toán nào?",
            "Hướng dẫn đổi phương thức thanh toán cho đơn hàng đã đặt?",
        ]
    else:
        suggestions = [
            "Các phương thức gửi hàng hoàn trả và quy định về phí hoàn trả?",
            "Cần chuẩn bị những bằng chứng nào khi bấm yêu cầu Trả hàng/Hoàn tiền?",
            "Quy định chung về đăng bán sản phẩm dành cho Người bán trên Shopee?",
        ]

    for q_text in suggestions:
        if st.button(f"👉 {q_text}", use_container_width=True, key=f"sug_{hash(q_text)}"):
            st.session_state["pending_query"] = q_text

    st.divider()

    if st.button("🗑️ Xóa Lịch Sử Hội Thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("<span style='font-size:0.75rem; color:var(--text-muted);'> Kiến trúc: <b>Hybrid Search</b> (bge-m3 + BM25) → <b>RRF Rerank</b> → <b>PageIndex Fallback</b> → <b>Citation Generation</b></span>", unsafe_allow_html=True)

# =============================================================================
# MAIN UI — HERO HEADER & STATS
# =============================================================================

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">
            <span class="pulse-dot"></span>
            RAG Pipeline v2.0 Active
        </div>
        <h1 class="hero-title-text">Shopee Support RAG AI Agent</h1>
        <p class="hero-desc">Hệ thống hỏi đáp thông minh chính sách Shopee, hỗ trợ đổi trả, thanh toán & quy định thương mại điện tử với độ chính xác cao và trích dẫn nguồn minh bạch.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render Metric Stats
st.markdown(
    f"""
    <div class="metric-grid">
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.messages)}</div>
            <div class="stat-label">Lượt Hỏi Đáp</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{top_k}</div>
            <div class="stat-label">Top-K Context Chunks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:var(--accent-cyan);">Hybrid RRF</div>
            <div class="stat-label">Retrieval Strategy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:var(--accent-emerald);">Ready</div>
            <div class="stat-label">PageIndex Fallback</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# HELPERS — CITATION DISPLAY
# =============================================================================

def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    
    with st.expander(f"📚 Nguồn tài liệu tham khảo ({len(sources)} chunks được trích xuất)"):
        # Giải thích các chỉ số tìm kiếm
        st.markdown(
            """
            <div style="background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 14px; padding: 14px 18px; margin-bottom: 16px; backdrop-filter: blur(10px);">
                <div style="font-weight: 700; color: #a78bfa; font-size: 0.9rem; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                    <span>🧮</span> Giải thích các chỉ số Retrieval (Hybrid Search & RRF Reranking):
                </div>
                <div style="font-size: 0.82rem; color: var(--text-body); line-height: 1.6;">
                    • <b>🎯 Điểm RRF (Reciprocal Rank Fusion):</b> Điểm tổng hợp xếp hạng <code>RRF = Σ [1 / (60 + Rank_i)]</code> gộp từ cả Dense Search & BM25.<br/>
                    • <b>⚡ Dense Search:</b> Cosine Similarity từ Vector Store (bge-m3 embedding).<br/>
                    • <b>🔤 BM25 Search:</b> Điểm khớp từ khóa theo thuật toán Lexical BM25 Okapi.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            source_name = meta.get("source") or src.get("source", "Tài liệu hệ thống")
            doc_type = meta.get("type", "legal")
            
            score = src.get("score", 0.0)
            dense_score = src.get("dense_score")
            bm25_score = src.get("bm25_score")

            # Helper safe float format
            def _safe_fmt(val, fmt_spec):
                if val is None:
                    return "N/A"
                try:
                    return f"{float(val):{fmt_spec}}"
                except (ValueError, TypeError):
                    return "N/A"

            # Formatted String values & badges (Only show badge if score is valid, hide if N/A)
            rrf_str = _safe_fmt(score, ".6f")
            dense_str = _safe_fmt(dense_score, ".4f")
            bm25_str = _safe_fmt(bm25_score, ".2f")

            chip_class = "chip-legal" if "legal" in str(doc_type).lower() else "chip-news"

            rrf_badge = f"""<div class="score-badge badge-rrf">🎯 Điểm RRF: <b>{rrf_str}</b></div>""" if rrf_str != "N/A" else ""
            dense_badge = f"""<div class="score-badge badge-dense">⚡ Dense search: <b>{dense_str}</b></div>""" if dense_str != "N/A" else ""
            bm25_badge = f"""<div class="score-badge badge-bm25">🔤 BM25 search: <b>{bm25_str}</b></div>""" if bm25_str != "N/A" else ""

            st.markdown(
                f"""
                <div style="background: rgba(18, 26, 43, 0.55); border: 1px solid var(--border-glass); border-radius: 14px; padding: 16px 20px; margin-bottom: 14px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span class="source-chip {chip_class}">DOC [{i}] • {doc_type.upper()}</span>
                            <strong style="color: var(--text-heading); font-size: 0.95rem;">{source_name}</strong>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; background: rgba(9, 13, 22, 0.45); padding: 10px 14px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.06);">
                        <span style="font-size: 0.78rem; color: var(--text-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-right: 4px;">Điểm tìm kiếm:</span>
                        {rrf_badge}
                        {dense_badge}
                        {bm25_badge}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code(src.get("content", "")[:350] + "...", language="markdown")
            st.divider()

# =============================================================================
# CHAT HISTORY DISPLAY
# =============================================================================

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# =============================================================================
# QUERY EXECUTION & LLM GENERATION
# =============================================================================

user_input = st.chat_input("Nhập thắc mắc của bạn về chính sách Shopee / đổi trả / thanh toán...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Hiển thị câu hỏi của User
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    # Sinh phản hồi từ RAG Agent
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("⚡ Đang thực thi Hybrid Search (Dense + BM25) & tổng hợp phản hồi..."):
            start_time = time.time()
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(query, top_k=top_k)
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])
                ret_source = response.get("retrieval_source", "hybrid")
                elapsed = time.time() - start_time

            except NotImplementedError:
                answer = "⚠️ **Task 9/10 chưa được kết nối hoàn chỉnh.** Hãy kiểm tra lại file `src/task9_retrieval_pipeline.py` và `src/task10_generation.py`!"
                sources = []
                ret_source = "none"
                elapsed = 0.0
            except Exception as e:
                answer = f"❌ **Đã xảy ra lỗi trong RAG Pipeline:** `{e}`"
                sources = []
                ret_source = "error"
                elapsed = 0.0

            st.markdown(answer)

            if elapsed > 0:
                st.markdown(
                    f"""
                    <div style="margin-top: 8px; font-size: 0.78rem; color: var(--text-muted); font-family: var(--font-mono);">
                        ⚡ <span>Thời gian phản hồi: <b>{elapsed:.2f}s</b></span> | 🔍 <span>Nguồn tìm kiếm: <b>{ret_source.upper()}</b></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if sources:
                render_sources(sources)

    # Lưu lịch sử chat
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
