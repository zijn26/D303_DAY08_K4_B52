"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PDF_CACHE_DIR = Path(__file__).parent.parent / "pageindex_pdfs"
DOC_IDS_FILE = Path(__file__).parent.parent / "pageindex_doc_ids.json"
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 120
HTTP_TIMEOUT_SECONDS = 60


def _get_client():
    """Tạo PageIndex client và báo lỗi rõ ràng khi thiếu API key."""
    normalized_key = PAGEINDEX_API_KEY.strip().lower()
    if (
        not normalized_key
        or "..." in normalized_key
        or "your_" in normalized_key
        or "replace" in normalized_key
    ):
        raise ValueError("Thiếu PAGEINDEX_API_KEY hợp lệ trong file .env")

    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_doc_ids() -> dict[str, str]:
    if not DOC_IDS_FILE.exists():
        return {}
    try:
        data = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(name): str(doc_id) for name, doc_id in data.items() if doc_id}


def _save_doc_ids(doc_ids: dict[str, str]) -> None:
    DOC_IDS_FILE.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _markdown_to_pdf(md_file: Path) -> Path:
    """Chuyển Markdown sang PDF Unicode để upload lên PageIndex cloud."""
    from fpdf import FPDF

    md_file = md_file.resolve()
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    relative_name = md_file.relative_to(STANDARDIZED_DIR).as_posix()
    pdf_name = relative_name.replace("/", "__").rsplit(".", 1)[0] + ".pdf"
    pdf_path = PDF_CACHE_DIR / pdf_name

    # Chỉ build lại khi Markdown mới hơn file PDF cache.
    if pdf_path.exists() and pdf_path.stat().st_mtime >= md_file.stat().st_mtime:
        return pdf_path

    font_regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font_bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if not font_regular.exists() or not font_bold.exists():
        raise FileNotFoundError("Không tìm thấy font DejaVu để tạo PDF Unicode")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("DejaVu", style="", fname=str(font_regular))
    pdf.add_font("DejaVu", style="B", fname=str(font_bold))
    pdf.add_page()

    for raw_line in md_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            pdf.ln(3)
            continue

        # Bỏ URL ảnh và markup cơ bản; nội dung text vẫn được giữ.
        line = re.sub(r"!\[[^]]*]\([^)]*\)", "", line)
        line = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", line)
        heading = len(line) - len(line.lstrip("#"))
        line = line.lstrip("#").strip()
        if not line:
            continue

        if heading:
            pdf.set_font("DejaVu", style="B", size=max(11, 17 - heading))
        else:
            pdf.set_font("DejaVu", size=10)
        pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(pdf_path))
    return pdf_path


def _submit_document(client, pdf_path: Path) -> dict:
    """Upload với timeout; SDK 0.1.x không truyền timeout cho requests."""
    import requests

    with pdf_path.open("rb") as pdf_file:
        response = requests.post(
            f"{client.BASE_URL}/doc/",
            headers=client._headers(),
            files={"file": pdf_file},
            data={"if_retrieval": True},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
    response.raise_for_status()
    return response.json()


def upload_documents() -> dict[str, str]:
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    client = _get_client()
    doc_ids = _load_doc_ids()
    markdown_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not markdown_files:
        print(f"⚠ Không có file Markdown trong {STANDARDIZED_DIR}")
        return doc_ids

    for md_file in markdown_files:
        source = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        if source in doc_ids:
            print(f"  → Skipped (cached): {source} -> {doc_ids[source]}")
            continue

        pdf_path = _markdown_to_pdf(md_file)
        response = _submit_document(client, pdf_path)
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex không trả doc_id cho {source}: {response}")

        doc_ids[source] = str(doc_id)
        _save_doc_ids(doc_ids)
        print(f"  ✓ Uploaded: {source} -> {doc_id}")

    return doc_ids


def _wait_for_retrieval(client, retrieval_id: str) -> dict:
    """Poll retrieval cho đến khi hoàn thành hoặc hết timeout."""
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = client.get_retrieval(retrieval_id)
        status = str(response.get("status", "")).lower()
        if status in {"completed", "complete", "succeeded", "success"}:
            return response
        if status in {"failed", "error", "cancelled"}:
            raise RuntimeError(f"PageIndex retrieval thất bại: {response}")
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"PageIndex retrieval quá {POLL_TIMEOUT_SECONDS} giây")


def _parse_retrieval(response: dict, doc_id: str, document: str) -> list[dict]:
    """Parse schema legacy `retrieved_nodes[].relevant_contents`."""
    parsed = []
    for node in response.get("retrieved_nodes", []):
        for group in node.get("relevant_contents", []):
            items = group if isinstance(group, list) else [group]
            for item in items:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("relevant_content") or "").strip()
                if not content:
                    continue
                parsed.append({
                    "content": content,
                    "metadata": {
                        "section": item.get("section_title") or node.get("title"),
                        "document": document,
                        "doc_id": doc_id,
                    },
                    "source": "pageindex",
                })
    return parsed


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    doc_ids = _load_doc_ids()
    if not doc_ids:
        return []

    client = _get_client()
    results = []
    for document, doc_id in doc_ids.items():
        if not client.is_retrieval_ready(doc_id):
            continue

        submitted = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
        if not retrieval_id:
            continue
        response = _wait_for_retrieval(client, str(retrieval_id))
        results.extend(_parse_retrieval(response, doc_id, document))

    # Legacy retrieval không cung cấp relevance score. Reciprocal rank tạo
    # score ổn định, giảm dần và phù hợp interface chung.
    for rank, result in enumerate(results, start=1):
        result["score"] = 1.0 / rank
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
