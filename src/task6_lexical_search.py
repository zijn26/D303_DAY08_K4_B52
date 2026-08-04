"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re

from .task4_chunking_indexing import chunk_documents, load_documents

# Corpus dùng chung cách chia chunk với Task 4 để dense search và BM25
# trả về các đơn vị tài liệu tương thích khi hybrid retrieval.
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
BM25_INDEX = None

# Query expansion nhẹ cho corpus song ngữ: vẫn là lexical retrieval, nhưng bổ sung
# các từ khóa tiếng Việt tương ứng khi người dùng hỏi bằng tiếng Anh.
QUERY_EXPANSIONS = {
    "return": ["trả", "hàng"],
    "refund": ["hoàn", "tiền"],
    "evidence": ["bằng", "chứng"],
    "policy": ["chính", "sách"],
    "payment": ["thanh", "toán"],
    "methods": ["phương", "thức"],
    "seller": ["người", "bán"],
    "listing": ["đăng", "bán"],
    "regulations": ["quy", "định"],
    "order": ["đơn", "hàng"],
    "tracking": ["theo", "dõi"],
    "guide": ["hướng", "dẫn"],
}


def tokenize(text: str) -> list[str]:
    """Tokenize không phân biệt hoa/thường, giữ chữ Việt và mã sản phẩm."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def tokenize_query(query: str) -> list[str]:
    """Tokenize query và mở rộng các thuật ngữ TMĐT Anh–Việt phổ biến."""
    tokens = tokenize(query)
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(QUERY_EXPANSIONS.get(token, []))
    return expanded


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    if not corpus:
        return None

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)


def initialize_index() -> None:
    """Lazy-load documents và chỉ tạo BM25 index một lần trong mỗi process."""
    global CORPUS, BM25_INDEX

    if BM25_INDEX is not None:
        return

    documents = load_documents()
    CORPUS = chunk_documents(documents) if documents else []
    BM25_INDEX = build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    initialize_index()
    if BM25_INDEX is None or not CORPUS:
        return []

    query_tokens = tokenize_query(query)
    if not query_tokens:
        return []

    scores = BM25_INDEX.get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: float(scores[idx]),
        reverse=True,
    )

    results = []
    for idx in ranked_indices:
        score = float(scores[idx])
        if score <= 0:
            continue

        document = CORPUS[idx]
        results.append({
            "content": document["content"],
            "score": score,
            "metadata": document.get("metadata", {}),
        })
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
