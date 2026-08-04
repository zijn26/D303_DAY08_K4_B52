"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.

Pipeline thật (task4-10: chunking, semantic/lexical search, reranking, generation) đã
implement xong — __main__ bên dưới dùng `RealRAGPipeline` (bọc task9.retrieve +
task10.generate_with_citation). `MockRAGPipeline` vẫn được giữ lại làm placeholder để
test nhanh luồng RAGAS mà không cần gọi LLM/embedding thật.

Cần OPENROUTER_API_KEY (hoặc OPENAI_API_KEY) trong `.env` cho cả 2 mục đích: (1) LLM sinh
câu trả lời trong task10, và (2) LLM giám khảo (judge) mà RAGAS dùng để chấm faithfulness/
answer_relevancy/context_recall/context_precision.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Console Windows mặc định dùng cp1252, không encode được tiếng Việt có dấu khi print().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Cho phép `import src...` chạy đúng dù script được gọi trực tiếp
# (python group_project/evaluation/eval_pipeline.py) hay qua `python -m ...`.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAG pipeline interface + Mock (dùng tạm khi task9/task10 chưa xong)
# =============================================================================

class MockRAGPipeline:
    """
    Pipeline giả lập dùng để test luồng evaluation (KHÔNG phải RAG thật).

    Trả lời bằng chính expected_answer/expected_context của golden dataset,
    chỉ để dữ liệu có shape hợp lệ (answer, contexts) cho RAGAS chạy hết pipeline.
    Điểm số vì vậy sẽ cao giả tạo — không đại diện cho chất lượng RAG thật.
    Thay bằng `RealRAGPipeline` (bọc src.task10_generation.generate_with_citation)
    ngay khi task4-10 implement xong.
    """

    def generate_with_citation(self, query: str, golden_item: dict | None = None) -> dict:
        answer = golden_item["expected_answer"] if golden_item else "Tôi không thể xác minh thông tin này từ nguồn hiện có"
        context = golden_item["expected_context"] if golden_item else "unknown"
        return {
            "answer": answer,
            "sources": [{"content": f"[MOCK] {context}: {answer}", "metadata": {"source": context}}],
            "retrieval_source": "mock",
        }


class RealRAGPipeline:
    """
    Bọc pipeline thật (task9 retrieve + task10 generation) vào interface chung.

    `use_reranking` cho phép compare_configs() làm A/B thật sự (hybrid+rerank vs
    dense-only), vì generate_with_citation() gốc không expose tham số này —
    ở đây ta gọi lại retrieve() trực tiếp với cờ tương ứng rồi tái dùng phần
    reorder/format/LLM-call của task10.
    """

    def __init__(self, use_reranking: bool = True):
        self.use_reranking = use_reranking

    def set_config(self, use_reranking: bool = True, **_ignored) -> None:
        self.use_reranking = use_reranking

    def generate_with_citation(self, query: str, golden_item: dict | None = None) -> dict:
        from src.task9_retrieval_pipeline import retrieve
        from src.task10_generation import (
            TOP_K, TOP_P, TEMPERATURE, LLM_MODEL, SYSTEM_PROMPT,
            reorder_for_llm, format_context,
        )

        chunks = retrieve(query, top_k=TOP_K, use_reranking=self.use_reranking)
        reordered = reorder_for_llm(chunks)
        context = format_context(reordered)
        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            ret_source = chunks[0].get("source", "hybrid") if chunks else "none"
            return {
                "answer": "⚠️ [Chưa cấu hình API Key] Vui lòng điền OPENROUTER_API_KEY hoặc OPENAI_API_KEY vào .env",
                "sources": chunks,
                "retrieval_source": ret_source,
            }

        from openai import OpenAI
        if os.getenv("OPENROUTER_API_KEY"):
            client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
            model_name = os.getenv("LLM_MODEL", LLM_MODEL)
        else:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model_name = "gpt-4o-mini"

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        }


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def _build_ragas_llm_and_embeddings():
    """
    Cấu hình LLM giám khảo (judge) + embeddings cho RAGAS.

    LLM: dùng OpenRouter (giống Task 10) qua ChatOpenAI với base_url tuỳ chỉnh —
    cần OPENROUTER_API_KEY (hoặc OPENAI_API_KEY để fallback thẳng OpenAI).

    Embeddings: gọi qua OpenRouter (model baai/bge-m3 — cùng model task4 đã dùng để
    build chroma_db/) cho metric answer_relevancy. KHÔNG dùng sentence-transformers
    local vì môi trường này có xung đột numpy/scipy làm sentence-transformers crash
    khi import (numpy 1.26 nhưng scipy cài đòi numpy>=2.0) — gọi API tránh được import chain đó.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if openrouter_key:
        chat = ChatOpenAI(
            model=os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-4o-mini"),
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )
        embeddings = OpenAIEmbeddings(
            model="baai/bge-m3",
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            check_embedding_ctx_length=False,
        )
    elif openai_key:
        chat = ChatOpenAI(
            model=os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            temperature=0,
        )
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
    else:
        raise RuntimeError(
            "Cần OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong .env để RAGAS gọi LLM giám khảo "
            "(xem .env.example)."
        )

    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embeddings)


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]):
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in golden_dataset:
        result = rag_pipeline.generate_with_citation(item["question"], golden_item=item)
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append([c["content"] for c in result["sources"]])
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)

    llm, embeddings = _build_ragas_llm_and_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
    )
    return result.to_pandas()


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="EcommerceSupport_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa ít nhất 2 configs bằng RAGAS.

    Configs so sánh:
    - hybrid_rerank: hybrid search + reranking (mặc định, use_reranking=True)
    - dense_only: chỉ dense/semantic search, không reranking (use_reranking=False)

    Lưu ý: chỉ RealRAGPipeline phân biệt được config (qua set_config). Nếu chạy với
    MockRAGPipeline (không có set_config), hai config sẽ cho điểm giống hệt nhau vì
    mock luôn trả expected_answer bất kể use_reranking.
    """
    configs = {
        "hybrid_rerank": {"use_reranking": True},
        "dense_only": {"use_reranking": False},
    }

    results = {}
    for config_name, params in configs.items():
        set_config = getattr(rag_pipeline, "set_config", None)
        if callable(set_config):
            set_config(**params)
        df = evaluate_with_ragas(rag_pipeline, golden_dataset)
        results[config_name] = df

    return results


# =============================================================================
# Export Results
# =============================================================================

def export_results(results, comparison: dict):
    """Export evaluation results to results.md"""
    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    content = "# RAG Evaluation Results\n\n"

    content += "## Overall Scores\n\n"
    content += "| Metric | Score |\n|--------|-------|\n"
    for col in metric_cols:
        if col in results.columns:
            content += f"| {col} | {results[col].mean():.3f} |\n"

    content += "\n## A/B Comparison\n\n"
    content += "| Config | " + " | ".join(metric_cols) + " |\n"
    content += "|--------|" + "--------|" * len(metric_cols) + "\n"
    for config_name, df in comparison.items():
        row = [f"{df[col].mean():.3f}" if col in df.columns else "n/a" for col in metric_cols]
        content += f"| {config_name} | " + " | ".join(row) + " |\n"

    content += "\n## Worst Performers\n\n"
    content += "| Question | " + " | ".join(metric_cols) + " |\n"
    content += "|----------|" + "--------|" * len(metric_cols) + "\n"
    if "faithfulness" in results.columns:
        worst = results.sort_values("faithfulness").head(3)
        for _, row in worst.iterrows():
            scores = [f"{row[col]:.3f}" if col in results.columns else "n/a" for col in metric_cols]
            content += f"| {row['question']} | " + " | ".join(scores) + " |\n"

    content += "\n## Recommendations\n\n"
    recs = []
    weak_metrics = [
        col for col in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
        if col in results.columns and results[col].mean() < 0.7
    ]
    if "context_precision" in weak_metrics or "context_recall" in weak_metrics:
        recs.append(
            "- context_precision/context_recall thấp → retriever (task9) chưa lấy đủ/đúng "
            "evidence; cân nhắc calibrate lại SCORE_THRESHOLD hoặc trọng số RRF."
        )
    if "faithfulness" in weak_metrics:
        recs.append(
            "- faithfulness thấp → câu trả lời của LLM (task10) không bám sát context được "
            "cấp; cân nhắc siết lại SYSTEM_PROMPT hoặc giảm temperature."
        )
    if "answer_relevancy" in weak_metrics:
        recs.append(
            "- answer_relevancy thấp → câu trả lời lạc đề so với câu hỏi; kiểm tra lại "
            "reorder_for_llm/format_context có giữ đúng câu hỏi gốc trong prompt không."
        )
    if not recs:
        recs.append("- Tất cả metric đều ≥ 0.7 trên golden dataset hiện tại — chưa phát hiện điểm yếu rõ rệt.")
    recs.append(
        "- Xem bảng A/B Comparison ở trên để quyết định có nên bật reranking "
        "(use_reranking=True) trong production hay không."
    )
    content += "\n".join(recs) + "\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"✓ Results exported to {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    pipeline = RealRAGPipeline()

    results = evaluate_with_ragas(pipeline, golden_dataset)
    print("\n=== RAGAS scores (per question) ===")
    print(results)

    comparison = compare_configs(pipeline, golden_dataset)
    export_results(results, comparison)
