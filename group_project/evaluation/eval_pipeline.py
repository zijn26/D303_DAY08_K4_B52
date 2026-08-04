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

⚠️ TRẠNG THÁI HIỆN TẠI: src/task9_retrieval_pipeline.py (retrieve) và
src/task10_generation.py (generate_with_citation) chưa được implement (các task nền
tảng task4-8 cũng vậy). Vì vậy __main__ bên dưới dùng MockRAGPipeline để có thể chạy thử
toàn bộ luồng RAGAS (question → answer/contexts → metrics → results.md) ngay bây giờ.
Khi pipeline thật (task4-10) hoàn thiện, đổi `pipeline = MockRAGPipeline()` thành
`from src.task10_generation import generate_with_citation` và bọc nó cùng interface
(xem `RealRAGPipeline` bên dưới) rồi chạy lại.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
    """Bọc src.task10_generation.generate_with_citation vào interface chung."""

    def generate_with_citation(self, query: str, golden_item: dict | None = None) -> dict:
        from src.task10_generation import generate_with_citation
        return generate_with_citation(query)


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

    Embeddings: dùng sentence-transformers CHẠY LOCAL (không tốn API call) cho
    metric answer_relevancy, vì OpenRouter không có endpoint embeddings.
    """
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
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
    elif openai_key:
        chat = ChatOpenAI(
            model=os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            temperature=0,
        )
    else:
        raise RuntimeError(
            "Cần OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong .env để RAGAS gọi LLM giám khảo "
            "(xem .env.example)."
        )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

    Lưu ý: MockRAGPipeline không phân biệt config (luôn trả expected_answer),
    nên hai config sẽ cho điểm gần như giống nhau khi chạy mock — đây là hạn chế
    kỳ vọng cho tới khi cắm RealRAGPipeline (retrieve() thật có tham số use_reranking).
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
    content += (
        "- Kết quả trên chạy với MockRAGPipeline (placeholder trả lời bằng chính "
        "expected_answer/expected_context) — điểm số CHƯA phản ánh RAG pipeline thật.\n"
        "- Sau khi task4-10 (chunking, semantic/lexical search, reranking, generation) "
        "hoàn thiện, chạy lại với `RealRAGPipeline` để có kết quả thực tế.\n"
        "- Khi có kết quả thật, tập trung phân tích các câu hỏi có context_precision/"
        "context_recall thấp trước — dấu hiệu retrieval chưa lấy đúng evidence.\n"
    )

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"✓ Results exported to {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    # ⚠ Pipeline thật (src.task10_generation.generate_with_citation) chưa implement
    # xong (phụ thuộc task4-9 cũng đang là TODO). Dùng MockRAGPipeline để test luồng
    # RAGAS end-to-end ngay bây giờ. Đổi sang `RealRAGPipeline()` khi pipeline sẵn sàng.
    pipeline = MockRAGPipeline()

    results = evaluate_with_ragas(pipeline, golden_dataset)
    print("\n=== RAGAS scores (per question) ===")
    print(results)

    comparison = compare_configs(pipeline, golden_dataset)
    export_results(results, comparison)
