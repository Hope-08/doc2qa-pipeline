"""Advanced QA 生成器：文档级配额 + 多跳/推理/条件问题路由。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import QAPair, SemanticChunk
    from utils.llm_client import LLMClient


class AdvancedQAGenerator:
    """按文档级配额生成高质量 QA，支持多种问题类型。

    注意：本类仅通过 generate_qa_task(qa_strategy="advanced") 路由调用，
    实际执行的是配额截断的简单生成，而非完整的 enrich→synthesize 两阶段链路。
    完整 advanced 路径由 pipeline/flow.py 中的 enrich_task + synthesize_qa_task 实现。

    Args:
        llm_client: LLM 调用客户端
        max_qa_per_doc: 每个文档最多生成的 QA 数，最小值 1
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_qa_per_doc: int = 10,
    ) -> None:
        if max_qa_per_doc < 1:
            raise ValueError(f"max_qa_per_doc 必须 >= 1，实际值：{max_qa_per_doc}")
        self.llm_client = llm_client
        self.max_qa_per_doc = max_qa_per_doc

    def generate_document_qa(self, chunks: list[SemanticChunk]) -> list[QAPair]:
        """为单个文档的 chunk 列表生成 QA，总数 <= max_qa_per_doc。

        Args:
            chunks: 属于同一文档的 SemanticChunk 列表

        Returns:
            QA 列表，长度 <= self.max_qa_per_doc

        Raises:
            ValueError: 如果 chunks 来自多个不同文档
        """
        from qa_generation.multi_type_generator import QAGenerator

        # 验证只有单文档
        doc_ids = {c.doc_id for c in chunks if c.doc_id}
        if len(doc_ids) > 1:
            raise ValueError(
                f"generate_document_qa 只接受单文档的 chunks，"
                f"但收到了 {len(doc_ids)} 个不同 doc_id：{doc_ids}"
            )

        # 使用现有 QAGenerator 生成候选 QA
        generator = QAGenerator(llm_client=self.llm_client)
        all_qa = generator.generate_batch(chunks)

        # 按 quality_score 排序（None 视为 0），取配额上限
        ranked = sorted(all_qa, key=lambda q: q.quality_score or 0.0, reverse=True)
        return ranked[: self.max_qa_per_doc]
