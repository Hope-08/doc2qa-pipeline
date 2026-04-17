"""去重过滤器单元测试。"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import numpy as np

from models import QAPair
from qa_filtering.deduplicator import Deduplicator


def _make_qa(
    question: str, answer: str = "答案内容。", qa_id: str = "qa_0", quality_score: float = 0.8
) -> QAPair:
    return dataclasses.replace(
        QAPair(
            qa_id=qa_id,
            chunk_id="chunk_0",
            doc_id="doc1",
            question=question,
            answer=answer,
            context="上下文内容。",
            question_type="definition",
            content_type="text",
        ),
        quality_score=quality_score,
    )


def _make_embedder_with_vectors(vectors: list[list[float]]) -> MagicMock:
    """构造返回指定向量序列的 mock 嵌入器。"""
    embedder = MagicMock()
    embedder.embed_documents.return_value = vectors
    return embedder


class TestDeduplicator:
    def test_single_item_returned_unchanged(self):
        """只有一个 QA 时，直接返回，不调用嵌入模型。"""
        qa = _make_qa("什么是深度学习？")
        dedup = Deduplicator()

        with patch.object(dedup, "_get_embedder") as mock_embed:
            result = dedup.deduplicate([qa])

        mock_embed.assert_not_called()
        assert len(result) == 1
        assert result[0] is qa

    def test_empty_list_returned_unchanged(self):
        """空列表输入返回空列表。"""
        dedup = Deduplicator()
        result = dedup.deduplicate([])
        assert result == []

    def test_identical_questions_keeps_one(self):
        """完全相同的两个问题（余弦相似度=1.0）应只保留一个。"""
        qa1 = _make_qa("什么是深度学习？", qa_id="qa_1")
        qa2 = _make_qa("什么是深度学习？", qa_id="qa_2")

        # 两个向量完全相同 → 余弦相似度 = 1.0 > 任何合理阈值
        vectors = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        dedup = Deduplicator(similarity_threshold=0.9)
        dedup._embedder = _make_embedder_with_vectors(vectors)

        result = dedup.deduplicate([qa1, qa2])

        assert len(result) == 1
        assert result[0].qa_id == "qa_1"

    def test_different_questions_all_kept(self):
        """完全不同的问题（余弦相似度=0.0）全部保留。"""
        qa1 = _make_qa("什么是深度学习？", qa_id="qa_1")
        qa2 = _make_qa("什么是强化学习？", qa_id="qa_2")
        qa3 = _make_qa("什么是自然语言处理？", qa_id="qa_3")

        # 三个正交向量 → 相互余弦相似度 = 0.0
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        dedup = Deduplicator(similarity_threshold=0.9)
        dedup._embedder = _make_embedder_with_vectors(vectors)

        result = dedup.deduplicate([qa1, qa2, qa3])

        assert len(result) == 3

    def test_greedy_keeps_first_occurrence(self):
        """贪心策略：相似对中保留先遍历到的（即质量分更高的，调用前应预先排序）。"""
        qa_high = _make_qa("什么是神经网络？", qa_id="qa_high", quality_score=0.95)
        qa_low = _make_qa("什么是神经网络训练？", qa_id="qa_low", quality_score=0.65)

        # qa_high 与 qa_low 高度相似
        vectors = [
            [1.0, 0.0],
            [0.99, 0.14],  # 与第一个余弦相似度约 0.99，超过阈值
        ]
        norms = [np.linalg.norm(v) for v in vectors]
        normed = [[v[i] / norms[j] for i in range(len(v))] for j, v in enumerate(vectors)]
        # 实际相似度 = dot(normed[0], normed[1])
        sim = sum(a * b for a, b in zip(normed[0], normed[1]))

        dedup = Deduplicator(similarity_threshold=sim - 0.01)  # 阈值略低于实际相似度 → 会去重
        dedup._embedder = _make_embedder_with_vectors(vectors)

        result = dedup.deduplicate([qa_high, qa_low])

        assert len(result) == 1
        assert result[0].qa_id == "qa_high"

    def test_embedding_unavailable_returns_original(self):
        """嵌入模型不可用时，跳过去重，返回原始列表。"""
        qa1 = _make_qa("问题一", qa_id="qa_1")
        qa2 = _make_qa("问题二", qa_id="qa_2")
        dedup = Deduplicator()

        with patch.object(dedup, "_get_embedder", side_effect=ImportError("no embedder")):
            result = dedup.deduplicate([qa1, qa2])

        assert len(result) == 2

    def test_threshold_boundary(self):
        """相似度低于阈值时应保留，高于阈值时应去重（严格小于比较）。"""
        qa1 = _make_qa("问题 A", qa_id="qa_1")
        qa2 = _make_qa("问题 B", qa_id="qa_2")

        # 两个正交向量：余弦相似度 = 0.0，远低于任何合理阈值，两个都应保留
        dedup_keep = Deduplicator(similarity_threshold=0.9)
        dedup_keep._embedder = _make_embedder_with_vectors([[1.0, 0.0], [0.0, 1.0]])
        result_keep = dedup_keep.deduplicate([qa1, qa2])
        assert len(result_keep) == 2

        # 两个完全相同向量：归一化后余弦相似度 ≈ 1.0，应去重（保留第一个）
        dedup_dedup = Deduplicator(similarity_threshold=0.9)
        dedup_dedup._embedder = _make_embedder_with_vectors([[1.0, 0.0], [1.0, 0.0]])
        result_dedup = dedup_dedup.deduplicate([qa1, qa2])
        assert len(result_dedup) == 1


# ---------------------------------------------------------------------------
# question + answer 联合去重（新增）
# ---------------------------------------------------------------------------


class TestQuestionAnswerDedup:
    """测试基于 question + answer 联合文本的去重行为。"""

    def _make_embedder_with_vectors(self, vectors):
        embedder = MagicMock()
        embedder.embed_documents.return_value = vectors
        return embedder

    def test_same_question_different_answer_kept(self):
        """相同问题但答案不同时，联合文本相似度低，应都保留。"""
        qa1 = _make_qa(
            "深度学习有哪些优势？", answer="深度学习能自动提取特征，减少人工工程。", qa_id="qa_1"
        )
        qa2 = _make_qa(
            "深度学习有哪些优势？",
            answer="深度学习在大数据场景下比传统方法准确率更高。",
            qa_id="qa_2",
        )

        # 模拟：联合文本 Q+A 的向量差异较大（正交）
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        dedup = Deduplicator(similarity_threshold=0.9)
        dedup._embedder = self._make_embedder_with_vectors(vectors)

        result = dedup.deduplicate([qa1, qa2])
        assert len(result) == 2

    def test_same_question_same_answer_deduped(self):
        """相同问题相同答案（联合文本完全一致），应只保留一个。"""
        qa1 = _make_qa("深度学习是什么？", answer="深度学习是机器学习的一个分支。", qa_id="qa_1")
        qa2 = _make_qa("深度学习是什么？", answer="深度学习是机器学习的一个分支。", qa_id="qa_2")

        vectors = [[1.0, 0.0], [1.0, 0.0]]  # 相同向量
        dedup = Deduplicator(similarity_threshold=0.9)
        dedup._embedder = self._make_embedder_with_vectors(vectors)

        result = dedup.deduplicate([qa1, qa2])
        assert len(result) == 1
