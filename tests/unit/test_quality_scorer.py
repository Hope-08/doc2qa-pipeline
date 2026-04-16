"""质量评分器单元测试。"""

from __future__ import annotations

import dataclasses
import json
from unittest.mock import MagicMock, patch

import pytest

from models import QAPair, QualityThresholds
from qa_filtering.quality_scorer import QualityScorer


def _make_qa(
    question: str = "什么是深度学习？",
    answer: str = "深度学习是机器学习的一个分支，使用多层神经网络处理数据。",
    context: str = "深度学习是机器学习的一个分支，使用多层神经网络处理数据，在图像识别领域取得突破。",
    qa_id: str = "qa_0",
) -> QAPair:
    return QAPair(
        qa_id=qa_id,
        chunk_id="chunk_0",
        doc_id="doc1",
        question=question,
        answer=answer,
        context=context,
        question_type="definition",
        content_type="text",
    )


def _langchain_patch():
    """使 langchain_core 在测试中可被动态导入。"""
    mock_lc = MagicMock()
    mock_lc_msg = MagicMock()
    mock_lc_msg.HumanMessage = MagicMock(side_effect=lambda content: content)
    return patch.dict("sys.modules", {
        "langchain_core": mock_lc,
        "langchain_core.messages": mock_lc_msg,
    })


def _make_mock_llm(scores: list[dict]) -> MagicMock:
    """构造返回特定 JSON 的 mock ChatOpenAI。"""
    resp = MagicMock()
    resp.content = json.dumps(scores, ensure_ascii=False)
    llm = MagicMock()
    llm.invoke.return_value = resp
    return llm


# ---------------------------------------------------------------------------
# _score_completeness（纯规则，无需 mock）
# ---------------------------------------------------------------------------


class TestScoreCompleteness:
    def test_very_short_answer(self):
        """少于 15 字符且无句号时，得分最低。"""
        score = QualityScorer._score_completeness("太短了")
        assert score < 0.5

    def test_short_answer_no_period(self):
        """15~50 字符之间、无句号，得分中等偏低。"""
        score = QualityScorer._score_completeness("这是一段没有句号的答案文本，长度在50字以内")
        assert 0.3 <= score <= 0.8

    def test_medium_answer_with_period(self):
        """50~200 字符、有句号，得分应接近最高。"""
        answer = "深度学习是机器学习的子领域，使用多层神经网络从原始数据中自动学习层次化特征表示。" * 2
        score = QualityScorer._score_completeness(answer)
        assert score >= 0.8

    def test_long_answer_with_period(self):
        """超过 200 字符且有句号，满分。"""
        answer = "深度学习在图像识别、语音识别和自然语言处理领域取得了显著成果。" * 10
        score = QualityScorer._score_completeness(answer)
        assert score == 1.0

    def test_period_detection_chinese(self):
        """中文句号应被识别。"""
        score_with = QualityScorer._score_completeness("这是一段有句号的答案。" * 5)
        score_without = QualityScorer._score_completeness("这是一段没有句号的答案" * 5)
        assert score_with > score_without

    def test_period_detection_english(self):
        """英文句点应被识别。"""
        score = QualityScorer._score_completeness("This is an answer with a period." * 5)
        assert score >= 0.8


# ---------------------------------------------------------------------------
# score_batch（需要 mock LLM + Embeddings）
# ---------------------------------------------------------------------------


class TestScoreBatch:
    def _make_scorer_with_mocks(
        self,
        llm_scores: list[dict],
        embed_vectors: list[list[float]],
        thresholds: QualityThresholds | None = None,
    ) -> QualityScorer:
        scorer = QualityScorer(
            nli_model="azure-openai/gpt-5.4",
            embedding_model="azure-openai/text-embedding-3-small",
            thresholds=thresholds,
        )
        scorer._llm = _make_mock_llm(llm_scores)
        embedder = MagicMock()
        embedder.embed_documents.return_value = embed_vectors
        scorer._embedder = embedder
        return scorer

    def test_returns_same_length_as_input(self):
        """输出列表长度应与输入相同。"""
        with _langchain_patch():
            scorer = self._make_scorer_with_mocks(
                llm_scores=[{"id": 0, "score": 0.9}],
                embed_vectors=[[1.0, 0.0], [1.0, 0.0]],
            )
            result = scorer.score_batch([_make_qa()])

        assert len(result) == 1

    def test_high_quality_qa_is_passed(self):
        """高质量 QA（高 faithfulness + relevance + completeness）应通过质检。"""
        qa = _make_qa(
            answer="深度学习是机器学习的一个分支，使用多层神经网络处理数据。它在图像识别和自然语言处理领域取得了显著进展。"
        )
        thresholds = QualityThresholds(
            faithfulness_min=0.5,
            relevance_min=0.3,
            answer_min_length=10,
            composite_pass=0.5,
        )

        with _langchain_patch():
            scorer = self._make_scorer_with_mocks(
                llm_scores=[{"id": 0, "score": 0.9}],
                embed_vectors=[[1.0, 0.0], [1.0, 0.0]],  # Q 和 A 相同向量 → 相似度 ≈ 1.0
                thresholds=thresholds,
            )
            result = scorer.score_batch([qa])

        assert result[0].is_passed is True
        assert result[0].fail_reasons == []

    def test_low_faithfulness_qa_fails(self):
        """faithfulness 低于阈值时，is_passed 应为 False 且 fail_reasons 非空。"""
        qa = _make_qa(answer="这个答案和上下文完全无关，纯属幻觉。")
        thresholds = QualityThresholds(faithfulness_min=0.7)

        with _langchain_patch():
            scorer = self._make_scorer_with_mocks(
                llm_scores=[{"id": 0, "score": 0.3}],  # 低分
                embed_vectors=[[1.0, 0.0], [1.0, 0.0]],
                thresholds=thresholds,
            )
            result = scorer.score_batch([qa])

        assert result[0].is_passed is False
        assert any("faithfulness" in r for r in result[0].fail_reasons)

    def test_short_answer_fails(self):
        """答案过短时，is_passed 应为 False，fail_reasons 含"答案过短"。"""
        qa = _make_qa(answer="短")
        thresholds = QualityThresholds(answer_min_length=15)

        with _langchain_patch():
            scorer = self._make_scorer_with_mocks(
                llm_scores=[{"id": 0, "score": 0.9}],
                embed_vectors=[[1.0, 0.0], [0.9, 0.1]],
                thresholds=thresholds,
            )
            result = scorer.score_batch([qa])

        assert result[0].is_passed is False
        assert any("答案过短" in r for r in result[0].fail_reasons)

    def test_scores_are_populated_on_result(self):
        """result QAPair 应填充所有评分字段，且值在合理范围内。"""
        qa = _make_qa(
            answer="深度学习是机器学习的一个分支，使用多层神经网络处理数据。它在图像识别领域取得了显著进展。"
        )

        with _langchain_patch():
            scorer = self._make_scorer_with_mocks(
                llm_scores=[{"id": 0, "score": 0.85}],
                embed_vectors=[[1.0, 0.0], [1.0, 0.0]],
            )
            result = scorer.score_batch([qa])

        qa_result = result[0]
        assert qa_result.faithfulness_score is not None
        assert qa_result.relevance_score is not None
        assert qa_result.completeness_score is not None
        assert qa_result.quality_score is not None
        assert 0.0 <= qa_result.faithfulness_score <= 1.0
        assert 0.0 <= qa_result.relevance_score <= 1.0

    def test_llm_unavailable_uses_default_faithfulness(self):
        """LLM 不可用时，faithfulness 使用默认值 0.5，评分仍正常执行。"""
        qa = _make_qa(
            answer="深度学习是机器学习的一个分支，使用多层神经网络处理数据。"
        )
        scorer = QualityScorer()
        embedder = MagicMock()
        embedder.embed_documents.return_value = [[1.0, 0.0], [1.0, 0.0]]
        scorer._embedder = embedder

        with patch("qa_filtering.quality_scorer.QualityScorer._get_llm",
                   side_effect=ImportError("no llm")):
            result = scorer.score_batch([qa])

        # 不报错，faithfulness 为 0.5
        assert result[0].faithfulness_score == pytest.approx(0.5)

    def test_empty_input(self):
        """空输入不报错，返回空列表。"""
        scorer = QualityScorer()
        result = scorer.score_batch([])
        assert result == []
