"""幻觉检测器单元测试。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from models import QAPair


def _make_qa(
    answer: str = "深度学习是机器学习分支。",
    context: str = "深度学习是机器学习的一个分支，使用多层神经网络。",
    qa_id: str = "qa_0",
) -> QAPair:
    return QAPair(
        qa_id=qa_id,
        chunk_id="chunk_0",
        doc_id="doc1",
        question="什么是深度学习？",
        answer=answer,
        context=context,
        question_type="definition",
        content_type="text",
    )


def _mock_llm_response(scores: list[dict]) -> MagicMock:
    """构造返回特定 JSON 的 mock LLM。"""
    resp = MagicMock()
    resp.content = json.dumps(scores, ensure_ascii=False)
    llm = MagicMock()
    llm.invoke.return_value = resp
    return llm


def _langchain_patch():
    """返回 patch.dict，使 langchain_core 在测试中可导入。"""
    mock_lc = MagicMock()
    mock_lc_msg = MagicMock()
    mock_lc_msg.HumanMessage = MagicMock(side_effect=lambda content: content)
    return patch.dict(
        "sys.modules",
        {
            "langchain_core": mock_lc,
            "langchain_core.messages": mock_lc_msg,
        },
    )


# ---------------------------------------------------------------------------
# _run_faithfulness_batch
# ---------------------------------------------------------------------------


class TestRunFaithfulnessBatch:
    def test_normal_scores_returned(self):
        """LLM 返回格式正确时，分数应与 LLM 返回值一致。"""
        with _langchain_patch():
            from qa_filtering.hallucination_detector import _run_faithfulness_batch

            qa = _make_qa()
            llm = _mock_llm_response([{"id": 0, "score": 0.9}])
            scores = _run_faithfulness_batch(llm, [qa])

        assert len(scores) == 1
        assert abs(scores[0] - 0.9) < 1e-6

    def test_empty_context_returns_zero_without_llm(self):
        """context 为空的 QA 直接返回 0.0，不调用 LLM。"""
        with _langchain_patch():
            from qa_filtering.hallucination_detector import _run_faithfulness_batch

            qa = _make_qa(context="   ")
            llm = MagicMock()
            scores = _run_faithfulness_batch(llm, [qa])

        assert scores == [0.0]
        llm.invoke.assert_not_called()

    def test_score_clamped_to_0_1(self):
        """LLM 返回超出 [0,1] 范围的分数应被截断。"""
        with _langchain_patch():
            from qa_filtering.hallucination_detector import _run_faithfulness_batch

            qa = _make_qa()
            llm = _mock_llm_response([{"id": 0, "score": 1.5}])
            scores = _run_faithfulness_batch(llm, [qa])

        assert scores[0] == 1.0

    def test_missing_item_in_response_defaults_to_0_5(self):
        """LLM 返回中遗漏某个条目时，该条目使用默认值 0.5。"""
        with _langchain_patch():
            from qa_filtering.hallucination_detector import _run_faithfulness_batch

            qa0 = _make_qa(qa_id="qa_0")
            qa1 = _make_qa(qa_id="qa_1")
            # 只返回 id=0，缺少 id=1
            llm = _mock_llm_response([{"id": 0, "score": 0.8}])
            scores = _run_faithfulness_batch(llm, [qa0, qa1])

        assert len(scores) == 2
        assert abs(scores[0] - 0.8) < 1e-6
        assert scores[1] == 0.5

    def test_item_with_missing_score_key_is_skipped(self):
        """LLM 返回条目缺少 score 字段时，该条目被跳过，不影响其他条目。"""
        with _langchain_patch():
            from qa_filtering.hallucination_detector import _run_faithfulness_batch

            qa0 = _make_qa(qa_id="qa_0")
            qa1 = _make_qa(qa_id="qa_1")
            # id=0 格式正常，id=1 缺少 score
            llm = _mock_llm_response(
                [
                    {"id": 0, "score": 0.7},
                    {"id": 1},  # 缺少 score
                ]
            )
            scores = _run_faithfulness_batch(llm, [qa0, qa1])

        assert scores[0] == pytest.approx(0.7)
        assert scores[1] == 0.5  # 跳过后使用默认值

    def test_malformed_json_response_falls_back_to_0_5(self):
        """LLM 返回非 JSON 内容时，所有条目回退到 0.5。"""
        with _langchain_patch():
            from qa_filtering.hallucination_detector import _run_faithfulness_batch

            qa = _make_qa()
            resp = MagicMock()
            resp.content = "我无法解析这个请求"
            llm = MagicMock()
            llm.invoke.return_value = resp
            scores = _run_faithfulness_batch(llm, [qa])

        assert scores == [0.5]

    def test_non_array_json_falls_back_to_0_5(self):
        """LLM 返回 JSON 对象（非数组）时，所有条目回退到 0.5。"""
        with _langchain_patch():
            from qa_filtering.hallucination_detector import _run_faithfulness_batch

            qa = _make_qa()
            resp = MagicMock()
            resp.content = '{"id": 0, "score": 0.9}'
            llm = MagicMock()
            llm.invoke.return_value = resp
            scores = _run_faithfulness_batch(llm, [qa])

        assert scores == [0.5]

    def test_mixed_empty_and_normal_context(self):
        """混合空 context 和正常 context 时，空的直接 0.0，其余走 LLM。"""
        with _langchain_patch():
            from qa_filtering.hallucination_detector import _run_faithfulness_batch

            qa_empty = _make_qa(context="", qa_id="qa_empty")
            qa_normal = _make_qa(qa_id="qa_normal")
            # needs_llm 只包含 qa_normal（index=1），LLM 返回 id=1
            llm = _mock_llm_response([{"id": 1, "score": 0.75}])
            scores = _run_faithfulness_batch(llm, [qa_empty, qa_normal])

        assert scores[0] == 0.0
        assert scores[1] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# HallucinationDetector
# ---------------------------------------------------------------------------


class TestHallucinationDetector:
    def test_detect_hallucination_below_threshold(self):
        """faithfulness 低于阈值的 QA 应标记为幻觉 True。"""
        with _langchain_patch():
            from qa_filtering.hallucination_detector import HallucinationDetector

            detector = HallucinationDetector(threshold=0.70)
            qa = _make_qa()
            detector._llm = _mock_llm_response([{"id": 0, "score": 0.5}])
            result = detector.detect([qa])

        assert result == [True]

    def test_detect_no_hallucination_above_threshold(self):
        """faithfulness 高于阈值的 QA 应标记为非幻觉 False。"""
        with _langchain_patch():
            from qa_filtering.hallucination_detector import HallucinationDetector

            detector = HallucinationDetector(threshold=0.60)
            qa = _make_qa()
            detector._llm = _mock_llm_response([{"id": 0, "score": 0.9}])
            result = detector.detect([qa])

        assert result == [False]

    def test_detect_llm_unavailable_returns_all_false(self):
        """LLM 不可用时，所有 QA 标记为非幻觉（放行），不报错。"""
        from qa_filtering.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()

        with patch.object(detector, "_get_llm", side_effect=ImportError("no llm")):
            result = detector.detect([_make_qa()])

        assert result == [False]

    def test_detect_empty_list(self):
        """空列表输入不报错，返回空列表。"""
        from qa_filtering.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()
        result = detector.detect([])
        assert result == []
