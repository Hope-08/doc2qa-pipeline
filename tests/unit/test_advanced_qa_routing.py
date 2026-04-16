"""测试 generate_qa_task 的 advanced/simple 路由逻辑。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_chunk_dict(doc_id: str = "doc1") -> dict:
    return {
        "chunk_id": f"{doc_id}_chunk_0",
        "doc_id": doc_id,
        "content": "故宫始建于明永乐四年，历时十四年建成，是中国古代宫廷建筑之精华。",
        "content_type": "text",
        "page_idx": 0,
        "source_items": [],
        "token_count": 30,
    }


# ---------------------------------------------------------------------------
# Fail-fast 校验测试（不应该触达 LLMClient）
# ---------------------------------------------------------------------------


def test_invalid_strategy_raises_before_llm():
    """非法 qa_strategy 应在进入 LLMClient 之前抛 ValueError。"""
    from pipeline.tasks import generate_qa_task

    with pytest.raises(ValueError, match="qa_strategy 无效"):
        generate_qa_task.fn(
            chunks_dict=[make_chunk_dict()],
            model="gpt-4o",
            temperature=0.7,
            qa_strategy="invalid_strategy",
        )


def test_invalid_advanced_quota_raises_before_llm():
    """advanced_max_qa_per_doc < 1 应在进入 LLMClient 之前抛 ValueError。"""
    from pipeline.tasks import generate_qa_task

    with pytest.raises(ValueError, match="advanced_max_qa_per_doc 必须 >= 1"):
        generate_qa_task.fn(
            chunks_dict=[make_chunk_dict()],
            model="gpt-4o",
            temperature=0.7,
            qa_strategy="advanced",
            advanced_max_qa_per_doc=0,
        )


# ---------------------------------------------------------------------------
# simple 路由测试
# ---------------------------------------------------------------------------


def test_simple_strategy_uses_qa_generator():
    """simple 路由应使用 QAGenerator，不使用 AdvancedQAGenerator。"""
    from models import QAPair
    from pipeline.tasks import generate_qa_task

    real_qa = QAPair(
        qa_id="qa1",
        chunk_id="doc1_chunk_0",
        doc_id="doc1",
        question="Q1",
        answer="A1",
        context="ctx",
        question_type="fact",
        content_type="text",
    )

    with (
        patch("utils.llm_client.LLMClient"),
        patch("qa_generation.multi_type_generator.QAGenerator") as mock_qagen_cls,
        patch("qa_generation.advanced_generator.AdvancedQAGenerator") as mock_adv_cls,
    ):
        mock_qagen_cls.return_value.generate_batch.return_value = [real_qa]
        result = generate_qa_task.fn(
            chunks_dict=[make_chunk_dict()],
            model="gpt-4o",
            temperature=0.7,
            qa_strategy="simple",
        )

    mock_qagen_cls.assert_called_once()
    mock_adv_cls.assert_not_called()
    assert len(result) == 1


# ---------------------------------------------------------------------------
# advanced 路由测试
# ---------------------------------------------------------------------------


def test_advanced_strategy_uses_advanced_generator():
    """advanced 路由应使用 AdvancedQAGenerator，不使用 QAGenerator。"""
    from models import QAPair
    from pipeline.tasks import generate_qa_task

    real_qa = QAPair(
        qa_id="qa1",
        chunk_id="doc1_chunk_0",
        doc_id="doc1",
        question="Q1",
        answer="A1",
        context="ctx",
        question_type="fact",
        content_type="text",
    )

    with (
        patch("utils.llm_client.LLMClient"),
        patch("qa_generation.advanced_generator.AdvancedQAGenerator") as mock_adv_cls,
        patch("qa_generation.multi_type_generator.QAGenerator") as mock_qagen_cls,
    ):
        mock_adv_cls.return_value.generate_document_qa.return_value = [real_qa]
        result = generate_qa_task.fn(
            chunks_dict=[make_chunk_dict()],
            model="gpt-4o",
            temperature=0.7,
            qa_strategy="advanced",
            advanced_max_qa_per_doc=5,
        )

    mock_adv_cls.assert_called_once()
    mock_qagen_cls.assert_not_called()
    assert len(result) == 1


# ---------------------------------------------------------------------------
# AdvancedQAGenerator 单元测试
# ---------------------------------------------------------------------------


def test_advanced_generator_init_rejects_zero_quota():
    """AdvancedQAGenerator 初始化时 max_qa_per_doc < 1 应抛 ValueError。"""
    from qa_generation.advanced_generator import AdvancedQAGenerator

    with pytest.raises(ValueError, match="max_qa_per_doc 必须 >= 1"):
        AdvancedQAGenerator(llm_client=MagicMock(), max_qa_per_doc=0)


def test_advanced_generator_rejects_multi_doc_chunks():
    """generate_document_qa 传入多 doc_id 的 chunks 应抛 ValueError。"""
    from models import SemanticChunk
    from qa_generation.advanced_generator import AdvancedQAGenerator

    chunks = [
        SemanticChunk(
            chunk_id="doc1_chunk_0",
            doc_id="doc1",
            content="内容1",
            content_type="text",
            page_idx=0,
            source_items=[],
            token_count=5,
        ),
        SemanticChunk(
            chunk_id="doc2_chunk_0",
            doc_id="doc2",
            content="内容2",
            content_type="text",
            page_idx=0,
            source_items=[],
            token_count=5,
        ),
    ]
    gen = AdvancedQAGenerator(llm_client=MagicMock(), max_qa_per_doc=5)
    with pytest.raises(ValueError, match="单文档"):
        gen.generate_document_qa(chunks)


def test_advanced_generator_respects_quota():
    """generate_document_qa 返回结果数 <= max_qa_per_doc。"""
    from models import QAPair, SemanticChunk
    from qa_generation.advanced_generator import AdvancedQAGenerator

    chunks = [
        SemanticChunk(
            chunk_id="doc1_chunk_0",
            doc_id="doc1",
            content="故宫始建于明永乐四年",
            content_type="text",
            page_idx=0,
            source_items=[],
            token_count=10,
        )
    ]
    # 模拟 QAGenerator 返回 20 条
    many_qa = [
        QAPair(
            qa_id=f"qa{i}",
            chunk_id="doc1_chunk_0",
            doc_id="doc1",
            question=f"Q{i}",
            answer=f"A{i}",
            context="ctx",
            question_type="fact",
            content_type="text",
        )
        for i in range(20)
    ]

    gen = AdvancedQAGenerator(llm_client=MagicMock(), max_qa_per_doc=5)
    with patch("qa_generation.multi_type_generator.QAGenerator") as mock_cls:
        mock_cls.return_value.generate_batch.return_value = many_qa
        result = gen.generate_document_qa(chunks)

    assert len(result) == 5
    mock_cls.return_value.generate_batch.assert_called_once()
