"""单元测试：models 模块。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from models import (
    ContentItem,
    EnrichedChunk,
    QAPair,
    QualityThresholds,
    SemanticChunk,
)


class TestContentItem:
    def test_create_text_item(self) -> None:
        item = ContentItem(
            item_id="doc1_item_0",
            doc_id="doc1",
            type="text",
            content="深度学习是机器学习的一个分支。",
            page_idx=0,
        )
        assert item.type == "text"
        assert item.page_idx == 0
        assert item.bbox is None

    def test_create_table_item(self) -> None:
        item = ContentItem(
            item_id="doc1_item_1",
            doc_id="doc1",
            type="table",
            content="<table><tr><td>数据</td></tr></table>",
            page_idx=1,
        )
        assert item.type == "table"
        assert "<table>" in item.content


class TestSemanticChunk:
    def test_create_text_chunk(self) -> None:
        chunk = SemanticChunk(
            chunk_id="doc1_chunk_0",
            doc_id="doc1",
            content_type="text",
            content="深度学习是机器学习的一个分支。",
            source_items=["doc1_item_0"],
            page_idx=0,
            token_count=16,
        )
        assert chunk.content_type == "text"
        assert chunk.token_count == 16

    def test_create_table_chunk(self) -> None:
        chunk = SemanticChunk(
            chunk_id="doc1_chunk_1",
            doc_id="doc1",
            content_type="table",
            content="<table></table>",
            source_items=["doc1_item_1"],
            page_idx=0,
            token_count=15,
        )
        assert chunk.content_type == "table"


class TestEnrichedChunk:
    def test_create_enriched_chunk_with_defaults(self) -> None:
        chunk = EnrichedChunk(
            chunk_id="doc1_chunk_0",
            doc_id="doc1",
            content_type="text",
            content="深度学习是机器学习的一个分支。",
            source_items=["doc1_item_0"],
            page_idx=0,
            token_count=16,
        )
        assert chunk.entities == []
        assert chunk.summary is None
        assert chunk.metadata == {}

    def test_enriched_chunk_stores_enriched_values(self) -> None:
        chunk = EnrichedChunk(
            chunk_id="doc1_chunk_2",
            doc_id="doc1",
            content_type="text",
            content="本节介绍检索增强生成的基本流程。",
            source_items=["doc1_item_2", "doc1_item_3"],
            page_idx=2,
            token_count=24,
            entities=["检索增强生成", "向量数据库"],
            summary="概述检索增强生成流程与关键组件。",
            metadata={"section": "2.1", "confidence": 0.93, "keywords": ["RAG", "embedding"]},
        )

        assert chunk.entities == ["检索增强生成", "向量数据库"]
        assert chunk.summary == "概述检索增强生成流程与关键组件。"
        assert chunk.metadata == {
            "section": "2.1",
            "confidence": 0.93,
            "keywords": ["RAG", "embedding"],
        }

    def test_enriched_chunk_mutable_defaults_are_isolated(self) -> None:
        chunk1 = EnrichedChunk(
            chunk_id="doc1_chunk_3",
            doc_id="doc1",
            content_type="text",
            content="第一段内容。",
            source_items=["doc1_item_4"],
            page_idx=3,
            token_count=8,
        )
        chunk2 = EnrichedChunk(
            chunk_id="doc1_chunk_4",
            doc_id="doc1",
            content_type="text",
            content="第二段内容。",
            source_items=["doc1_item_5"],
            page_idx=4,
            token_count=8,
        )

        chunk1.entities.append("RAG")
        chunk1.metadata["keywords"] = ["embedding"]

        assert chunk1.entities == ["RAG"]
        assert chunk1.metadata == {"keywords": ["embedding"]}
        assert chunk2.entities == []
        assert chunk2.metadata == {}


class TestQAPair:
    def test_default_scores_none(self) -> None:
        qa = QAPair(
            qa_id="qa_0",
            chunk_id="doc1_chunk_0",
            doc_id="doc1",
            question="什么是深度学习？",
            answer="深度学习是机器学习的一个分支，使用多层神经网络。",
            context="深度学习是机器学习的一个分支，使用多层神经网络处理数据。",
            question_type="definition",
            content_type="text",
        )
        assert qa.quality_score is None
        assert qa.is_passed is False
        assert qa.fail_reasons == []

    def test_fail_reasons_mutable_default(self) -> None:
        qa1 = QAPair(
            qa_id="1", chunk_id="c1", doc_id="d1",
            question="q", answer="a", context="ctx",
            question_type="definition", content_type="text",
        )
        qa2 = QAPair(
            qa_id="2", chunk_id="c2", doc_id="d1",
            question="q2", answer="a2", context="ctx2",
            question_type="reasoning", content_type="text",
        )
        qa1.fail_reasons.append("test")
        # qa2 的 fail_reasons 不应受影响（dataclass field(default_factory=list)）
        assert qa2.fail_reasons == []


class TestQualityThresholds:
    def test_frozen(self) -> None:
        thr = QualityThresholds()
        with pytest.raises(FrozenInstanceError):
            thr.faithfulness_min = 0.9  # type: ignore[misc]

    def test_custom_values(self) -> None:
        thr = QualityThresholds(faithfulness_min=0.7, composite_pass=0.65)
        assert thr.faithfulness_min == 0.7
        assert thr.composite_pass == 0.65
