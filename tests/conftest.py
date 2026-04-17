"""pytest 共享 Fixture。"""

from __future__ import annotations

import dataclasses

import pytest

from models import ContentItem, ParsedDocument, QAPair, SemanticChunk


@pytest.fixture
def sample_content_item() -> ContentItem:
    return ContentItem(
        item_id="doc1_item_0",
        doc_id="doc1",
        type="text",
        content="深度学习是机器学习的一个分支，使用多层神经网络处理数据，在图像识别和自然语言处理等领域取得了突破性进展。",
        page_idx=0,
    )


@pytest.fixture
def sample_table_item() -> ContentItem:
    return ContentItem(
        item_id="doc1_item_1",
        doc_id="doc1",
        type="table",
        content="<table><tr><th>产品</th><th>价格</th></tr><tr><td>A</td><td>100</td></tr></table>",
        page_idx=1,
    )


@pytest.fixture
def sample_parsed_document(
    sample_content_item: ContentItem, sample_table_item: ContentItem
) -> ParsedDocument:
    return ParsedDocument(
        doc_id="doc1",
        source_path="doc1.pdf",
        markdown="# 示例文档\n\n深度学习是机器学习的一个分支...",
        content_list=[sample_content_item, sample_table_item],
        metadata={"backend": "pipeline", "total_items": 2},
    )


@pytest.fixture
def sample_text_chunk() -> SemanticChunk:
    return SemanticChunk(
        chunk_id="doc1_chunk_0",
        doc_id="doc1",
        content_type="text",
        content="深度学习是机器学习的一个分支，使用多层神经网络处理数据。",
        source_items=["doc1_item_0"],
        page_idx=0,
        token_count=30,
    )


@pytest.fixture
def sample_table_chunk() -> SemanticChunk:
    return SemanticChunk(
        chunk_id="doc1_chunk_1",
        doc_id="doc1",
        content_type="table",
        content="<table><tr><th>产品</th><th>价格</th></tr><tr><td>A</td><td>100</td></tr></table>",
        source_items=["doc1_item_1"],
        page_idx=1,
        token_count=50,
    )


@pytest.fixture
def sample_qa_pair() -> QAPair:
    return QAPair(
        qa_id="qa_0",
        chunk_id="doc1_chunk_0",
        doc_id="doc1",
        question="什么是深度学习？",
        answer="深度学习是机器学习的一个分支，使用多层神经网络处理数据。",
        context="深度学习是机器学习的一个分支，使用多层神经网络处理数据，在图像识别和NLP领域取得突破。",
        question_type="definition",
        content_type="text",
    )


@pytest.fixture
def passed_qa_pair(sample_qa_pair: QAPair) -> QAPair:
    """已通过质检的 QA 对（返回副本，不修改原 fixture）。"""
    return dataclasses.replace(
        sample_qa_pair,
        quality_score=0.85,
        faithfulness_score=0.90,
        relevance_score=0.80,
        completeness_score=0.85,
        is_passed=True,
    )
