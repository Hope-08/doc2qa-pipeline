"""HeadlineSplitter 单元测试。"""

from __future__ import annotations

from chunking.headline_splitter import HeadlineSplitter
from chunking.type_aware_chunker import TypeAwareChunker
from models import ContentItem, ParsedDocument


def test_split_uses_heading_boundaries() -> None:
    """HeadlineSplitter 应按 heading 类型的 ContentItem 切分 section。"""
    splitter = HeadlineSplitter(
        embedding_model="test-model",
        chunk_size=800,
        similarity_threshold=0.3,
    )
    items = [
        ContentItem("d_item_0", "d", "heading", "第一章", 0),
        ContentItem("d_item_1", "d", "text", "乌鲁木齐段内容", 0),
        ContentItem("d_item_2", "d", "heading", "第二章", 1),
        ContentItem("d_item_3", "d", "text", "喀纳斯段内容", 1),
    ]
    chunks = splitter.split(doc_id="d", items=items)
    assert [chunk.content for chunk in chunks] == [
        "第一章\n乌鲁木齐段内容",
        "第二章\n喀纳斯段内容",
    ]


def test_split_large_section_with_semantic_chunker(monkeypatch) -> None:
    """超长 section 应交给 SemanticChunker 二次切分。"""
    splitter = HeadlineSplitter(
        embedding_model="test-model",
        chunk_size=10,
        similarity_threshold=0.3,
    )
    monkeypatch.setattr(
        splitter._semantic_chunker,
        "chunk_texts_batch",
        lambda texts: [["第一段", "第二段"]],
    )
    items = [
        ContentItem("d_item_0", "d", "heading", "第一章", 0),
        ContentItem("d_item_1", "d", "text", "很长很长的内容" * 10, 0),
    ]
    chunks = splitter.split(doc_id="d", items=items)
    assert [chunk.content for chunk in chunks] == ["第一段", "第二段"]


# ─── Task 4 tests ───────────────────────────────────────────────────────


def test_chunk_document_uses_headline_splitter_when_enough_headings(monkeypatch) -> None:
    """heading 数量 >= headline_min_count 时应走 HeadlineSplitter。"""
    parsed = ParsedDocument(
        doc_id="d",
        source_path="sample.pdf",
        markdown="",
        content_list=[
            ContentItem("d_item_0", "d", "heading", "第一章", 0),
            ContentItem("d_item_1", "d", "text", "内容一", 0),
            ContentItem("d_item_2", "d", "heading", "第二章", 1),
            ContentItem("d_item_3", "d", "text", "内容二", 1),
        ],
    )
    chunker = TypeAwareChunker(
        embedding_model="test-model",
        chunk_size=800,
        similarity_threshold=0.3,
        headline_min_count=2,
    )
    monkeypatch.setattr(chunker._headline_splitter, "split", lambda doc_id, items: ["headline"])
    monkeypatch.setattr(chunker, "_chunk_with_legacy_logic", lambda parsed_doc: ["legacy"])

    chunks = chunker.chunk_document(parsed)

    assert chunks == ["headline"]


def test_split_sub_chunks_have_correct_source_items(monkeypatch) -> None:
    """超长 section 切分后，每个子 chunk 的 source_items 只含实际贡献内容的 item_id。"""
    splitter = HeadlineSplitter(
        embedding_model="test-model",
        chunk_size=5,
        similarity_threshold=0.3,
    )
    items = [
        ContentItem("d_item_0", "d", "text", "第一节内容", 0),
        ContentItem("d_item_1", "d", "text", "第二节内容", 1),
    ]
    # section_text = "第一节内容\n第二节内容" (远超 chunk_size*1.5=7.5)
    monkeypatch.setattr(
        splitter._semantic_chunker,
        "chunk_texts_batch",
        lambda texts: [["第一节内容", "第二节内容"]],
    )
    chunks = splitter.split(doc_id="d", items=items)
    assert len(chunks) == 2
    assert chunks[0].source_items == [
        "d_item_0"
    ], f"第一子 chunk 应只引用 d_item_0，实际：{chunks[0].source_items}"
    assert chunks[1].source_items == [
        "d_item_1"
    ], f"第二子 chunk 应只引用 d_item_1，实际：{chunks[1].source_items}"


def test_chunk_document_falls_back_to_legacy_when_headings_not_enough(monkeypatch) -> None:
    """heading 数量 < headline_min_count 时应走旧逻辑。"""
    parsed = ParsedDocument(
        doc_id="d",
        source_path="sample.pdf",
        markdown="",
        content_list=[ContentItem("d_item_0", "d", "text", "内容一", 0)],
    )
    chunker = TypeAwareChunker(
        embedding_model="test-model",
        chunk_size=800,
        similarity_threshold=0.3,
        headline_min_count=2,
    )
    monkeypatch.setattr(chunker, "_chunk_with_legacy_logic", lambda parsed_doc: ["legacy"])

    chunks = chunker.chunk_document(parsed)

    assert chunks == ["legacy"]
