"""单元测试：doc_relation_index 文档关系索引构建。"""

from __future__ import annotations

from models import EnrichedChunk
from qa_generation.doc_relation_index import build_relation_index


def _make_chunk(
    chunk_id: str,
    content: str = "内容",
    entities: list[str] | None = None,
) -> EnrichedChunk:
    return EnrichedChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        content_type="text",
        content=content,
        source_items=[f"{chunk_id}_item"],
        page_idx=0,
        token_count=10,
        entities=entities or [],
    )


def test_build_relation_index_creates_entity_overlap_edge() -> None:
    """两个 chunk 共享实体时生成 entities_overlap 关系。"""
    chunks = [
        _make_chunk("doc_chunk_0", entities=["乌鲁木齐", "天山"]),
        _make_chunk("doc_chunk_1", entities=["乌鲁木齐", "机场"]),
    ]
    relations = build_relation_index(chunks, embedding_vectors={})
    assert len(relations) == 1
    assert relations[0]["relation_type"] == "entities_overlap"
    assert relations[0]["left_chunk_id"] == "doc_chunk_0"
    assert relations[0]["right_chunk_id"] == "doc_chunk_1"
    assert "乌鲁木齐" in relations[0]["signal"]


def test_build_relation_index_creates_similarity_edge_when_no_entity_overlap() -> None:
    """无实体重叠但 embedding 相似度足够时生成 summary_similarity 关系。"""
    chunks = [
        _make_chunk("doc_chunk_0", entities=["酒店"]),
        _make_chunk("doc_chunk_1", entities=["宾馆"]),
    ]
    relations = build_relation_index(
        chunks,
        embedding_vectors={
            "doc_chunk_0": [1.0, 0.0],
            "doc_chunk_1": [0.9, 0.1],
        },
        similarity_threshold=0.6,
    )
    assert len(relations) == 1
    assert relations[0]["relation_type"] == "summary_similarity"


def test_build_relation_index_no_edge_when_both_conditions_fail() -> None:
    """无实体重叠且相似度低于阈值时不生成关系。"""
    chunks = [
        _make_chunk("doc_chunk_0", entities=["天山"]),
        _make_chunk("doc_chunk_1", entities=["大海"]),
    ]
    relations = build_relation_index(
        chunks,
        embedding_vectors={
            "doc_chunk_0": [1.0, 0.0],
            "doc_chunk_1": [0.0, 1.0],
        },
        similarity_threshold=0.6,
    )
    assert relations == []
