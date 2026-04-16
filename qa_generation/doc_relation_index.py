"""文档级 chunk 关系索引：基于实体重叠和 embedding 相似度构建关系边。"""
from __future__ import annotations

import math
from itertools import combinations

from models import EnrichedChunk


def build_relation_index(
    chunks: list[EnrichedChunk],
    embedding_vectors: dict[str, list[float]],
    similarity_threshold: float = 0.6,
) -> list[dict]:
    """构建 chunk 对之间的关系索引。

    优先检查实体重叠（entities_overlap），无重叠时检查 embedding 余弦相似度
    （summary_similarity）。两个条件都不满足则不生成关系边。

    Args:
        chunks: 文档的富化 chunk 列表
        embedding_vectors: chunk_id → 向量的映射（可为空 dict）
        similarity_threshold: 余弦相似度阈值，默认 0.6

    Returns:
        关系边列表，每条边为 dict：
        {left_chunk_id, right_chunk_id, relation_type, signal}
    """
    relations: list[dict] = []
    for left, right in combinations(chunks, 2):
        overlap = sorted(set(left.entities) & set(right.entities))
        if overlap:
            relations.append({
                "left_chunk_id": left.chunk_id,
                "right_chunk_id": right.chunk_id,
                "relation_type": "entities_overlap",
                "signal": overlap,
            })
            continue
        left_vec = embedding_vectors.get(left.chunk_id)
        right_vec = embedding_vectors.get(right.chunk_id)
        similarity = _cosine(left_vec, right_vec)
        if similarity >= similarity_threshold:
            relations.append({
                "left_chunk_id": left.chunk_id,
                "right_chunk_id": right.chunk_id,
                "relation_type": "summary_similarity",
                "signal": [round(similarity, 4)],
            })
    return relations


def _cosine(vec_a: list[float] | None, vec_b: list[float] | None) -> float:
    """计算两个向量的余弦相似度。向量为 None 或零向量时返回 0.0。"""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
