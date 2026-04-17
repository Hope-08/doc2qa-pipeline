"""去重过滤器：基于 question embedding 相似度去除重复 QA。"""

from __future__ import annotations

import numpy as np

from models import QAPair
from utils.logger import get_logger

logger = get_logger(__name__)


class Deduplicator:
    """基于 OpenAI Embedding 相似度的 QA 去重器。

    对已通过质检的 QA 对，计算问题两两相似度，
    超过阈值则视为重复，保留质量分更高的一个。

    Args:
        embedding_model:      OpenAI embedding 模型名，如 azure-openai/text-embedding-3-small。
        similarity_threshold: 两问题相似度超过此值视为重复（推荐 0.85~0.95）。
    """

    def __init__(
        self,
        embedding_model: str = "azure-openai/text-embedding-3-small",
        similarity_threshold: float = 0.90,
    ) -> None:
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self._embedder = None

    def deduplicate(self, qa_pairs: list[QAPair]) -> list[QAPair]:
        """去重，返回新列表（不修改原列表）。

        使用 question + answer 联合文本做 embedding，相比单独对 question 去重
        能区分"相同问题不同答案"（信息互补，应保留）与"相同问题相同答案"（真正重复）。

        Args:
            qa_pairs: 输入 QA 对，按 quality_score 降序排列效果更好。

        Returns:
            去重后的 QA 对列表。
        """
        if len(qa_pairs) <= 1:
            return list(qa_pairs)

        try:
            embedder = self._get_embedder()
        except Exception:
            logger.warning("embedding 模型不可用，跳过去重")
            return list(qa_pairs)

        # 联合 question + answer 文本以区分"相同问题不同答案"
        combined_texts = [f"{qa.question} {qa.answer}" for qa in qa_pairs]
        embs = np.array(embedder.embed_documents(combined_texts))
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs = embs / norms  # 归一化，点积即余弦相似度

        # 预计算全矩阵相似度
        sim_matrix = embs @ embs.T  # shape: (n, n)

        # 贪心去重：按顺序遍历，若与已保留的任意条目相似度过高则丢弃
        kept_indices: list[int] = []
        for i in range(len(qa_pairs)):
            if not kept_indices or float(np.max(sim_matrix[i, kept_indices])) < self.similarity_threshold:
                kept_indices.append(i)

        removed = len(qa_pairs) - len(kept_indices)
        if removed:
            logger.info("去重：移除 %d 个重复 QA（共 %d → %d）", removed, len(qa_pairs), len(kept_indices))

        return [qa_pairs[i] for i in kept_indices]

    def _get_embedder(self):
        if self._embedder is None:
            from langchain_openai import OpenAIEmbeddings
            logger.info("初始化 OpenAI embedding 模型：%s", self.embedding_model)
            self._embedder = OpenAIEmbeddings(model=self.embedding_model)
        return self._embedder
