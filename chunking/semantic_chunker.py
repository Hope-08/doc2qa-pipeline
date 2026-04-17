"""语义分块核心实现，基于 OpenAI Embeddings 的语义相似度切分。"""

from __future__ import annotations

import re

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# chunk 内容最短字符数，低于此值不做语义切分，直接保留
_MIN_TEXT_FOR_SPLIT = 50
# 语义切分后单个 chunk 的最短字符数；更短的 chunk 会被合并到相邻 chunk
_MIN_CHUNK_CHARS = 30


class SemanticChunker:
    """基于 OpenAI Embeddings 语义相似度的分块器。

    算法：
    1. 按标点切句
    2. 用 OpenAI Embeddings 计算相邻句子的余弦相似度
    3. 相似度低于 similarity_threshold 处作为语义边界切分
    4. 超过 chunk_size 的片段按字符数二次切分

    Args:
        embedding_model:      OpenAI embedding 模型名，如 azure-openai/text-embedding-3-small。
        chunk_size:           单个 chunk 的字符数上限。
        similarity_threshold: 相邻句子相似度低于此值时切分（推荐 0.3 ~ 0.6）。
    """

    def __init__(
        self,
        embedding_model: str = "azure-openai/text-embedding-3-small",
        chunk_size: int = 512,
        similarity_threshold: float = 0.3,
    ) -> None:
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self._embedder = None  # 延迟加载

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def chunk_text(self, text: str) -> list[str]:
        """对纯文本进行语义切分，返回子文本列表。

        Args:
            text: 输入文本，长度不限。

        Returns:
            切分后的子文本列表。若 text 过短，直接返回 ``[text]``。
        """
        if len(text) < _MIN_TEXT_FOR_SPLIT:
            return [text]

        sentences = self._split_sentences(text)
        if len(sentences) <= 2:
            return [text]

        try:
            embedder = self._get_embedder()
            all_embs = np.array(embedder.embed_documents(sentences))
        except Exception as exc:
            logger.warning("embedding API 调用失败，降级为字符切分：%s", exc)
            return self._char_split(text)
        norms = np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8
        all_embs = all_embs / norms

        return self._chunks_from_embeddings(sentences, all_embs)

    def chunk_texts_batch(self, texts: list[str]) -> list[list[str]]:
        """批量切分多个文本，整批句子仅发起一次 embedding API 调用。

        相比逐文本调用 ``chunk_text``，对有大量 text ContentItem 的文档可显著减少 API 调用次数。

        Args:
            texts: 待切分文本列表。

        Returns:
            与 texts 等长的列表，每项为该文本切分后的子文本列表。
        """
        if not texts:
            return []

        # 区分需要 embedding 的文本和直接返回的短文本
        results: list[tuple[int, list[str]]] = []  # (原始 idx, chunks)
        to_embed: list[tuple[int, list[str]]] = []  # (原始 idx, sentences)

        for i, text in enumerate(texts):
            if len(text) < _MIN_TEXT_FOR_SPLIT:
                results.append((i, [text]))
                continue
            sentences = self._split_sentences(text)
            if len(sentences) <= 2:
                results.append((i, [text]))
                continue
            to_embed.append((i, sentences))

        if to_embed:
            # 收集所有句子到一个列表，记录每段的起始偏移量
            all_sentences: list[str] = []
            offsets: list[int] = []
            for _, sentences in to_embed:
                offsets.append(len(all_sentences))
                all_sentences.extend(sentences)
            offsets.append(len(all_sentences))  # 末尾哨兵

            try:
                embedder = self._get_embedder()
                raw_embs = np.array(embedder.embed_documents(all_sentences))
            except Exception as exc:
                logger.warning("批量 embedding API 调用失败，降级为字符切分：%s", exc)
                for orig_idx, text in ((i, texts[i]) for i, _ in to_embed):
                    results.append((orig_idx, self._char_split(text)))
                results.sort(key=lambda x: x[0])
                return [chunks for _, chunks in results]
            norms = np.linalg.norm(raw_embs, axis=1, keepdims=True) + 1e-8
            raw_embs = raw_embs / norms

            for embed_idx, (orig_idx, sentences) in enumerate(to_embed):
                start = offsets[embed_idx]
                end = offsets[embed_idx + 1]
                embs = raw_embs[start:end]
                chunks = self._chunks_from_embeddings(sentences, embs)
                results.append((orig_idx, chunks))

        results.sort(key=lambda x: x[0])
        return [chunks for _, chunks in results]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _chunks_from_embeddings(self, sentences: list[str], embs: np.ndarray) -> list[str]:
        """根据预计算的归一化 embedding 矩阵执行语义切分。

        Args:
            sentences: 句子列表。
            embs:      已归一化的 embedding 矩阵，shape=(len(sentences), dim)。

        Returns:
            切分后的子文本列表。
        """
        cut_points = [0]
        for i in range(1, len(sentences)):
            sim = float(np.dot(embs[i - 1], embs[i]))
            if sim < self.similarity_threshold:
                cut_points.append(i)
        cut_points.append(len(sentences))

        chunks: list[str] = []
        for start, end in zip(cut_points[:-1], cut_points[1:]):
            segment = "".join(sentences[start:end])
            if len(segment) > self.chunk_size:
                for j in range(0, len(segment), self.chunk_size):
                    sub = segment[j : j + self.chunk_size]
                    if sub.strip():
                        chunks.append(sub)
            elif segment.strip():
                chunks.append(segment)

        if not chunks:
            return ["".join(sentences)]

        # 后合并：消灭微小 chunk（< _MIN_CHUNK_CHARS），合并到相邻 chunk
        merged: list[str] = [chunks[0]]
        for chunk in chunks[1:]:
            if len(merged[-1]) < _MIN_CHUNK_CHARS or len(chunk) < _MIN_CHUNK_CHARS:
                merged[-1] += chunk
            else:
                merged.append(chunk)
        # 若最后一个 chunk 仍过短，合并到倒数第二个
        if len(merged) >= 2 and len(merged[-1]) < _MIN_CHUNK_CHARS:
            merged[-2] += merged.pop()
        return merged

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """按中英文标点切句。

        注：不在 \\n 处切分——PDF 提取的换行往往是排版行折，并非句子边界。
        真正的段落分隔已由 TypeAwareChunker._merge_short_items 在合并时用 \\n
        连接并保留，此处仅按句末标点切分即可。

        中文引号后的标点（如 。"）先切分在 。 处，下一句以 " 开头，
        属于常见场景，对语义边界检测影响可接受。
        """
        parts = re.split(r"(?<=[。！？.!?])\s*", text)
        return [p for p in parts if p.strip()]

    def _get_embedder(self):
        """延迟初始化 OpenAIEmbeddings。"""
        if self._embedder is None:
            from langchain_openai import OpenAIEmbeddings

            logger.info("初始化 OpenAI embedding 模型：%s", self.embedding_model)
            self._embedder = OpenAIEmbeddings(model=self.embedding_model)
        return self._embedder

    def _char_split(self, text: str) -> list[str]:
        """纯字符切分降级策略（embedding API 不可用时使用）。"""
        if len(text) <= self.chunk_size:
            return [text]
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
            if text[i : i + self.chunk_size].strip()
        ]
