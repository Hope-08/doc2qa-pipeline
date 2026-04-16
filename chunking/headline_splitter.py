"""按 heading 类型切分文档章节的分块器。

策略：
- 遇到 heading 类型的 ContentItem 时开始新 section
- section 内文本合并为单个 chunk
- 超长 section（> chunk_size * 1.5 字符）交给 SemanticChunker 二次切分
"""

from __future__ import annotations

from chunking.semantic_chunker import SemanticChunker
from models import ContentItem, SemanticChunk
from utils.logger import get_logger

logger = get_logger(__name__)

# 超长 section 判定倍数
_LONG_SECTION_MULTIPLIER = 1.5


class HeadlineSplitter:
    """基于 heading ContentItem 边界切分文档章节的分块器。

    Args:
        embedding_model:      语义二次切分使用的 embedding 模型。
        chunk_size:           单个 chunk 的字符数上限。
        similarity_threshold: SemanticChunker 语义切分阈值。
    """

    def __init__(
        self,
        embedding_model: str = "azure-openai/text-embedding-3-small",
        chunk_size: int = 512,
        similarity_threshold: float = 0.3,
    ) -> None:
        self._chunk_size = chunk_size
        self._semantic_chunker = SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
        )

    def split(self, doc_id: str, items: list[ContentItem]) -> list[SemanticChunk]:
        """将 content_list 按 heading 边界切分为章节 chunk 列表。

        Args:
            doc_id: 所属文档 ID。
            items:  文档的 ContentItem 列表。

        Returns:
            SemanticChunk 列表。
        """
        sections = self._collect_sections(items)
        chunks: list[SemanticChunk] = []
        chunk_idx = 0

        for section_items in sections:
            section_text = "\n".join(item.content for item in section_items if item.content.strip())
            if not section_text.strip():
                continue

            source_ids = [item.item_id for item in section_items]
            page_idx = section_items[0].page_idx
            long_threshold = self._chunk_size * _LONG_SECTION_MULTIPLIER

            if len(section_text) > long_threshold:
                sub_texts_list = self._semantic_chunker.chunk_texts_batch([section_text])
                sub_texts = sub_texts_list[0] if sub_texts_list else [section_text]
            else:
                sub_texts = [section_text]

            # 构建 item 在 section_text 中的字符偏移映射（仅非空内容的 item）
            item_offsets: list[tuple[int, int, str]] = []
            pos = 0
            for item in section_items:
                if not item.content.strip():
                    continue
                item_offsets.append((pos, pos + len(item.content), item.item_id))
                pos += len(item.content) + 1  # +1 对应 "\n".join 中的换行

            search_cursor = 0
            for sub in sub_texts:
                if not sub.strip():
                    continue

                # 在 section_text 中定位当前子文本的起止位置
                sub_start = section_text.find(sub, search_cursor)
                if sub_start == -1:
                    sub_start = search_cursor  # 降级：找不到时按顺序推进
                sub_end = sub_start + len(sub)

                # 仅保留与当前子文本字符范围有重叠的 item
                sub_source_ids = [
                    iid for (start, end, iid) in item_offsets if start < sub_end and end > sub_start
                ] or source_ids  # 无匹配时保底使用全量

                search_cursor = sub_end

                chunks.append(
                    SemanticChunk(
                        chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                        doc_id=doc_id,
                        content_type="text",
                        content=sub,
                        source_items=sub_source_ids,
                        page_idx=page_idx,
                        token_count=len(sub),
                    )
                )
                chunk_idx += 1

        logger.info(
            "HeadlineSplitter 分块完成 [%s]：%d items → %d sections → %d chunks",
            doc_id,
            len(items),
            len(sections),
            len(chunks),
        )
        return chunks

    @staticmethod
    def _collect_sections(items: list[ContentItem]) -> list[list[ContentItem]]:
        """按 heading 边界将 items 分组为 section 列表。

        每遇到一个 heading 类型的 item 即开启新 section。
        heading 本身作为该 section 的第一个 item。

        Args:
            items: 原始 ContentItem 列表。

        Returns:
            section 列表，每个 section 是 ContentItem 列表。
        """
        sections: list[list[ContentItem]] = []
        current: list[ContentItem] = []

        for item in items:
            if item.type == "heading":
                if current:
                    sections.append(current)
                current = [item]
            else:
                current.append(item)

        if current:
            sections.append(current)

        return sections
