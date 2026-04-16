"""类型感知分块器：根据 ContentItem 类型选择不同的分块策略。

策略：
- text          → 相邻小块预合并后交给 SemanticChunker 语义切分
- table         → 整体保留（1 item = 1 chunk，不切分）
- image_caption → 整体保留
- equation      → 跳过（公式不适合直接生成 QA）
"""

from __future__ import annotations

from chunking.headline_splitter import HeadlineSplitter
from chunking.semantic_chunker import SemanticChunker
from models import ContentItem, ParsedDocument, SemanticChunk
from utils.logger import get_logger

logger = get_logger(__name__)

# 跳过的内容类型
_SKIP_TYPES = frozenset({"equation"})
# 不切分直接保留的类型
_PRESERVE_TYPES = frozenset({"table", "image"})
# image 类型在分块后统一标记为 image_caption
_IMAGE_CONTENT_TYPE = "image_caption"

# 预合并阈值：单个 ContentItem 低于此长度视为"短块"，需与邻近短块合并
# 避免每个段落标题/行程条目单独成 chunk 导致上下文碎片化
_MERGE_THRESHOLD = 200


class TypeAwareChunker:
    """类型感知的文档分块器。

    Args:
        embedding_model:      语义分块使用的 embedding 模型。
        chunk_size:           文本块字符数上限。
        similarity_threshold: 语义切分阈值。
    """

    def __init__(
        self,
        embedding_model: str = "azure-openai/text-embedding-3-small",
        chunk_size: int = 512,
        similarity_threshold: float = 0.3,
        headline_min_count: int = 2,
    ) -> None:
        self._semantic_chunker = SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
        )
        self._headline_splitter = HeadlineSplitter(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
        )
        self._chunk_size = chunk_size
        self._headline_min_count = headline_min_count

    def chunk_document(self, parsed_doc: ParsedDocument) -> list[SemanticChunk]:
        """对整个文档的 content_list 进行类型感知分块。

        当 heading 数量 >= headline_min_count 时，路由到 HeadlineSplitter；
        否则走原有逻辑（_chunk_with_legacy_logic）。

        Args:
            parsed_doc: 文档解析结果。

        Returns:
            SemanticChunk 列表，可直接传入 QA 生成模块。
        """
        heading_count = sum(1 for item in parsed_doc.content_list if item.type == "heading")
        if heading_count >= self._headline_min_count:
            return self._headline_splitter.split(parsed_doc.doc_id, parsed_doc.content_list)
        return self._chunk_with_legacy_logic(parsed_doc)

    def _chunk_with_legacy_logic(self, parsed_doc: ParsedDocument) -> list[SemanticChunk]:
        """原有类型感知分块逻辑。

        text 类型的相邻短 ContentItem 先合并成足够大的文本组，
        再批量发起一次 embedding API 调用，避免碎片 chunk 和过多 API 请求。

        Args:
            parsed_doc: 文档解析结果。

        Returns:
            SemanticChunk 列表，可直接传入 QA 生成模块。
        """
        chunks: list[SemanticChunk] = []
        chunk_idx = 0

        # 将 content_list 分组：连续的 text 类型聚合成组，非 text 类型单独处理
        groups = self._group_text_items(parsed_doc.content_list)

        # 对每个 text 组进行预合并，再收集待 embedding 的文本
        merged_groups: list[tuple[list[ContentItem], str]] = []  # (源 items, 合并后文本)
        for is_text, items in groups:
            if not is_text:
                merged_groups.append((items, ""))
            else:
                for merged_items, merged_text in self._merge_short_items(items):
                    merged_groups.append((merged_items, merged_text))

        # 批量对所有 text 组做语义切分（单次 embedding API）
        text_groups = [(items, text) for items, text in merged_groups if text]
        if text_groups:
            texts = [text for _, text in text_groups]
            batched_results = self._semantic_chunker.chunk_texts_batch(texts)
            text_chunk_map: dict[int, list[str]] = {
                i: result for i, result in enumerate(batched_results)
            }
        else:
            text_chunk_map = {}

        text_group_idx = 0
        for items, text in merged_groups:
            if not text:
                # 非 text 类型（table / image）
                item = items[0]
                if item.type in _SKIP_TYPES:
                    continue
                content_type = _IMAGE_CONTENT_TYPE if item.type == "image" else item.type
                chunks.append(SemanticChunk(
                    chunk_id=f"{parsed_doc.doc_id}_chunk_{chunk_idx}",
                    doc_id=parsed_doc.doc_id,
                    content_type=content_type,
                    content=item.content,
                    source_items=[item.item_id],
                    page_idx=item.page_idx,
                    token_count=len(item.content),
                ))
                chunk_idx += 1
            else:
                sub_texts = text_chunk_map.get(text_group_idx, [text])
                source_ids = [item.item_id for item in items]
                page_idx = items[0].page_idx
                for sub in sub_texts:
                    if sub.strip():
                        chunks.append(SemanticChunk(
                            chunk_id=f"{parsed_doc.doc_id}_chunk_{chunk_idx}",
                            doc_id=parsed_doc.doc_id,
                            content_type="text",
                            content=sub,
                            source_items=source_ids,
                            page_idx=page_idx,
                            token_count=len(sub),
                        ))
                        chunk_idx += 1
                text_group_idx += 1

        logger.info(
            "分块完成 [%s]：%d items → %d chunks",
            parsed_doc.doc_id,
            len(parsed_doc.content_list),
            len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _group_text_items(
        items: list[ContentItem],
    ) -> list[tuple[bool, list[ContentItem]]]:
        """将 content_list 按 text/非text 分组，连续 text 项合为一组。"""
        groups: list[tuple[bool, list[ContentItem]]] = []
        for item in items:
            is_text = item.type not in _SKIP_TYPES and item.type not in _PRESERVE_TYPES
            if groups and groups[-1][0] == is_text:
                prev_is_text, prev_items = groups[-1]
                groups[-1] = (prev_is_text, [*prev_items, item])
            else:
                groups.append((is_text, [item]))
        return groups

    def _merge_short_items(
        self,
        items: list[ContentItem],
    ) -> list[tuple[list[ContentItem], str]]:
        """将相邻短 text ContentItem 合并为较大的文本组。

        合并规则：
        - 当前组累计长度 < chunk_size 时，继续追加下一个短块（< _MERGE_THRESHOLD）
        - 遇到长块（>= _MERGE_THRESHOLD）或累计超过 chunk_size 时，新开一组
        - 保证每组至少包含一个 item

        Returns:
            list of (源 items 列表, 合并后文本)
        """
        result: list[tuple[list[ContentItem], str]] = []
        current_items: list[ContentItem] = []
        current_text = ""

        for item in items:
            text = item.content
            is_short = len(text) < _MERGE_THRESHOLD
            would_exceed = len(current_text) + len(text) > self._chunk_size

            if current_items and (not is_short or would_exceed):
                # 先提交当前组
                result.append((current_items, current_text))
                current_items = []
                current_text = ""

            current_items.append(item)
            current_text = (current_text + "\n" + text).strip() if current_text else text

        if current_items:
            result.append((current_items, current_text))

        return result
