"""内容标准化：清洗解析结果，去除噪声内容。"""

from __future__ import annotations

import re

from models import ContentItem
from utils.logger import get_logger

logger = get_logger(__name__)

# 匹配连续空白行（3 行以上）
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
# 匹配常见页眉页脚模式（如"第 X 页"、"- X -"）
_PAGE_MARKER = re.compile(r"^[-–\s]*第\s*\d+\s*页[-–\s]*$|^[-–\s]*\d+\s*[-–\s]*$", re.MULTILINE)


class ContentNormalizer:
    """文本内容标准化。

    功能：
    - 去除多余空白行
    - 去除页眉页脚标记
    - 标准化 Unicode 空白字符
    - 丢弃内容过短的 item
    """

    def __init__(self, min_text_length: int = 20) -> None:
        self.min_text_length = min_text_length

    def normalize_items(self, items: list[ContentItem]) -> list[ContentItem]:
        """对整个 content_list 批量标准化，返回新的列表（不修改原数据）。"""
        normalized: list[ContentItem] = []
        dropped = 0

        for item in items:
            clean = self._normalize_single(item)
            if clean is not None:
                normalized.append(clean)
            else:
                dropped += 1

        if dropped:
            logger.debug("标准化丢弃 %d 个过短/噪声 item", dropped)

        return normalized

    def _normalize_single(self, item: ContentItem) -> ContentItem | None:
        """标准化单个 ContentItem，内容无效时返回 None。"""
        if item.type != "text":
            # 表格、公式、图片 caption 不做文本清洗
            return item

        content = item.content
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        content = _PAGE_MARKER.sub("", content)
        content = _EXCESS_BLANK_LINES.sub("\n\n", content)
        content = content.strip()

        if len(content) < self.min_text_length:
            return None

        # 创建新对象（保持不可变原则）
        return ContentItem(
            item_id=item.item_id,
            doc_id=item.doc_id,
            type=item.type,
            content=content,
            page_idx=item.page_idx,
            bbox=item.bbox,
            metadata=item.metadata,
        )
