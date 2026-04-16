"""单元测试：ContentNormalizer。"""

from __future__ import annotations

from models import ContentItem
from parsing.content_normalizer import ContentNormalizer


def _make_item(content: str, item_type: str = "text") -> ContentItem:
    return ContentItem(
        item_id="test_0",
        doc_id="test",
        type=item_type,
        content=content,
        page_idx=0,
    )


class TestContentNormalizer:
    def setup_method(self) -> None:
        self.normalizer = ContentNormalizer(min_text_length=20)

    def test_removes_excess_blank_lines(self) -> None:
        item = _make_item("段落一。\n\n\n\n\n段落二。这是足够长的测试内容。")
        result = self.normalizer.normalize_items([item])
        assert result
        assert "\n\n\n" not in result[0].content

    def test_drops_short_text(self) -> None:
        item = _make_item("太短了。")
        result = self.normalizer.normalize_items([item])
        assert result == []

    def test_preserves_table_without_cleaning(self) -> None:
        item = _make_item("<table><tr><td>数据</td></tr></table>", item_type="table")
        result = self.normalizer.normalize_items([item])
        assert len(result) == 1
        assert result[0].content == item.content  # 表格内容不做清洗

    def test_returns_new_objects(self) -> None:
        item = _make_item("这是一段测试文本，用于验证标准化器返回新对象的行为。")
        result = self.normalizer.normalize_items([item])
        assert result[0] is not item  # 返回新对象
