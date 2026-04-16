"""DocumentParser 单元测试：Mock 各本地解析引擎验证路由和解析逻辑。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from parsing.document_parser import DocumentParser

# ---------------------------------------------------------------------------
# 辅助：构造 fitz.open mock（PyMuPDF）
# ---------------------------------------------------------------------------

def _make_fitz_doc(pages: list[list[tuple]]) -> MagicMock:
    """构造 fitz.open 返回的 mock Document。

    pages: 每页的 block 列表，每个 block 是 (x0, y0, x1, y1, text, block_no, block_type)。
    """
    mock_pages = []
    for blocks in pages:
        page = MagicMock()
        page.get_text.return_value = " ".join(b[4] for b in blocks if len(b) >= 5)
        page.__iter__ = MagicMock(return_value=iter([]))
        page.get_text = MagicMock(side_effect=lambda mode="", b=blocks: (
            " ".join(blk[4] for blk in b if len(blk) >= 5) if mode != "blocks" else b
        ))
        mock_pages.append(page)

    doc = MagicMock()
    doc.__len__ = MagicMock(return_value=len(mock_pages))
    doc.__iter__ = MagicMock(return_value=iter(mock_pages))
    doc.__enter__ = MagicMock(return_value=doc)
    doc.__exit__ = MagicMock(return_value=False)
    return doc


# ---------------------------------------------------------------------------
# 辅助：构造 docling 结果 mock
# ---------------------------------------------------------------------------

def _make_docling_result(items: list[dict]) -> MagicMock:
    """构造 DocumentConverter().convert() 的返回值 mock。

    items: [{"label": "text"|"table", "text": str, "page_no": int}, ...]
    """
    dl_items = []
    for item_def in items:
        item = MagicMock()
        item.label = item_def.get("label", "text")
        item.text = item_def.get("text", "")
        item.export_to_dict.return_value = {
            "prov": [{"page_no": item_def.get("page_no", 0)}],
            "data": {},
            "captions": [],
        }
        if item.label == "table":
            item.export_to_html.return_value = item_def.get("text", "")
        dl_items.append((item, None))

    dl_doc = MagicMock()
    dl_doc.iterate_items.return_value = dl_items
    dl_doc.export_to_markdown.return_value = "\n\n".join(
        i.get("text", "") for i in items if i.get("label", "text") == "text"
    )

    result = MagicMock()
    result.document = dl_doc
    return result


# ---------------------------------------------------------------------------
# 辅助：构造 unstructured partition mock
# ---------------------------------------------------------------------------

def _make_unstructured_elements(texts: list[str]) -> list[MagicMock]:
    elements = []
    for i, text in enumerate(texts):
        el = MagicMock()
        el.__str__ = MagicMock(return_value=text)
        type(el).__name__ = "NarrativeText"
        el.metadata.page_number = i + 1
        elements.append(el)
    return elements


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------


class TestDocumentParser:
    """DocumentParser 单元测试。"""

    # ------------------------------------------------------------------
    # 数字版 PDF → PyMuPDF 路径
    # ------------------------------------------------------------------

    def test_parse_digital_pdf_uses_pymupdf(self, tmp_path: Path) -> None:
        """数字版 PDF 应路由到 PyMuPDF 引擎。"""
        doc_file = tmp_path / "digital.pdf"
        doc_file.write_bytes(b"%PDF-1.4 fake content")

        long_text = "这是一段长度超过100字符的测试文本，用于验证数字版PDF的PyMuPDF解析路径。" * 2
        blocks = [(0, 0, 100, 20, long_text, 0, 0)]

        fitz_doc = MagicMock()
        fitz_doc.__len__ = MagicMock(return_value=1)
        page = MagicMock()
        page.get_text = MagicMock(side_effect=lambda mode="": (
            long_text if mode == "" else blocks
        ))
        fitz_doc.__iter__ = MagicMock(return_value=iter([page]))
        fitz_doc.__enter__ = MagicMock(return_value=fitz_doc)
        fitz_doc.__exit__ = MagicMock(return_value=False)

        with patch("fitz.open", return_value=fitz_doc):
            parser = DocumentParser(backend="auto")
            result = parser.parse(str(doc_file))

        assert result.doc_id == "digital"
        assert result.metadata["parsed_via"] == "pymupdf"

    def test_pymupdf_filters_short_text(self, tmp_path: Path) -> None:
        """PyMuPDF 引擎应过滤长度小于 _MIN_CONTENT_LENGTH 的 block。"""
        doc_file = tmp_path / "test.pdf"
        doc_file.write_bytes(b"%PDF")

        short = "短"  # 长度 < _MIN_CONTENT_LENGTH
        long = "这段文本足够长，超过最短长度阈值，不会被过滤掉。" * 2
        blocks = [
            (0, 0, 100, 20, short, 0, 0),
            (0, 20, 100, 40, long, 1, 0),
        ]

        fitz_doc = MagicMock()
        fitz_doc.__len__ = MagicMock(return_value=1)
        page = MagicMock()
        page.get_text = MagicMock(side_effect=lambda mode="": (
            long if mode == "" else blocks
        ))
        fitz_doc.__iter__ = MagicMock(return_value=iter([page]))
        fitz_doc.__enter__ = MagicMock(return_value=fitz_doc)
        fitz_doc.__exit__ = MagicMock(return_value=False)

        with patch("fitz.open", return_value=fitz_doc):
            parser = DocumentParser(backend="pymupdf")
            result = parser.parse(str(doc_file))

        assert len(result.content_list) == 1
        assert result.content_list[0].content == long

    # ------------------------------------------------------------------
    # 扫描版 PDF → ocrmypdf + docling 路径
    # ------------------------------------------------------------------

    def test_scanned_pdf_routes_to_ocrmypdf_then_docling(self, tmp_path: Path) -> None:
        """扫描版 PDF（平均字符数 <= 100）应先过 ocrmypdf 再交给 docling。"""
        doc_file = tmp_path / "scanned.pdf"
        doc_file.write_bytes(b"%PDF-1.4")

        # fitz 检测字符数很少 → 扫描版
        fitz_doc = MagicMock()
        fitz_doc.__len__ = MagicMock(return_value=1)
        page = MagicMock()
        page.get_text = MagicMock(return_value="X")  # 平均 1 字符/页
        fitz_doc.__iter__ = MagicMock(return_value=iter([page]))
        fitz_doc.__enter__ = MagicMock(return_value=fitz_doc)
        fitz_doc.__exit__ = MagicMock(return_value=False)

        docling_result = _make_docling_result([
            {"label": "text", "text": "OCR后识别出的内容，这是一段来自扫描件的文字。", "page_no": 0}
        ])

        with patch("fitz.open", return_value=fitz_doc), \
             patch("ocrmypdf.ocr") as mock_ocr, \
             patch("docling.document_converter.DocumentConverter") as MockConverter:
            MockConverter.return_value.convert.return_value = docling_result
            parser = DocumentParser(backend="auto", ocr_lang="chi_sim+eng")
            result = parser.parse(str(doc_file))

        mock_ocr.assert_called_once()
        call_kwargs = mock_ocr.call_args[1]
        # ocrmypdf Python API 要求 language 为列表，"chi_sim+eng" 应被拆分为 ["chi_sim", "eng"]
        assert call_kwargs["language"] == ["chi_sim", "eng"]
        assert result.metadata["parsed_via"] == "docling"

    # ------------------------------------------------------------------
    # DOCX / 图片 → docling 路径
    # ------------------------------------------------------------------

    def test_docx_uses_docling(self, tmp_path: Path) -> None:
        """DOCX 文件应路由到 docling。"""
        doc_file = tmp_path / "report.docx"
        doc_file.write_bytes(b"PK fake docx")

        docling_result = _make_docling_result([
            {"label": "text", "text": "这是一份Word文档的测试内容，用于验证docling解析。", "page_no": 0}
        ])

        with patch("docling.document_converter.DocumentConverter") as MockConverter:
            MockConverter.return_value.convert.return_value = docling_result
            parser = DocumentParser(backend="auto")
            result = parser.parse(str(doc_file))

        assert result.metadata["parsed_via"] == "docling"
        assert result.doc_id == "report"

    def test_image_uses_docling(self, tmp_path: Path) -> None:
        """图片文件应路由到 docling。"""
        img_file = tmp_path / "chart.png"
        img_file.write_bytes(b"\x89PNG\r\n fake")

        docling_result = _make_docling_result([
            {"label": "text", "text": "图片中识别出的文字内容，来自OCR引擎处理结果。", "page_no": 0}
        ])

        with patch("docling.document_converter.DocumentConverter") as MockConverter:
            MockConverter.return_value.convert.return_value = docling_result
            parser = DocumentParser(backend="auto")
            result = parser.parse(str(img_file))

        assert result.metadata["parsed_via"] == "docling"

    # ------------------------------------------------------------------
    # unstructured 兜底路径
    # ------------------------------------------------------------------

    def test_unknown_format_uses_unstructured(self, tmp_path: Path) -> None:
        """未知格式文件应兜底到 unstructured。"""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("这是一段纯文本内容，用于测试unstructured解析引擎。")

        elements = _make_unstructured_elements(["这是一段纯文本内容，用于测试unstructured解析引擎。"])

        with patch("unstructured.partition.auto.partition", return_value=elements):
            parser = DocumentParser(backend="auto")
            result = parser.parse(str(txt_file))

        assert result.metadata["parsed_via"] == "unstructured"
        assert len(result.content_list) == 1

    def test_forced_unstructured_backend(self, tmp_path: Path) -> None:
        """backend='unstructured' 时强制使用 unstructured，即使是 PDF。"""
        doc_file = tmp_path / "doc.pdf"
        doc_file.write_bytes(b"%PDF-1.4")

        elements = _make_unstructured_elements(["这段内容来自unstructured强制解析PDF的结果。"])

        with patch("unstructured.partition.auto.partition", return_value=elements):
            parser = DocumentParser(backend="unstructured")
            result = parser.parse(str(doc_file))

        assert result.metadata["parsed_via"] == "unstructured"

    # ------------------------------------------------------------------
    # 通用行为
    # ------------------------------------------------------------------

    def test_file_not_found_raises(self) -> None:
        """文件不存在时应抛出 FileNotFoundError。"""
        parser = DocumentParser()
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            parser.parse("/nonexistent/path/file.pdf")

    def test_parsed_document_structure(self, tmp_path: Path) -> None:
        """解析结果应包含正确的 ParsedDocument 字段。"""
        doc_file = tmp_path / "sample.pdf"
        doc_file.write_bytes(b"%PDF-1.4")

        long_text = "验证ParsedDocument字段结构的测试文本，内容足够长。" * 3
        blocks = [(0, 0, 100, 20, long_text, 0, 0)]

        fitz_doc = MagicMock()
        fitz_doc.__len__ = MagicMock(return_value=1)
        page = MagicMock()
        page.get_text = MagicMock(side_effect=lambda mode="": (
            long_text if mode == "" else blocks
        ))
        fitz_doc.__iter__ = MagicMock(return_value=iter([page]))
        fitz_doc.__enter__ = MagicMock(return_value=fitz_doc)
        fitz_doc.__exit__ = MagicMock(return_value=False)

        with patch("fitz.open", return_value=fitz_doc):
            parser = DocumentParser(backend="pymupdf")
            result = parser.parse(str(doc_file))

        assert result.doc_id.startswith("sample_")  # doc_id = stem + "_" + path hash
        assert result.source_path == str(doc_file)
        assert isinstance(result.markdown, str)
        assert isinstance(result.content_list, list)
        assert isinstance(result.metadata, dict)
        assert all(hasattr(item, "item_id") for item in result.content_list)
        assert all(hasattr(item, "type") for item in result.content_list)
        assert all(hasattr(item, "content") for item in result.content_list)
        assert all(hasattr(item, "page_idx") for item in result.content_list)

    def test_pymupdf_filters_image_blocks(self, tmp_path: Path) -> None:
        """PyMuPDF 引擎应跳过 block_type == 1（图片 block），避免混入无效 xref 字符串。"""
        doc_file = tmp_path / "test.pdf"
        doc_file.write_bytes(b"%PDF")

        text = "这段文字是真实文本，应该被保留下来。" * 2
        # block_type == 1 表示图片 block
        image_block = (0, 50, 100, 100, "<image: DeviceRGB ...>", 1, 1)
        text_block = (0, 0, 100, 20, text, 0, 0)
        blocks = [image_block, text_block]

        fitz_doc = MagicMock()
        fitz_doc.__len__ = MagicMock(return_value=1)
        page = MagicMock()
        page.get_text = MagicMock(side_effect=lambda mode="": (
            text if mode == "" else blocks
        ))
        fitz_doc.__iter__ = MagicMock(return_value=iter([page]))
        fitz_doc.__enter__ = MagicMock(return_value=fitz_doc)
        fitz_doc.__exit__ = MagicMock(return_value=False)

        with patch("fitz.open", return_value=fitz_doc):
            parser = DocumentParser(backend="pymupdf")
            result = parser.parse(str(doc_file))

        # 只有文本 block 被保留，图片 block 被过滤
        assert len(result.content_list) == 1
        assert "<image:" not in result.content_list[0].content

    def test_ocrmypdf_language_is_list(self, tmp_path: Path) -> None:
        """ocrmypdf 应收到语言列表而非 + 分隔字符串。"""
        doc_file = tmp_path / "scanned.pdf"
        doc_file.write_bytes(b"%PDF-1.4")

        fitz_doc = MagicMock()
        fitz_doc.__len__ = MagicMock(return_value=1)
        page = MagicMock()
        page.get_text = MagicMock(return_value="X")
        fitz_doc.__iter__ = MagicMock(return_value=iter([page]))
        fitz_doc.__enter__ = MagicMock(return_value=fitz_doc)
        fitz_doc.__exit__ = MagicMock(return_value=False)

        docling_result = _make_docling_result([
            {"label": "text", "text": "OCR内容文字足够长。" * 3, "page_no": 1}
        ])

        with patch("fitz.open", return_value=fitz_doc), \
             patch("ocrmypdf.ocr") as mock_ocr, \
             patch("docling.document_converter.DocumentConverter") as MockConverter:
            MockConverter.return_value.convert.return_value = docling_result
            parser = DocumentParser(backend="auto", ocr_lang="chi_sim+eng")
            parser.parse(str(doc_file))

        call_kwargs = mock_ocr.call_args[1]
        assert isinstance(call_kwargs["language"], list), "language 参数必须是列表"
        assert call_kwargs["language"] == ["chi_sim", "eng"]

    def test_docling_page_no_is_zero_indexed(self, tmp_path: Path) -> None:
        """docling 的 page_no 是 1-indexed，解析器应转换为 0-indexed。"""
        doc_file = tmp_path / "report.docx"
        doc_file.write_bytes(b"PK")

        # docling 真实 page_no 从 1 开始
        docling_result = _make_docling_result([
            {"label": "text", "text": "第二页的内容，足够长的文字。" * 2, "page_no": 2}
        ])

        with patch("docling.document_converter.DocumentConverter") as MockConverter:
            MockConverter.return_value.convert.return_value = docling_result
            parser = DocumentParser(backend="auto")
            result = parser.parse(str(doc_file))

        # page_no=2 应转换为 page_idx=1
        if result.content_list:
            assert result.content_list[0].page_idx == 1

    def test_default_backend(self) -> None:
        """默认 backend 应为 'auto'。"""
        parser = DocumentParser()
        assert parser.backend == "auto"

    def test_default_ocr_lang(self) -> None:
        """默认 OCR 语言应为 'chi_sim+eng'。"""
        parser = DocumentParser()
        assert parser.ocr_lang == "chi_sim+eng"

    def test_custom_ocr_lang(self) -> None:
        """自定义 OCR 语言应被保存。"""
        parser = DocumentParser(ocr_lang="eng")
        assert parser.ocr_lang == "eng"

    def test_content_items_have_valid_ids(self, tmp_path: Path) -> None:
        """所有 ContentItem 的 item_id 应包含 doc_id。"""
        doc_file = tmp_path / "myfile.pdf"
        doc_file.write_bytes(b"%PDF")

        text = "足够长的测试内容，验证item_id的格式是否正确包含doc_id前缀。"
        blocks = [(0, 0, 100, 20, text, 0, 0)]

        fitz_doc = MagicMock()
        fitz_doc.__len__ = MagicMock(return_value=1)
        page = MagicMock()
        page.get_text = MagicMock(side_effect=lambda mode="": (
            text if mode == "" else blocks
        ))
        fitz_doc.__iter__ = MagicMock(return_value=iter([page]))
        fitz_doc.__enter__ = MagicMock(return_value=fitz_doc)
        fitz_doc.__exit__ = MagicMock(return_value=False)

        with patch("fitz.open", return_value=fitz_doc):
            parser = DocumentParser(backend="pymupdf")
            result = parser.parse(str(doc_file))

        for item in result.content_list:
            assert "myfile" in item.item_id

    def test_docling_maps_section_header_to_heading(self, tmp_path: Path) -> None:
        """docling 的 section_header label 应映射为 heading 类型。"""
        doc_file = tmp_path / "report.docx"
        doc_file.write_bytes(b"PK fake docx")

        # 构造包含 section_header label 的 docling 结果
        items_def = [
            {"label": "section_header", "text": "第一章：引言与背景", "page_no": 1},
            {"label": "text", "text": "这是引言部分的正文内容，足够长的文字供测试使用。", "page_no": 1},
        ]
        docling_result = _make_docling_result(items_def)

        # section_header label 需要以带 .value 属性的 mock 对象表示（模拟 DocItemLabel 枚举）
        for (item, _), item_def in zip(docling_result.document.iterate_items(), items_def):
            label_mock = MagicMock()
            label_mock.value = item_def["label"]
            item.label = label_mock

        with patch("docling.document_converter.DocumentConverter") as MockConverter:
            MockConverter.return_value.convert.return_value = docling_result
            parser = DocumentParser(backend="docling")
            result = parser.parse(str(doc_file))

        heading_items = [item for item in result.content_list if item.type == "heading"]
        assert len(heading_items) >= 1, "应至少有一个 heading 类型的 ContentItem"
        assert heading_items[0].content == "第一章：引言与背景"

    def test_pymupdf_marks_large_font_block_as_heading(self, tmp_path: Path) -> None:
        """PyMuPDF 较大字号的 block 应映射为 heading 类型。"""
        doc_file = tmp_path / "doc.pdf"
        doc_file.write_bytes(b"%PDF")

        heading_text = "第一节：系统架构设计"
        body_text = "这是正文段落，字号正常，内容足够长用于验证分类逻辑。" * 2

        # block 格式：(x0, y0, x1, y1, text, block_no, block_type)
        blocks = [
            (0, 0, 200, 30, heading_text, 0, 0),
            (0, 40, 200, 80, body_text, 1, 0),
        ]

        # get_text("dict") 返回包含字号信息的结构，用于区分标题和正文
        # heading block 字号 24，body block 字号 12（比例 >= 1.2 应被判定为 heading）
        dict_result = {
            "blocks": [
                {
                    "lines": [{"spans": [{"size": 24.0, "text": heading_text}]}]
                },
                {
                    "lines": [{"spans": [{"size": 12.0, "text": body_text}]}]
                },
            ]
        }

        fitz_doc = MagicMock()
        fitz_doc.__len__ = MagicMock(return_value=1)
        page = MagicMock()

        def get_text_side_effect(mode=""):
            if mode == "blocks":
                return blocks
            if mode == "dict":
                return dict_result
            # mode == "" 返回纯文本（用于 _is_digital_pdf 检测）
            return heading_text + " " + body_text

        page.get_text = MagicMock(side_effect=get_text_side_effect)
        fitz_doc.__iter__ = MagicMock(return_value=iter([page]))
        fitz_doc.__enter__ = MagicMock(return_value=fitz_doc)
        fitz_doc.__exit__ = MagicMock(return_value=False)

        with patch("fitz.open", return_value=fitz_doc):
            parser = DocumentParser(backend="pymupdf")
            result = parser.parse(str(doc_file))

        heading_items = [item for item in result.content_list if item.type == "heading"]
        text_items = [item for item in result.content_list if item.type == "text"]
        assert len(heading_items) >= 1, "较大字号的 block 应被标记为 heading"
        assert heading_items[0].content == heading_text
        assert len(text_items) >= 1, "正常字号的 block 应保留为 text"
