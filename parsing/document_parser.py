"""文档解析器：本地多引擎策略，将多格式文档转为结构化内容列表。

支持格式：PDF（数字版 + 扫描版）、DOCX、图片（PNG/JPG）。

引擎路由（auto 模式）：
    PDF → PyMuPDF 检测文本量
          ├─ 数字版（平均 >100 字符/页）→ PyMuPDF 直接提取
          └─ 扫描版 → ocrmypdf 生成文字层 → docling 精解析
    DOCX / 图片 → docling（内置 OCR）
    其他格式    → unstructured（兜底）

系统依赖（需预先安装）：
    Tesseract OCR（Windows: https://github.com/UB-Mannheim/tesseract/wiki）
    Ghostscript  （Windows: https://www.ghostscript.com/releases/）
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Optional

from models import ContentItem, ParsedDocument
from utils.logger import get_logger

logger = get_logger(__name__)

# 数字版 PDF 判断阈值：每页平均字符数
_DIGITAL_PDF_CHARS_PER_PAGE = 100
# 内容最短长度（字符），低于此值丢弃
_MIN_CONTENT_LENGTH = 10

# 支持的格式集合
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
_OFFICE_SUFFIXES = {".docx", ".doc"}

# docling 可映射为 text 类型的所有已知 label（用于检测 API 变更）
_KNOWN_TEXT_LABELS = frozenset({
    "text", "paragraph", "section_header", "list_item",
    "code", "formula", "footnote", "caption", "page_header",
    "page_footer", "title",
})


def _make_doc_id(path: Path) -> str:
    """生成文档唯一 ID，附加路径哈希后缀以避免同名文件碰撞。

    例：report.pdf（不同目录）→ report_3a7f2c1e / report_9b4d0a5f
    """
    path_hash = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:8]
    return f"{path.stem}_{path_hash}"


class DocumentParser:
    """本地多引擎文档解析器。

    Args:
        backend:  解析引擎选择。
                  - ``"auto"``        自动路由（推荐）
                  - ``"pymupdf"``     强制使用 PyMuPDF（仅适合数字版 PDF）
                  - ``"docling"``     强制使用 docling
                  - ``"unstructured"`` 强制使用 unstructured（兜底）
        ocr_lang: Tesseract 语言包，多语言用 ``+`` 分隔，如 ``"chi_sim+eng"``。
    """

    def __init__(
        self,
        backend: str = "auto",
        ocr_lang: str = "chi_sim+eng",
    ) -> None:
        self.backend = backend
        self.ocr_lang = ocr_lang

    def parse(self, file_path: str) -> ParsedDocument:
        """解析单个文档。

        Args:
            file_path: 输入文档路径。

        Returns:
            ParsedDocument，包含 markdown 全文和带类型标注的 content_list。

        Raises:
            FileNotFoundError: 文件不存在。
            RuntimeError:      解析失败。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{file_path}")

        logger.info("开始解析 [%s] backend=%s", path.name, self.backend)

        if self.backend == "auto":
            return self._parse_auto(path)
        if self.backend == "pymupdf":
            return self._parse_with_pymupdf(path)
        if self.backend == "docling":
            return self._parse_with_docling(path)
        return self._parse_with_unstructured(path)

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------

    def _parse_auto(self, path: Path) -> ParsedDocument:
        """auto 模式：按文件类型和内容特征自动选择引擎。"""
        suffix = path.suffix.lower()

        if suffix in _IMAGE_SUFFIXES:
            return self._parse_with_docling(path)

        if suffix in _OFFICE_SUFFIXES:
            return self._parse_with_docling(path)

        if suffix == ".pdf":
            if self._is_digital_pdf(path):
                logger.debug("[%s] 识别为数字版 PDF，使用 PyMuPDF", path.name)
                return self._parse_with_pymupdf(path)
            else:
                logger.debug("[%s] 识别为扫描版 PDF，使用 ocrmypdf + docling", path.name)
                return self._parse_scanned_pdf(path)

        logger.debug("[%s] 未知格式，兜底使用 unstructured", path.name)
        return self._parse_with_unstructured(path)

    # ------------------------------------------------------------------
    # 引擎适配层
    # ------------------------------------------------------------------

    def _is_digital_pdf(self, path: Path) -> bool:
        """使用 PyMuPDF 快速检测 PDF 是否为数字版。"""
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError("请安装 PyMuPDF：uv sync --extra parsing") from exc

        with fitz.open(str(path)) as doc:
            if len(doc) == 0:
                return False
            total_chars = sum(len(page.get_text()) for page in doc)
            avg_chars = total_chars / len(doc)

        return avg_chars > _DIGITAL_PDF_CHARS_PER_PAGE

    def _parse_with_pymupdf(self, path: Path) -> ParsedDocument:
        """PyMuPDF 引擎：适合数字版 PDF，毫秒级，零 ML 依赖。"""
        try:
            import fitz
        except ImportError as exc:
            raise ImportError("请安装 PyMuPDF：uv sync --extra parsing") from exc

        doc_id = _make_doc_id(path)
        content_items: list[ContentItem] = []
        markdown_parts: list[str] = []
        idx = 0

        with fitz.open(str(path)) as doc:
            for page_num, page in enumerate(doc):
                blocks = page.get_text("blocks")
                for block in blocks:
                    # blocks 格式：(x0, y0, x1, y1, text, block_no, block_type)
                    # block_type == 1 表示图片 block，无有效文本，跳过
                    if len(block) < 7 or block[6] == 1:
                        continue
                    text: str = block[4].strip()
                    if not text or len(text) < _MIN_CONTENT_LENGTH:
                        continue

                    content_items.append(ContentItem(
                        item_id=f"{doc_id}_item_{idx}",
                        doc_id=doc_id,
                        type="text",
                        content=text,
                        page_idx=page_num,
                        bbox=tuple(block[:4]),
                        metadata={},
                    ))
                    markdown_parts.append(text)
                    idx += 1

        logger.info("PyMuPDF 解析完成 [%s]：%d 个内容项", doc_id, len(content_items))

        return ParsedDocument(
            doc_id=doc_id,
            source_path=str(path),
            markdown="\n\n".join(markdown_parts),
            content_list=content_items,
            metadata={"parsed_via": "pymupdf", "total_items": len(content_items)},
        )

    def _parse_with_docling(self, path: Path) -> ParsedDocument:
        """docling 引擎：支持 PDF/DOCX/图片，内置 OCR，CPU 可运行。"""
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise ImportError("请安装 docling：uv sync --extra parsing") from exc

        doc_id = _make_doc_id(path)
        converter = DocumentConverter()
        result = converter.convert(str(path))
        dl_doc = result.document

        content_items: list[ContentItem] = []
        markdown_parts: list[str] = []
        idx = 0

        for item, _ in dl_doc.iterate_items():
            # iterate_items() 返回 (item, level) 元组，level 为层级深度整数
            item_dict = item.export_to_dict() if hasattr(item, "export_to_dict") else {}
            label = getattr(item, "label", "text")
            # DocItemLabel 是 str 枚举（str, Enum），用 str() 确保与字符串字面量比较正确
            label_str = str(label.value) if hasattr(label, "value") else str(label)

            if label_str == "table":
                content = item_dict.get("data", {}).get("table_cells", "")
                if hasattr(item, "export_to_html"):
                    content = item.export_to_html()
                item_type = "table"
            elif label_str == "picture":
                # captions 是 RefItem 列表，取第一个的文本表示
                captions = item_dict.get("captions", [])
                content = str(captions[0]) if captions else ""
                item_type = "image"
            else:
                content = getattr(item, "text", "") or ""
                item_type = "text"
                # 记录非标准 text 类型的 label，便于发现 docling API 变更
                if label_str not in _KNOWN_TEXT_LABELS:
                    logger.debug("[%s] 未识别的 docling label '%s'，按 text 处理", doc_id, label_str)

            content = str(content).strip()
            if not content or len(content) < _MIN_CONTENT_LENGTH:
                continue

            prov = item_dict.get("prov", [{}])
            # docling page_no 是 1-indexed，转换为 0-indexed
            raw_page_no = prov[0].get("page_no", 1) if prov else 1
            page_idx = max(raw_page_no - 1, 0)

            content_items.append(ContentItem(
                item_id=f"{doc_id}_item_{idx}",
                doc_id=doc_id,
                type=item_type,
                content=content,
                page_idx=page_idx,
                metadata={},
            ))
            if item_type == "text":
                markdown_parts.append(content)
            idx += 1

        markdown = dl_doc.export_to_markdown() if hasattr(dl_doc, "export_to_markdown") else "\n\n".join(markdown_parts)

        logger.info("docling 解析完成 [%s]：%d 个内容项", doc_id, len(content_items))

        return ParsedDocument(
            doc_id=doc_id,
            source_path=str(path),
            markdown=markdown,
            content_list=content_items,
            metadata={"parsed_via": "docling", "total_items": len(content_items)},
        )

    def _parse_scanned_pdf(self, path: Path) -> ParsedDocument:
        """扫描版 PDF：先用 ocrmypdf 生成文字层，再用 docling 解析。"""
        try:
            import ocrmypdf
        except ImportError as exc:
            raise ImportError("请安装 ocrmypdf：uv sync --extra parsing") from exc

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_pdf = Path(tmp_dir) / f"{path.stem}_ocr.pdf"
            logger.debug("ocrmypdf 处理 [%s] lang=%s", path.name, self.ocr_lang)
            # ocrmypdf Python API 要求 language 为列表，而非 Tesseract CLI 的 "+" 分隔字符串
            lang_list = self.ocr_lang.split("+")
            ocrmypdf.ocr(
                str(path),
                str(output_pdf),
                language=lang_list,
                skip_text=True,
                progress_bar=False,
            )
            return self._parse_with_docling(output_pdf)

    def _parse_with_unstructured(self, path: Path) -> ParsedDocument:
        """unstructured 引擎：多格式兜底，CPU 运行。"""
        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:
            raise ImportError(
                "请安装 unstructured：uv sync --extra parsing"
            ) from exc

        doc_id = _make_doc_id(path)
        elements = partition(filename=str(path))

        _TYPE_MAP = {
            "Table": "table",
            "Image": "image",
            "FigureCaption": "image",
        }

        content_items: list[ContentItem] = []
        markdown_parts: list[str] = []
        idx = 0

        for element in elements:
            text = str(element).strip()
            if not text or len(text) < _MIN_CONTENT_LENGTH:
                continue

            category = type(element).__name__
            item_type = _TYPE_MAP.get(category, "text")
            # page_number 可能为 None（部分 element 无页码信息），显式 None 检查
            raw_page = element.metadata.page_number
            page_idx = max((raw_page - 1), 0) if raw_page is not None else 0

            content_items.append(ContentItem(
                item_id=f"{doc_id}_item_{idx}",
                doc_id=doc_id,
                type=item_type,
                content=text,
                page_idx=page_idx,
                metadata={},
            ))
            if item_type == "text":
                markdown_parts.append(text)
            idx += 1

        logger.info("unstructured 解析完成 [%s]：%d 个内容项", doc_id, len(content_items))

        return ParsedDocument(
            doc_id=doc_id,
            source_path=str(path),
            markdown="\n\n".join(markdown_parts),
            content_list=content_items,
            metadata={"parsed_via": "unstructured", "total_items": len(content_items)},
        )


# ---------------------------------------------------------------------------
# 模块级便捷函数
# ---------------------------------------------------------------------------


def parse_document(
    file_path: str,
    backend: str = "auto",
    ocr_lang: str = "chi_sim+eng",
) -> ParsedDocument:
    """解析单个文档的便捷函数。

    等价于 ``DocumentParser(backend, ocr_lang).parse(file_path)``。
    """
    return DocumentParser(backend=backend, ocr_lang=ocr_lang).parse(file_path)
