"""测试文件名映射功能。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from openpyxl import load_workbook

from export.excel_exporter import ExcelExporter
from models import QAPair


def _make_qa(doc_id: str) -> QAPair:
    return QAPair(
        qa_id=f"qa_{doc_id}",
        chunk_id=f"chunk_{doc_id}",
        doc_id=doc_id,
        question="测试问题",
        answer="测试答案",
        context="测试上下文",
        question_type="definition",
        content_type="text",
        reason="测试理由",
        reasoning_steps="步骤1\n步骤2",
        quality_score=0.85,
        is_passed=True,
    )


def test_excel_shows_original_filename_not_doc_id():
    """Excel "来源文件"列应显示原始文件名，而非 doc_id。"""
    qa_pairs = [
        _make_qa("doc_001"),
        _make_qa("doc_002"),
    ]

    # 模拟真实场景：doc_id 映射到原始文件名
    doc_id_to_filename = {
        "doc_001": "产品手册.pdf",
        "doc_002": "技术文档.docx",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output = str(Path(tmpdir) / "test.xlsx")
        ExcelExporter.export(qa_pairs, output, doc_id_to_filename)

        # 验证 Excel 内容
        wb = load_workbook(output)
        ws = wb.active

        # 第 2 行（数据行 1）应显示"产品手册.pdf"而非"doc_001"
        assert ws.cell(2, 1).value == "产品手册.pdf", "来源文件列应显示原始文件名"
        assert ws.cell(3, 1).value == "技术文档.docx", "来源文件列应显示原始文件名"

        # 不应该显示 doc_id
        assert ws.cell(2, 1).value != "doc_001"
        assert ws.cell(3, 1).value != "doc_002"


def test_excel_fallback_to_doc_id_when_mapping_missing():
    """当映射不存在时，应回退到显示 doc_id（向下兼容）。"""
    qa_pairs = [_make_qa("doc_999")]

    doc_id_to_filename = {}  # 空映射

    with tempfile.TemporaryDirectory() as tmpdir:
        output = str(Path(tmpdir) / "test.xlsx")
        ExcelExporter.export(qa_pairs, output, doc_id_to_filename)

        wb = load_workbook(output)
        ws = wb.active

        # 回退到 doc_id
        assert ws.cell(2, 1).value == "doc_999"
