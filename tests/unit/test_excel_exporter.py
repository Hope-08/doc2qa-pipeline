"""单元测试：ExcelExporter。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from openpyxl import load_workbook

from export.excel_exporter import ExcelExporter
from models import QAPair


def _make_qa(idx: int = 0, doc_id: str = "doc1") -> QAPair:
    """创建测试用 QAPair。"""
    return QAPair(
        qa_id=f"qa_{idx}",
        chunk_id=f"chunk_{idx}",
        doc_id=doc_id,
        question=f"问题 {idx}：什么是深度学习？",
        answer=f"答案 {idx}：深度学习是机器学习的一个分支，使用多层神经网络处理数据。",
        context=f"上下文 {idx}：这是原文依据内容。",
        question_type="definition",
        content_type="text",
        reason=f"出题理由 {idx}：该段落包含核心概念定义。",
        reasoning_steps="步骤1：识别定义\n步骤2：提取关键信息",
        quality_score=0.85,
        faithfulness_score=0.90,
        relevance_score=0.75,
        completeness_score=0.80,
        is_passed=True,
    )


class TestExcelExporter:
    """ExcelExporter 测试类。"""

    def test_export_excel_basic(self) -> None:
        """验证基本导出功能：文件创建。"""
        qa_pairs = [_make_qa(i) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "test.xlsx")
            result_path = ExcelExporter.export(qa_pairs, output)

            # 验证文件创建
            assert Path(result_path).exists()
            assert result_path == output

            # 验证可以打开为 Excel 文件
            wb = load_workbook(result_path)
            ws = wb.active
            assert ws is not None

            # 验证行数：1 行表头 + 3 行数据
            assert ws.max_row == 4

    def test_export_excel_columns(self) -> None:
        """验证列结构和数据内容。"""
        qa_pairs = [_make_qa(0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "test.xlsx")
            ExcelExporter.export(qa_pairs, output)

            wb = load_workbook(output)
            ws = wb.active
            assert ws is not None

            # 验证表头列名
            expected_headers = [
                "来源文件",
                "问题类型",
                "问题",
                "答案",
                "生成原因",
                "答案流程",
                "综合质量分",
                "原文依据",
            ]
            actual_headers = [ws.cell(row=1, column=i + 1).value for i in range(8)]
            assert actual_headers == expected_headers

            # 验证数据行内容
            qa = qa_pairs[0]
            assert ws.cell(row=2, column=1).value == qa.doc_id  # 来源文件
            assert ws.cell(row=2, column=2).value == qa.question_type  # 问题类型
            assert ws.cell(row=2, column=3).value == qa.question  # 问题
            assert ws.cell(row=2, column=4).value == qa.answer  # 答案
            assert ws.cell(row=2, column=5).value == qa.reason  # 生成原因
            assert ws.cell(row=2, column=6).value == qa.reasoning_steps  # 答案流程
            assert ws.cell(row=2, column=7).value == qa.quality_score  # 综合质量分
            assert ws.cell(row=2, column=8).value == qa.context  # 原文依据

    def test_export_excel_with_filename_mapping(self) -> None:
        """验证 doc_id 到文件名的映射。"""
        qa_pairs = [_make_qa(0, doc_id="abc123"), _make_qa(1, doc_id="def456")]
        doc_id_to_filename = {
            "abc123": "产品手册.pdf",
            "def456": "技术文档.docx",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "test.xlsx")
            ExcelExporter.export(qa_pairs, output, doc_id_to_filename)

            wb = load_workbook(output)
            ws = wb.active
            assert ws is not None

            # 验证来源文件列使用映射后的文件名
            assert ws.cell(row=2, column=1).value == "产品手册.pdf"
            assert ws.cell(row=3, column=1).value == "技术文档.docx"

    def test_export_excel_without_filename_mapping(self) -> None:
        """验证无映射时使用 doc_id。"""
        qa_pairs = [_make_qa(0, doc_id="my_doc_id")]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "test.xlsx")
            ExcelExporter.export(qa_pairs, output)

            wb = load_workbook(output)
            ws = wb.active
            assert ws is not None

            # 无映射时应使用 doc_id
            assert ws.cell(row=2, column=1).value == "my_doc_id"

    def test_export_excel_empty_list(self) -> None:
        """验证空列表导出。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "empty.xlsx")
            result_path = ExcelExporter.export([], output)

            assert Path(result_path).exists()

            wb = load_workbook(result_path)
            ws = wb.active
            assert ws is not None

            # 只有表头行
            assert ws.max_row == 1

    def test_export_excel_creates_parent_dirs(self) -> None:
        """验证自动创建父目录。"""
        qa_pairs = [_make_qa(0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "nested" / "deep" / "output.xlsx")
            ExcelExporter.export(qa_pairs, output)
            assert Path(output).exists()

    def test_export_excel_header_styling(self) -> None:
        """验证表头样式（加粗、居中）。"""
        qa_pairs = [_make_qa(0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "styled.xlsx")
            ExcelExporter.export(qa_pairs, output)

            wb = load_workbook(output)
            ws = wb.active
            assert ws is not None

            # 验证表头单元格加粗
            header_cell = ws.cell(row=1, column=1)
            assert header_cell.font.bold is True

            # 验证表头居中
            assert header_cell.alignment.horizontal == "center"

    def test_export_excel_multiline_content(self) -> None:
        """验证多行内容处理（reasoning_steps 含换行）。"""
        qa = _make_qa(0)
        qa_pairs = [qa]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "multiline.xlsx")
            ExcelExporter.export(qa_pairs, output)

            wb = load_workbook(output)
            ws = wb.active
            assert ws is not None

            # reasoning_steps 包含换行符，应正确保存
            reasoning_cell = ws.cell(row=2, column=6)
            assert "\n" in str(reasoning_cell.value)
            # 验证启用了文本换行
            assert reasoning_cell.alignment.wrap_text is True

    def test_export_excel_none_values(self) -> None:
        """验证 None 值处理。"""
        qa = QAPair(
            qa_id="qa_0",
            chunk_id="chunk_0",
            doc_id="doc1",
            question="问题",
            answer="答案",
            context="上下文",
            question_type="definition",
            content_type="text",
            reason="",  # 空字符串
            reasoning_steps="",  # 空字符串
            quality_score=None,  # None 值
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "none_values.xlsx")
            ExcelExporter.export([qa], output)

            wb = load_workbook(output)
            ws = wb.active
            assert ws is not None

            # quality_score 为 None 时应为空
            assert ws.cell(row=2, column=7).value is None
