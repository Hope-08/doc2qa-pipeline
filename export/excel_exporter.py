"""Excel 格式数据集导出器。"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.worksheet import Worksheet

from models import QAPair
from utils.logger import get_logger

logger = get_logger(__name__)


class ExcelExporter:
    """Excel 格式导出器。

    导出包含以下列的 Excel 文件：
    - 来源文件、问题类型、问题、答案、生成原因、答案流程、综合质量分、原文依据
    """

    # 表头定义
    HEADERS = [
        "来源文件",
        "问题类型",
        "问题",
        "答案",
        "生成原因",
        "答案流程",
        "综合质量分",
        "原文依据",
    ]

    # 列宽配置（字符数）
    COLUMN_WIDTHS = {
        "来源文件": 20,
        "问题类型": 12,
        "问题": 50,
        "答案": 60,
        "生成原因": 40,
        "答案流程": 50,
        "综合质量分": 12,
        "原文依据": 60,
    }

    # 需要自动换行的列（长文本）
    WRAP_TEXT_COLUMNS = {"问题", "答案", "生成原因", "答案流程", "原文依据"}

    @staticmethod
    def export(
        qa_pairs: list[QAPair],
        output_path: str,
        doc_id_to_filename: dict[str, str] | None = None,
    ) -> str:
        """导出 QA 数据集到 Excel。

        Args:
            qa_pairs: QA 对列表
            output_path: 输出文件路径（.xlsx）
            doc_id_to_filename: doc_id -> 原始文件名映射表（用于"来源文件"列）

        Returns:
            实际输出路径字符串
        """
        # 确保父目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 创建工作簿
        wb = Workbook()
        ws = wb.active
        if ws is None:
            ws = wb.create_sheet()
        ws.title = "QA数据集"

        # 写入表头
        ExcelExporter._write_headers(ws)

        # 写入数据行
        ExcelExporter._write_data_rows(ws, qa_pairs, doc_id_to_filename)

        # 设置列宽
        ExcelExporter._set_column_widths(ws)

        # 保存文件
        wb.save(output_path)

        logger.info("Excel 导出完成：%s（%d 条）", output_path, len(qa_pairs))
        return output_path

    @staticmethod
    def _write_headers(ws: Worksheet) -> None:
        """写入表头并设置样式。

        Args:
            ws: Excel 工作表
        """
        header_font = Font(bold=True)
        header_alignment = Alignment(horizontal="center", vertical="center")

        for col_idx, header in enumerate(ExcelExporter.HEADERS, start=1):
            cell = ws.cell(row=1, column=col_idx, value=header)
            cell.font = header_font
            cell.alignment = header_alignment

    @staticmethod
    def _write_data_rows(
        ws: Worksheet,
        qa_pairs: list[QAPair],
        doc_id_to_filename: dict[str, str] | None,
    ) -> None:
        """写入数据行。

        Args:
            ws: Excel 工作表
            qa_pairs: QA 对列表
            doc_id_to_filename: doc_id → 文件名映射
        """
        wrap_alignment = Alignment(wrap_text=True, vertical="top")

        for row_idx, qa in enumerate(qa_pairs, start=2):
            # 获取来源文件名
            source_filename = (
                doc_id_to_filename.get(qa.doc_id, qa.doc_id) if doc_id_to_filename else qa.doc_id
            )

            # 构建行数据
            row_data = [
                source_filename,
                qa.question_type,
                qa.question,
                qa.answer,
                qa.reason,
                qa.reasoning_steps,
                qa.quality_score,
                qa.context,
            ]

            # 写入每个单元格
            for col_idx, value in enumerate(row_data, start=1):
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                # 对长文本列启用自动换行（通过 header 名称判断）
                header_name = ExcelExporter.HEADERS[col_idx - 1]
                if header_name in ExcelExporter.WRAP_TEXT_COLUMNS:
                    cell.alignment = wrap_alignment

    @staticmethod
    def _set_column_widths(ws: Worksheet) -> None:
        """设置列宽。

        Args:
            ws: Excel 工作表
        """
        for col_idx, header in enumerate(ExcelExporter.HEADERS, start=1):
            col_letter = ws.cell(row=1, column=col_idx).column_letter
            ws.column_dimensions[col_letter].width = ExcelExporter.COLUMN_WIDTHS.get(header, 15)
