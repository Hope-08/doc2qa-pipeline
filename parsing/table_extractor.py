"""表格补充提取器：针对数字原生 PDF 使用 camelot 提取高精度表格。

作为主解析引擎的补充方案，当主引擎在复杂表格场景下识别不准确时启用。

注意：
  camelot-py[cv] 依赖 ghostscript（系统级），Windows 需单独安装。
  下载地址：https://www.ghostscript.com/releases/gsdnld.html
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class TableExtractor:
    """基于 camelot 的 PDF 表格补充提取器。

    Args:
        flavor:      ``"lattice"``（有线框表格）或 ``"stream"``（无线框）。
        accuracy_min: 保留表格的最低精度分（0~100），低于此值丢弃。
    """

    def __init__(
        self,
        flavor: str = "lattice",
        accuracy_min: float = 80.0,
    ) -> None:
        self.flavor = flavor
        self.accuracy_min = accuracy_min

    def extract(
        self,
        pdf_path: str,
        pages: str = "all",
    ) -> list[dict]:
        """从 PDF 中提取表格。

        Args:
            pdf_path: PDF 文件路径（仅支持数字原生 PDF）。
            pages:    页码范围，camelot 格式，例如 ``"1,3,5-7"``。

        Returns:
            表格列表，每项包含：
            - ``html``:     HTML 格式表格字符串
            - ``page``:     所在页码
            - ``accuracy``: camelot 精度分
            - ``df``:       pandas DataFrame（供调试）
        """
        try:
            import camelot
        except ImportError:
            logger.warning(
                "camelot-py 未安装，跳过表格补充提取。"
                "安装：uv add 'camelot-py[cv]>=0.11.0'"
            )
            return []

        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"文件不存在：{pdf_path}")

        logger.info("camelot 提取表格 [%s] pages=%s", Path(pdf_path).name, pages)

        try:
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor=self.flavor)
        except Exception as exc:
            logger.warning("camelot 提取失败，跳过：%s", exc)
            return []

        results = []
        for table in tables:
            if table.accuracy < self.accuracy_min:
                logger.debug(
                    "丢弃低精度表格（accuracy=%.1f < %.1f）",
                    table.accuracy,
                    self.accuracy_min,
                )
                continue

            results.append(
                {
                    "html": table.df.to_html(index=False, border=0),
                    "page": table.page,
                    "accuracy": table.accuracy,
                    # 注意：df（pandas DataFrame）不包含在返回值中以避免序列化问题
                }
            )

        logger.info("camelot 提取到 %d 个有效表格", len(results))
        return results
