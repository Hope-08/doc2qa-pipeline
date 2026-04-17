"""doc2qa-pipeline CLI 入口。

使用示例::

    # 基础运行（解析引擎从 .env 或环境变量读取）
    python main.py run --input docs/

    # 指定 LLM 模型
    python main.py run --input docs/ --model openai/gpt-4o

    # 使用 uv run
    uv run python main.py run --input docs/ --format alpaca
"""

from __future__ import annotations

import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()  # 将 .env 全量注入 os.environ（langchain/openai 等库直接读 os.environ）

from utils.logger import configure_root_logger


@click.group()
@click.option("--debug", is_flag=True, default=False, help="开启 DEBUG 日志")
def cli(debug: bool) -> None:
    """企业知识库 QA 数据自动生成平台。"""
    configure_root_logger(logging.DEBUG if debug else logging.INFO)


@cli.command("run")
@click.option(
    "--input",
    "-i",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help="输入文档路径，可多次指定。支持文件或目录（目录下所有 PDF/DOCX）。",
)
@click.option("--output", "-o", default=None, help="输出数据集路径")
@click.option(
    "--model",
    "-m",
    default=None,
    help="LLM 模型名称，如 gpt-4o、qwen2.5-72b-instruct",
)
@click.option(
    "--format",
    "-f",
    "export_format",
    default="jsonl",
    type=click.Choice(["jsonl", "hf_dataset", "alpaca", "ragas"]),
    show_default=True,
    help="输出格式",
)
def run_cmd(
    input: tuple[str, ...],
    output: str | None,
    model: str | None,
    export_format: str,
) -> None:
    """运行 QA 生成 Pipeline。

    解析引擎和 OCR 语言通过配置文件或环境变量指定：
    PARSING_BACKEND=auto
    OCR_LANG=chi_sim+eng
    """
    from pipeline.flow import doc2qa_flow

    # 展开目录输入
    file_paths: list[str] = []
    for p in input:
        path = Path(p)
        if path.is_dir():
            for ext in ("*.pdf", "*.docx", "*.doc", "*.png", "*.jpg", "*.jpeg"):
                file_paths.extend(str(f) for f in path.rglob(ext))
        else:
            file_paths.append(str(path))

    if not file_paths:
        click.echo("未找到任何支持的文档文件，退出。", err=True)
        sys.exit(1)

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{random.randint(1000, 9999)}"
    click.echo(f"共找到 {len(file_paths)} 个文档，开始处理...  job_id={job_id}")

    result = doc2qa_flow(
        file_paths=file_paths,
        output_path=output,
        model=model,
        export_format=export_format,
        job_id=job_id,
    )

    from config.settings import settings

    job_dir = Path(settings.output_dir).resolve() / job_id
    click.echo(f"完成！生成 {len(result)} 个高质量 QA 对。")
    click.echo(f"中间结果目录：{job_dir}")


if __name__ == "__main__":
    cli()
