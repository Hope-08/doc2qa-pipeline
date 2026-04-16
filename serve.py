"""API 服务启动入口。

使用方法::

    # 开发模式（热重载）
    uv run python serve.py --reload

    # 生产模式
    uv run python serve.py

    # 或直接用 uvicorn
    uv run uvicorn api.app:app --host 0.0.0.0 --port 8080

访问交互文档：http://localhost:8080/docs
"""

from __future__ import annotations

import click
import uvicorn
from dotenv import load_dotenv

load_dotenv()  # 将 .env 全量注入 os.environ（langchain/openai 等库直接读 os.environ）


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="监听地址")
@click.option("--port", default=8080, show_default=True, help="监听端口")
@click.option("--reload", is_flag=True, default=False, help="开启热重载（开发模式）")
@click.option("--workers", default=1, show_default=True, help="worker 进程数（生产环境建议 2-4）")
def main(host: str, port: int, reload: bool, workers: int) -> None:
    """启动 doc2qa-pipeline API 服务。"""
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,  # reload 模式不支持多 worker
        log_level="info",
    )


if __name__ == "__main__":
    main()
