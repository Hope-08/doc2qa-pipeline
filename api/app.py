"""FastAPI 应用工厂。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.logger import configure_root_logger, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子。"""
    # 启动时
    configure_root_logger()
    logger.info("doc2qa-pipeline API 服务启动")

    from config.settings import settings

    logger.info("解析引擎: %s  OCR语言: %s", settings.parsing_backend, settings.ocr_lang)
    logger.info("LLM 模型: %s", settings.llm_model)
    logger.info("CORS 允许来源: %s", settings.cors_origins)

    yield

    # 关闭时
    logger.info("doc2qa-pipeline API 服务关闭")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。"""
    app = FastAPI(
        title="doc2qa-pipeline API",
        description=(
            "企业知识库 QA 数据自动生成接口。\n\n"
            "**使用流程**：\n"
            "1. `POST /api/v1/jobs` 上传文档（支持多文件），获得 `job_id`\n"
            "2. `GET /api/v1/jobs/{job_id}` 轮询任务状态，直到 `completed`\n"
            "3. `GET /api/v1/jobs/{job_id}/result` 获取 QA 结果（JSON）\n"
            "4. `GET /api/v1/jobs/{job_id}/export/excel` 下载 Excel 文件"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS：来源由配置文件控制（CORS_ORIGINS 环境变量或 .env）
    from config.settings import settings as _settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    # 注册路由
    from api.routers import health, jobs

    app.include_router(health.router)
    app.include_router(jobs.router)

    return app


app = create_app()
