"""健康检查路由。"""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/v1/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """服务健康检查。

    返回服务状态和关键配置信息，可用于负载均衡探活。
    """
    from config.settings import settings

    return HealthResponse(
        status="ok",
        parsing_backend=settings.parsing_backend,
    )
