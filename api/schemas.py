"""API 请求/响应 Pydantic 模型。

与项目根目录的 models.py（dataclass）分离，专用于 FastAPI 序列化。
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal["pending", "running", "completed", "failed"]


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------


class JobCreatedResponse(BaseModel):
    """POST /api/v1/jobs 响应。"""

    job_id: str
    status: JobStatus
    file_count: int
    created_at: float = Field(default_factory=time.time)

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "file_count": 1,
                "created_at": 1744617600.0,
            }
        }
    }


class JobStatusResponse(BaseModel):
    """GET /api/v1/jobs/{job_id} 响应。"""

    job_id: str
    status: JobStatus
    progress: str
    error: str | None = None
    created_at: float
    updated_at: float


class QAPairResponse(BaseModel):
    """单个 QA 对（用于结果接口）。"""

    qa_id: str
    doc_id: str
    question: str
    answer: str
    question_type: str
    content_type: str
    quality_score: float | None = None
    faithfulness_score: float | None = None
    relevance_score: float | None = None
    is_passed: bool = True
    reason: str = ""
    reasoning_steps: str = ""


class JobResultResponse(BaseModel):
    """GET /api/v1/jobs/{job_id}/result 响应。"""

    job_id: str
    status: JobStatus
    total_qa: int
    qa_pairs: list[QAPairResponse]
    error: str | None = None


class JobListItem(BaseModel):
    """任务列表条目。"""

    job_id: str
    status: JobStatus
    progress: str
    file_count: int
    created_at: float
    updated_at: float


class JobListResponse(BaseModel):
    """GET /api/v1/jobs 响应。"""

    total: int
    jobs: list[JobListItem]


class HealthResponse(BaseModel):
    """GET /api/v1/health 响应。"""

    status: str
    version: str = "1.0.0"
    parsing_backend: str
