"""任务管理路由：文档上传、任务状态查询、结果获取。"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, UploadFile

from api.job_store import job_store
from api.schemas import (
    JobCreatedResponse,
    JobListItem,
    JobListResponse,
    JobResultResponse,
    JobStatusResponse,
    QAPairResponse,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

# 支持上传的文件后缀
_ALLOWED_SUFFIXES = {".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg"}
# 单文件上传大小上限（100 MB）
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024


# ---------------------------------------------------------------------------
# POST /api/v1/jobs  —  创建任务
# ---------------------------------------------------------------------------


@router.post("", response_model=JobCreatedResponse, status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    model: str | None = None,
    export_format: str = "jsonl",
) -> JobCreatedResponse:
    """上传文档，创建 QA 生成任务。

    - **file**：支持 PDF / DOCX / 图片（单文件）
    - **model**：LLM 模型名称，不填则使用服务端默认配置
    - **export_format**：输出格式，可选 `jsonl` / `hf_dataset` / `alpaca` / `ragas`

    返回 `job_id`，通过 `GET /api/v1/jobs/{job_id}` 轮询状态。
    """
    # 校验文件类型
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型：{file.filename}，支持：{sorted(_ALLOWED_SUFFIXES)}",
        )

    # 校验文件大小（size 可能为 None，需实际读取计量）
    if file.size is not None and file.size > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大：{file.size} 字节，上限 {_MAX_UPLOAD_BYTES} 字节（100 MB）",
        )

    # 校验 export_format
    valid_formats = {"jsonl", "hf_dataset", "alpaca", "ragas"}
    if export_format not in valid_formats:
        raise HTTPException(
            status_code=422,
            detail=f"不支持的格式：{export_format}，可选：{sorted(valid_formats)}",
        )

    # 创建任务记录
    job = job_store.create()
    job_store.update(job.job_id, file_count=1)

    # 保存上传文件到临时目录（防路径遍历：只保留文件名，替换非法字符，禁止连续点）
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"doc2qa_{job.job_id}_"))
    raw_name = Path(file.filename or "upload").name
    safe_name = re.sub(r"[^A-Za-z0-9._\u4e00-\u9fff-]", "_", raw_name) or "upload"
    # 禁止连续点（防止 .. 绕过路径限制）
    safe_name = re.sub(r"\.{2,}", "_", safe_name)
    dest = tmp_dir / safe_name
    written = 0
    try:
        with open(dest, "wb") as out:
            while chunk := file.file.read(1024 * 1024):  # 分块写入，逐步计量大小
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件过大，上限 {_MAX_UPLOAD_BYTES} 字节（100 MB）",
                    )
                out.write(chunk)

        # 路径穿越验证：确保写入路径在沙盒目录内
        if not dest.resolve().is_relative_to(tmp_dir.resolve()):
            raise HTTPException(status_code=400, detail="非法文件名")

    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        job_store.update(job.job_id, status="failed", error="上传失败")
        raise
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        job_store.update(job.job_id, status="failed", error="上传失败")
        logger.exception("文件写入失败 job_id=%s：%s", job.job_id, exc)
        raise HTTPException(status_code=500, detail="文件上传失败，请重试")

    saved_paths = [str(dest)]
    job_store.update(job.job_id, file_paths=saved_paths)

    logger.info("任务创建 job_id=%s，文件=%s", job.job_id, safe_name)

    # 后台运行 Pipeline（不阻塞当前请求）
    background_tasks.add_task(
        _run_pipeline,
        job.job_id,
        saved_paths,
        model,
        export_format,
        tmp_dir,
    )

    return JobCreatedResponse(
        job_id=job.job_id,
        status="pending",
        file_count=1,
        created_at=job.created_at,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/jobs  —  任务列表
# ---------------------------------------------------------------------------


@router.get("", response_model=JobListResponse)
def list_jobs() -> JobListResponse:
    """列出所有任务（按创建时间倒序）。"""
    jobs = job_store.list_all()
    return JobListResponse(
        total=len(jobs),
        jobs=[
            JobListItem(
                job_id=j.job_id,
                status=j.status,
                progress=j.progress,
                file_count=j.file_count,
                created_at=j.created_at,
                updated_at=j.updated_at,
            )
            for j in jobs
        ],
    )


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}  —  任务状态
# ---------------------------------------------------------------------------


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """查询任务当前状态。

    - `pending`：等待处理
    - `running`：正在处理
    - `completed`：处理完成，可调用 `/result` 获取结果
    - `failed`：处理失败，`error` 字段含错误信息
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{job_id}")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/result  —  获取结果
# ---------------------------------------------------------------------------


@router.get("/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str) -> JobResultResponse:
    """获取任务生成的 QA 结果。

    任务未完成时返回 `status` 和空 `qa_pairs`，不报错。
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{job_id}")

    qa_responses: list[QAPairResponse] = []
    for qa in job.result:
        qa_responses.append(
            QAPairResponse(
                qa_id=qa.get("qa_id", ""),
                doc_id=qa.get("doc_id", ""),
                question=qa.get("question", ""),
                answer=qa.get("answer", ""),
                question_type=qa.get("question_type", "unknown"),
                content_type=qa.get("content_type", "text"),
                quality_score=qa.get("quality_score"),
                faithfulness_score=qa.get("faithfulness_score"),
                relevance_score=qa.get("relevance_score"),
                is_passed=qa.get("is_passed", True),
            )
        )

    return JobResultResponse(
        job_id=job.job_id,
        status=job.status,
        total_qa=len(qa_responses),
        qa_pairs=qa_responses,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# DELETE /api/v1/jobs/{job_id}  —  删除任务
# ---------------------------------------------------------------------------


@router.delete("/{job_id}", status_code=204)
def delete_job(job_id: str) -> Response:
    """删除任务记录。

    若任务仍在运行（status=running），返回 HTTP 409 拒绝删除，
    避免删除后后台线程继续运行但无法追踪的竞态问题。
    """
    deleted, reason = job_store.delete_if_not_running(job_id)
    if not deleted and reason is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{job_id}")
    if reason == "running":
        raise HTTPException(
            status_code=409,
            detail="任务正在运行中，无法删除。请等待任务完成或失败后再删除。",
        )
    logger.info("任务已删除 job_id=%s", job_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# 后台 Pipeline 执行函数
# ---------------------------------------------------------------------------


def _run_pipeline(
    job_id: str,
    file_paths: list[str],
    model: str | None,
    export_format: str,
    tmp_dir: Path,
) -> None:
    """在 FastAPI BackgroundTask 线程中运行完整 Pipeline。"""
    try:
        job_store.update(job_id, status="running", progress="Pipeline 启动中...")
        logger.info("Pipeline 开始 job_id=%s，文件数=%d", job_id, len(file_paths))

        from pipeline.flow import doc2qa_flow

        job_store.update(job_id, progress=f"处理 {len(file_paths)} 个文档...")
        result = doc2qa_flow(
            file_paths=file_paths,
            model=model,
            export_format=export_format,
            job_id=job_id,
        )

        from config.settings import settings as _settings
        job_output = Path(_settings.output_dir).resolve() / job_id
        job_store.update(
            job_id,
            status="completed",
            result=result,
            progress=f"完成，生成 {len(result)} 个 QA 对（中间结果：{job_output}/）",
        )
        logger.info("Pipeline 完成 job_id=%s，QA 数=%d", job_id, len(result))

    except Exception as exc:
        # 对客户端只暴露通用错误分类，详细堆栈仅写服务端日志
        exc_type = type(exc).__name__
        if "FileNotFoundError" in exc_type:
            client_msg = "文档解析失败：文件不存在"
        elif "ImportError" in exc_type or "ModuleNotFoundError" in exc_type:
            client_msg = "服务端依赖未就绪，请联系管理员"
        elif "TimeoutError" in exc_type or "Timeout" in exc_type:
            client_msg = "处理超时，请稍后重试"
        else:
            client_msg = "处理失败，请查看服务端日志"
        job_store.update(job_id, status="failed", error=client_msg, progress="处理失败")
        logger.exception("Pipeline 失败 job_id=%s：%s", job_id, exc)

    finally:
        # 清理上传的临时文件
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("临时目录已清理：%s", tmp_dir)
