"""线程安全的内存任务状态存储。

进程重启后数据丢失，适用于单机部署场景。
如需持久化，可将 JobStore 替换为 Redis 或数据库实现。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

JobStatus = Literal["pending", "running", "completed", "failed"]


@dataclass
class Job:
    """单个处理任务的状态记录。"""

    job_id: str
    status: JobStatus = "pending"
    progress: str = ""
    file_paths: list[str] = field(default_factory=list)  # 上传的临时文件路径
    file_count: int = 0
    result: list[dict] = field(default_factory=list)  # 最终 QA 对列表
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class JobStore:
    """线程安全的内存任务存储。

    所有写操作通过锁保护，适用于多线程 uvicorn（workers=1）场景。
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 写操作
    # ------------------------------------------------------------------

    def create(self) -> Job:
        """创建新任务，返回 Job 对象。

        job_id 格式：{YYYYMMDD_HHMMSS}_{4位随机十六进制}，便于按时间排序和肉眼识别。
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:4]
        job = Job(job_id=f"{ts}_{suffix}")
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def update(self, job_id: str, **kwargs: object) -> None:
        """更新任务字段，自动刷新 updated_at。

        Raises:
            AttributeError: 传入了 Job 不存在的字段名时立即报错，防止拼写错误静默丢失。
        """
        invalid = [k for k in kwargs if k not in Job.__dataclass_fields__]
        if invalid:
            raise AttributeError(f"Job 不存在以下字段：{invalid}，请检查字段名拼写")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)
            job.updated_at = time.time()

    def delete(self, job_id: str) -> bool:
        """删除任务，返回是否存在。"""
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def delete_if_not_running(self, job_id: str) -> tuple[bool, str | None]:
        """原子化检查并删除任务。

        在单一锁内完成状态检查与删除，避免 TOCTOU 竞态。

        Returns:
            (deleted, error_reason)
            - (True, None)：删除成功
            - (False, None)：任务不存在
            - (False, "running")：任务正在运行，拒绝删除
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False, None
            if job.status == "running":
                return False, "running"
            self._jobs.pop(job_id)
            return True, None

    # ------------------------------------------------------------------
    # 读操作
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        """查询单个任务，不存在返回 None。"""
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list[Job]:
        """返回所有任务列表（按创建时间倒序）。"""
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)


# 全局单例，在 app 启动时由 app.py 初始化后注入各路由
job_store = JobStore()
