"""POST /api/v1/jobs 多文件上传集成测试。"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.job_store import job_store


@pytest.fixture
def client() -> TestClient:
    """创建测试客户端，每个测试使用独立的 app 实例。"""
    app = create_app()
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_job_store():
    """每个测试前后清空 job_store，避免测试污染。"""
    job_store._jobs.clear()
    yield
    job_store._jobs.clear()


def make_test_file(filename: str, content: bytes = b"x" * 150) -> tuple[str, io.BytesIO, str]:
    """创建测试文件元组，格式为 (filename, file_object, content_type)。

    默认内容为 150 字节（满足最小文件大小要求 100 字节）。
    """
    return (filename, io.BytesIO(content), "application/octet-stream")


class TestMultiFileUpload:
    """测试多文件上传功能。"""

    def test_upload_multiple_files_success(self, client: TestClient):
        """验证可以成功上传多个文件。"""
        files = [
            ("files", make_test_file("doc1.pdf")),
            ("files", make_test_file("doc2.pdf")),
            ("files", make_test_file("doc3.docx")),
        ]

        with patch("api.routers.jobs._run_pipeline"):
            response = client.post("/api/v1/jobs", files=files)

        assert response.status_code == 202
        data = response.json()
        assert data["file_count"] == 3
        assert data["status"] == "pending"
        assert "job_id" in data

    def test_upload_single_file_still_works(self, client: TestClient):
        """验证单文件上传仍然兼容。"""
        files = [("files", make_test_file("single.pdf"))]

        with patch("api.routers.jobs._run_pipeline"):
            response = client.post("/api/v1/jobs", files=files)

        assert response.status_code == 202
        data = response.json()
        assert data["file_count"] == 1

    def test_upload_no_files_returns_422(self, client: TestClient):
        """验证不上传文件返回 422 错误。"""
        response = client.post("/api/v1/jobs", files=[])

        assert response.status_code == 422
        # FastAPI 返回 validation error，格式为 list[dict] 或 string
        detail = response.json()["detail"]
        if isinstance(detail, list):
            # FastAPI validation error 格式
            assert any("files" in str(d) or "required" in str(d).lower() for d in detail)
        else:
            # 自定义错误消息
            assert "至少" in detail or "必须" in detail

    def test_upload_invalid_file_type_returns_415(self, client: TestClient):
        """验证上传不支持的文件类型返回 415 错误。"""
        files = [
            ("files", make_test_file("doc1.pdf")),
            ("files", make_test_file("invalid.exe")),  # 不支持的类型
        ]

        response = client.post("/api/v1/jobs", files=files)

        assert response.status_code == 415
        assert "invalid.exe" in response.json()["detail"]

    def test_upload_oversized_file_returns_413(self, client: TestClient):
        """验证上传过大文件返回 413 错误。"""
        # 创建超过 100MB 的大文件模拟
        large_content = b"x" * (101 * 1024 * 1024)  # 101 MB
        files = [
            ("files", make_test_file("small.pdf")),
            ("files", ("large.pdf", io.BytesIO(large_content), "application/pdf")),
        ]

        response = client.post("/api/v1/jobs", files=files)

        assert response.status_code == 413
        assert "过大" in response.json()["detail"]

    def test_upload_files_saved_to_same_directory(self, client: TestClient):
        """验证所有文件保存到同一临时目录。"""
        files = [
            ("files", make_test_file("doc1.pdf")),
            ("files", make_test_file("doc2.pdf")),
        ]

        saved_paths = []

        def capture_paths(job_id, file_paths, *args, **kwargs):
            saved_paths.extend(file_paths)

        with patch("api.routers.jobs._run_pipeline", side_effect=capture_paths):
            response = client.post("/api/v1/jobs", files=files)

        assert response.status_code == 202
        assert len(saved_paths) == 2

        # 验证所有文件在同一目录
        from pathlib import Path

        dirs = {Path(p).parent for p in saved_paths}
        assert len(dirs) == 1, "所有文件应保存在同一目录"

    def test_upload_mixed_types_success(self, client: TestClient):
        """验证可以混合上传不同类型的文件。"""
        files = [
            ("files", make_test_file("document.pdf")),
            ("files", make_test_file("report.docx")),
            ("files", make_test_file("screenshot.png")),
            ("files", make_test_file("photo.jpg")),
        ]

        with patch("api.routers.jobs._run_pipeline"):
            response = client.post("/api/v1/jobs", files=files)

        assert response.status_code == 202
        assert response.json()["file_count"] == 4

    def test_file_count_in_job_store_matches_uploaded(self, client: TestClient):
        """验证 job_store 中的 file_count 与上传文件数一致。"""
        files = [
            ("files", make_test_file("doc1.pdf")),
            ("files", make_test_file("doc2.pdf")),
            ("files", make_test_file("doc3.pdf")),
        ]

        with patch("api.routers.jobs._run_pipeline"):
            response = client.post("/api/v1/jobs", files=files)

        job_id = response.json()["job_id"]
        job = job_store.get(job_id)
        assert job is not None
        assert job.file_count == 3

    def test_all_file_paths_stored_in_job(self, client: TestClient):
        """验证所有文件路径都存储在 job 记录中。"""
        files = [
            ("files", make_test_file("doc1.pdf")),
            ("files", make_test_file("doc2.pdf")),
        ]

        with patch("api.routers.jobs._run_pipeline"):
            response = client.post("/api/v1/jobs", files=files)

        job_id = response.json()["job_id"]
        job = job_store.get(job_id)
        assert job is not None
        assert len(job.file_paths) == 2
        # 验证文件名在路径中
        path_names = [p.split("/")[-1].split("\\")[-1] for p in job.file_paths]
        assert "doc1.pdf" in path_names
        assert "doc2.pdf" in path_names

    def test_validation_stops_at_first_invalid_file(self, client: TestClient):
        """验证遇到无效文件时立即停止，不保存任何文件。"""
        files = [
            ("files", make_test_file("valid1.pdf")),
            ("files", make_test_file("invalid.txt")),  # 第二个文件无效
            ("files", make_test_file("valid2.pdf")),
        ]

        response = client.post("/api/v1/jobs", files=files)

        assert response.status_code == 415
        # 确保没有创建任何 job
        assert len(job_store._jobs) == 0
