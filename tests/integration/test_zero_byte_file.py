"""测试零字节文件验证。"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient


class TestZeroByteFileValidation:
    """验证 API 拒绝空文件（0 字节）。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前清空 job_store。"""
        from api.job_store import job_store

        job_store._jobs.clear()
        yield
        job_store._jobs.clear()

    def test_zero_byte_file_rejected(self):
        """0 字节文件应被拒绝并返回 422。"""
        from api.app import create_app

        app = create_app()
        client = TestClient(app)

        # 创建 0 字节文件
        empty_file = io.BytesIO(b"")
        empty_file.name = "empty.pdf"

        response = client.post(
            "/api/v1/jobs",
            files=[("files", ("empty.pdf", empty_file, "application/pdf"))],
            data={"export_format": "jsonl"},
        )

        assert response.status_code == 422
        assert "过小" in response.json()["detail"] or "empty" in response.json()["detail"].lower()

    def test_very_small_file_rejected(self):
        """小于最小大小的文件应被拒绝。"""
        from api.app import create_app

        app = create_app()
        client = TestClient(app)

        # 创建 50 字节文件（小于 100 字节阈值）
        small_file = io.BytesIO(b"x" * 50)

        response = client.post(
            "/api/v1/jobs",
            files=[("files", ("tiny.pdf", small_file, "application/pdf"))],
            data={"export_format": "jsonl"},
        )

        assert response.status_code == 422
        assert "过小" in response.json()["detail"]

    def test_minimum_size_file_accepted(self):
        """刚好达到最小大小的文件应被接受。"""
        from unittest.mock import patch

        from api.app import create_app

        app = create_app()
        client = TestClient(app)

        # 创建 100 字节文件（最小阈值）
        valid_file = io.BytesIO(b"x" * 100)

        with patch("api.routers.jobs._run_pipeline"):
            response = client.post(
                "/api/v1/jobs",
                files=[("files", ("valid.pdf", valid_file, "application/pdf"))],
                data={"export_format": "jsonl"},
            )

            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
