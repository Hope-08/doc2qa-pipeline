"""GET /jobs/{id}/export/excel 端点集成测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.job_store import job_store


@pytest.fixture
def client() -> TestClient:
    """创建测试客户端。"""
    app = create_app()
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_job_store():
    """每个测试前后清空 job_store。"""
    job_store._jobs.clear()
    yield
    job_store._jobs.clear()


class TestExportExcelEndpoint:
    """GET /api/v1/jobs/{job_id}/export/excel 测试。"""

    def test_export_excel_returns_file(self, client: TestClient):
        """验证成功下载 Excel 文件。"""
        job = job_store.create()
        job_store.update(job.job_id, status="completed")

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建假的 Excel 文件
            job_dir = Path(tmpdir) / job.job_id
            job_dir.mkdir(parents=True)
            excel_path = job_dir / "qa_dataset.xlsx"
            excel_content = b"PK\x03\x04fake xlsx content"  # XLSX 实际是 ZIP 格式
            excel_path.write_bytes(excel_content)

            # Mock settings.output_dir 指向临时目录
            with patch("api.routers.jobs.settings") as mocksettings:
                mocksettings.output_dir = tmpdir

                response = client.get(f"/api/v1/jobs/{job.job_id}/export/excel")

                assert response.status_code == 200
                assert (
                    response.headers["content-type"]
                    == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                # 验证 Content-Disposition 包含文件名
                content_disp = response.headers.get("content-disposition", "")
                assert f"qa_{job.job_id}.xlsx" in content_disp
                # 验证返回内容与文件一致
                assert response.content == excel_content

    def test_export_excel_job_not_found(self, client: TestClient):
        """验证任务不存在时返回 404。"""
        response = client.get("/api/v1/jobs/nonexistent_job_id/export/excel")
        assert response.status_code == 404
        assert "不存在" in response.json()["detail"]

    def test_export_excel_job_not_completed(self, client: TestClient):
        """验证任务未完成时返回 409。"""
        job = job_store.create()
        # 任务保持 pending 状态

        response = client.get(f"/api/v1/jobs/{job.job_id}/export/excel")
        assert response.status_code == 409
        assert "尚未完成" in response.json()["detail"]

    def test_export_excel_job_running(self, client: TestClient):
        """验证任务正在运行时返回 409。"""
        job = job_store.create()
        job_store.update(job.job_id, status="running")

        response = client.get(f"/api/v1/jobs/{job.job_id}/export/excel")
        assert response.status_code == 409
        assert "尚未完成" in response.json()["detail"]

    def test_export_excel_file_not_exists(self, client: TestClient):
        """验证 Excel 文件不存在时返回 410。"""
        job = job_store.create()
        job_store.update(job.job_id, status="completed")

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建 job 目录但不创建 Excel 文件
            job_dir = Path(tmpdir) / job.job_id
            job_dir.mkdir(parents=True)

            with patch("api.routers.jobs.settings") as mocksettings:
                mocksettings.output_dir = tmpdir

                response = client.get(f"/api/v1/jobs/{job.job_id}/export/excel")

                assert response.status_code == 410
                assert "Excel 文件不存在" in response.json()["detail"]

    def test_export_excel_job_failed(self, client: TestClient):
        """验证任务失败时返回 409。"""
        job = job_store.create()
        job_store.update(job.job_id, status="failed", error="处理失败")

        response = client.get(f"/api/v1/jobs/{job.job_id}/export/excel")
        assert response.status_code == 409
        assert "尚未完成" in response.json()["detail"]
