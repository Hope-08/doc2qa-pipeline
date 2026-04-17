"""jobs router 集成测试：验证 API 端点返回正确字段。"""

from __future__ import annotations

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


class TestGetJobResult:
    """GET /api/v1/jobs/{job_id}/result 测试。"""

    def test_result_includes_reason_and_reasoning_steps(self, client: TestClient):
        """验证 result 端点返回 reason 和 reasoning_steps 字段。"""
        # 创建一个已完成的任务，结果中包含 reason 和 reasoning_steps
        job = job_store.create()
        job_store.update(
            job.job_id,
            status="completed",
            result=[
                {
                    "qa_id": "qa_001",
                    "doc_id": "doc_001",
                    "question": "深度学习的核心是什么？",
                    "answer": "多层神经网络",
                    "question_type": "definition",
                    "content_type": "text",
                    "quality_score": 0.85,
                    "faithfulness_score": 0.90,
                    "relevance_score": 0.80,
                    "is_passed": True,
                    "reason": "该问题涵盖了深度学习的核心概念",
                    "reasoning_steps": "1. 识别定义类问题\n2. 提取关键概念\n3. 生成简洁答案",
                },
            ],
        )

        response = client.get(f"/api/v1/jobs/{job.job_id}/result")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "completed"
        assert data["total_qa"] == 1
        assert len(data["qa_pairs"]) == 1

        qa = data["qa_pairs"][0]
        assert qa["qa_id"] == "qa_001"
        assert qa["question"] == "深度学习的核心是什么？"
        assert qa["answer"] == "多层神经网络"
        # 验证新字段存在且值正确
        assert qa["reason"] == "该问题涵盖了深度学习的核心概念"
        assert qa["reasoning_steps"] == "1. 识别定义类问题\n2. 提取关键概念\n3. 生成简洁答案"

    def test_result_default_values_for_missing_fields(self, client: TestClient):
        """验证当 QA 数据缺少 reason/reasoning_steps 时返回默认空字符串（向后兼容）。"""
        job = job_store.create()
        job_store.update(
            job.job_id,
            status="completed",
            result=[
                {
                    "qa_id": "qa_002",
                    "doc_id": "doc_002",
                    "question": "什么是机器学习？",
                    "answer": "机器学习是人工智能的一个分支",
                    "question_type": "definition",
                    "content_type": "text",
                    # 注意：没有 reason 和 reasoning_steps 字段
                },
            ],
        )

        response = client.get(f"/api/v1/jobs/{job.job_id}/result")
        assert response.status_code == 200

        data = response.json()
        qa = data["qa_pairs"][0]
        # 验证默认值为空字符串
        assert qa["reason"] == ""
        assert qa["reasoning_steps"] == ""

    def test_result_job_not_found(self, client: TestClient):
        """验证查询不存在的任务返回 404。"""
        response = client.get("/api/v1/jobs/nonexistent_job_id/result")
        assert response.status_code == 404
        assert "不存在" in response.json()["detail"]

    def test_result_pending_job_returns_empty_qa_pairs(self, client: TestClient):
        """验证未完成的任务返回空 qa_pairs 列表。"""
        job = job_store.create()
        # 任务保持 pending 状态，没有 result

        response = client.get(f"/api/v1/jobs/{job.job_id}/result")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "pending"
        assert data["total_qa"] == 0
        assert data["qa_pairs"] == []
