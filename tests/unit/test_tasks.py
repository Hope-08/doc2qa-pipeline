"""export_task 单元测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path


def test_export_task_excel_format():
    """export_task 支持 excel 格式。"""
    from pipeline.tasks import export_task

    qa_dicts = [
        {
            "qa_id": "qa_1",
            "chunk_id": "chunk_1",
            "doc_id": "doc1",
            "question": "问题",
            "answer": "答案",
            "context": "上下文",
            "question_type": "definition",
            "content_type": "text",
            "reason": "理由",
            "reasoning_steps": "步骤",
            "is_passed": True,
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output = str(Path(tmpdir) / "test.xlsx")
        result = export_task.fn(qa_dicts, output, "excel")

        assert Path(result).exists()
        assert result.endswith(".xlsx")
