"""单元测试：DatasetExporter。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from export.dataset_exporter import DatasetExporter
from models import QAPair


def _make_qa(idx: int = 0) -> QAPair:
    return QAPair(
        qa_id=f"qa_{idx}",
        chunk_id=f"chunk_{idx}",
        doc_id="doc1",
        question=f"问题 {idx}",
        answer=f"答案 {idx}，这是一个较为完整的答案内容。",
        context=f"上下文 {idx}",
        question_type="definition",
        content_type="text",
        quality_score=0.85,
        faithfulness_score=0.90,
        relevance_score=0.75,
        completeness_score=0.80,
        reason=f"出题理由 {idx}",
        reasoning_steps="步骤1: ...\n步骤2: ...",
        is_passed=True,
    )


class TestDatasetExporter:
    def test_export_jsonl(self) -> None:
        qa_pairs = [_make_qa(i) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "test.jsonl")
            result_path = DatasetExporter.export(qa_pairs, output, "jsonl")

            lines = Path(result_path).read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 3

            first = json.loads(lines[0])
            assert "messages" in first
            assert first["messages"][0]["role"] == "user"
            assert first["messages"][1]["role"] == "assistant"

    def test_export_alpaca(self) -> None:
        qa_pairs = [_make_qa(i) for i in range(2)]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "test.json")
            result_path = DatasetExporter.export(qa_pairs, output, "alpaca")

            data = json.loads(Path(result_path).read_text(encoding="utf-8"))
            assert len(data) == 2
            assert "instruction" in data[0]
            assert "output" in data[0]

    def test_export_ragas(self) -> None:
        qa_pairs = [_make_qa(0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "ragas.json")
            result_path = DatasetExporter.export(qa_pairs, output, "ragas")

            data = json.loads(Path(result_path).read_text(encoding="utf-8"))
            assert "question" in data[0]
            assert "ground_truth" in data[0]
            assert "contexts" in data[0]
            assert isinstance(data[0]["contexts"], list)

    def test_unsupported_format_raises(self) -> None:
        with pytest.raises(ValueError, match="不支持的格式"):
            DatasetExporter.export([], "out.txt", "csv")  # type: ignore[arg-type]

    def test_creates_parent_dirs(self) -> None:
        qa_pairs = [_make_qa(0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "nested" / "deep" / "out.jsonl")
            DatasetExporter.export(qa_pairs, output, "jsonl")
            assert Path(output).exists()
