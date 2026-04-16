"""单元测试：doc2qa_flow 的 QA 策略路由（simple vs advanced）。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 共享辅助
# ---------------------------------------------------------------------------


def _make_mock_settings(qa_strategy: str) -> MagicMock:
    """构造用于替换 pipeline.flow._settings 的 mock 对象。"""
    cfg = MagicMock()
    cfg.llm_model = "gpt-4o"
    cfg.parsing_backend = "auto"
    cfg.ocr_lang = "chi_sim+eng"
    cfg.chunk_size = 800
    cfg.similarity_threshold = 0.3
    cfg.embedding_model = "azure-openai/text-embedding-3-small"
    cfg.llm_temperature = 0.3
    cfg.faithfulness_min = 0.6
    cfg.relevance_min = 0.5
    cfg.answer_min_length = 15
    cfg.composite_pass = 0.55
    cfg.nli_model = "azure-openai/gpt-5.4"
    cfg.output_dir = "./output"
    cfg.qa_strategy = qa_strategy
    cfg.max_qa_per_doc = 10
    cfg.entity_extractor_model = "azure-openai/gpt-4o"
    cfg.synthesize_model = "azure-openai/gpt-5.4"
    cfg.self_consistency_check = True
    return cfg


def _stub_chunk() -> dict:
    return {
        "chunk_id": "c0",
        "doc_id": "doc",
        "content_type": "text",
        "content": "测试内容" * 30,
        "source_items": ["item_0"],
        "page_idx": 0,
        "token_count": 120,
        "metadata": {},
    }


def _stub_enriched_chunk() -> dict:
    return {
        **_stub_chunk(),
        "entities": ["实体A"],
        "summary": None,
    }


def _stub_qa() -> dict:
    return {
        "qa_id": "qa_0",
        "chunk_id": "c0",
        "doc_id": "doc",
        "question": "问题",
        "answer": "答案",
        "context": "上下文",
        "question_type": "single_chunk",
        "content_type": "text",
        "quality_score": None,
        "faithfulness_score": None,
        "relevance_score": None,
        "completeness_score": None,
        "is_passed": True,
        "fail_reasons": [],
    }


# ---------------------------------------------------------------------------
# 测试 1：advanced 策略
# ---------------------------------------------------------------------------


def test_advanced_strategy_calls_enrich_and_synthesize_not_generate(tmp_path) -> None:
    """cfg.qa_strategy == 'advanced' 时：
    - enrich_task 和 synthesize_qa_task 必须被调用
    - generate_qa_task 不得被调用
    """
    from pipeline.flow import doc2qa_flow

    mock_cfg = _make_mock_settings("advanced")
    mock_cfg.output_dir = str(tmp_path)

    parsed = {
        "doc_id": "doc",
        "source_path": "doc.pdf",
        "markdown": "",
        "content_list": [],
        "metadata": {},
    }
    cleaned = dict(parsed)
    chunks = [_stub_chunk()]
    enriched = [_stub_enriched_chunk()]
    qa_raw = [_stub_qa()]
    qa_filtered = [_stub_qa()]

    with (
        patch("pipeline.flow._settings", mock_cfg),
        patch("pipeline.flow.parse_task") as mock_parse,
        patch("pipeline.flow.normalize_task") as mock_normalize,
        patch("pipeline.flow.chunk_task") as mock_chunk,
        patch("pipeline.flow.enrich_task") as mock_enrich,
        patch("pipeline.flow.synthesize_qa_task") as mock_synthesize,
        patch("pipeline.flow.generate_qa_task") as mock_generate,
        patch("pipeline.flow.score_filter_task") as mock_score,
        patch("pipeline.flow.export_task") as mock_export,
        patch("qa_generation.doc_qa_synthesizer.calc_doc_quota", return_value=5),
    ):
        mock_parse.fn = MagicMock(return_value=parsed)
        mock_parse.return_value = parsed
        mock_normalize.fn = MagicMock(return_value=cleaned)
        mock_normalize.return_value = cleaned
        mock_chunk.fn = MagicMock(return_value=chunks)
        mock_chunk.return_value = chunks
        mock_enrich.fn = MagicMock(return_value=enriched)
        mock_enrich.return_value = enriched
        mock_synthesize.fn = MagicMock(return_value=qa_raw)
        mock_synthesize.return_value = qa_raw
        mock_generate.fn = MagicMock(return_value=[])
        mock_generate.return_value = []
        mock_score.fn = MagicMock(return_value=qa_filtered)
        mock_score.return_value = qa_filtered
        mock_export.fn = MagicMock(return_value=str(tmp_path / "qa_dataset.jsonl"))
        mock_export.return_value = str(tmp_path / "qa_dataset.jsonl")

        result = doc2qa_flow.fn(
            file_paths=["doc.pdf"],
            job_id="test_advanced_job",
        )

    mock_enrich.assert_called_once()
    mock_synthesize.assert_called_once()
    mock_generate.assert_not_called()
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 测试 2：simple 策略
# ---------------------------------------------------------------------------


def test_simple_strategy_calls_generate_not_enrich(tmp_path) -> None:
    """cfg.qa_strategy == 'simple' 时：
    - generate_qa_task 必须被调用
    - enrich_task 和 synthesize_qa_task 不得被调用
    """
    from pipeline.flow import doc2qa_flow

    mock_cfg = _make_mock_settings("simple")
    mock_cfg.output_dir = str(tmp_path)

    parsed = {
        "doc_id": "doc",
        "source_path": "doc.pdf",
        "markdown": "",
        "content_list": [],
        "metadata": {},
    }
    cleaned = dict(parsed)
    chunks = [_stub_chunk()]
    qa_raw = [_stub_qa()]
    qa_filtered = [_stub_qa()]

    with (
        patch("pipeline.flow._settings", mock_cfg),
        patch("pipeline.flow.parse_task") as mock_parse,
        patch("pipeline.flow.normalize_task") as mock_normalize,
        patch("pipeline.flow.chunk_task") as mock_chunk,
        patch("pipeline.flow.enrich_task") as mock_enrich,
        patch("pipeline.flow.synthesize_qa_task") as mock_synthesize,
        patch("pipeline.flow.generate_qa_task") as mock_generate,
        patch("pipeline.flow.score_filter_task") as mock_score,
        patch("pipeline.flow.export_task") as mock_export,
    ):
        mock_parse.fn = MagicMock(return_value=parsed)
        mock_parse.return_value = parsed
        mock_normalize.fn = MagicMock(return_value=cleaned)
        mock_normalize.return_value = cleaned
        mock_chunk.fn = MagicMock(return_value=chunks)
        mock_chunk.return_value = chunks
        mock_enrich.fn = MagicMock(return_value=[])
        mock_enrich.return_value = []
        mock_synthesize.fn = MagicMock(return_value=[])
        mock_synthesize.return_value = []
        mock_generate.fn = MagicMock(return_value=qa_raw)
        mock_generate.return_value = qa_raw
        mock_score.fn = MagicMock(return_value=qa_filtered)
        mock_score.return_value = qa_filtered
        mock_export.fn = MagicMock(return_value=str(tmp_path / "qa_dataset.jsonl"))
        mock_export.return_value = str(tmp_path / "qa_dataset.jsonl")

        result = doc2qa_flow.fn(
            file_paths=["doc.pdf"],
            job_id="test_simple_job",
        )

    mock_generate.assert_called_once()
    mock_enrich.assert_not_called()
    mock_synthesize.assert_not_called()
    assert isinstance(result, list)
