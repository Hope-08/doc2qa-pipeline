"""单元测试：EntityExtractor 实体提取器。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from models import SemanticChunk


def _make_chunk() -> SemanticChunk:
    return SemanticChunk(
        chunk_id="doc_chunk_0",
        doc_id="doc",
        content_type="text",
        content="乌鲁木齐到喀纳斯行程介绍",
        source_items=["doc_item_0"],
        page_idx=0,
        token_count=20,
    )


def test_extract_entities_returns_enriched_chunk() -> None:
    """成功路径：LLM 返回有效 JSON，实体正确解析。"""
    from qa_generation.entity_extractor import EntityExtractor

    mock_response = MagicMock()
    mock_response.content = '{"entities": ["乌鲁木齐", "喀纳斯"]}'

    with patch("qa_generation.entity_extractor.ChatOpenAI") as mock_llm_cls:
        mock_llm_cls.return_value.invoke.return_value = mock_response
        extractor = EntityExtractor(model="azure-openai/gpt-4o")
        enriched = extractor.extract(_make_chunk())

    assert enriched.chunk_id == "doc_chunk_0"
    assert enriched.entities == ["乌鲁木齐", "喀纳斯"]
    assert enriched.doc_id == "doc"
    assert enriched.content == "乌鲁木齐到喀纳斯行程介绍"


def test_extract_entities_returns_empty_list_on_invalid_json() -> None:
    """JSON 解析失败时返回空实体列表，不抛异常。"""
    from qa_generation.entity_extractor import EntityExtractor

    mock_response = MagicMock()
    mock_response.content = "not-json"

    with patch("qa_generation.entity_extractor.ChatOpenAI") as mock_llm_cls:
        mock_llm_cls.return_value.invoke.return_value = mock_response
        extractor = EntityExtractor(model="azure-openai/gpt-4o")
        enriched = extractor.extract(_make_chunk())

    assert enriched.entities == []


def test_extract_entities_returns_empty_list_on_missing_key() -> None:
    """JSON 有效但缺少 entities 字段时返回空列表。"""
    from qa_generation.entity_extractor import EntityExtractor

    mock_response = MagicMock()
    mock_response.content = '{"result": []}'

    with patch("qa_generation.entity_extractor.ChatOpenAI") as mock_llm_cls:
        mock_llm_cls.return_value.invoke.return_value = mock_response
        extractor = EntityExtractor(model="azure-openai/gpt-4o")
        enriched = extractor.extract(_make_chunk())

    assert enriched.entities == []


# ─── 实体清洗 ─────────────────────────────────────────────────────────────────


def test_parse_entities_deduplicates() -> None:
    """_parse_entities 应对重复实体去重，保持首次出现顺序。"""
    from qa_generation.entity_extractor import EntityExtractor

    raw = '{"entities": ["乌鲁木齐", "喀纳斯", "乌鲁木齐", "天山"]}'
    result = EntityExtractor._parse_entities(raw)
    assert result == ["乌鲁木齐", "喀纳斯", "天山"]


def test_parse_entities_filters_too_short() -> None:
    """过短的实体（1~2字符且非专有名词特征）应被过滤。"""
    from qa_generation.entity_extractor import EntityExtractor

    raw = '{"entities": ["X", "Y", "乌鲁木齐", "A"]}'
    result = EntityExtractor._parse_entities(raw)
    # 单字符英文字母被过滤，中文专有名词保留
    assert "乌鲁木齐" in result
    assert "X" not in result


def test_parse_entities_filters_pure_numbers() -> None:
    """纯数字实体应被过滤。"""
    from qa_generation.entity_extractor import EntityExtractor

    raw = '{"entities": ["2024", "天山", "100", "喀纳斯"]}'
    result = EntityExtractor._parse_entities(raw)
    assert "2024" not in result
    assert "100" not in result
    assert "天山" in result


def test_parse_entities_filters_generic_terms() -> None:
    """泛词（如'文本'、'内容'、'信息'等）应被过滤。"""
    from qa_generation.entity_extractor import EntityExtractor

    raw = '{"entities": ["文本", "内容", "天山", "信息", "乌鲁木齐"]}'
    result = EntityExtractor._parse_entities(raw)
    assert "文本" not in result
    assert "内容" not in result
    assert "信息" not in result
    assert "天山" in result
    assert "乌鲁木齐" in result


def test_parse_entities_filters_punctuation_only() -> None:
    """纯标点/符号条目应被过滤。"""
    from qa_generation.entity_extractor import EntityExtractor

    raw = '{"entities": ["...", "——", "天山", "、"]}'
    result = EntityExtractor._parse_entities(raw)
    assert "天山" in result
    assert "..." not in result


# ─── Task 8: enrich_task & synthesize_qa_task ──────────────────────────────


def test_enrich_task_serializes_enriched_chunks(monkeypatch) -> None:
    """enrich_task 应返回序列化的 enriched dict 列表。"""
    from models import EnrichedChunk
    from pipeline.tasks import enrich_task

    chunk_dict = {
        "chunk_id": "doc_chunk_0",
        "doc_id": "doc",
        "content_type": "text",
        "content": "乌鲁木齐内容",
        "source_items": ["item_0"],
        "page_idx": 0,
        "token_count": 10,
        "metadata": {},
    }

    class _FakeExtractor:
        def extract(self, chunk: SemanticChunk) -> EnrichedChunk:
            return EnrichedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content_type=chunk.content_type,
                content=chunk.content,
                source_items=list(chunk.source_items),
                page_idx=chunk.page_idx,
                token_count=chunk.token_count,
                entities=["乌鲁木齐"],
            )

    monkeypatch.setattr(
        "pipeline.tasks.EntityExtractor",
        lambda model, **kwargs: _FakeExtractor(),
    )

    result = enrich_task.fn([chunk_dict], model="azure-openai/gpt-4o")

    assert result[0]["entities"] == ["乌鲁木齐"]
    assert result[0]["chunk_id"] == "doc_chunk_0"


def test_synthesize_qa_task_returns_qa_dicts(monkeypatch) -> None:
    """synthesize_qa_task 应返回序列化的 QA dict 列表。"""
    from models import QAPair
    from pipeline.tasks import synthesize_qa_task

    enriched_dict = {
        "chunk_id": "doc_chunk_0",
        "doc_id": "doc",
        "content_type": "text",
        "content": "内容" * 200,
        "source_items": ["item_0"],
        "page_idx": 0,
        "token_count": 400,
        "entities": ["实体A"],
        "summary": None,
        "metadata": {},
    }

    qa = QAPair(
        qa_id="qa_0",
        chunk_id="doc_chunk_0",
        doc_id="doc",
        question="问题",
        answer="答案",
        context="上下文",
        question_type="single_chunk",
        content_type="text",
    )

    class _FakeSynthesizer:
        def synthesize(self, chunks, relations, quota):
            return [qa]

    monkeypatch.setattr("pipeline.tasks.build_relation_index", lambda **kwargs: [])
    monkeypatch.setattr(
        "pipeline.tasks.DocumentQASynthesizer",
        lambda **kwargs: _FakeSynthesizer(),
    )

    result = synthesize_qa_task.fn(
        enriched_chunks=[enriched_dict],
        quota=5,
        synthesize_model="azure-openai/gpt-5.4",
        verifier_model="azure-openai/gpt-4o",
        embedding_model="azure-openai/text-embedding-3-small",
        self_consistency_check=False,
    )

    assert len(result) == 1
    assert result[0]["question"] == "问题"
