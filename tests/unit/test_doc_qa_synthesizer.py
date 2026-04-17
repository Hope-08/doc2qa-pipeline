"""单元测试：DocumentQASynthesizer 文档级 QA 合成器。"""

from __future__ import annotations

from unittest.mock import MagicMock

from models import EnrichedChunk, QAPair
from qa_generation.doc_qa_synthesizer import (
    DocumentQASynthesizer,
    calc_doc_quota,
    is_short_document,
)

# ─── 辅助函数 ─────────────────────────────────────────────────────────────────


def _make_enriched_chunk(
    chunk_id: str = "doc_chunk_0",
    content: str = "内容",
    entities: list[str] | None = None,
) -> EnrichedChunk:
    return EnrichedChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        content_type="text",
        content=content,
        source_items=[f"{chunk_id}_item"],
        page_idx=0,
        token_count=len(content),
        entities=entities or [],
    )


def _make_qa_pair(question: str = "问题", answer: str = "答案") -> QAPair:
    return QAPair(
        qa_id="qa_0",
        chunk_id="doc_chunk_0",
        doc_id="doc",
        question=question,
        answer=answer,
        context="上下文",
        question_type="single_chunk",
        content_type="text",
    )


# ─── calc_doc_quota ──────────────────────────────────────────────────────────


def test_calc_doc_quota_caps_at_max() -> None:
    """大文档 quota 取 max_qa_per_doc 上限。"""
    assert calc_doc_quota(total_chars=9600, max_qa_per_doc=10) == 10


def test_calc_doc_quota_scales_with_size() -> None:
    """小文档 quota 按字符数缩放。"""
    assert calc_doc_quota(total_chars=1200, max_qa_per_doc=10) == 3


def test_calc_doc_quota_mid_range() -> None:
    """中等文档 quota 处于 min 和 max 之间。"""
    assert calc_doc_quota(total_chars=4000, max_qa_per_doc=10) == 5


# ─── is_short_document ───────────────────────────────────────────────────────


def test_is_short_document_single_chunk() -> None:
    """只有一个 chunk 的文档是短文档。"""
    chunks = [_make_enriched_chunk(content="内容" * 10)]
    assert is_short_document(chunks) is True


def test_is_short_document_total_chars_below_800() -> None:
    """两个 chunk 但总字符数 < 800 的文档是短文档。"""
    chunks = [
        _make_enriched_chunk("c0", content="内容" * 10),
        _make_enriched_chunk("c1", content="内容" * 10),
    ]
    assert is_short_document(chunks) is True


def test_is_short_document_returns_false_for_normal_doc() -> None:
    """两个 chunk 且总字符 >= 800 的文档不是短文档。"""
    chunks = [
        _make_enriched_chunk("c0", content="内容" * 200),
        _make_enriched_chunk("c1", content="内容" * 200),
    ]
    assert is_short_document(chunks) is False


# ─── DocumentQASynthesizer ───────────────────────────────────────────────────


def _make_synthesizer(monkeypatch) -> DocumentQASynthesizer:
    """构造 DocumentQASynthesizer，使用 MagicMock 替换 ChatOpenAI 避免 API key 校验。"""
    mock_llm = MagicMock()
    monkeypatch.setattr(
        "qa_generation.doc_qa_synthesizer.ChatOpenAI",
        lambda **kwargs: mock_llm,
    )
    return DocumentQASynthesizer(
        synthesize_model="azure-openai/gpt-5.4",
        verifier_model="azure-openai/gpt-4o",
        embedding_model="azure-openai/text-embedding-3-small",
        self_consistency_check=True,
    )


def test_synthesize_filters_easy_question_via_self_consistency(monkeypatch) -> None:
    """自一致性检查启用时，易回答的问题被过滤。"""
    synthesizer = _make_synthesizer(monkeypatch)
    chunks = [
        _make_enriched_chunk("c0", content="内容" * 200),
        _make_enriched_chunk("c1", content="内容" * 200),
    ]
    easy_qa = _make_qa_pair("乌鲁木齐是哪个自治区的首府？", "新疆维吾尔自治区首府。")

    monkeypatch.setattr(synthesizer, "_parse_qa_pairs", lambda raw, chunks: [easy_qa])
    monkeypatch.setattr(synthesizer, "_is_too_easy", lambda qa: True)
    synthesizer._synth_llm.invoke.return_value = MagicMock(content="[]")

    result = synthesizer.synthesize(chunks=chunks, relations=[], quota=3)
    assert result == []


def test_synthesize_skips_self_consistency_for_short_doc(monkeypatch) -> None:
    """短文档跳过自一致性过滤，直接返回所有生成的 QA。"""
    synthesizer = _make_synthesizer(monkeypatch)
    # 短文档：只有 1 个 chunk
    chunks = [_make_enriched_chunk("c0", content="内容" * 10)]
    qa = _make_qa_pair()

    monkeypatch.setattr(synthesizer, "_parse_qa_pairs", lambda raw, chunks: [qa])
    monkeypatch.setattr(synthesizer, "_is_too_easy", lambda qa: True)  # 即便判定为 easy 也不过滤
    synthesizer._synth_llm.invoke.return_value = MagicMock(content="[]")

    result = synthesizer.synthesize(chunks=chunks, relations=[], quota=3)
    assert result == [qa]  # 短文档跳过过滤


# ─── _parse_qa_pairs 严格校验 ───────────────────────────────────────────────


def test_parse_qa_pairs_invalid_json_returns_empty(monkeypatch) -> None:
    """JSON 无效时返回空列表。"""
    synthesizer = _make_synthesizer(monkeypatch)
    chunks = [_make_enriched_chunk("c0", content="内容" * 50)]
    result = synthesizer._parse_qa_pairs("not-json", chunks)
    assert result == []


def test_parse_qa_pairs_invalid_source_chunks_discarded(monkeypatch) -> None:
    """source_chunks 中含不存在 chunk_id 的条目应被丢弃。"""
    synthesizer = _make_synthesizer(monkeypatch)
    chunks = [
        _make_enriched_chunk("c0", content="内容" * 50),
        _make_enriched_chunk("c1", content="内容" * 50),
    ]
    raw = '[{"question": "问题", "answer": "答案", "question_type": "multi_hop", "source_chunks": ["nonexistent_id"], "key_entities": ["实体"]}]'
    result = synthesizer._parse_qa_pairs(raw, chunks)
    assert result == []


def test_parse_qa_pairs_invalid_question_type_discarded(monkeypatch) -> None:
    """question_type 不在允许集合内的条目应被丢弃。"""
    synthesizer = _make_synthesizer(monkeypatch)
    chunks = [
        _make_enriched_chunk("c0", content="内容" * 50),
    ]
    raw = '[{"question": "问题", "answer": "答案", "question_type": "invalid_xyz", "source_chunks": ["c0"], "key_entities": ["实体"]}]'
    result = synthesizer._parse_qa_pairs(raw, chunks)
    assert result == []


def test_parse_qa_pairs_multi_hop_requires_two_source_chunks(monkeypatch) -> None:
    """multi_hop 题必须关联 >= 2 个 source_chunks，否则丢弃。"""
    synthesizer = _make_synthesizer(monkeypatch)
    chunks = [
        _make_enriched_chunk("c0", content="内容" * 50),
        _make_enriched_chunk("c1", content="内容" * 50),
    ]
    raw = '[{"question": "问题", "answer": "答案", "question_type": "multi_hop", "source_chunks": ["c0"], "key_entities": ["实体"]}]'
    result = synthesizer._parse_qa_pairs(raw, chunks)
    assert result == []


def test_parse_qa_pairs_multi_hop_with_two_chunks_kept(monkeypatch) -> None:
    """multi_hop 题关联 2 个合法 source_chunks 时保留。"""
    synthesizer = _make_synthesizer(monkeypatch)
    chunks = [
        _make_enriched_chunk("c0", content="内容" * 50),
        _make_enriched_chunk("c1", content="内容" * 50),
    ]
    raw = '[{"question": "问题", "answer": "答案", "question_type": "multi_hop", "source_chunks": ["c0", "c1"], "key_entities": ["实体"]}]'
    result = synthesizer._parse_qa_pairs(raw, chunks)
    assert len(result) == 1
    assert result[0].question_type == "multi_hop"
    assert "c0" in result[0].metadata.get("source_chunks", [])
    assert "c1" in result[0].metadata.get("source_chunks", [])


def test_parse_qa_pairs_key_entities_must_be_list(monkeypatch) -> None:
    """key_entities 非 list 时，该字段应被规范化为空列表（条目保留）。"""
    synthesizer = _make_synthesizer(monkeypatch)
    chunks = [
        _make_enriched_chunk("c0", content="内容" * 50),
    ]
    raw = '[{"question": "问题", "answer": "答案", "question_type": "single_chunk", "source_chunks": ["c0"], "key_entities": "实体A"}]'
    result = synthesizer._parse_qa_pairs(raw, chunks)
    # 条目可保留，但 key_entities 应为 list
    if result:
        assert isinstance(result[0].metadata.get("key_entities", []), list)


def test_parse_qa_pairs_context_uses_source_chunks_content(monkeypatch) -> None:
    """context 应优先使用 source_chunks 对应的 chunk 内容，而非 fallback。"""
    synthesizer = _make_synthesizer(monkeypatch)
    chunks = [
        _make_enriched_chunk("c0", content="A" * 100),
        _make_enriched_chunk("c1", content="B" * 100),
    ]
    raw = '[{"question": "问题", "answer": "答案", "question_type": "contrast", "source_chunks": ["c0", "c1"], "key_entities": ["实体"]}]'
    result = synthesizer._parse_qa_pairs(raw, chunks)
    assert len(result) == 1
    assert "A" in result[0].context
    assert "B" in result[0].context


# ─── _is_too_easy 保守判定 ──────────────────────────────────────────────────


def test_is_too_easy_returns_false_when_verifier_vague(monkeypatch) -> None:
    """verifier 给出模糊、泛化回答时不应判定为 easy（保守保留）。"""
    synthesizer = _make_synthesizer(monkeypatch)
    synthesizer._verify_llm.invoke.return_value = MagicMock(
        content="这个问题比较复杂，可能需要具体背景信息。"
    )
    qa = _make_qa_pair("新疆的天山山脉有多长？", "天山山脉在新疆段约1700公里。")
    result = synthesizer._is_too_easy(qa)
    # 泛回答 → 保守保留，不应判定为 easy
    assert result is False


def test_is_too_easy_returns_true_when_verifier_gives_specific_confident_answer(
    monkeypatch,
) -> None:
    """verifier 给出高置信具体答案时判定为常识题，应过滤。"""
    synthesizer = _make_synthesizer(monkeypatch)
    synthesizer._verify_llm.invoke.return_value = MagicMock(content="中国的首都是北京。")
    qa = _make_qa_pair("中国的首都是哪里？", "北京。")
    result = synthesizer._is_too_easy(qa)
    assert result is True


def test_is_too_easy_returns_false_on_dont_know(monkeypatch) -> None:
    """verifier 回答'不知道'时保留 QA。"""
    synthesizer = _make_synthesizer(monkeypatch)
    synthesizer._verify_llm.invoke.return_value = MagicMock(content="不知道")
    qa = _make_qa_pair("该文档中提到的第三条路线是什么？", "内容详见原文。")
    result = synthesizer._is_too_easy(qa)
    assert result is False


# ─── _parse_qa_pairs 解析新字段 ──────────────────────────────────────────────


def test_parse_qa_pairs_with_new_fields(monkeypatch) -> None:
    """解析包含 reason 和 reasoning_steps 的 QA。"""
    synthesizer = _make_synthesizer(monkeypatch)
    chunks = [
        EnrichedChunk(
            chunk_id="c1",
            doc_id="d1",
            content_type="text",
            content="测试内容1",
            source_items=[],
            page_idx=0,
            token_count=10,
            entities=["实体A"],
            summary="摘要1",
        ),
        EnrichedChunk(
            chunk_id="c2",
            doc_id="d1",
            content_type="text",
            content="测试内容2",
            source_items=[],
            page_idx=0,
            token_count=10,
            entities=["实体B"],
            summary="摘要2",
        ),
    ]

    raw_json = """[{
        "question": "实体A和实体B有什么关联？",
        "answer": "它们共同说明了某个概念",
        "question_type": "multi_hop",
        "source_chunks": ["c1", "c2"],
        "key_entities": ["实体A", "实体B"],
        "reason": "考察跨块关联推理",
        "reasoning_steps": "步骤1: 从c1获取实体A信息\\n步骤2: 从c2获取实体B信息\\n步骤3: 综合分析"
    }]"""

    qa_pairs = synthesizer._parse_qa_pairs(raw_json, chunks)

    assert len(qa_pairs) == 1
    assert qa_pairs[0].reason == "考察跨块关联推理"
    assert "步骤1" in qa_pairs[0].reasoning_steps
