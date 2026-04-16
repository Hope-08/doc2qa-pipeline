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
