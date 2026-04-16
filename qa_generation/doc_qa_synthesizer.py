"""文档级 QA 合成器：基于 chunk 关系图生成高难度压测题目。"""

from __future__ import annotations

import json
import logging
import uuid

from langchain_openai import ChatOpenAI

from config.prompts import DOC_QA_SYNTHESIS_PROMPT, SELF_CONSISTENCY_PROMPT
from models import EnrichedChunk, QAPair

logger = logging.getLogger(__name__)

_DONT_KNOW_PATTERNS = frozenset(
    {
        "不知道",
        "不清楚",
        "无法回答",
        "没有相关",
        "don't know",
        "do not know",
        "i don't",
        "cannot answer",
        "i cannot",
        "i'm not sure",
        "i am not sure",
        "no information",
    }
)


def calc_doc_quota(total_chars: int, max_qa_per_doc: int) -> int:
    """计算文档级 QA 配额：min(max_qa_per_doc, max(3, total_chars // 800))。"""
    return min(max_qa_per_doc, max(3, total_chars // 800))


def is_short_document(chunks: list[EnrichedChunk]) -> bool:
    """判断是否为短文档：chunk 数 < 2 或总字符 < 800。"""
    if len(chunks) < 2:
        return True
    return sum(len(c.content) for c in chunks) < 800


class DocumentQASynthesizer:
    """文档级 QA 合成器：调用 LLM 合成 QA，并可选执行自一致性验证。

    Args:
        synthesize_model: QA 合成模型
        verifier_model: 自一致性验证模型（不用 context 直接回答）
        embedding_model: embedding 模型（预留，暂未使用）
        self_consistency_check: 是否启用自一致性过滤
    """

    def __init__(
        self,
        synthesize_model: str,
        verifier_model: str,
        embedding_model: str,
        self_consistency_check: bool,
    ) -> None:
        self._synth_llm = ChatOpenAI(model=synthesize_model, temperature=0.3)
        self._verify_llm = ChatOpenAI(model=verifier_model, temperature=0.0)
        self._embedding_model = embedding_model
        self._self_consistency_check = self_consistency_check

    def synthesize(
        self,
        chunks: list[EnrichedChunk],
        relations: list[dict],
        quota: int,
    ) -> list[QAPair]:
        """合成文档级 QA，可选执行自一致性过滤。"""
        prompt = DOC_QA_SYNTHESIS_PROMPT.format(
            quota=quota,
            multi_hop_count=max(1, quota // 3),
            chunk_summaries=self._format_chunk_summaries(chunks),
            relations=self._format_relations(relations),
        )
        raw = self._synth_llm.invoke(prompt).content
        qa_pairs = self._parse_qa_pairs(raw, chunks)

        if not self._self_consistency_check or is_short_document(chunks):
            return qa_pairs
        return [qa for qa in qa_pairs if not self._is_too_easy(qa)]

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _format_chunk_summaries(chunks: list[EnrichedChunk]) -> str:
        lines = []
        for c in chunks:
            entities_str = "、".join(c.entities) if c.entities else "无"
            lines.append(f"[{c.chunk_id}] 实体：{entities_str}\n内容摘要：{c.content[:200]}")
        return "\n\n".join(lines)

    @staticmethod
    def _format_relations(relations: list[dict]) -> str:
        if not relations:
            return "（无关系边）"
        lines = []
        for rel in relations:
            signal = rel.get("signal", [])
            lines.append(
                f"{rel['left_chunk_id']} —[{rel['relation_type']}]→ "
                f"{rel['right_chunk_id']}  信号：{signal}"
            )
        return "\n".join(lines)

    def _parse_qa_pairs(self, raw: str, chunks: list[EnrichedChunk]) -> list[QAPair]:
        """解析 LLM 返回的 JSON QA 列表，失败时返回空列表。"""
        doc_id = chunks[0].doc_id if chunks else "unknown"
        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                return []
        except (json.JSONDecodeError, ValueError):
            logger.warning("_parse_qa_pairs: JSON decode failed, raw=%r", raw[:200])
            return []

        qa_pairs = []
        for item in items:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not question or not answer:
                continue
            source_chunks = item.get("source_chunks") or []
            context = " | ".join(
                c.content[:200] for c in chunks if c.chunk_id in source_chunks
            ) or (chunks[0].content[:200] if chunks else "")
            qa_pairs.append(
                QAPair(
                    qa_id=str(uuid.uuid4()),
                    chunk_id=(
                        source_chunks[0]
                        if source_chunks
                        else (chunks[0].chunk_id if chunks else "")
                    ),
                    doc_id=doc_id,
                    question=question,
                    answer=answer,
                    context=context,
                    question_type=str(item.get("question_type", "single_chunk")),
                    content_type="text",
                    metadata={
                        "source_chunks": source_chunks,
                        "key_entities": item.get("key_entities", []),
                    },
                )
            )
        return qa_pairs

    def _is_too_easy(self, qa: QAPair) -> bool:
        """自一致性验证：让 verifier 不看 context 直接回答，若答对则为常识题，过滤。"""
        try:
            prompt = SELF_CONSISTENCY_PROMPT.format(question=qa.question)
            response = self._verify_llm.invoke(prompt).content.strip()
        except Exception as exc:
            logger.warning("自一致性验证失败，保留 QA [%s]：%s", qa.question[:60], exc)
            return False  # 失败时保守保留
        # 检测中英文"不知道"模式；verifier 能给出非"不知道"回答则认为是常识题
        response_lower = response.lower()
        is_easy = bool(response) and not any(p in response_lower for p in _DONT_KNOW_PATTERNS)
        if is_easy:
            logger.debug("self-consistency: filtering question=%r", qa.question[:80])
        return is_easy
