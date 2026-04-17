"""文档级 QA 合成器：基于 chunk 关系图生成高难度压测题目。"""

from __future__ import annotations

import json
import logging
import re
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
        "不确定",
        "无从得知",
        "需要更多信息",
        "需要具体",
        "需要特定",
    }
)

# 模糊/泛化回答特征（含这些词的回答不算"确定性回答"）
_VAGUE_PATTERNS = re.compile(
    r"(可能|也许|大概|一般来说|通常情况|具体情况|因情况而异|视情况|"
    r"比较复杂|需要具体|需要更多|取决于|depending on|it depends|"
    r"generally|typically|usually|in general|may vary)",
    re.IGNORECASE,
)

# 合法的 question_type 值
_VALID_QUESTION_TYPES = frozenset({"multi_hop", "contrast", "hypothetical", "single_chunk"})


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
        embedding_model: embedding 模型（预留）
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
        """解析 LLM 返回的 JSON QA 列表，执行严格校验，不合法条目直接丢弃。

        校验规则：
        1. source_chunks 中每个 chunk_id 必须存在于 chunks 中
        2. question_type 必须在 _VALID_QUESTION_TYPES 内
        3. multi_hop 题的 source_chunks 必须 >= 2
        4. key_entities 必须为 list，若不是则规范化为 []
        5. context 优先使用 source_chunks 对应内容（不做 fallback 兜底的截断版）
        """
        doc_id = chunks[0].doc_id if chunks else "unknown"
        valid_chunk_ids = {c.chunk_id for c in chunks}
        chunk_map = {c.chunk_id: c for c in chunks}

        # 提取并解析 JSON（兼容前后有说明文字的情况）
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            logger.warning("_parse_qa_pairs: JSON 数组未找到，raw=%r", raw[:200])
            return []
        try:
            items = json.loads(raw[start:end])
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

            # 校验 question_type
            q_type = str(item.get("question_type", "single_chunk"))
            if q_type not in _VALID_QUESTION_TYPES:
                logger.debug("丢弃非法 question_type=%r：%s", q_type, question[:60])
                continue

            # 校验 source_chunks：所有 id 必须存在
            source_chunks = item.get("source_chunks") or []
            if not isinstance(source_chunks, list):
                source_chunks = []
            valid_sources = [cid for cid in source_chunks if cid in valid_chunk_ids]
            if source_chunks and not valid_sources:
                # source_chunks 中无任何合法 id → 丢弃
                logger.debug("丢弃无效 source_chunks=%r：%s", source_chunks, question[:60])
                continue

            # multi_hop 必须有 >= 2 个合法 source chunks
            if q_type == "multi_hop" and len(valid_sources) < 2:
                logger.debug("丢弃伪多跳（source_chunks 数量=%d）：%s", len(valid_sources), question[:60])
                continue

            # 规范化 key_entities
            raw_entities = item.get("key_entities", [])
            if not isinstance(raw_entities, list):
                key_entities: list[str] = []
            else:
                key_entities = [str(e) for e in raw_entities if str(e).strip()]

            # 构建 context：优先使用 valid_sources 对应内容（保留更多字符以支撑 faithfulness 评分）
            if valid_sources:
                context = " | ".join(
                    chunk_map[cid].content[:800] for cid in valid_sources if cid in chunk_map
                )
            elif chunks:
                context = chunks[0].content[:800]
            else:
                context = ""

            qa_pairs.append(
                QAPair(
                    qa_id=str(uuid.uuid4()),
                    chunk_id=(
                        valid_sources[0]
                        if valid_sources
                        else (chunks[0].chunk_id if chunks else "")
                    ),
                    doc_id=doc_id,
                    question=question,
                    answer=answer,
                    context=context,
                    question_type=q_type,
                    content_type="text",
                    metadata={
                        "source_chunks": valid_sources,
                        "key_entities": key_entities,
                    },
                )
            )
        return qa_pairs

    def _is_too_easy(self, qa: QAPair) -> bool:
        """自一致性验证：保守判定——只过滤明确常识题/百科题。

        过滤条件（同时满足）：
        1. verifier 给出了非"不知道"回答
        2. 回答不含模糊/泛化特征词
        3. 回答长度适中（说明 verifier 给出了具体答案，而非敷衍）

        若 verifier 给出模糊泛化的回答，保守保留 QA（不视为 easy）。
        """
        try:
            prompt = SELF_CONSISTENCY_PROMPT.format(question=qa.question)
            response = self._verify_llm.invoke(prompt).content.strip()
        except Exception as exc:
            logger.warning("自一致性验证失败，保留 QA [%s]：%s", qa.question[:60], exc)
            return False  # 失败时保守保留

        if not response:
            return False

        response_lower = response.lower()

        # 含"不知道"模式 → 保留
        if any(p in response_lower for p in _DONT_KNOW_PATTERNS):
            return False

        # 含模糊/泛化特征词 → 保留（不是确定性回答）
        if _VAGUE_PATTERNS.search(response):
            return False

        # 长度范围 [2, 200]：
        # - 下限 2：排除空响应（空字符串、单标点）
        # - 上限 200：覆盖简洁的百科/常识答案（通常 ≤ 100 字符），同时避免
        #   把含有上下文判断的较长专业答案（> 200 字符）误判为 easy
        is_easy = 2 <= len(response) <= 200
        if is_easy:
            logger.debug("self-consistency: filtering question=%r", qa.question[:80])
        return is_easy
