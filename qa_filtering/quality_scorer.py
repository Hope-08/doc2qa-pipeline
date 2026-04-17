"""质量评分器：LLM 幻觉检测 + OpenAI embedding 相关性评分 + 规则完整性评分。"""

from __future__ import annotations

import dataclasses

import numpy as np

from models import QAPair, QualityThresholds
from qa_filtering.hallucination_detector import _run_faithfulness_batch
from utils.logger import get_logger
from utils.text_filters import META_TEXT_RE as _META_TEXT_RE

logger = get_logger(__name__)

# 评分权重
_WEIGHTS = {"faithfulness": 0.50, "relevance": 0.30, "completeness": 0.20}
# faithfulness 评分每批条目数
_FAITHFULNESS_BATCH_SIZE = 5


class QualityScorer:
    """多维度 QA 质量评分器。

    评分维度：
    - faithfulness（忠实度）：调用 LLM（OpenAI API）判断 answer 是否由 context 支撑
    - relevance（相关性）：OpenAI Embeddings，Q-A 余弦相似度
    - completeness（完整性）：规则，根据答案长度和结构

    Args:
        nli_model:       用于 faithfulness 评分的 LLM 模型名，如 azure-openai/gpt-5.4。
        embedding_model: OpenAI embedding 模型名，用于 relevance 计算。
        thresholds:      质量过滤阈值配置，默认使用 QualityThresholds()。
        batch_size:      embedding 批量推理大小。
    """

    def __init__(
        self,
        nli_model: str = "azure-openai/gpt-5.4",
        embedding_model: str = "azure-openai/text-embedding-3-small",
        thresholds: QualityThresholds | None = None,
        batch_size: int = 32,
    ) -> None:
        self.nli_model = nli_model
        self.embedding_model = embedding_model
        self.thresholds = thresholds or QualityThresholds()
        self.batch_size = batch_size

        self._llm: object = None       # 延迟加载
        self._embedder: object = None  # 延迟加载

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def score_batch(self, qa_pairs: list[QAPair]) -> list[QAPair]:
        """批量评分，返回携带评分字段的新 QAPair 列表（不修改原对象）。

        Args:
            qa_pairs: 待评分的 QAPair 列表。

        Returns:
            新的 QAPair 列表（每个对象均为原对象的副本，带有填充好的评分字段）。
        """
        if not qa_pairs:
            return []

        logger.info("开始质量评分：%d 个 QA 对", len(qa_pairs))

        faithfulness = self._score_faithfulness(qa_pairs)
        relevance = self._score_relevance(qa_pairs)
        completeness = [self._score_completeness(qa.answer) for qa in qa_pairs]

        thr = self.thresholds
        scored: list[QAPair] = []
        passed = 0

        for idx, qa in enumerate(qa_pairs):
            fs, rs, cs = faithfulness[idx], relevance[idx], completeness[idx]
            composite = (
                fs * _WEIGHTS["faithfulness"]
                + rs * _WEIGHTS["relevance"]
                + cs * _WEIGHTS["completeness"]
            )

            fail_reasons: list[str] = []
            if fs < thr.faithfulness_min:
                fail_reasons.append(f"幻觉风险 faithfulness={fs:.2f}")
            if rs < thr.relevance_min:
                fail_reasons.append(f"相关性不足 relevance={rs:.2f}")
            if len(qa.answer) < thr.answer_min_length:
                fail_reasons.append(f"答案过短 {len(qa.answer)} 字符")
            if composite < thr.composite_pass:
                fail_reasons.append(f"综合分不足 {composite:.2f}")

            # ── 规则层惩罚 ──────────────────────────────────────────────
            rule_reasons = _apply_rule_penalties(qa)
            fail_reasons.extend(rule_reasons)

            is_passed = len(fail_reasons) == 0
            if is_passed:
                passed += 1

            scored.append(dataclasses.replace(
                qa,
                faithfulness_score=round(fs, 4),
                relevance_score=round(rs, 4),
                completeness_score=round(cs, 4),
                quality_score=round(composite, 4),
                fail_reasons=fail_reasons,
                is_passed=is_passed,
            ))

        logger.info(
            "评分完成：通过 %d/%d（%.1f%%）",
            passed,
            len(qa_pairs),
            passed / len(qa_pairs) * 100,
        )
        return scored

    # ------------------------------------------------------------------
    # 各维度评分
    # ------------------------------------------------------------------

    def _score_faithfulness(self, qa_pairs: list[QAPair]) -> list[float]:
        """调用 LLM 批量判断 answer 是否由 context 支撑，返回 0~1 分数列表。"""
        try:
            llm = self._get_llm()
        except Exception:
            logger.warning("LLM 不可用，faithfulness 默认 0.5")
            return [0.5] * len(qa_pairs)

        all_scores: list[float] = []
        for i in range(0, len(qa_pairs), _FAITHFULNESS_BATCH_SIZE):
            batch = qa_pairs[i : i + _FAITHFULNESS_BATCH_SIZE]
            all_scores.extend(_run_faithfulness_batch(llm, batch))
        return all_scores

    def _score_relevance(self, qa_pairs: list[QAPair]) -> list[float]:
        """Q-A OpenAI Embedding 余弦相似度。

        将 questions 和 answers 拼接后一次性发出，仅需一次 API 调用。
        """
        try:
            embedder = self._get_embedder()
        except Exception:
            logger.warning("embedding 模型不可用，relevance 默认 0.5")
            return [0.5] * len(qa_pairs)

        n = len(qa_pairs)
        questions = [qa.question for qa in qa_pairs]
        answers = [qa.answer for qa in qa_pairs]

        # 一次 API 调用同时获取 Q 和 A 的 embedding
        all_embs = np.array(embedder.embed_documents(questions + answers))
        q_embs, a_embs = all_embs[:n], all_embs[n:]

        similarities: list[float] = []
        for q, a in zip(q_embs, a_embs):
            norm = np.linalg.norm(q) * np.linalg.norm(a) + 1e-8
            similarities.append(float(np.dot(q, a) / norm))
        return similarities

    @staticmethod
    def _score_completeness(answer: str) -> float:
        """基于答案长度和句号的规则评分。"""
        length = len(answer.strip())
        if length < 15:
            length_score = 0.1
        elif length < 50:
            length_score = 0.6
        elif length < 200:
            length_score = 0.9
        else:
            length_score = 1.0

        has_period = "。" in answer or "." in answer
        structure_score = 1.0 if has_period else 0.5

        return (length_score + structure_score) / 2

    # ------------------------------------------------------------------
    # 延迟加载
    # ------------------------------------------------------------------

    def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            logger.info("初始化 faithfulness LLM：%s", self.nli_model)
            self._llm = ChatOpenAI(model=self.nli_model, temperature=0)
        return self._llm

    def _get_embedder(self):
        if self._embedder is None:
            from langchain_openai import OpenAIEmbeddings
            logger.info("初始化 OpenAI embedding 模型：%s", self.embedding_model)
            self._embedder = OpenAIEmbeddings(model=self.embedding_model)
        return self._embedder


# ---------------------------------------------------------------------------
# 规则层惩罚（模块级）
# ---------------------------------------------------------------------------


def _apply_rule_penalties(qa: QAPair) -> list[str]:
    """对 QA 执行轻量规则检测，返回 fail_reasons 追加列表。

    检测项：
    1. 元文本问题：问题或答案含元文本指代词
    2. 伪多跳题：question_type 为 multi_hop 但 source_chunks 只有一个
    """
    reasons: list[str] = []

    # 元文本检测
    if _META_TEXT_RE.search(qa.question) or _META_TEXT_RE.search(qa.answer):
        reasons.append("元文本指代（问题或答案含'文中'/'根据材料'等指代词）")

    # 伪多跳检测
    if qa.question_type == "multi_hop":
        source_chunks = qa.metadata.get("source_chunks", []) if qa.metadata else []
        if len(source_chunks) < 2:
            reasons.append(f"伪多跳题：multi_hop 但 source_chunks 数量={len(source_chunks)}")

    return reasons
