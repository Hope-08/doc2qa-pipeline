"""多类型 QA 生成器：根据 chunk 类型分发到对应 Prompt，统一调用 LLM。"""

from __future__ import annotations

import json
import re
import uuid
from typing import Optional

from config.prompts import PROMPT_BY_TYPE
from models import QAPair, SemanticChunk
from utils.llm_client import LLMClient
from utils.logger import get_logger
from utils.text_filters import META_TEXT_RE as _META_TEXT_PATTERNS

logger = get_logger(__name__)

# chunk 内容最短有效长度（字符）
_MIN_CHUNK_CHARS = 50
# 日期格式字符串占比超过此比例 → 视为纯日期列表
_MAX_DATE_RATIO = 0.6
_DATE_PATTERN = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")

# 目录碎片检测：大量 "章/节/……数字" 行（用 [^\n] 限制单行匹配，避免回溯放大）
_TOC_LINE_PATTERN = re.compile(r"(第[一二三四五六七八九十百\d]+[章节篇]|[一二三四五六七八九十\d]+[、.．])[^\n]{0,30}[…\.]{3,}\s*\d+")
_TOC_LINE_MIN = 3  # 超过此行数判定为目录

# 联系方式碎片检测
_CONTACT_PATTERN = re.compile(
    r"(电话|手机|传真|邮箱|E-?mail|Tel|Fax|微信|QQ)[：:：]?\s*[\d\w@.\-+（）()]+",
    re.IGNORECASE,
)
_CONTACT_MIN_RATIO = 0.5  # 联系方式行占总行数比例超过此值 → 联系方式碎片

# 允许的 question_type 值（与 PROMPT_BY_TYPE 生成内容保持一致）
_ALLOWED_TYPES = frozenset({"definition", "reasoning", "procedure", "fact", "comparison", "summary", "description", "context", "unknown"})

# Q-A 重叠比率（question token 在 answer 中的占比超过此阈值 → 复述题）
_QA_OVERLAP_THRESHOLD = 0.7


class QAGenerator:
    """多类型 QA 生成器。

    根据 SemanticChunk 的 content_type 选择对应 Prompt，
    通过 LLMClient 调用 LLM，解析返回的 JSON 生成 QAPair 列表。

    Args:
        llm_client: LLMClient 实例，由调用方注入（便于测试替换）。
                    若不传则用默认配置自动创建。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm = llm_client or LLMClient()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def generate_batch(self, chunks: list[SemanticChunk]) -> list[QAPair]:
        """批量为 chunks 生成 QA 对。

        跳过不支持的 content_type，单个 chunk 失败时记录日志并继续。
        """
        all_qa: list[QAPair] = []
        skipped = 0

        for chunk in chunks:
            if chunk.content_type not in PROMPT_BY_TYPE:
                logger.debug("跳过不支持的类型 [%s]", chunk.content_type)
                continue

            skip_reason = self._low_quality_reason(chunk)
            if skip_reason:
                logger.debug("跳过低质量 chunk [%s]：%s", chunk.chunk_id, skip_reason)
                skipped += 1
                continue

            try:
                qa_list = self._generate_for_chunk(chunk)
                all_qa.extend(qa_list)
                logger.debug(
                    "chunk [%s] → %d 个 QA 对", chunk.chunk_id, len(qa_list)
                )
            except Exception as exc:
                logger.warning("chunk [%s] 生成失败，跳过：%s", chunk.chunk_id, exc)

        logger.info(
            "批量生成完成：%d chunks（跳过 %d 低质量）→ %d QA 对",
            len(chunks), skipped, len(all_qa),
        )
        return all_qa

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _low_quality_reason(chunk: SemanticChunk) -> str | None:
        """检查 chunk 是否为低质量内容，返回跳过原因；None 表示可正常生成。

        过滤规则：
        1. 内容过短（< _MIN_CHUNK_CHARS 字符）
        2. 纯日期列表（日期格式字符串占比超过 _MAX_DATE_RATIO）
        3. 目录碎片（大量"第N章……M"格式行）
        4. 联系方式碎片（电话/邮箱行占比过高）
        """
        content = chunk.content.strip()

        if len(content) < _MIN_CHUNK_CHARS:
            return f"内容过短（{len(content)} 字符）"

        # 纯日期列表检测
        date_matches = _DATE_PATTERN.findall(content)
        if date_matches:
            date_chars = sum(len(m) for m in date_matches)
            if date_chars / max(len(content), 1) > _MAX_DATE_RATIO:
                return f"纯日期列表（日期字符占比 {date_chars/len(content):.0%}）"

        # 目录碎片检测
        toc_lines = _TOC_LINE_PATTERN.findall(content)
        if len(toc_lines) >= _TOC_LINE_MIN:
            return f"目录碎片（检测到 {len(toc_lines)} 条目录行）"

        # 联系方式碎片检测
        lines = [line for line in content.splitlines() if line.strip()]
        if lines:
            contact_lines = sum(1 for line in lines if _CONTACT_PATTERN.search(line))
            if contact_lines / len(lines) >= _CONTACT_MIN_RATIO and contact_lines >= 2:
                return f"联系方式碎片（联系方式行占比 {contact_lines/len(lines):.0%}）"

        return None

    def _generate_for_chunk(self, chunk: SemanticChunk) -> list[QAPair]:
        """为单个 chunk 生成 QA，并执行后处理过滤。"""
        prompt = PROMPT_BY_TYPE[chunk.content_type].format(context=chunk.content)
        result = self._llm.complete(prompt)
        raw_list = self._parse_json(result.content, chunk.chunk_id)

        qa_pairs: list[QAPair] = []
        for qa in raw_list:
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            if not question or not answer:
                continue
            # 元文本指代词过滤
            if _META_TEXT_PATTERNS.search(question) or _META_TEXT_PATTERNS.search(answer):
                logger.debug(
                    "chunk [%s] 过滤元文本问答：%s", chunk.chunk_id, question[:60]
                )
                continue
            # Q-A 高重叠复述题过滤
            if _is_qa_overlap(question, answer):
                logger.debug(
                    "chunk [%s] 过滤复述题：%s", chunk.chunk_id, question[:60]
                )
                continue

            qa_pairs.append(
                QAPair(
                    qa_id=str(uuid.uuid4()),
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    question=question,
                    answer=answer,
                    context=chunk.content,
                    question_type=qa.get("type", "unknown"),
                    content_type=chunk.content_type,
                    metadata={"page_idx": chunk.page_idx},
                )
            )
        return qa_pairs

    @staticmethod
    def _parse_json(content: str, chunk_id: str) -> list[dict]:
        """从 LLM 响应中提取 JSON 数组，校验必须为 list[dict]。"""
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            logger.warning("chunk [%s] LLM 响应中未找到 JSON 数组", chunk_id)
            return []
        try:
            parsed = json.loads(content[start:end])
        except json.JSONDecodeError as exc:
            logger.warning("chunk [%s] JSON 解析失败：%s", chunk_id, exc)
            return []

        if not isinstance(parsed, list):
            logger.warning("chunk [%s] LLM 返回非列表 JSON，跳过", chunk_id)
            return []

        result = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            # question / answer 必须存在且非空
            if not item.get("question") or not item.get("answer"):
                continue
            result.append(item)
        return result


# ------------------------------------------------------------------
# 模块级工具函数
# ------------------------------------------------------------------

def _is_qa_overlap(question: str, answer: str) -> bool:
    """判断答案是否只是问题的复述（高 token 重叠率）。

    将问题和答案按字切分，计算问题 tokens 在答案中出现的比例。
    中文不需要分词，字级别已足够捕获复述题。
    """
    q_chars = set(c for c in question if c.strip() and c not in "？?。！!，,、：:")
    if not q_chars:
        return False
    a_chars = set(c for c in answer if c.strip())
    overlap = q_chars & a_chars
    overlap_ratio = len(overlap) / len(q_chars)
    # 同时检查答案是否比问题包含足够多的新信息
    new_in_answer = a_chars - q_chars
    if overlap_ratio >= _QA_OVERLAP_THRESHOLD and len(new_in_answer) < 5:
        return True
    return False
