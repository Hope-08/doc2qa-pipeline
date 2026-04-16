"""多类型 QA 生成器：根据 chunk 类型分发到对应 Prompt，统一调用 LLM。"""

from __future__ import annotations

import json
import re
import uuid
from typing import Optional

from models import QAPair, SemanticChunk
from config.prompts import PROMPT_BY_TYPE
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

# chunk 内容最短有效长度（字符）：太短的 chunk 没有足够上下文生成有意义的 QA
_MIN_CHUNK_CHARS = 50
# 若 chunk 中"日期格式字符串"占比超过此比例，视为纯日期列表，跳过
_MAX_DATE_RATIO = 0.6
_DATE_PATTERN = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")


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

        Args:
            chunks: 语义分块列表。

        Returns:
            所有成功生成的 QAPair（未评分，quality_score 为 None）。
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
        1. 内容过短（< _MIN_CHUNK_CHARS 字符）：没有足够上下文
        2. 纯日期列表：日期格式字符串占比超过 _MAX_DATE_RATIO
        """
        content = chunk.content.strip()

        if len(content) < _MIN_CHUNK_CHARS:
            return f"内容过短（{len(content)} 字符）"

        # 计算日期格式字符串占总内容的比例
        date_matches = _DATE_PATTERN.findall(content)
        if date_matches:
            date_chars = sum(len(m) for m in date_matches)
            if date_chars / max(len(content), 1) > _MAX_DATE_RATIO:
                return f"纯日期列表（日期字符占比 {date_chars/len(content):.0%}）"

        return None

    def _generate_for_chunk(self, chunk: SemanticChunk) -> list[QAPair]:
        """为单个 chunk 生成 QA。"""
        prompt = PROMPT_BY_TYPE[chunk.content_type].format(context=chunk.content)
        result = self._llm.complete(prompt)
        raw_list = self._parse_json(result.content, chunk.chunk_id)

        return [
            QAPair(
                qa_id=str(uuid.uuid4()),
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                question=qa["question"].strip(),
                answer=qa["answer"].strip(),
                context=chunk.content,
                question_type=qa.get("type", "unknown"),
                content_type=chunk.content_type,
                metadata={"page_idx": chunk.page_idx},
            )
            for qa in raw_list
            if qa.get("question") and qa.get("answer")
        ]

    @staticmethod
    def _parse_json(content: str, chunk_id: str) -> list[dict]:
        """从 LLM 响应中提取 JSON 数组，容错处理多余文字。"""
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            logger.warning("chunk [%s] LLM 响应中未找到 JSON 数组", chunk_id)
            return []
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError as exc:
            logger.warning("chunk [%s] JSON 解析失败：%s", chunk_id, exc)
            return []
