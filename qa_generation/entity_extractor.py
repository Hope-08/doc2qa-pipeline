"""实体提取器：从 SemanticChunk 中提取命名实体，返回 EnrichedChunk。"""

from __future__ import annotations

import json
import logging
import re

from langchain_openai import ChatOpenAI

from config.prompts import ENTITY_EXTRACTION_PROMPT
from models import EnrichedChunk, SemanticChunk

logger = logging.getLogger(__name__)

# 泛词黑名单（在实体中出现时直接过滤）
_GENERIC_TERMS = frozenset({
    # 中文泛词
    "文本", "内容", "信息", "数据", "方法", "情况", "问题", "文章", "文件", "材料",
    "结果", "条件", "过程", "方式", "系统", "功能", "特点", "特征", "要求",
    "说明", "介绍", "描述", "分析", "处理", "管理", "操作", "使用", "服务", "工作",
    "部分", "形式", "类型", "领域", "方面", "情形", "规则", "标准",
    # 时间词（Prompt 已明确要求过滤，黑名单补齐）
    "年份", "月份", "季度", "年度", "日期", "时间",
    # 英文泛词
    "text", "content", "data", "information", "method", "system", "type",
})

# 纯数字模式
_PURE_NUMBER_PATTERN = re.compile(r"^\d+\.?\d*$")
# 纯标点/符号模式
_PURE_PUNCT_PATTERN = re.compile(r"^[\s\W_]+$")


class EntityExtractor:
    """调用 LLM 从文本块中提取命名实体。

    Args:
        model: LiteLLM 格式的模型名，如 "azure-openai/gpt-4o"
        temperature: 生成温度，实体提取建议 0.0
    """

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self._llm = ChatOpenAI(model=model, temperature=temperature)

    def extract(self, chunk: SemanticChunk) -> EnrichedChunk:
        """提取 chunk 中的命名实体，返回富化后的 EnrichedChunk。

        LLM 调用失败时实体列表为空，不抛出异常。
        """
        try:
            prompt = ENTITY_EXTRACTION_PROMPT.format(chunk_text=chunk.content)
            response = self._llm.invoke(prompt)
            entities = self._parse_entities(response.content)
        except Exception as exc:
            logger.warning("实体提取失败 [%s]：%s", chunk.chunk_id, exc)
            entities = []
        return EnrichedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            content_type=chunk.content_type,
            content=chunk.content,
            source_items=list(chunk.source_items),
            page_idx=chunk.page_idx,
            token_count=chunk.token_count,
            entities=entities,
        )

    @staticmethod
    def _parse_entities(raw: str) -> list[str]:
        """解析 LLM 返回的 JSON 实体列表，并清洗去噪。

        清洗规则：
        1. 去重（保持首次出现顺序）
        2. 过滤过短实体（单字符英文字母等）
        3. 过滤纯数字
        4. 过滤纯标点/符号
        5. 过滤泛词黑名单
        """
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
        entities = data.get("entities", [])
        if not isinstance(entities, list):
            return []

        seen: set[str] = set()
        result: list[str] = []
        for e in entities:
            s = str(e).strip()
            if not s:
                continue
            # 纯标点/符号过滤
            if _PURE_PUNCT_PATTERN.match(s):
                continue
            # 纯数字过滤
            if _PURE_NUMBER_PATTERN.match(s):
                continue
            # 过短过滤：单字符英文字母（或类似无意义短词）
            if len(s) <= 1 and s.isascii():
                continue
            # 泛词过滤
            if s in _GENERIC_TERMS:
                continue
            # 去重（大小写无关）
            lower = s.lower()
            if lower in seen:
                continue
            seen.add(lower)
            result.append(s)
        return result
