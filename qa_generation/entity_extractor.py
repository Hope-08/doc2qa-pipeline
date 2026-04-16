"""实体提取器：从 SemanticChunk 中提取命名实体，返回 EnrichedChunk。"""

from __future__ import annotations

import json
import logging

from langchain_openai import ChatOpenAI

from config.prompts import ENTITY_EXTRACTION_PROMPT
from models import EnrichedChunk, SemanticChunk

logger = logging.getLogger(__name__)


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
        """解析 LLM 返回的 JSON 实体列表，失败时返回空列表。"""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
        entities = data.get("entities", [])
        if not isinstance(entities, list):
            return []
        return [str(e).strip() for e in entities if str(e).strip()]
