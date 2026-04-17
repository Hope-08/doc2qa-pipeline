"""
核心数据模型定义。

所有模块共享的数据类，避免循环导入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypeAlias

# ---------------------------------------------------------------------------
# 文档解析层
# ---------------------------------------------------------------------------


@dataclass
class ContentItem:
    """解析器输出的单个内容项。

    Attributes:
        item_id:  内容项唯一 ID，格式 {doc_id}_item_{idx}
        doc_id:   所属文档 ID
        type:     内容类型，取值 text | table | image | equation
        content:  实际内容（文本 / HTML 表格 / LaTeX 公式 / 图片 caption）
        page_idx: 所在页码（从 0 开始）
        bbox:     版面边界框坐标 (x0, y0, x1, y1)，可选
        metadata: 额外元数据
    """

    item_id: str
    doc_id: str
    type: str
    content: str
    page_idx: int
    bbox: Optional[tuple[float, float, float, float]] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """文档解析结果。

    Attributes:
        doc_id:       文档唯一 ID（通常为文件名 stem）
        source_path:  原始文件路径
        markdown:     完整 Markdown 内容
        content_list: 结构化内容列表
        metadata:     文档级元数据（页数、语言、后端等）
    """

    doc_id: str
    source_path: str
    markdown: str
    content_list: list[ContentItem]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 分块层
# ---------------------------------------------------------------------------


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list[JSONScalar]
ChunkMetadata: TypeAlias = dict[str, JSONValue]


@dataclass
class SemanticChunk:
    """语义分块结果。

    Attributes:
        chunk_id:     唯一 ID，格式 {doc_id}_chunk_{idx}
        doc_id:       所属文档 ID
        content_type: 内容类型，取值 text | table | image_caption
        content:      分块后的文本内容
        source_items: 来源 ContentItem 的 ID 列表
        page_idx:     所在页码
        token_count:  估算 token 数
    """

    chunk_id: str
    doc_id: str
    content_type: str
    content: str
    source_items: list[str]
    page_idx: int
    token_count: int


@dataclass
class EnrichedChunk:
    """高级 QA 生成使用的富化分块。

    Attributes:
        chunk_id:     唯一 ID，格式 {doc_id}_chunk_{idx}
        doc_id:       所属文档 ID
        content_type: 内容类型，取值 text | table | image_caption
        content:      分块后的文本内容
        source_items: 来源 ContentItem 的 ID 列表
        page_idx:     所在页码
        token_count:  估算 token 数
        entities:     从 chunk 中抽取的关键实体名称列表；仅保存字符串，便于
                      直接进入后续 prompt 或序列化结果
        summary:      对 chunk 关键信息的简短摘要；无摘要时为 None
        metadata:     面向高级 QA 生成的补充元数据，必须保持 JSON 可序列化；
                      当前约定支持标量值以及标量列表，例如 section、confidence、
                      keywords 等轻量字段
    """

    chunk_id: str
    doc_id: str
    content_type: str
    content: str
    source_items: list[str]
    page_idx: int
    token_count: int
    entities: list[str] = field(default_factory=list)
    summary: Optional[str] = None
    metadata: ChunkMetadata = field(default_factory=dict)


# ---------------------------------------------------------------------------
# QA 生成层
# ---------------------------------------------------------------------------


@dataclass
class QAPair:
    """QA 问答对。

    质量评分字段（faithfulness_score 等）初始为 None，
    由 qa_filtering 模块填充后再做过滤。
    """

    qa_id: str
    chunk_id: str
    doc_id: str

    question: str
    answer: str
    context: str  # 原始 chunk 内容，作为 RAG 上下文

    question_type: str  # definition / reasoning / comparison /
    #                                  procedure / fact / summary / description
    content_type: str  # text / table / image_caption

    reason: str = ""  # LLM 出题理由
    reasoning_steps: str = ""  # 推理步骤 / 多跳推理链（多行文本）

    # 质量评分（qa_filtering 填充）
    quality_score: Optional[float] = None
    faithfulness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    completeness_score: Optional[float] = None

    is_passed: bool = False
    fail_reasons: list[str] = field(default_factory=list)

    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 质量控制层
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityThresholds:
    """质量过滤阈值配置（不可变）。"""

    faithfulness_min: float = 0.60  # NLI entailment 最低分
    relevance_min: float = 0.50  # Q-A 余弦相似度最低分
    answer_min_length: int = 15  # 答案最少字符数（中文）
    composite_pass: float = 0.55  # 综合加权分通过线
