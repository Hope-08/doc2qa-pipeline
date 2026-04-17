"""全局配置，基于 pydantic-settings，支持 .env 文件覆盖。"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """所有模块共享的全局配置。

    优先级：环境变量 > .env 文件 > 字段默认值。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM 配置
    # ------------------------------------------------------------------
    llm_model: str = Field(
        default="azure-openai/gpt-5.4",
        description="模型名称，如 gpt-4o、qwen2.5-72b-instruct",
    )
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0)

    # ------------------------------------------------------------------
    # 文档解析（本地多引擎）
    # ------------------------------------------------------------------
    parsing_backend: str = Field(
        default="auto",
        description="解析引擎：auto | docling | pymupdf | unstructured",
    )
    ocr_lang: str = Field(
        default="chi_sim+eng",
        description="Tesseract OCR 语言包（仅扫描 PDF 模式使用）",
    )

    # ------------------------------------------------------------------
    # 语义分块
    # ------------------------------------------------------------------
    chunk_size: int = Field(
        default=800, description="单个 chunk 的字符数上限（中文约 1 字 = 1~1.5 token）"
    )
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="相邻句子相似度低于此值时视为语义边界",
    )
    embedding_model: str = Field(
        default="azure-openai/text-embedding-3-small",
        description="OpenAI embedding 模型，用于语义分块、相关性评分和去重",
    )

    # ------------------------------------------------------------------
    # 质量评估
    # ------------------------------------------------------------------
    nli_model: str = Field(
        default="azure-openai/gpt-5.4",
        description="用于 faithfulness 评分的 LLM 模型（调用 OpenAI API，无需本地下载）",
    )
    faithfulness_min: float = Field(default=0.60, ge=0.0, le=1.0)
    relevance_min: float = Field(default=0.50, ge=0.0, le=1.0)
    answer_min_length: int = Field(default=15, ge=1)
    composite_pass: float = Field(default=0.55, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # QA 生成策略（两阶段高质量方案）
    # ------------------------------------------------------------------
    qa_strategy: str = Field(
        default="advanced",
        description="QA 生成策略：simple（旧方案，3条/chunk）| advanced（两阶段，≤N条/文档）",
    )
    max_qa_per_doc: int = Field(
        default=10,
        gt=0,
        description="advanced 策略下单文档 QA 上限",
    )
    headline_min_count: int = Field(
        default=2,
        gt=0,
        description="触发 HeadlineSplitter 按标题切节的最少标题数",
    )
    self_consistency_check: bool = Field(
        default=True,
        description="是否启用自一致性验证（用 entity_extractor_model 回答问题，过滤常识题）",
    )
    entity_extractor_model: str = Field(
        default="azure-openai/gpt-4o",
        description="实体提取 + 自一致性验证使用的模型（litellm 格式）",
    )
    synthesize_model: str = Field(
        default="azure-openai/gpt-5.4",
        description="QA 合成使用的模型（litellm 格式），追求最高质量",
    )

    # ------------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------------
    output_dir: str = Field(default="./output", description="数据集输出目录")

    # ------------------------------------------------------------------
    # API 服务
    # ------------------------------------------------------------------
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS 允许来源列表，生产环境应设置为实际域名",
    )


# 全局单例，供各模块直接 import 使用
settings = Settings()
