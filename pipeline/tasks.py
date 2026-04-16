"""Prefect Task 定义：每个 Task 对应 Pipeline 的一个处理阶段。

所有 Task 均配置：
- retries：失败自动重试（LLM 限流场景）
- cache_key_fn：相同输入直接复用缓存，支持断点续跑
- cache_expiration：缓存有效期
"""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import timedelta
from typing import Any

from prefect import task
from prefect.tasks import task_input_hash

# ---------------------------------------------------------------------------
# 延迟导入（monkeypatch 需要在模块命名空间可见）
# ---------------------------------------------------------------------------

try:
    from qa_generation.entity_extractor import EntityExtractor
except ImportError:  # pragma: no cover
    EntityExtractor = None  # type: ignore[assignment,misc]

try:
    from qa_generation.doc_qa_synthesizer import DocumentQASynthesizer
    from qa_generation.doc_relation_index import build_relation_index
except ImportError:  # pragma: no cover
    DocumentQASynthesizer = None  # type: ignore[assignment,misc]
    build_relation_index = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 模块级常量
# ---------------------------------------------------------------------------

_VALID_QA_STRATEGIES: frozenset[str] = frozenset({"simple", "advanced"})

# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------


def _serialize(obj: Any) -> Any:
    """将 dataclass 序列化为 dict（Prefect 需要可 JSON 化的返回值）。"""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    return obj


def _deserialize_content_item(d: dict) -> Any:
    """从 dict 还原 ContentItem，将 bbox list 转回 tuple。"""
    from models import ContentItem

    if d.get("bbox") is not None:
        d = {**d, "bbox": tuple(d["bbox"])}
    return ContentItem(**d)


def _file_content_hash(context: Any, arguments: dict) -> str:
    """自定义 parse_task 缓存 key：基于文件内容 SHA-256，而非文件路径字符串。

    使用文件内容哈希的优势：
    - 文件移动/重命名后仍能命中缓存
    - 文件内容变更后自动失效（路径相同但内容不同的文件不会误用旧缓存）
    """
    file_path = arguments.get("file_path", "")
    try:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for _chunk in iter(lambda: f.read(65536), b""):
                h.update(_chunk)
        content_digest = h.hexdigest()
    except OSError:
        content_digest = file_path  # 文件不可读时退化为路径字符串

    other = f"{arguments.get('parsing_backend', '')},{arguments.get('ocr_lang', '')}"
    return f"parse-document_{content_digest}_{other}"


# ---------------------------------------------------------------------------
# Task 定义
# ---------------------------------------------------------------------------


@task(
    name="parse-document",
    description="使用本地多引擎解析文档，提取结构化内容",
    retries=2,
    retry_delay_seconds=10,
    cache_key_fn=_file_content_hash,
    cache_expiration=timedelta(days=7),
)
def parse_task(file_path: str, parsing_backend: str, ocr_lang: str) -> dict:
    """文档解析 Task（结果缓存 7 天）。"""
    from parsing.document_parser import DocumentParser

    parser = DocumentParser(backend=parsing_backend, ocr_lang=ocr_lang)
    doc = parser.parse(file_path)
    return _serialize(doc)


@task(
    name="normalize-content",
    description="清洗解析结果，去除噪声",
    retries=1,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=7),
)
def normalize_task(parsed_dict: dict) -> dict:
    """内容标准化 Task。"""
    from parsing.content_normalizer import ContentNormalizer

    items = [_deserialize_content_item(item) for item in parsed_dict["content_list"]]
    normalizer = ContentNormalizer()
    clean_items = normalizer.normalize_items(items)

    return {**parsed_dict, "content_list": [_serialize(i) for i in clean_items]}


@task(
    name="chunk-document",
    description="语义分块",
    retries=1,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=7),
)
def chunk_task(
    parsed_dict: dict, chunk_size: int, similarity_threshold: float, embedding_model: str
) -> list[dict]:
    """语义分块 Task（结果缓存 7 天）。"""
    from chunking.type_aware_chunker import TypeAwareChunker
    from models import ParsedDocument

    items = [_deserialize_content_item(item) for item in parsed_dict["content_list"]]
    doc = ParsedDocument(
        doc_id=parsed_dict["doc_id"],
        source_path=parsed_dict["source_path"],
        markdown=parsed_dict.get("markdown", ""),
        content_list=items,
        metadata=parsed_dict.get("metadata", {}),
    )
    chunker = TypeAwareChunker(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        similarity_threshold=similarity_threshold,
    )
    chunks = chunker.chunk_document(doc)
    return [_serialize(c) for c in chunks]


@task(
    name="generate-qa",
    description="调用 LLM 生成 QA 对",
    retries=3,
    retry_delay_seconds=60,  # LLM 限流时等待 1 分钟
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=3),  # QA 缓存 3 天（模型可能更新）
)
def generate_qa_task(
    chunks_dict: list[dict],
    model: str,
    temperature: float,
    qa_strategy: str = "simple",
    advanced_max_qa_per_doc: int = 10,
) -> list[dict]:
    """QA 生成 Task（LLM 失败自动重试 3 次）。"""
    if qa_strategy not in _VALID_QA_STRATEGIES:
        raise ValueError(f"qa_strategy 无效：{qa_strategy!r}，合法值：{_VALID_QA_STRATEGIES}")

    if qa_strategy == "advanced" and advanced_max_qa_per_doc < 1:
        raise ValueError(f"advanced_max_qa_per_doc 必须 >= 1，实际值：{advanced_max_qa_per_doc}")

    from models import SemanticChunk
    from utils.llm_client import LLMClient

    chunks = [SemanticChunk(**c) for c in chunks_dict]
    llm = LLMClient(model=model, temperature=temperature)

    if qa_strategy == "advanced":
        from qa_generation.advanced_generator import AdvancedQAGenerator

        generator = AdvancedQAGenerator(
            llm_client=llm,
            max_qa_per_doc=advanced_max_qa_per_doc,
        )
        qa_pairs = generator.generate_document_qa(chunks)
    else:  # simple
        from qa_generation.multi_type_generator import QAGenerator

        generator = QAGenerator(llm_client=llm)
        qa_pairs = generator.generate_batch(chunks)

    return [_serialize(qa) for qa in qa_pairs]


@task(
    name="score-and-filter-qa",
    description="质量评分 + 过滤低质量 QA",
    retries=1,
)
def score_filter_task(
    qa_dicts: list[dict],
    faithfulness_min: float,
    relevance_min: float,
    answer_min_length: int,
    composite_pass: float,
    nli_model: str,
    embedding_model: str,
) -> list[dict]:
    """质量评分 + 过滤 Task。"""
    from models import QAPair, QualityThresholds
    from qa_filtering.deduplicator import Deduplicator
    from qa_filtering.quality_scorer import QualityScorer

    qa_pairs = [QAPair(**qa) for qa in qa_dicts]
    thresholds = QualityThresholds(
        faithfulness_min=faithfulness_min,
        relevance_min=relevance_min,
        answer_min_length=answer_min_length,
        composite_pass=composite_pass,
    )
    scorer = QualityScorer(
        nli_model=nli_model, embedding_model=embedding_model, thresholds=thresholds
    )
    scored = scorer.score_batch(qa_pairs)

    # 按质量分降序排列后去重，确保保留相似问题中质量最高的
    passed = sorted(
        [qa for qa in scored if qa.is_passed],
        key=lambda qa: qa.quality_score or 0,
        reverse=True,
    )

    dedup = Deduplicator(embedding_model=embedding_model)
    final = dedup.deduplicate(passed)

    return [_serialize(qa) for qa in final]


def _deserialize_semantic_chunk(d: dict) -> Any:
    """从 dict 还原 SemanticChunk（忽略未知字段）。"""
    import dataclasses

    from models import SemanticChunk

    known = {f.name for f in dataclasses.fields(SemanticChunk)}
    return SemanticChunk(**{k: v for k, v in d.items() if k in known})


def _deserialize_enriched_chunk(d: dict) -> Any:
    """从 dict 还原 EnrichedChunk（忽略未知字段，防止 schema 漂移导致 TypeError）。"""
    import dataclasses

    from models import EnrichedChunk

    known = {f.name for f in dataclasses.fields(EnrichedChunk)}
    return EnrichedChunk(**{k: v for k, v in d.items() if k in known})


@task(
    name="enrich-chunks",
    description="调用 LLM 对每个 chunk 提取实体（并发执行）",
    retries=2,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=7),
)
def enrich_task(chunks: list[dict], model: str) -> list[dict]:
    """实体提取 Task：并发调用 EntityExtractor，结果可 JSON 化。"""
    from concurrent.futures import ThreadPoolExecutor

    extractor = EntityExtractor(model=model)
    semantic_chunks = [_deserialize_semantic_chunk(chunk) for chunk in chunks]
    with ThreadPoolExecutor(max_workers=5) as executor:
        enriched = list(executor.map(extractor.extract, semantic_chunks))
    return [_serialize(e) for e in enriched]


@task(
    name="synthesize-qa",
    description="文档级 QA 合成：构建关系索引并调用合成器",
    retries=2,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=3),
)
def synthesize_qa_task(
    enriched_chunks: list[dict],
    quota: int,
    synthesize_model: str,
    verifier_model: str,
    embedding_model: str,
    self_consistency_check: bool,
) -> list[dict]:
    """文档级 QA 合成 Task（LLM 失败自动重试 2 次）。"""
    chunks = [_deserialize_enriched_chunk(chunk) for chunk in enriched_chunks]
    # embedding_vectors 传空 dict：当前版本仅启用实体重叠边，语义相似边暂不计算。
    relations = build_relation_index(chunks=chunks, embedding_vectors={})
    synthesizer = DocumentQASynthesizer(
        synthesize_model=synthesize_model,
        verifier_model=verifier_model,
        embedding_model=embedding_model,
        self_consistency_check=self_consistency_check,
    )
    qa_pairs = synthesizer.synthesize(chunks=chunks, relations=relations, quota=quota)
    return [_serialize(qa) for qa in qa_pairs]


@task(name="export-dataset", description="导出数据集")
def export_task(
    qa_dicts: list[dict],
    output_path: str,
    format_type: str,
) -> str:
    """数据集导出 Task。"""
    from export.dataset_exporter import DatasetExporter
    from models import QAPair

    qa_pairs = [QAPair(**qa) for qa in qa_dicts]
    return DatasetExporter.export(qa_pairs, output_path, format_type)  # type: ignore[arg-type]
