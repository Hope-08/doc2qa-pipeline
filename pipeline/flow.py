"""主 Pipeline Flow：编排所有 Task，支持断点续跑。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from prefect import flow

from config.settings import settings as _settings
from pipeline.tasks import (
    parse_task,
    normalize_task,
    chunk_task,
    generate_qa_task,
    score_filter_task,
    export_task,
)
from utils.logger import get_logger

logger = get_logger(__name__)


@flow(
    name="doc2qa-pipeline",
    description="企业知识库 QA 数据自动生成 Pipeline",
    log_prints=True,
)
def doc2qa_flow(
    file_paths: list[str],
    output_path: str | None = None,
    model: str | None = None,
    export_format: str = "jsonl",
    job_id: str | None = None,
) -> list[dict]:
    """主 Pipeline Flow。

    Args:
        file_paths:    输入文档路径列表（支持 PDF / DOCX / 图片）。
        output_path:   输出数据集路径，默认写入 job 目录下。
        model:         LLM 模型名称，默认读取 Settings.llm_model。
        export_format: 导出格式（jsonl / hf_dataset / alpaca / ragas）。
        job_id:        任务 ID，用于创建中间结果目录。不传时自动生成时间戳 ID。

    Returns:
        通过质检的 QA 对列表（dict 格式，可序列化）。

    中间结果目录结构::

        output/{job_id}/
          01_parsed_{doc}.json       # 文档解析原始结果
          02_normalized_{doc}.json   # 清洗后结果
          03_chunks_{doc}.json       # 语义分块结果
          04_qa_raw_{doc}.json       # 各文档未过滤 QA
          04_qa_raw_all.json         # 全部文档汇总（多文档时）
          05_qa_filtered.json        # 质检+去重后 QA
          qa_dataset.jsonl           # 最终导出文件

    断点续跑：重新执行同一 flow，Prefect 自动跳过已缓存的 Task。

    监控面板：运行 ``prefect server start`` 后访问 http://localhost:4200
    """
    cfg = _settings
    model = model or cfg.llm_model

    # 创建 job 专属输出目录（并验证 job_id 不会逃逸出 output_dir）
    _job_id = job_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(cfg.output_dir).resolve()
    job_dir = (base_dir / _job_id).resolve()
    if not job_dir.is_relative_to(base_dir):
        raise ValueError(f"job_id 超出输出目录范围：{_job_id!r}")
    job_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_path or str(
        job_dir / f"qa_dataset.{_format_ext(export_format)}"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("doc2qa-pipeline 启动  job_id=%s", _job_id)
    logger.info(
        "文档数：%d  模型：%s  解析引擎：%s",
        len(file_paths), model, cfg.parsing_backend,
    )
    logger.info("中间结果目录：%s", job_dir)
    logger.info("=" * 60)

    all_qa_raw: list[dict] = []

    for file_path in file_paths:
        # 用路径 MD5 前缀保证同名文件的中间结果不互相覆盖
        doc_hash = hashlib.md5(file_path.encode()).hexdigest()[:6]
        doc_stem = f"{Path(file_path).stem}_{doc_hash}"
        logger.info("处理文档：%s", Path(file_path).name)

        # 1. 解析（本地多引擎）
        parsed = parse_task(
            file_path,
            cfg.parsing_backend,
            cfg.ocr_lang,
        )
        _save_step(job_dir, f"01_parsed_{doc_stem}.json", parsed)

        # 2. 清洗
        cleaned = normalize_task(parsed)
        _save_step(job_dir, f"02_normalized_{doc_stem}.json", cleaned)

        # 3. 分块
        chunks = chunk_task(cleaned, cfg.chunk_size, cfg.similarity_threshold, cfg.embedding_model)
        _save_step(job_dir, f"03_chunks_{doc_stem}.json", chunks)

        if not chunks:
            logger.warning("文档 [%s] 未产生有效 chunk，跳过", Path(file_path).name)
            continue

        # 4. QA 生成
        qa_list = generate_qa_task(
            chunks,
            model=model,
            temperature=cfg.llm_temperature,
        )
        _save_step(job_dir, f"04_qa_raw_{doc_stem}.json", qa_list)
        all_qa_raw.extend(qa_list)

    # 多文档时额外保存汇总的原始 QA，方便对比各文档贡献量
    if len(file_paths) > 1:
        _save_step(job_dir, "04_qa_raw_all.json", all_qa_raw)

    if not all_qa_raw:
        logger.warning("未生成任何 QA 对，Pipeline 结束")
        return []

    # 5. 质量评分 + 过滤 + 去重
    passed_qa = score_filter_task(
        all_qa_raw,
        faithfulness_min=cfg.faithfulness_min,
        relevance_min=cfg.relevance_min,
        answer_min_length=cfg.answer_min_length,
        composite_pass=cfg.composite_pass,
        nli_model=cfg.nli_model,
        embedding_model=cfg.embedding_model,
    )
    _save_step(job_dir, "05_qa_filtered.json", passed_qa)

    # 6. 导出
    final_path = export_task(passed_qa, output_path, export_format)

    logger.info("=" * 60)
    logger.info("Pipeline 完成！")
    logger.info(
        "总 QA 数：%d  通过质检：%d  通过率：%.1f%%",
        len(all_qa_raw), len(passed_qa),
        len(passed_qa) / max(len(all_qa_raw), 1) * 100,
    )
    logger.info("中间结果目录：%s", job_dir)
    logger.info("最终输出：%s", final_path)
    logger.info("=" * 60)

    return passed_qa


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _save_step(job_dir: Path, filename: str, data: Any) -> None:
    """将某一步骤的中间结果保存为 JSON 文件到 job 目录。

    写入失败时只记录 warning，不影响主流程。
    """
    target = job_dir / filename
    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug("步骤结果已保存：%s", target.name)
    except (OSError, IOError) as exc:
        logger.warning("步骤结果保存失败（IO 错误）[%s]：%s", filename, exc)
    except TypeError as exc:
        logger.error("步骤结果无法序列化 [%s]：%s", filename, exc, exc_info=True)


def _format_ext(format_type: str) -> str:
    """根据格式类型返回文件扩展名。"""
    return {
        "jsonl": "jsonl",
        "hf_dataset": "dataset",
        "alpaca": "json",
        "ragas": "json",
    }.get(format_type, "jsonl")
