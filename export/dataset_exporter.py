"""数据集导出器：支持 JSONL、HuggingFace Dataset、Alpaca、RAGAS 四种格式。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from models import QAPair
from utils.logger import get_logger

logger = get_logger(__name__)

FormatType = Literal["jsonl", "hf_dataset", "alpaca", "ragas"]


class DatasetExporter:
    """多格式 QA 数据集导出器。

    所有导出方法均为静态方法，无需实例化。
    """

    @staticmethod
    def export(
        qa_pairs: list[QAPair],
        output_path: str,
        format_type: FormatType = "jsonl",
    ) -> str:
        """导出 QA 数据集。

        Args:
            qa_pairs:    已过滤（is_passed=True）的 QA 对列表。
            output_path: 输出路径（文件或目录，取决于 format_type）。
            format_type: 导出格式。

        Returns:
            实际输出路径字符串。

        Raises:
            ValueError: format_type 不支持时抛出。
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        dispatch = {
            "jsonl": DatasetExporter._export_jsonl,
            "hf_dataset": DatasetExporter._export_hf,
            "alpaca": DatasetExporter._export_alpaca,
            "ragas": DatasetExporter._export_ragas,
        }
        if format_type not in dispatch:
            raise ValueError(f"不支持的格式：{format_type}，可选 {list(dispatch)}")

        return dispatch[format_type](qa_pairs, output_path)

    # ------------------------------------------------------------------
    # 各格式实现
    # ------------------------------------------------------------------

    @staticmethod
    def _export_jsonl(qa_pairs: list[QAPair], output_path: str) -> str:
        """OpenAI fine-tune 格式（messages 数组）。"""
        with open(output_path, "w", encoding="utf-8") as f:
            for qa in qa_pairs:
                record = {
                    "messages": [
                        {"role": "user", "content": qa.question},
                        {"role": "assistant", "content": qa.answer},
                    ],
                    "metadata": {
                        "qa_id": qa.qa_id,
                        "doc_id": qa.doc_id,
                        "question_type": qa.question_type,
                        "quality_score": qa.quality_score,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("JSONL 导出完成：%s（%d 条）", output_path, len(qa_pairs))
        return output_path

    @staticmethod
    def _export_hf(qa_pairs: list[QAPair], output_path: str) -> str:
        """HuggingFace Dataset（Arrow 格式）。"""
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError("请安装 datasets：uv add 'datasets>=2.20.0'") from exc

        # save_to_disk 写入目录结构，若 output_path 已作为文件存在则会失败
        dest = Path(output_path)
        if dest.exists() and not dest.is_dir():
            raise RuntimeError(
                f"导出路径 {output_path} 已作为文件存在，hf_dataset 格式需要写入目录。"
                " 请删除该文件或使用其他输出路径。"
            )
        dest.mkdir(parents=True, exist_ok=True)

        data = {
            "qa_id": [qa.qa_id for qa in qa_pairs],
            "question": [qa.question for qa in qa_pairs],
            "answer": [qa.answer for qa in qa_pairs],
            "context": [qa.context for qa in qa_pairs],
            "doc_id": [qa.doc_id for qa in qa_pairs],
            "question_type": [qa.question_type for qa in qa_pairs],
            "content_type": [qa.content_type for qa in qa_pairs],
            "quality_score": [qa.quality_score for qa in qa_pairs],
            "faithfulness_score": [qa.faithfulness_score for qa in qa_pairs],
            "relevance_score": [qa.relevance_score for qa in qa_pairs],
        }
        dataset = Dataset.from_dict(data)
        dataset.save_to_disk(output_path)

        logger.info("HF Dataset 导出完成：%s（%d 条）", output_path, len(qa_pairs))
        return output_path

    @staticmethod
    def _export_alpaca(qa_pairs: list[QAPair], output_path: str) -> str:
        """Alpaca instruction-following 格式。"""
        records = [
            {
                "instruction": qa.question,
                "input": "",
                "output": qa.answer,
                "metadata": {
                    "qa_id": qa.qa_id,
                    "doc_id": qa.doc_id,
                    "context": qa.context,
                    "quality_score": qa.quality_score,
                },
            }
            for qa in qa_pairs
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        logger.info("Alpaca 导出完成：%s（%d 条）", output_path, len(qa_pairs))
        return output_path

    @staticmethod
    def _export_ragas(qa_pairs: list[QAPair], output_path: str) -> str:
        """RAGAS 评测格式（question / ground_truth / contexts）。"""
        records = [
            {
                "question": qa.question,
                "ground_truth": qa.answer,
                "contexts": [qa.context],
                "metadata": {"qa_id": qa.qa_id, "doc_id": qa.doc_id},
            }
            for qa in qa_pairs
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        logger.info("RAGAS 导出完成：%s（%d 条）", output_path, len(qa_pairs))
        return output_path
