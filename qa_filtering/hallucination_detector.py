"""幻觉检测器：LLM 专项封装，可单独使用。

模块级函数 `_run_faithfulness_batch` 同时被 QualityScorer 使用，
避免重复维护相同的 Prompt 和解析逻辑。
"""

from __future__ import annotations

import json

from models import QAPair
from utils.logger import get_logger

logger = get_logger(__name__)

# 每批最多条目数（避免超出 LLM 上下文）
_BATCH_SIZE = 5
# context 截断长度
_CONTEXT_MAX_CHARS = 1500


def _run_faithfulness_batch(
    llm: object,
    batch: list[QAPair],
    context_max_chars: int = _CONTEXT_MAX_CHARS,
) -> list[float]:
    """调用 LLM 对一批 QA 打 faithfulness 分，返回 0~1 列表。

    若某个 QA 的 context 为空，直接返回 0.0（无依据支撑）。
    若 LLM 返回格式异常或遗漏某些条目，记录 warning 并使用默认值 0.5。

    Args:
        llm:               已初始化的 ChatOpenAI 实例。
        batch:             当前批次的 QAPair 列表。
        context_max_chars: context 截断字符数。

    Returns:
        与 batch 等长的浮点列表。
    """
    from langchain_core.messages import HumanMessage

    # 预处理：空 context 直接标为 0.0，其余进 LLM
    needs_llm: list[tuple[int, QAPair]] = []  # (batch_idx, qa)
    preset: dict[int, float] = {}
    for j, qa in enumerate(batch):
        if not qa.context.strip():
            logger.warning("QA 第 %d 条 context 为空，faithfulness=0.0", j)
            preset[j] = 0.0
        else:
            needs_llm.append((j, qa))

    if not needs_llm:
        return [preset[j] for j in range(len(batch))]

    items = [
        {"id": orig_j, "context": qa.context[:context_max_chars], "answer": qa.answer}
        for orig_j, qa in needs_llm
    ]
    prompt = (
        '请判断以下每个条目中，"回答"是否完全基于"上下文"所支撑。\n'
        '返回 JSON 数组，每项格式为 {"id": <编号>, "score": <0~1 浮点数>}，'
        "1 = 完全支撑，0 = 完全不支撑。只返回 JSON，不要其他内容。\n\n"
        f"条目：\n{json.dumps(items, ensure_ascii=False, indent=2)}"
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        # 提取 JSON 数组子串：LLM 有时会在 [...] 前后附加说明文字
        start_idx = raw.find("[")
        end_idx = raw.rfind("]")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            raise ValueError(f"响应中未找到 JSON 数组：{raw[:100]}")
        results = json.loads(raw[start_idx : end_idx + 1])

        if not isinstance(results, list):
            raise ValueError(f"LLM 返回格式非数组：{type(results)}")

        scores_map: dict[int, float] = {}
        for r in results:
            item_id = r.get("id")
            item_score = r.get("score")
            if item_id is None or item_score is None:
                logger.warning("LLM 返回条目缺少 id 或 score 字段，跳过：%s", r)
                continue
            try:
                scores_map[int(item_id)] = max(0.0, min(1.0, float(item_score)))
            except (ValueError, TypeError) as exc:
                logger.warning("LLM 返回条目值无法解析，跳过：%s（%s）", r, exc)

        # 检查是否有遗漏条目
        expected_ids = {orig_j for orig_j, _ in needs_llm}
        missing = expected_ids - scores_map.keys()
        if missing:
            logger.warning(
                "LLM 返回缺少 %d 个条目（id=%s），使用默认值 0.5", len(missing), sorted(missing)
            )

        for orig_j, _ in needs_llm:
            preset.setdefault(orig_j, scores_map.get(orig_j, 0.5))

    except Exception as exc:
        logger.warning("LLM faithfulness 评分失败：%s，默认 0.5", exc)
        for orig_j, _ in needs_llm:
            preset.setdefault(orig_j, 0.5)

    return [preset[j] for j in range(len(batch))]


class HallucinationDetector:
    """基于 LLM（OpenAI API）的幻觉检测器。

    是 QualityScorer 中 faithfulness 维度的独立封装，
    可在不需要完整质量评分时单独使用。

    Args:
        model:     ChatOpenAI 模型名，如 azure-openai/gpt-5.4。
        threshold: faithfulness 分数低于此值视为幻觉。
    """

    def __init__(
        self,
        model: str = "azure-openai/gpt-5.4",
        threshold: float = 0.60,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self._llm = None

    def detect(self, qa_pairs: list[QAPair]) -> list[bool]:
        """检测 QA 对是否存在幻觉。

        Args:
            qa_pairs: 待检测的 QA 对列表。

        Returns:
            布尔列表，``True`` 表示存在幻觉（不忠实于 context）。
        """
        try:
            llm = self._get_llm()
        except Exception:
            logger.warning("LLM 不可用，全部标记为非幻觉（放行）")
            return [False] * len(qa_pairs)

        all_scores: list[float] = []
        for i in range(0, len(qa_pairs), _BATCH_SIZE):
            batch = qa_pairs[i : i + _BATCH_SIZE]
            all_scores.extend(_run_faithfulness_batch(llm, batch))

        return [s < self.threshold for s in all_scores]

    def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI

            logger.info("初始化幻觉检测 LLM：%s", self.model)
            self._llm = ChatOpenAI(model=self.model, temperature=0)
        return self._llm
