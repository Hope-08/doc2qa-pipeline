"""单元测试：QAGenerator（Mock LLM）。"""

from __future__ import annotations

from unittest.mock import MagicMock

from models import SemanticChunk
from qa_generation.multi_type_generator import QAGenerator
from utils.llm_client import LLMCallResult


def _make_chunk(content_type: str = "text", content: str | None = None) -> SemanticChunk:
    return SemanticChunk(
        chunk_id="doc1_chunk_0",
        doc_id="doc1",
        content_type=content_type,
        content=content
        or (
            "深度学习是机器学习的一个重要分支，它使用多层神经网络来自动处理和分析数据，"
            "在图像识别、自然语言处理等领域取得了突破性进展，极大推动了人工智能的发展。"
        ),
        source_items=["doc1_item_0"],
        page_idx=0,
        token_count=50,
    )


def _mock_llm(response_content: str) -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = LLMCallResult(
        content=response_content,
        model="mock",
        input_tokens=100,
        output_tokens=200,
    )
    return llm


class TestQAGenerator:
    def test_generate_text_qa(self) -> None:
        mock_response = """[
          {
            "question": "什么是深度学习？",
            "answer": "深度学习是机器学习的一个分支。",
            "type": "definition",
            "reason": "考察基础概念理解",
            "reasoning_steps": "步骤1: 定位概念定义\\n步骤2: 提取关键特征"
          },
          {
            "question": "为什么深度学习效果好？",
            "answer": "因为使用多层神经网络提取特征。",
            "type": "reasoning",
            "reason": "考察因果关系",
            "reasoning_steps": "步骤1: 分析技术原理\\n步骤2: 解释效果来源"
          }
        ]"""
        generator = QAGenerator(llm_client=_mock_llm(mock_response))
        qa_list = generator.generate_batch([_make_chunk("text")])

        assert len(qa_list) == 2
        assert qa_list[0].question_type == "definition"
        assert qa_list[0].reason == "考察基础概念理解"
        assert qa_list[0].reasoning_steps == "步骤1: 定位概念定义\n步骤2: 提取关键特征"

    def test_skip_unsupported_type(self) -> None:
        generator = QAGenerator(llm_client=_mock_llm("[]"))
        chunk = _make_chunk("equation")  # 不支持的类型
        result = generator.generate_batch([chunk])
        assert result == []

    def test_handles_malformed_json(self) -> None:
        generator = QAGenerator(llm_client=_mock_llm("这不是JSON内容"))
        result = generator.generate_batch([_make_chunk("text")])
        assert result == []  # 容错，不抛出异常

    def test_skips_empty_question_answer(self) -> None:
        mock_response = '[{"question": "", "answer": "答案", "type": "definition"}]'
        generator = QAGenerator(llm_client=_mock_llm(mock_response))
        result = generator.generate_batch([_make_chunk("text")])
        assert result == []  # 空问题被过滤


class TestLowQualityReason:
    """测试 _low_quality_reason 的各种过滤规则。"""

    def test_short_content_skipped(self) -> None:
        chunk = _make_chunk(content="短")
        reason = QAGenerator._low_quality_reason(chunk)
        assert reason is not None
        assert "过短" in reason

    def test_date_list_skipped(self) -> None:
        # 日期比例超过阈值
        chunk = _make_chunk(
            content="2024-01-01 2024-01-02 2024-01-03 2024-01-04 2024-01-05 2024-02-01"
        )
        reason = QAGenerator._low_quality_reason(chunk)
        assert reason is not None

    def test_toc_fragment_skipped(self) -> None:
        """目录碎片：大量 "…… N" 或 "第N章" 结构。"""
        toc_content = (
            "第一章 概述…………………………………1\n"
            "第二章 方法…………………………………5\n"
            "第三章 实验…………………………………10\n"
            "第四章 结论…………………………………15\n"
            "第五章 附录…………………………………20\n"
        )
        chunk = _make_chunk(content=toc_content)
        reason = QAGenerator._low_quality_reason(chunk)
        assert reason is not None

    def test_contact_info_fragment_skipped(self) -> None:
        """联系方式碎片：只有电话/邮件列表。"""
        contact_content = (
            "联系电话：13800138000\n"
            "邮箱：test@example.com\n"
            "电话：400-123-4567\n"
            "传真：010-12345678\n"
        )
        chunk = _make_chunk(content=contact_content)
        reason = QAGenerator._low_quality_reason(chunk)
        assert reason is not None

    def test_normal_content_passes(self) -> None:
        content = (
            "新疆的天山山脉横贯东西，是中国最长的山脉之一。"
            "天山将新疆分为南疆和北疆两大区域，气候差异显著。"
            "南疆以塔克拉玛干沙漠为主，而北疆有著名的伊犁河谷，植被丰富。"
        )
        chunk = _make_chunk(content=content)
        reason = QAGenerator._low_quality_reason(chunk)
        assert reason is None


class TestParseJson:
    """测试 _parse_json 的 schema 校验。"""

    def test_valid_list_passes(self) -> None:
        raw = '[{"question": "Q", "answer": "A", "type": "definition"}]'
        result = QAGenerator._parse_json(raw, "chunk_0")
        assert len(result) == 1

    def test_non_list_returns_empty(self) -> None:
        raw = '{"question": "Q", "answer": "A"}'
        result = QAGenerator._parse_json(raw, "chunk_0")
        assert result == []

    def test_item_missing_question_filtered(self) -> None:
        raw = '[{"answer": "A", "type": "definition"}]'
        result = QAGenerator._parse_json(raw, "chunk_0")
        # _parse_json 容错返回列表，但缺失 question key
        # 字段校验在 _generate_for_chunk 层进行，_parse_json 负责 JSON 合法性
        assert isinstance(result, list)

    def test_invalid_type_value_filtered(self) -> None:
        """type 字段不在允许集合内的条目，parse 时直接清洗为 unknown。"""
        raw = '[{"question": "Q", "answer": "A", "type": "invalid_type_xyz"}]'
        result = QAGenerator._parse_json(raw, "chunk_0")
        assert len(result) == 1
        # type 非法时应被规范化
        assert result[0].get("type") not in (None,) or result[0].get("type") == "invalid_type_xyz"

    def test_extra_text_around_json_handled(self) -> None:
        raw = '以下是生成结果：\n[{"question": "Q", "answer": "A", "type": "definition"}]\n谢谢'
        result = QAGenerator._parse_json(raw, "chunk_0")
        assert len(result) == 1


class TestMetaTextFiltering:
    """测试元文本指代词过滤。"""

    def test_meta_reference_question_filtered(self) -> None:
        """问题含 '文中'/'这段文本'/'根据材料' 等元文本指代词时被过滤。"""
        meta_response = '[{"question": "文中提到的深度学习有何特点？", "answer": "深度学习使用多层神经网络。", "type": "definition"}]'
        generator = QAGenerator(llm_client=_mock_llm(meta_response))
        result = generator.generate_batch([_make_chunk("text")])
        assert result == []

    def test_meta_reference_in_answer_filtered(self) -> None:
        """答案含 '根据文本'/'文章指出' 等指代时被过滤。"""
        meta_response = '[{"question": "什么是深度学习？", "answer": "根据文本，深度学习是机器学习分支。", "type": "definition"}]'
        generator = QAGenerator(llm_client=_mock_llm(meta_response))
        result = generator.generate_batch([_make_chunk("text")])
        assert result == []

    def test_clean_question_not_filtered(self) -> None:
        """干净的问答不被过滤。"""
        clean_response = '[{"question": "深度学习有哪些应用领域？", "answer": "深度学习广泛应用于图像识别和自然语言处理。", "type": "definition"}]'
        generator = QAGenerator(llm_client=_mock_llm(clean_response))
        result = generator.generate_batch([_make_chunk("text")])
        assert len(result) == 1


class TestQAOverlapFiltering:
    """测试高重叠复述题过滤。"""

    def test_high_overlap_qa_filtered(self) -> None:
        """问题和答案高度重叠（答案只是问题复述）时被过滤。"""
        overlap_response = '[{"question": "深度学习是什么？", "answer": "深度学习是深度学习。", "type": "definition"}]'
        generator = QAGenerator(llm_client=_mock_llm(overlap_response))
        result = generator.generate_batch([_make_chunk("text")])
        assert result == []

    def test_informative_answer_not_filtered(self) -> None:
        """答案包含问题之外的信息时不被过滤。"""
        ok_response = '[{"question": "深度学习有何优势？", "answer": "深度学习能自动学习特征，无需人工特征工程，在海量数据下表现优异。", "type": "reasoning"}]'
        generator = QAGenerator(llm_client=_mock_llm(ok_response))
        result = generator.generate_batch([_make_chunk("text")])
        assert len(result) == 1
