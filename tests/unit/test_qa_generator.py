"""单元测试：QAGenerator（Mock LLM）。"""

from __future__ import annotations

from unittest.mock import MagicMock

from models import SemanticChunk
from qa_generation.multi_type_generator import QAGenerator
from utils.llm_client import LLMCallResult


def _make_chunk(content_type: str = "text") -> SemanticChunk:
    return SemanticChunk(
        chunk_id="doc1_chunk_0",
        doc_id="doc1",
        content_type=content_type,
        content="深度学习是机器学习的一个分支，使用多层神经网络处理数据，在图像识别和NLP等领域取得了突破。",
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
        mock_response = """
        [
          {"question": "什么是深度学习？", "answer": "深度学习是机器学习的一个分支。", "type": "definition"},
          {"question": "为什么深度学习效果好？", "answer": "因为使用多层神经网络提取特征。", "type": "reasoning"},
          {"question": "如何应用深度学习？", "answer": "可以应用于图像识别和NLP。", "type": "procedure"}
        ]
        """
        generator = QAGenerator(llm_client=_mock_llm(mock_response))
        qa_list = generator.generate_batch([_make_chunk("text")])

        assert len(qa_list) == 3
        assert qa_list[0].question_type == "definition"
        assert qa_list[0].doc_id == "doc1"
        assert qa_list[0].quality_score is None  # 未评分

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
