"""API schemas 单元测试。"""

from api.schemas import QAPairResponse


def test_qapair_response_new_fields():
    """QAPairResponse 包含新字段。"""
    data = {
        "qa_id": "qa_1",
        "doc_id": "doc_1",
        "question": "问题",
        "answer": "答案",
        "question_type": "definition",
        "content_type": "text",
        "reason": "出题理由",
        "reasoning_steps": "步骤1\n步骤2",
    }
    response = QAPairResponse(**data)
    assert response.reason == "出题理由"
    assert response.reasoning_steps == "步骤1\n步骤2"


def test_qapair_response_default_values():
    """QAPairResponse 新字段有默认值（向后兼容）。"""
    data = {
        "qa_id": "qa_1",
        "doc_id": "doc_1",
        "question": "问题",
        "answer": "答案",
        "question_type": "definition",
        "content_type": "text",
    }
    response = QAPairResponse(**data)
    assert response.reason == ""
    assert response.reasoning_steps == ""
