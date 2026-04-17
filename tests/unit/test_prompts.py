"""Tests for QA generation prompt templates."""

from __future__ import annotations

from config.prompts import (
    CAPTION_QA_PROMPT,
    DOC_QA_SYNTHESIS_PROMPT,
    TABLE_QA_PROMPT,
    TEXT_QA_PROMPT,
)


def test_text_qa_prompt_requires_new_fields():
    """TEXT_QA_PROMPT requires output of reason and reasoning_steps fields."""
    prompt = TEXT_QA_PROMPT.format(context="test content")
    # Verify JSON example format includes the new fields as keys
    assert '"reason":' in prompt, "TEXT_QA_PROMPT should include '\"reason\":' in JSON example"
    assert (
        '"reasoning_steps":' in prompt
    ), "TEXT_QA_PROMPT should include '\"reasoning_steps\":' in JSON example"


def test_table_qa_prompt_requires_new_fields():
    """TABLE_QA_PROMPT requires output of reason and reasoning_steps fields."""
    prompt = TABLE_QA_PROMPT.format(context="<table></table>")
    # Verify JSON example format includes the new fields as keys
    assert '"reason":' in prompt, "TABLE_QA_PROMPT should include '\"reason\":' in JSON example"
    assert (
        '"reasoning_steps":' in prompt
    ), "TABLE_QA_PROMPT should include '\"reasoning_steps\":' in JSON example"


def test_caption_qa_prompt_requires_new_fields():
    """CAPTION_QA_PROMPT requires output of reason and reasoning_steps fields."""
    prompt = CAPTION_QA_PROMPT.format(context="图片描述")
    # Verify JSON example format includes the new fields as keys
    assert '"reason":' in prompt, "CAPTION_QA_PROMPT should include '\"reason\":' in JSON example"
    assert (
        '"reasoning_steps":' in prompt
    ), "CAPTION_QA_PROMPT should include '\"reasoning_steps\":' in JSON example"


def test_doc_qa_synthesis_prompt_requires_new_fields():
    """DOC_QA_SYNTHESIS_PROMPT requires output of reason and reasoning_steps fields."""
    prompt = DOC_QA_SYNTHESIS_PROMPT.format(
        quota=5,
        multi_hop_count=2,
        chunk_summaries="[chunk_1] 摘要",
        relations="chunk_1 -> chunk_2",
    )
    # Verify JSON example format includes the new fields as keys
    assert (
        '"reason":' in prompt
    ), "DOC_QA_SYNTHESIS_PROMPT should include '\"reason\":' in JSON example"
    assert (
        '"reasoning_steps":' in prompt
    ), "DOC_QA_SYNTHESIS_PROMPT should include '\"reasoning_steps\":' in JSON example"
