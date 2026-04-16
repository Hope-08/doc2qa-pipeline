"""单元测试：settings 默认值。"""

from __future__ import annotations

from config.settings import Settings


class TestSettingsDefaults:
    def test_advanced_qa_defaults(self) -> None:
        settings = Settings(_env_file=None)

        assert settings.qa_strategy == "advanced"
        assert settings.chunk_size == 800
        assert settings.max_qa_per_doc == 10
        assert settings.entity_extractor_model == "azure-openai/gpt-4o"
        assert settings.synthesize_model == "azure-openai/gpt-5.4"
        assert settings.nli_model == "azure-openai/gpt-5.4"
