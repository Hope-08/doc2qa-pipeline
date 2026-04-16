"""LangChain OpenAI LLM 客户端封装。

自动从环境变量读取：
  OPENAI_API_KEY   - API 密钥
  OPENAI_API_BASE  - 自定义端点（兼容 OpenAI 协议的服务）
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMCallResult:
    """单次 LLM 调用结果。"""

    content: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMClient:
    """基于 langchain-openai 的 LLM 客户端。

    Args:
        model:       模型名称，如 ``"gpt-4o"``、``"qwen2.5-72b-instruct"``。
        temperature: 生成温度（0.0 ~ 2.0）。
        max_tokens:  最大输出 token 数。
    """

    def __init__(
        self,
        model: str = "azure-openai/gpt-5.4",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._call_count: int = 0
        self._llm = None  # 延迟初始化，避免启动时校验 API Key

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def complete(self, prompt: str) -> LLMCallResult:
        """发起单次 completion 请求。

        Args:
            prompt: 用户侧完整 prompt 文本。

        Returns:
            LLMCallResult，包含生成内容和 token 用量。

        Raises:
            ImportError: langchain-openai 未安装时抛出。
        """
        try:
            from langchain_core.messages import HumanMessage
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "请安装 langchain-openai：uv add 'langchain-openai>=0.1.0'"
            ) from exc

        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        self._call_count += 1
        response = self._llm.invoke([HumanMessage(content=prompt)])

        usage = response.usage_metadata or {}
        result = LLMCallResult(
            content=response.content or "",
            model=self.model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

        logger.debug(
            "LLM call #%d | tokens=%d+%d",
            self._call_count,
            result.input_tokens,
            result.output_tokens,
        )

        return result

    @property
    def call_count(self) -> int:
        return self._call_count
