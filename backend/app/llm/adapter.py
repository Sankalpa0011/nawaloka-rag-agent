"""Thin LangChain/LlamaIndex wrappers that forward every call through our LLMRouter
(get_router()) so failover, token budgets and the circuit breaker still apply. Async only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

from app.llm.base import Msg, Tier
from app.llm.router import get_router
from app.logging_conf import get_logger

log = get_logger(__name__)

# Rough ceiling for the models we actually use. LlamaIndex needs a number for
# its context-window bookkeeping; nothing here gets close to it.
CONTEXT_WINDOW = 32_000


def _to_msgs(messages: Sequence[BaseMessage] | Sequence[ChatMessage]) -> list[Msg]:
    """Framework messages -> our Msg. Anything unrecognised becomes a user turn."""
    out: list[Msg] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, (HumanMessage, BaseMessage)) and not isinstance(m, ChatMessage):
            role = "user"
        else:
            # LlamaIndex ChatMessage carries its role as an enum-ish value
            role = str(getattr(m, "role", "user")).split(".")[-1].lower()
            if role not in ("system", "user", "assistant"):
                role = "user"

        content = m.content if isinstance(m.content, str) else str(m.content)
        out.append(Msg(role, content))  # type: ignore[arg-type]
    return out


_SYNC_MESSAGE = (
    "{what} is sync and this adapter is async-only - use {use} instead. "
    "The provider SDKs cache their httpx clients, and an httpx client is bound "
    "to the event loop that created it, so driving one from a second loop "
    "either errors immediately or hangs until the request times out. Measured: "
    "a single sync call cost 54s and a needless failover to the backup provider. "
    "Every caller in this system is async, and LlamaIndex offers aselect / "
    "achat / acomplete, so there is no reason to reach for the sync path."
)


def _no_sync(what: str, use: str):
    raise NotImplementedError(_SYNC_MESSAGE.format(what=what, use=use))


# LangChain
class NawalokaChatModel(BaseChatModel):
    """LangChain chat model backed by our router."""

    tier: Tier = "fast"
    temperature: float = 0.0
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "nawaloka-router"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"tier": self.tier, "temperature": self.temperature}

    def _result(self, text: str, provider: str, model: str) -> ChatResult:
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(
            generations=[generation],
            llm_output={"provider": provider, "model": model, "tier": self.tier},
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = await get_router().complete(
            _to_msgs(messages),
            tier=kwargs.get("tier", self.tier),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return self._result(response.text, response.provider, response.model)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _no_sync("invoke()", "ainvoke()")


# LlamaIndex
class NawalokaLlamaLLM(CustomLLM):
    """LlamaIndex LLM backed by the same router."""

    tier: Tier = "fast"
    temperature: float = 0.0
    max_tokens: int = 1024

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=CONTEXT_WINDOW,
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=f"nawaloka-router:{self.tier}",
        )

    async def _acomplete_text(self, prompt: str) -> str:
        response = await get_router().complete(
            [Msg("user", prompt)],
            tier=self.tier,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.text

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        _no_sync("complete()", "acomplete()")

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(text=await self._acomplete_text(prompt))

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        response = await get_router().complete(
            _to_msgs(messages),
            tier=self.tier,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return ChatResponse(
            message=ChatMessage(role="assistant", content=response.text)
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        _no_sync("chat()", "achat()")

    # CustomLLM makes stream_complete abstract so it has to exist. Nothing here
    # streams through LlamaIndex - synthesis is one non-streaming call because
    # the answer gets post-checked for unsupported prices first - so this yields
    # the finished text in one go rather than pretending to stream.
    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        text = await self._acomplete_text(prompt)

        async def gen():
            yield CompletionResponse(text=text, delta=text)

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        _no_sync("stream_complete()", "astream_complete()")

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        _no_sync("stream_chat()", "achat()")


# Cached per tier - these are stateless wrappers, so one each is plenty.
@lru_cache(maxsize=4)
def get_lc_llm(tier: Tier = "fast", max_tokens: int = 1024) -> NawalokaChatModel:
    return NawalokaChatModel(tier=tier, max_tokens=max_tokens)


@lru_cache(maxsize=4)
def get_li_llm(tier: Tier = "fast", max_tokens: int = 1024) -> NawalokaLlamaLLM:
    return NawalokaLlamaLLM(tier=tier, max_tokens=max_tokens)
