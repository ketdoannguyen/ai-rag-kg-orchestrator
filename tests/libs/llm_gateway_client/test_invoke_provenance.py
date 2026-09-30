"""What an ``InvokeResponse`` reports about the call that produced it.

A request names a logical model; the gateway resolves that name to a provider
model, and the provider answers. Only what comes back says who actually served
the call and what it billed, so the response has to carry both -- and carry a
count the provider left out as absent rather than as a zero it never claimed.

The gateway retries on its own behalf, so it also has to say how many times it
asked: an answer that came back after a retry holds the counts of its last
attempt alone, and nothing in it states what the attempts before it billed.
"""

from __future__ import annotations

from typing import Any

import llm_gateway_client.client as client_module
import pytest
from llm_gateway_client.client import AIGatewayClient
from llm_gateway_client.models import GatewayChatTurn, GatewayInvokeRequest
from loguru import logger
from tenacity import wait_none

REQUESTED = "merge"
SERVED = "deepseek-v4.1-flash"


class _ProviderResponse:
    """As much of a LiteLLM response as the client reads."""

    def __init__(self, **envelope: Any) -> None:
        self._envelope = envelope

    def model_dump(self) -> dict[str, Any]:
        return self._envelope


def _answer(**envelope: Any) -> dict[str, Any]:
    """One provider envelope with a readable choice in it."""
    choice = {
        "message": {"role": "assistant", "content": "{}"},
        "finish_reason": "stop",
    }
    return {"model": SERVED, "choices": [choice], **envelope}


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch):
    """Install a provider that answers every call with one canned envelope."""
    envelope: dict[str, Any] = {}

    async def acompletion(**_kwargs: Any) -> _ProviderResponse:
        return _ProviderResponse(**envelope)

    monkeypatch.setattr("llm_gateway_client.client.acompletion", acompletion)
    return envelope


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real 3-attempt policy but drop the exponential sleep."""
    monkeypatch.setattr(client_module._CHAT_RETRY, "wait", wait_none())


class _FlakyProvider:
    """A provider that fails its first calls, then answers one canned envelope."""

    def __init__(self, failures: int, error: str = "upstream refused the request") -> None:
        self.failures = failures
        self.error = error
        self.calls = 0

    async def __call__(self, **_kwargs: Any) -> _ProviderResponse:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(self.error)
        return _ProviderResponse(
            **_answer(usage={"prompt_tokens": 1200, "completion_tokens": 800})
        )


async def _invoke() -> Any:
    client = AIGatewayClient(chat_models={REQUESTED: f"openai/{SERVED}"})
    return await client.invoke(
        GatewayInvokeRequest(
            model=REQUESTED,
            messages=[GatewayChatTurn(role="user", content="Gộp các đoạn")],
        )
    )


async def test_the_response_carries_the_model_that_answered(provider: dict[str, Any]) -> None:
    provider.update(
        _answer(usage={"prompt_tokens": 1200, "completion_tokens": 800, "total_tokens": 2000})
    )

    response = await _invoke()

    assert response.model == SERVED
    assert response.usage == {
        "prompt_tokens": 1200,
        "completion_tokens": 800,
        "total_tokens": 2000,
    }


async def test_the_provider_envelope_is_carried_verbatim(provider: dict[str, Any]) -> None:
    stated = {
        "prompt_tokens": 1200,
        "completion_tokens": 800,
        "prompt_tokens_details": {"cached_tokens": 900},
    }
    provider.update(_answer(usage=stated))

    response = await _invoke()

    assert response.stated_usage == stated
    assert response.stated_usage["prompt_tokens_details"] == {"cached_tokens": 900}


async def test_a_response_without_usage_states_none(provider: dict[str, Any]) -> None:
    provider.update(_answer())

    response = await _invoke()

    # The normalized view zero-fills the counters a merge prices; only the
    # envelope itself can say the provider stated nothing at all.
    assert response.stated_usage is None
    assert response.usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


async def test_a_response_without_a_model_names_none(provider: dict[str, Any]) -> None:
    envelope = _answer(usage={"prompt_tokens": 1, "completion_tokens": 1})
    del envelope["model"]
    provider.update(envelope)

    response = await _invoke()

    assert response.model is None


async def test_a_call_answered_first_try_reports_one_attempt(provider: dict[str, Any]) -> None:
    provider.update(_answer(usage={"prompt_tokens": 1200, "completion_tokens": 800}))

    response = await _invoke()

    assert response.attempts == 1


async def test_a_retried_call_reports_every_attempt_it_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flaky = _FlakyProvider(failures=1)
    monkeypatch.setattr("llm_gateway_client.client.acompletion", flaky)

    response = await _invoke()

    assert flaky.calls == 2
    assert response.attempts == 2  # the count a caller needs to distrust the bill
    assert response.usage["completion_tokens"] == 800  # still the answer that came back


async def test_a_retry_log_names_the_error_type_but_not_its_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A provider error can quote the request or the book page it rejected, so the
    # retry log has to say what kind of error it was and repeat none of it.
    flaky = _FlakyProvider(failures=1, error="rejected: 'Đoạn một Đoạn hai'")
    monkeypatch.setattr("llm_gateway_client.client.acompletion", flaky)
    logged: list[str] = []
    sink = logger.add(logged.append, level="WARNING")
    try:
        await _invoke()
    finally:
        logger.remove(sink)

    assert flaky.calls == 2  # the retry the log is about
    assert any("RuntimeError" in line for line in logged)
    assert all("Đoạn một" not in line for line in logged)
