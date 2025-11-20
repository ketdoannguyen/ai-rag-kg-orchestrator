# LLM Gateway Client

Provider-agnostic chat + embedding gateway built on top of [LiteLLM](https://docs.litellm.ai/docs/) that keeps your application code decoupled from any single vendor interface.

## Overview

The gateway uses LiteLLM’s OpenAI-compatible schema to target 100+ models while exposing a compact, typed API:

- **`AIGatewayClient`** – unified client for chat completions, streaming, and embeddings
- **`GatewayModelConfig`** – declarative per-provider configuration (model id, api key, base URL, etc.)
- **Request/Response Models** – stable message and embedding payloads used throughout the workspace
- **Automatic Retries** – Tenacity-powered exponential backoff around every provider call
- **Tool & Streaming Support** – send OpenAI-format tool definitions and consume delta chunks

## Quick Start

```python
import os

from llm_gateway_client import (
    AIGatewayClient,
    GatewayModelConfig,
    GatewayInvokeRequest,
    GatewayChatTurn,
    EmbeddingRequest,
)

client = AIGatewayClient(
    chat_models={
        "default": GatewayModelConfig(
            model="openai/gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    },
    embedding_models={
        "default": GatewayModelConfig(
            model="openai/text-embedding-3-small",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    },
)
```

### Chat Invocation

```python
async def ask_capital():
    request = GatewayInvokeRequest(
        messages=[
            GatewayChatTurn(role="system", content="You are a helpful assistant."),
            GatewayChatTurn(role="user", content="What is the capital of France?"),
        ],
        provider_options={"temperature": 0.2},
    )
    response = await client.invoke(request)
    print(response.message.content)
```

### Streaming

```python
async def stream_story():
    request = GatewayInvokeRequest(
        messages=[GatewayChatTurn(role="user", content="Tell me a short story.")],
        provider_options={"temperature": 0.7},
    )
    async for chunk in client.astream(request):
        if chunk.content_delta:
            print(chunk.content_delta, end="", flush=True)
```

### Tooling

```python
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Fetches the current weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}

tool_client = client.bind_tools([weather_tool])
```

### Embeddings

```python
async def embed_texts():
    request = EmbeddingRequest(input=["Hello world", "Goodbye"])
    response = await client.embed(request)
    print(len(response.embeddings))  # -> 2
```

## Configuration Tips

- Set provider credentials via environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).
- Per-request overrides go into `provider_options` (temperature, max tokens, Azure API version, routing metadata, …). These values merge with any defaults defined in `GatewayModelConfig.extra`.
- Model routing works by exact key or prefix. For example, registering `"gpt-4o": GatewayModelConfig(...)` will be selected when `request.model` starts with `"gpt-4o"`.

## Dependencies

- [LiteLLM](https://github.com/BerriAI/litellm) – multi-provider OpenAI-compatible shim
- Pydantic – request/response validation
- Loguru + Tenacity – structured logging and retries

## Project Layout

```
llm-gateway-client/
├── pyproject.toml
├── README.md
└── src/llm_gateway_client
    ├── client.py          # LiteLLM-backed gateway
    ├── models/            # Request / response / config models
    ├── utils.py           # Message + stream conversion helpers
    └── exceptions.py
```