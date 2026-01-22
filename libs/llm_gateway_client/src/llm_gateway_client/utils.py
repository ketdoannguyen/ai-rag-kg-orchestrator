"""Utility functions for the LLM Gateway Client."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from .models.request import GatewayChatTurn
from .models.response import InvokeStreamChunk
from .models.tool import ToolCall, ToolCallChunk


class MessageConverter:
    """Utilities for translating between gateway and LiteLLM formats."""

    @staticmethod
    def to_litellm_messages(messages: list[GatewayChatTurn]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            payload: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.role == "tool":
                if not msg.tool_call_id:
                    raise ValueError("tool_call_id must be provided for tool messages")
                payload["tool_call_id"] = msg.tool_call_id
            if msg.role == "assistant" and msg.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            formatted.append(payload)
        return formatted

    @staticmethod
    def parse_assistant_message(
        message_payload: dict[str, Any],
    ) -> tuple[GatewayChatTurn, list[ToolCall] | None]:
        content = MessageConverter._normalize_content(message_payload.get("content"))
        tool_calls_payload = message_payload.get("tool_calls") or []
        tool_calls = MessageConverter._parse_tool_calls(tool_calls_payload)
        chat_turn = GatewayChatTurn(
            role="assistant",
            content=content,
            tool_calls=tool_calls or None,
        )
        return chat_turn, tool_calls or None

    @staticmethod
    def parse_stream_chunk(chunk: dict[str, Any]) -> InvokeStreamChunk:
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        content_delta = MessageConverter._normalize_content(delta.get("content")) or None
        tool_call_chunks_payload = delta.get("tool_calls") or []
        tool_call_chunks = MessageConverter._parse_tool_call_chunks(tool_call_chunks_payload)
        return InvokeStreamChunk(
            content_delta=content_delta,
            tool_call_chunks=tool_call_chunks or None,
            finish_reason=choice.get("finish_reason"),
            usage=chunk.get("usage"),
        )

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if content is None:
            logger.debug("Content is None, returning empty string")
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text = ""
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "unknown")
                    if item_type == "text":
                        text += item.get("text", "")
                    elif item_type == "thinking":
                        # Qwen3 models may use thinking blocks - skip these
                        thinking_text = item.get("thinking", "")
                        logger.debug(f"Skipping thinking block ({len(thinking_text)} chars)")
                        continue
                    else:
                        # Log unknown content block types
                        logger.warning(f"Unknown content block type: {item_type}, item: {item}")
                elif isinstance(item, str):
                    # Handle list of strings
                    text += item
            if not text:
                logger.warning(f"No text content extracted from list ({len(content)} items)")
                # Debug: log first few items
                for i, item in enumerate(content[:3]):
                    if isinstance(item, dict):
                        logger.debug(
                            f"  Item {i}: type={item.get('type')}, keys={list(item.keys())}"
                        )
                    else:
                        logger.debug(f"  Item {i}: {type(item).__name__}")
            return text
        # Unknown type - log and try to convert
        logger.warning(
            f"Unknown content type: {type(content).__name__}, value: {str(content)[:200]}"
        )
        return str(content) if content else ""

    @staticmethod
    def _parse_tool_calls(payload: list[Any]) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for item in payload:
            call = MessageConverter._parse_tool_call(item)
            if call:
                tool_calls.append(call)
        return tool_calls

    @staticmethod
    def _parse_tool_call(item: Any) -> ToolCall | None:
        if not isinstance(item, dict):
            return None
        tool_id = item.get("id")
        fn_payload = item.get("function") or {}
        name = fn_payload.get("name")
        arguments_raw = fn_payload.get("arguments") or "{}"
        if not tool_id or not name:
            return None
        try:
            arguments = json.loads(arguments_raw)
        except json.JSONDecodeError:
            arguments = {"_raw": arguments_raw}
        return ToolCall(id=tool_id, name=name, arguments=arguments)

    @staticmethod
    def _parse_tool_call_chunks(payload: list[Any]) -> list[ToolCallChunk]:
        chunks: list[ToolCallChunk] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            fn_payload = item.get("function") or {}
            chunks.append(
                ToolCallChunk(
                    id=item.get("id"),
                    name=fn_payload.get("name"),
                    arguments_delta=fn_payload.get("arguments") or "",
                    index=item.get("index"),
                )
            )
        return chunks
