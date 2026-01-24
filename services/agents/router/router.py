"""Router Agent implementation.

This module contains the main RouterAgent class that analyzes user queries
and produces structured search objects for downstream retrieval.
"""

from __future__ import annotations

import asyncio
from typing import Any

from llm_gateway_client import AIGatewayClient
from llm_gateway_client.exceptions import GatewayAPIError
from llm_gateway_client.models import GatewayChatTurn, GatewayInvokeRequest
from loguru import logger
from prompts import (
    build_router_system_prompt,
    build_router_user_prompt,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .helper import parse_router_response
from .models import (
    ChatMessage,
    DomainConfig,
    FilterFieldConfig,
    Intent,
    RouterConfig,
    RouterInput,
    RouterOutput,
    SubQuestion,
)


class RouterAgent:
    """Domain-agnostic Router Agent for query analysis.

    Example:
        ```python
        router = RouterAgent(llm_client)
        result = await router.route(RouterInput(
            user_query="Học phí ngành IT năm nay?",
            domain_config=config,
        ))

        for sq in result.sub_questions:
            print(sq.query, sq.keywords, sq.filters)
        ```
    """

    __slots__ = ("config", "llm_client")

    def __init__(
        self,
        llm_client: AIGatewayClient,
        config: RouterConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or RouterConfig()
        logger.info(
            "RouterAgent initialized",
            llm_model=self.config.llm_model,
            temperature=self.config.temperature,
        )

    async def route(self, input: RouterInput) -> RouterOutput:
        """Route a user query to produce structured search object."""
        logger.info(
            "Routing query",
            query=input.user_query[:100],
            domain=input.domain_config.domain_name,
            history_length=len(input.conversation_history),
        )

        # Build prompts
        system_prompt = build_router_system_prompt(
            domain_name=input.domain_config.domain_name,
            filter_fields=input.domain_config.get_filter_fields_dict(),
            current_year=input.current_year,
            custom_intent_descriptions=input.domain_config.custom_intent_descriptions,
        )
        user_prompt = build_router_user_prompt(
            user_query=input.user_query,
            conversation_history=input.get_history_as_dicts(),
        )

        # Build LLM request
        request = GatewayInvokeRequest(
            model=self.config.llm_model,
            messages=[
                GatewayChatTurn(role="system", content=system_prompt),
                GatewayChatTurn(role="user", content=user_prompt),
            ],
            provider_options={"temperature": self.config.temperature},
        )

        # Invoke LLM
        try:
            response = await self._invoke_with_retry(request)
            raw_content = response.message.content if response.message else ""
            logger.debug("LLM response received", content_length=len(raw_content))
        except Exception as e:
            logger.error("LLM invocation failed: {}", str(e))
            return RouterOutput.create_chit_chat_response(input.user_query)

        # Parse and build output
        parsed = parse_router_response(raw_content, input.user_query)
        output = RouterOutput(
            intent=Intent(parsed["intent"]),
            rewritten_query=parsed["rewritten_query"],
            sub_questions=[SubQuestion(**sq) for sq in parsed["sub_questions"]],
            raw_llm_response=raw_content,
        )

        logger.info(
            "Query routed",
            intent=output.intent.value,
            sub_questions_count=len(output.sub_questions),
        )
        return output

    @retry(
        retry=retry_if_exception_type((GatewayAPIError, Exception)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=lambda rs: logger.warning(
            "Retrying LLM call, attempt={}",
            rs.attempt_number,
        ),
    )
    async def _invoke_with_retry(self, request: GatewayInvokeRequest):
        """Invoke LLM with exponential backoff retry."""
        return await self.llm_client.invoke(request)

    async def route_batch(
        self,
        inputs: list[RouterInput],
        max_concurrency: int = 10,
    ) -> list[RouterOutput]:
        """Route multiple queries with concurrency control."""
        logger.info("Routing batch", size=len(inputs), max_concurrency=max_concurrency)

        semaphore = asyncio.Semaphore(max_concurrency)

        async def route_with_semaphore(inp: RouterInput) -> RouterOutput:
            async with semaphore:
                return await self.route(inp)

        results = await asyncio.gather(
            *[route_with_semaphore(inp) for inp in inputs],
            return_exceptions=True,
        )

        outputs: list[RouterOutput] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch item {} failed: {}", i, str(result))
                outputs.append(RouterOutput.create_chit_chat_response(inputs[i].user_query))
            else:
                outputs.append(result)

        return outputs


async def route_query(
    llm_client: AIGatewayClient,
    user_query: str,
    domain_config: dict[str, Any],
    conversation_history: list[dict[str, str]] | None = None,
) -> RouterOutput:
    """Convenience function to route a single query from dict config."""
    # Build domain config
    filter_fields = {
        name: FilterFieldConfig(**cfg) if isinstance(cfg, dict) else cfg
        for name, cfg in domain_config.get("filter_fields", {}).items()
    }
    config = DomainConfig(
        domain_name=domain_config.get("domain_name", "General"),
        filter_fields=filter_fields,
        custom_intent_descriptions=domain_config.get("custom_intent_descriptions"),
    )

    # Build history
    history = [ChatMessage(**msg) for msg in (conversation_history or [])]

    # Route
    router = RouterAgent(llm_client)
    return await router.route(
        RouterInput(
            user_query=user_query,
            conversation_history=history,
            domain_config=config,
        )
    )
