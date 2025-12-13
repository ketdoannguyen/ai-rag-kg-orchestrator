"""Router Agent package.

Domain-agnostic Router Agent for query analysis in RAG systems.

Example:
    ```python
    from services.agents.router import RouterAgent, RouterInput, DomainConfig
    
    router = RouterAgent(llm_client)
    result = await router.route(RouterInput(
        user_query="Học phí ngành IT năm nay?",
        domain_config=DomainConfig(domain_name="Tuyển sinh", filter_fields={...}),
    ))
    
    for sq in result.sub_questions:
        print(sq.query, sq.keywords, sq.filters)
    ```
"""

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
from .router import RouterAgent, route_query

__all__ = [
    "RouterAgent",
    "route_query",
    "RouterInput",
    "RouterOutput",
    "RouterConfig",
    "DomainConfig",
    "FilterFieldConfig",
    "ChatMessage",
    "Intent",
    "SubQuestion",
]
