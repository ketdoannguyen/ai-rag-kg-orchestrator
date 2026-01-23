"""Chunking configuration for RAG optimization."""

# Maximum characters per chunk segment.
# LLM should merge semantically related texts to approach this limit,
# but NEVER merge unrelated texts just to reach max.
MAX_CHUNK_SIZE: int = 1024
