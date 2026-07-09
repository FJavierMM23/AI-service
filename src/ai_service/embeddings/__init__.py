from ai_service.embeddings.ollama_embedder import (
    EmbeddingError,
    embed_batch,
    embed_text,
    embedding_dimension,
    health_check,
)

__all__ = [
    "EmbeddingError",
    "embed_batch",
    "embed_text",
    "embedding_dimension",
    "health_check",
]