"""Endpoint de estado del servicio."""
from fastapi import APIRouter

from ai_service.api.schemas import HealthResponse
from ai_service.config import settings
from ai_service.embeddings import health_check
from ai_service.vectorstore import get_store

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    """Estado del servicio: Ollama accesible y tamaño del índice."""
    ollama_ok = health_check()
    total = get_store().count() if ollama_ok else 0

    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama_available=ollama_ok,
        total_chunks=total,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )