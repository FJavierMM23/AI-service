"""Endpoint para listar los modelos LLM disponibles en Ollama."""
from fastapi import APIRouter, HTTPException

from ai_service.config import settings
from ai_service.llm.ollama_client import GenerationError, list_models

router = APIRouter(tags=["models"])


@router.get("/models")
def get_models():
    """Modelos de generación descargados en Ollama (excluye el de embeddings)."""
    try:
        todos = list_models()
    except GenerationError as e:
        raise HTTPException(status_code=503, detail=str(e))

    generacion = [m for m in todos if m != settings.embedding_model]

    return {
        "models": generacion,
        "default": settings.llm_model,
    }