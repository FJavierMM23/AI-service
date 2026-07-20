"""Aplicación FastAPI del ai-service."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_service.api.routes import documents, health, query, models
from ai_service.embeddings import EmbeddingError, health_check
from ai_service.llm import GenerationError


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Arranque: avisar si Ollama no está accesible (no impedimos arrancar;
    # el healthcheck reflejará el estado degradado).
    if health_check():
        print("✓ Ollama accesible. ai-service listo.")
    else:
        print("⚠ Ollama NO accesible. El servicio arrancará en modo degradado.")
    yield
    print("ai-service detenido.")


app = FastAPI(
    title="AI Service",
    description="Servicio RAG local: consulta documentos indexados usando Ollama.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(models.router)


# --- Manejadores globales de excepciones (el @ControllerAdvice) ---

@app.exception_handler(EmbeddingError)
def embedding_error_handler(request: Request, exc: EmbeddingError):
    return JSONResponse(
        status_code=503,
        content={"detail": f"Servicio de embeddings no disponible: {exc}"},
    )


@app.exception_handler(GenerationError)
def generation_error_handler(request: Request, exc: GenerationError):
    return JSONResponse(
        status_code=503,
        content={"detail": f"Servicio de generación no disponible: {exc}"},
    )


@app.exception_handler(ValueError)
def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )