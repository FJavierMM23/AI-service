"""DTOs de la API (modelos Pydantic de request/response).

Separados de los modelos de dominio (models.py): la API expone una vista
controlada, no las estructuras internas.
"""
from pydantic import BaseModel, Field


# ---------- /query ----------

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)


class SourceInfo(BaseModel):
    source: str
    page: int | None = None
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    question: str
    model: str


# ---------- /documents ----------

class DocumentInfo(BaseModel):
    source: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class UploadResponse(BaseModel):
    source: str
    chunks_created: int
    elapsed_seconds: float


class DeleteResponse(BaseModel):
    source: str
    deleted_chunks: int


# ---------- /health ----------

class HealthResponse(BaseModel):
    status: str                 # "ok" | "degraded"
    ollama_available: bool
    total_chunks: int
    llm_model: str
    embedding_model: str