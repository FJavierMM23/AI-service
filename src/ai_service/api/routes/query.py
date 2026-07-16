"""Endpoint de consulta RAG."""
from fastapi import APIRouter

from ai_service.api.schemas import QueryRequest, QueryResponse, SourceInfo
from ai_service.rag import ask

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Hace una pregunta y responde usando los documentos indexados."""
    result = ask(
        question=request.question,
        top_k=request.top_k,
        min_score=request.min_score,
    )

    sources = [
        SourceInfo(
            source=r.chunk.source,
            page=r.chunk.metadata.get("page"),
            chunk_index=r.chunk.chunk_index,
            score=round(r.score, 4),
        )
        for r in result.sources
    ]

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        question=result.question,
        model=result.model,
    )