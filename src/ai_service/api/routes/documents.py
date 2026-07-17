"""Endpoints de gestión de documentos."""
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, Form

import json

from ai_service.api.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    UploadResponse,
)
from ai_service.config import settings
from ai_service.ingestion import ingest_path
from ai_service.vectorstore import get_store

router = APIRouter(prefix="/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = (".pdf", ".md", ".markdown", ".txt", ".docx", ".pptx", ".xlsx", ".html", ".htm")


@router.get("", response_model=DocumentListResponse)
def list_documents():
    """Lista los documentos indexados con su número de chunks."""
    store = get_store()
    sources = store.list_sources()
    documents = [
        DocumentInfo(source=src, chunks=n)
        for src, n in sorted(sources.items())
    ]
    return DocumentListResponse(documents=documents, total_chunks=store.count())


@router.post("", response_model=UploadResponse, status_code=201)
def upload_document(file: UploadFile, metadata: str | None = Form(None)):
    """Sube un fichero, lo guarda en el directorio de datos y lo indexa."""
    extra = json.loads(metadata) if metadata else None
    if not file.filename:
        raise HTTPException(status_code=422, detail="El fichero no tiene nombre.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Extensión '{suffix}' no soportada. "
                   f"Soportadas: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Guardamos el fichero en el directorio de datos (el mismo que usa la CLI).
    data_dir = Path(settings.documents_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    # Path().name elimina cualquier ruta que venga en el nombre (seguridad básica).
    dest = data_dir / Path(file.filename).name

    content = file.file.read()
    dest.write_bytes(content)

    report = ingest_path(dest, extra_metadata=extra)

    return UploadResponse(
        source=dest.name,
        chunks_created=report.chunks_created,
        elapsed_seconds=round(report.elapsed_seconds, 2),
    )


@router.delete("/{source}", response_model=DeleteResponse)
def delete_document(source: str):
    """Borra todos los chunks de un documento del índice."""
    store = get_store()
    deleted = store.delete_by_source(source)
    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No existe ningún documento indexado con nombre '{source}'.",
        )
    return DeleteResponse(source=source, deleted_chunks=deleted)