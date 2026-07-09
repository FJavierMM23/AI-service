"""Pipeline de ingesta: documento(s) → chunks → vector store.

Coordina loaders, chunker y vectorstore. Es la capa de negocio
reutilizable tanto por la CLI como por la futura API REST.
"""
from dataclasses import dataclass
from pathlib import Path

from ai_service.chunker import split_document
from ai_service.loaders import load_directory, load_document
from ai_service.vectorstore import get_store


@dataclass
class IngestionReport:
    """Resumen del resultado de una operación de ingesta."""
    documents_processed: int
    chunks_created: int
    sources: list[str]
    elapsed_seconds: float


def ingest_path(
    path: str | Path,
    chunk_size: int = 2000,
    overlap: int = 300,
    replace_existing: bool = True,
) -> IngestionReport:
    """Ingesta un fichero o un directorio completo en el vector store.
    
    Flujo por cada documento:
      1. Cargar con el loader correspondiente (según extensión).
      2. Trocear con split_document.
      3. Si replace_existing, borrar chunks previos del mismo source.
      4. Insertar chunks en el vector store (vectorización incluida).
    
    Args:
        path: Ruta a un fichero soportado o a un directorio (se recorre recursivo).
        chunk_size: Tamaño máximo de chunk en caracteres.
        overlap: Solapamiento entre chunks consecutivos.
        replace_existing: Si True, borra chunks previos del mismo source antes de insertar.
    
    Returns:
        IngestionReport con el resumen.
    
    Raises:
        FileNotFoundError: Si path no existe.
        ValueError: Si no se encuentra ningún documento soportado.
    """
    ...