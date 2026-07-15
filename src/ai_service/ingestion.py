"""Pipeline de ingesta: documento(s) → chunks → vector store.

Coordina loaders, chunker y vectorstore. Es la capa de negocio
reutilizable tanto por la CLI como por la futura API REST.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

from ai_service.chunker import split_document
from ai_service.loaders import load_directory, load_document
from ai_service.models import Document
from ai_service.vectorstore import ChromaStore, get_store


@dataclass
class IngestionReport:
    """Resumen del resultado de una operación de ingesta."""
    documents_processed: int = 0
    chunks_created: int = 0
    sources: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def ingest_path(
    path: str | Path,
    chunk_size: int = 1000,
    overlap: int = 150,
    replace_existing: bool = True,
    store: ChromaStore | None = None,
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
        store: Store a usar. Por defecto, la instancia global (get_store()).
               Útil pasar uno propio en tests.

    Returns:
        IngestionReport con el resumen.

    Raises:
        FileNotFoundError: Si path no existe.
        ValueError: Si no se encuentra ningún documento soportado.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"No existe la ruta: {target}")

    if store is None:
        store = get_store()

    documents = _load_documents(target)
    if not documents:
        raise ValueError(f"No se encontraron documentos soportados en: {target}")

    report = IngestionReport()
    started = time.perf_counter()

    for doc in documents:
        chunks = split_document(doc, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            continue

        if replace_existing:
            store.delete_by_source(doc.source)

        inserted = store.add_chunks(chunks)

        report.documents_processed += 1
        report.chunks_created += inserted
        report.sources.append(doc.source)

    report.elapsed_seconds = time.perf_counter() - started
    return report


def _load_documents(target: Path) -> list[Document]:
    """Carga los documentos apuntados por target (fichero o directorio)."""
    if target.is_dir():
        return load_directory(target)
    return [load_document(target)]