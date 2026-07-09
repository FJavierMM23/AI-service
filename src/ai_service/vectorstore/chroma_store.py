"""Almacén vectorial basado en ChromaDB.

Encapsula toda la interacción con Chroma para que el resto del código
trabaje solo con los tipos del dominio (Chunk, SearchResult).
"""
import hashlib
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from ai_service.config import settings
from ai_service.embeddings import embed_batch, embed_text
from ai_service.models import Chunk, SearchResult


COLLECTION_NAME = "manuales"


class ChromaStore:
    """Wrapper sobre una colección de ChromaDB.

    Reglas de uso:
      - Instancia única por proceso (usa get_store()).
      - Los métodos add_chunks/search llaman a Ollama para vectorizar.
      - Los IDs de chunks son deterministas para permitir re-ingesta idempotente.
    """

    def __init__(self, persist_path: str | Path, collection_name: str = COLLECTION_NAME):
        """Abre (o crea) la BD en disco y la colección indicada."""
        persist_path = Path(persist_path)
        persist_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Vectoriza y persiste una lista de chunks.

        Antes de insertar, borra cualquier chunk previo con el mismo ID
        para que re-ingerir un documento no genere duplicados.
        """
        if not chunks:
            return 0

        ids = [self._make_chunk_id(c) for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._build_metadata(c) for c in chunks]

        # Idempotencia: borramos primero por si ya existían con esos IDs.
        # Chroma no lanza si el ID no existe, así que es seguro.
        self._collection.delete(ids=ids)

        embeddings = embed_batch(documents)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Busca los top_k chunks más similares a la query."""
        if top_k <= 0:
            return []
        if self.count() == 0:
            return []

        query_vector = embed_text(query)
        # No pedimos más resultados de los que hay
        effective_k = min(top_k, self.count())

        raw = self._collection.query(
            query_embeddings=[query_vector],
            n_results=effective_k,
        )

        # Chroma devuelve listas anidadas [ [resultados_query_1], [resultados_query_2], ... ]
        # Como pedimos una sola query, cogemos siempre el índice 0.
        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        results: list[SearchResult] = []
        for _id, text, meta, dist in zip(ids, documents, metadatas, distances):
            chunk = Chunk(
                text=text,
                source=meta.get("source", "unknown"),
                chunk_index=int(meta.get("chunk_index", 0)),
                metadata=self._extract_extra_metadata(meta),
            )
            score = self._distance_to_score(dist)
            results.append(SearchResult(chunk=chunk, score=score, distance=dist))

        return results

    def delete_by_source(self, source: str) -> int:
        """Elimina todos los chunks cuyo campo source coincida."""
        # Contamos antes para poder devolver el número
        matches = self._collection.get(where={"source": source})
        num = len(matches["ids"])
        if num > 0:
            self._collection.delete(where={"source": source})
        return num

    def list_sources(self) -> dict[str, int]:
        """Devuelve un dict {nombre_documento: número_de_chunks}."""
        all_items = self._collection.get()  # sin filtro = todo
        counter: Counter[str] = Counter()
        for meta in all_items["metadatas"]:
            source = meta.get("source", "unknown")
            counter[source] += 1
        return dict(counter)

    def count(self) -> int:
        """Número total de chunks en la colección."""
        return self._collection.count()

    def reset(self) -> None:
        """Vacía la colección entera. Irreversible."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # --- Métodos internos ---

    @staticmethod
    def _make_chunk_id(chunk: Chunk) -> str:
        """Genera un ID determinista basado en source, índice y hash del inicio del texto.

        La combinación source + chunk_index sería suficiente en teoría, pero
        añadimos hash del texto para que un cambio de contenido genere un ID nuevo.
        """
        text_head = chunk.text[:200]
        payload = f"{chunk.source}::{chunk.chunk_index}::{text_head}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{chunk.source}::{chunk.chunk_index}::{digest[:16]}"

    @staticmethod
    def _build_metadata(chunk: Chunk) -> dict:
        """Aplana los metadatos del chunk al formato que Chroma acepta.

        Chroma solo admite valores escalares (str, int, float, bool) en metadatas.
        Aquí extraemos lo importante y descartamos estructuras anidadas.
        """
        meta: dict = {
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
        }
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                meta[key] = value
        return meta

    @staticmethod
    def _extract_extra_metadata(meta: dict) -> dict:
        """Devuelve los metadatos originales del chunk sin source ni chunk_index."""
        return {k: v for k, v in meta.items() if k not in ("source", "chunk_index")}

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        """Convierte distancia coseno (0-2) a un score de similitud (0-1).

        distance=0 → score=1 (idénticos)
        distance=1 → score=0.5 (ortogonales)
        distance=2 → score=0 (opuestos)
        """
        score = 1.0 - (distance / 2.0)
        # Clamp por si Chroma devuelve valores fuera de rango por precisión numérica.
        return max(0.0, min(1.0, score))


# --- Singleton lazy ---

_store: ChromaStore | None = None


def get_store() -> ChromaStore:
    """Devuelve la instancia global del store (singleton lazy).

    Se inicializa la primera vez que se llama, usando settings.chroma_path.
    """
    global _store
    if _store is None:
        _store = ChromaStore(persist_path=settings.chroma_path)
    return _store