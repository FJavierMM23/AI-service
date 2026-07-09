"""Almacén vectorial basado en ChromaDB.

Encapsula toda la interacción con Chroma para que el resto del código
trabaje solo con los tipos del dominio (Chunk, SearchResult).
"""
import hashlib
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
        """Abre (o crea) la BD en disco y la colección indicada.
        
        Args:
            persist_path: Directorio donde ChromaDB guarda los datos.
            collection_name: Nombre lógico de la colección (equivalente a una tabla).
        """
        ...
    
    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Vectoriza y persiste una lista de chunks.
        
        Antes de insertar, borra cualquier chunk previo con el mismo (source, chunk_index)
        para que re-ingerir un documento no genere duplicados.
        
        Args:
            chunks: Lista de Chunk. Si está vacía, no hace nada.
        
        Returns:
            Número de chunks efectivamente insertados.
        """
        ...
    
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Busca los top_k chunks más similares a la query.
        
        Args:
            query: Texto de búsqueda en lenguaje natural.
            top_k: Máximo de resultados a devolver.
        
        Returns:
            Lista de SearchResult ordenada de mayor a menor score.
            Puede tener menos de top_k elementos si la BD tiene pocos chunks.
        """
        ...
    
    def delete_by_source(self, source: str) -> int:
        """Elimina todos los chunks cuyo campo source coincida.
        
        Returns:
            Número de chunks eliminados.
        """
        ...
    
    def list_sources(self) -> dict[str, int]:
        """Devuelve un dict {nombre_documento: número_de_chunks}."""
        ...
    
    def count(self) -> int:
        """Número total de chunks en la colección."""
        ...
    
    def reset(self) -> None:
        """Vacía la colección entera. Irreversible."""
        ...
    
    # --- Métodos internos ---
    
    @staticmethod
    def _make_chunk_id(chunk: Chunk) -> str:
        """Genera un ID determinista basado en source, índice y hash del inicio del texto."""
        ...


def get_store() -> ChromaStore:
    """Devuelve la instancia global del store (singleton lazy).
    
    Se inicializa la primera vez que se llama, usando settings.chroma_path.
    """
    ...