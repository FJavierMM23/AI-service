from dataclasses import dataclass, field


@dataclass
class Document:
    """Un documento cargado desde disco, antes de trocear."""
    text: str                    # Contenido completo en texto plano
    source: str                  # Nombre del archivo (ej: "manual_docker.pdf")
    metadata: dict = field(default_factory=dict)  # Info extra (nº páginas, etc.)


@dataclass
class Chunk:
    """Un fragmento de un documento, listo para vectorizar."""
    text: str                    # El contenido del fragmento
    source: str                  # De qué documento viene
    chunk_index: int             # Posición del chunk dentro del documento (0, 1, 2...)
    metadata: dict = field(default_factory=dict)  # Página de origen, etc.


@dataclass
class SearchResult:
    """Un resultado de búsqueda semántica: chunk + métricas de similitud."""
    chunk: Chunk
    score: float          # Similitud normalizada 0-1 (más alto = más parecido)
    distance: float       # Distancia bruta devuelta por la BD vectorial