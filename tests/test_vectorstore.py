"""Tests del vector store con ChromaDB.

Cada test usa un tmp_path distinto para tener una BD aislada.
Requieren Ollama corriendo (para los embeddings).
"""
import pytest

from ai_service.embeddings import health_check
from ai_service.models import Chunk
from ai_service.vectorstore import ChromaStore


@pytest.fixture(scope="module", autouse=True)
def _require_ollama():
    if not health_check():
        pytest.skip(
            "Ollama no está disponible o el modelo de embeddings no está descargado.",
            allow_module_level=True,
        )


@pytest.fixture
def store(tmp_path):
    """Un ChromaStore vacío sobre un directorio temporal."""
    return ChromaStore(persist_path=tmp_path / "chroma")


def _mk_chunk(text: str, source: str, index: int, **extra) -> Chunk:
    return Chunk(text=text, source=source, chunk_index=index, metadata=extra)


def test_store_vacio_tiene_count_cero(store):
    assert store.count() == 0
    assert store.list_sources() == {}


def test_search_en_store_vacio_devuelve_lista_vacia(store):
    assert store.search("cualquier cosa") == []


def test_add_chunks_incrementa_count(store):
    chunks = [
        _mk_chunk("El perro es un mamífero doméstico.", "animales.md", 0),
        _mk_chunk("Los volcanes son formaciones geológicas.", "geografia.md", 0),
    ]
    inserted = store.add_chunks(chunks)
    assert inserted == 2
    assert store.count() == 2


def test_search_devuelve_chunk_mas_similar(store):
    store.add_chunks([
        _mk_chunk("El perro es un animal doméstico común.", "animales.md", 0),
        _mk_chunk("El monte Everest es la montaña más alta.", "geografia.md", 0),
        _mk_chunk("La integral de una función mide su área.", "matematicas.md", 0),
    ])
    results = store.search("mascotas y animales", top_k=3)
    assert len(results) == 3
    # El chunk del perro debe estar el primero
    assert results[0].chunk.source == "animales.md"


def test_search_ordena_por_score_descendente(store):
    store.add_chunks([
        _mk_chunk(f"Contenido de prueba número {i}.", "test.md", i)
        for i in range(5)
    ])
    results = store.search("contenido", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_score_esta_entre_cero_y_uno(store):
    store.add_chunks([_mk_chunk("un texto cualquiera", "t.md", 0)])
    results = store.search("otro texto", top_k=1)
    assert 0.0 <= results[0].score <= 1.0


def test_delete_by_source_elimina_solo_ese_documento(store):
    store.add_chunks([
        _mk_chunk("Chunk uno de A.", "a.md", 0),
        _mk_chunk("Chunk dos de A.", "a.md", 1),
        _mk_chunk("Chunk único de B.", "b.md", 0),
    ])
    deleted = store.delete_by_source("a.md")
    assert deleted == 2
    assert store.count() == 1
    assert store.list_sources() == {"b.md": 1}


def test_reingestar_mismo_source_no_duplica(store):
    chunks = [
        _mk_chunk("Contenido idéntico.", "manual.md", 0),
        _mk_chunk("Otro contenido idéntico.", "manual.md", 1),
    ]
    store.add_chunks(chunks)
    store.add_chunks(chunks)  # segunda vez
    assert store.count() == 2


def test_list_sources_agrupa_por_documento(store):
    store.add_chunks([
        _mk_chunk("a1", "a.md", 0),
        _mk_chunk("a2", "a.md", 1),
        _mk_chunk("a3", "a.md", 2),
        _mk_chunk("b1", "b.md", 0),
    ])
    sources = store.list_sources()
    assert sources == {"a.md": 3, "b.md": 1}


def test_reset_vacia_la_coleccion(store):
    store.add_chunks([
        _mk_chunk("algo", "x.md", 0),
        _mk_chunk("otra cosa", "y.md", 0),
    ])
    assert store.count() == 2
    store.reset()
    assert store.count() == 0
    assert store.list_sources() == {}


def test_metadatos_del_chunk_se_preservan(store):
    store.add_chunks([_mk_chunk("texto con página", "doc.pdf", 0, page=42, type="pdf")])
    results = store.search("texto", top_k=1)
    assert results[0].chunk.metadata.get("page") == 42
    assert results[0].chunk.metadata.get("type") == "pdf"