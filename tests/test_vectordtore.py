"""Tests del vector store con ChromaDB.

Cada test usa un tmp_path distinto para tener una BD aislada.
También requieren Ollama corriendo (para los embeddings).
"""
import pytest

from ai_service.models import Chunk
from ai_service.vectorstore import ChromaStore


@pytest.fixture
def store(tmp_path):
    """Un ChromaStore vacío sobre un directorio temporal."""
    ...


def test_store_vacio_tiene_count_cero(store):
    ...


def test_add_chunks_incrementa_count(store):
    ...


def test_search_devuelve_chunk_mas_similar(store):
    ...


def test_search_ordena_por_score_descendente(store):
    ...


def test_delete_by_source_elimina_solo_ese_documento(store):
    ...


def test_reingestar_mismo_source_no_duplica(store):
    ...


def test_list_sources_agrupa_por_documento(store):
    ...


def test_reset_vacia_la_coleccion(store):
    ...