"""Tests del pipeline RAG.

Requieren Ollama corriendo con ambos modelos (embeddings + LLM).
Los tests que llaman al LLM tardan varios segundos cada uno: es normal.
"""
import pytest

from ai_service.embeddings import health_check
from ai_service.models import Chunk
from ai_service.rag import ask
from ai_service.rag.prompts import NO_CONTEXT_ANSWER
from ai_service.vectorstore import ChromaStore


@pytest.fixture(scope="module", autouse=True)
def _require_ollama():
    if not health_check():
        pytest.skip(
            "Ollama no está disponible.",
            allow_module_level=True,
        )


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_path=tmp_path / "chroma")


@pytest.fixture
def store_con_datos(store):
    """Store con un chunk de contenido conocido y verificable."""
    store.add_chunks([
        Chunk(
            text=(
                "El algoritmo Compare-And-Swap (CAS) es una instrucción atómica "
                "de hardware que compara el valor de una posición de memoria con "
                "un valor esperado y, solo si coinciden, lo sustituye por uno nuevo. "
                "En Java lo utilizan las clases del paquete java.util.concurrent.atomic, "
                "como AtomicInteger y AtomicLong."
            ),
            source="concurrencia.md",
            chunk_index=0,
            metadata={"page": 3},
        ),
    ])
    return store


def test_ask_pregunta_vacia_lanza_valueerror(store):
    with pytest.raises(ValueError):
        ask("", store=store)
    with pytest.raises(ValueError):
        ask("   ", store=store)


def test_ask_sin_documentos_devuelve_no_context_sin_fuentes(store):
    result = ask("¿qué es un semáforo?", store=store)
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []


def test_ask_con_documento_relevante_incluye_fuentes(store_con_datos):
    result = ask("¿qué es el algoritmo CAS?", store=store_con_datos)
    assert result.answer != NO_CONTEXT_ANSWER
    assert len(result.sources) >= 1
    assert result.sources[0].chunk.source == "concurrencia.md"


def test_ask_respuesta_contiene_informacion_del_contexto(store_con_datos):
    """La respuesta debería mencionar conceptos del chunk (test laxo:
    no comprobamos redacción exacta porque el LLM no es determinista)."""
    result = ask("¿qué clases de Java usan CAS?", store=store_con_datos)
    answer_lower = result.answer.lower()
    assert "atomic" in answer_lower


def test_ask_min_score_imposible_devuelve_no_context(store_con_datos):
    result = ask("¿qué es CAS?", min_score=0.99, store=store_con_datos)
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []


def test_ask_pregunta_fuera_de_dominio_no_inventa(store_con_datos):
    """Pregunta sobre algo que NO está en los documentos. Con el filtro
    de score debería devolver no-context; si algún chunk pasa el filtro,
    al menos el LLM no debería responder la pregunta con datos externos."""
    result = ask("¿cuál es la capital de Mongolia?", store=store_con_datos)
    assert "ulán bator" not in result.answer.lower()
    assert "ulan bator" not in result.answer.lower()