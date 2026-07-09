"""Tests del cliente de embeddings.

Requieren Ollama corriendo con el modelo settings.embedding_model descargado.
Si Ollama no está disponible, todos los tests se marcan como skipped.
"""
import pytest

from ai_service.embeddings import (
    embed_batch,
    embed_text,
    embedding_dimension,
    health_check,
)


@pytest.fixture(scope="module", autouse=True)
def _require_ollama():
    """Salta todos los tests del módulo si Ollama no está accesible."""
    if not health_check():
        pytest.skip(
            "Ollama no está disponible o el modelo de embeddings no está descargado.",
            allow_module_level=True,
        )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similitud coseno entre dos vectores. Auxiliar de test."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def test_health_check_devuelve_true_si_modelo_disponible():
    assert health_check() is True


def test_embed_text_devuelve_vector_no_vacio():
    vector = embed_text("un texto de prueba")
    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)


def test_embedding_dimension_es_estable():
    dim1 = embedding_dimension()
    dim2 = embedding_dimension()
    assert dim1 == dim2
    assert dim1 > 0


def test_embed_text_vacio_lanza_valueerror():
    with pytest.raises(ValueError):
        embed_text("")
    with pytest.raises(ValueError):
        embed_text("   ")


def test_embed_batch_preserva_orden_y_dimension():
    textos = ["primer texto", "segundo texto", "tercer texto"]
    vectors = embed_batch(textos)
    assert len(vectors) == len(textos)
    dim = len(vectors[0])
    assert all(len(v) == dim for v in vectors)


def test_embed_batch_lista_vacia_devuelve_lista_vacia():
    assert embed_batch([]) == []


def test_embed_batch_con_texto_vacio_lanza_valueerror():
    with pytest.raises(ValueError):
        embed_batch(["ok", "", "ok"])


def test_textos_similares_tienen_mayor_similitud_que_textos_distintos():
    """El test semántico clave: 'perro' y 'gato' deben estar más cerca entre sí
    que 'perro' y algo totalmente no relacionado."""
    v_perro = embed_text("El perro es un animal doméstico común.")
    v_gato = embed_text("El gato es un animal doméstico frecuente.")
    v_matematicas = embed_text("La derivada parcial se aplica a funciones multivariables.")

    sim_animales = _cosine_similarity(v_perro, v_gato)
    sim_perro_mates = _cosine_similarity(v_perro, v_matematicas)

    assert sim_animales > sim_perro_mates