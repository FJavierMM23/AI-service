"""Tests de la API REST.

Usan TestClient (sin levantar servidor). Requieren Ollama para los
endpoints que tocan embeddings/LLM.
"""
import pytest
from fastapi.testclient import TestClient

from ai_service.api.app import app
from ai_service.embeddings import health_check

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _require_ollama():
    if not health_check():
        pytest.skip("Ollama no está disponible.", allow_module_level=True)


def test_health_devuelve_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ollama_available"] is True


def test_query_pregunta_vacia_devuelve_422():
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_top_k_fuera_de_rango_devuelve_422():
    response = client.post("/query", json={"question": "hola", "top_k": 999})
    assert response.status_code == 422


def test_query_devuelve_estructura_correcta():
    response = client.post("/query", json={"question": "¿qué es CAS?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert isinstance(body["sources"], list)


def test_list_documents_devuelve_estructura_correcta():
    response = client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert "documents" in body
    assert "total_chunks" in body


def test_delete_documento_inexistente_devuelve_404():
    response = client.delete("/documents/no-existe-para-nada.pdf")
    assert response.status_code == 404


def test_upload_extension_no_soportada_devuelve_422():
    response = client.post(
        "/documents",
        files={"file": ("malware.exe", b"contenido", "application/octet-stream")},
    )
    assert response.status_code == 422