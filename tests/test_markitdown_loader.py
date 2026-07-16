"""Tests del loader MarkItDown.

No requieren Ollama: la conversión es local y pura.
"""
import pytest

from ai_service.loaders import load_document
from ai_service.loaders.markitdown_loader import load_with_markitdown


def test_fichero_inexistente_lanza_filenotfound():
    with pytest.raises(FileNotFoundError):
        load_with_markitdown("no_existe.docx")


def test_extension_desconocida_lanza_valueerror(tmp_path):
    raro = tmp_path / "fichero.xyz"
    raro.write_text("contenido")
    with pytest.raises(ValueError):
        load_document(raro)


def test_html_se_convierte_a_markdown(tmp_path):
    html = tmp_path / "pagina.html"
    html.write_text(
        "<html><body><h1>Título principal</h1>"
        "<p>Un párrafo con <b>negrita</b>.</p>"
        "<ul><li>elemento uno</li><li>elemento dos</li></ul>"
        "</body></html>"
    )
    doc = load_document(html)
    assert "Título principal" in doc.text
    assert "elemento uno" in doc.text
    assert doc.metadata["converted_by"] == "markitdown"
    assert doc.source == "pagina.html"


def test_imagenes_base64_se_eliminan(tmp_path):
    html = tmp_path / "con_imagen.html"
    html.write_text(
        '<html><body><h1>Doc</h1>'
        '<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==">'
        '<p>Texto útil.</p></body></html>'
    )
    doc = load_document(html)
    assert "base64" not in doc.text
    assert "Texto útil" in doc.text