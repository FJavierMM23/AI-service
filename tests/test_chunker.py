from ai_service.chunker import split_document
from ai_service.models import Document


def _make_doc(text: str) -> Document:
    return Document(text=text, source="test.md", metadata={"type": "markdown"})


def test_documento_pequeno_genera_un_chunk():
    doc = _make_doc("Un párrafo corto que cabe entero en un chunk.")
    chunks = split_document(doc)
    assert len(chunks) == 1
    assert chunks[0].source == "test.md"


def test_documento_largo_genera_varios_chunks():
    parrafo = "Esta es una frase de relleno para el test. " * 20  # ~880 chars
    doc = _make_doc("\n\n".join([parrafo] * 5))  # ~4400 chars
    chunks = split_document(doc, chunk_size=2000, overlap=300)
    assert len(chunks) > 1


def test_ningun_chunk_supera_el_tamano_maximo():
    parrafo = "Frase de relleno para comprobar límites de tamaño. " * 30
    doc = _make_doc("\n\n".join([parrafo] * 4))
    chunk_size = 1500
    chunks = split_document(doc, chunk_size=chunk_size, overlap=200)
    for chunk in chunks:
        # margen pequeño por el solapamiento añadido al inicio
        assert len(chunk.text) <= chunk_size + 300


def test_los_chunks_conservan_el_orden():
    doc = _make_doc("\n\n".join(f"Párrafo número {i}. " + "relleno " * 50 for i in range(10)))
    chunks = split_document(doc, chunk_size=1000, overlap=100)
    indices = [c.chunk_index for c in chunks]
    assert indices == sorted(indices)