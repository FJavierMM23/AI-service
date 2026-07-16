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


def test_documento_con_encabezados_trocea_por_secciones():
    texto = (
        "# Título del documento\n\n"
        "Introducción breve.\n\n"
        "## Sección A\n\n"
        "Contenido de la sección A.\n\n"
        "## Sección B\n\n"
        "Contenido de la sección B."
    )
    doc = Document(text=texto, source="doc.md", metadata={"type": "markdown"})
    chunks = split_document(doc)

    secciones = [c.metadata["section"] for c in chunks]
    assert "Sección A" in secciones
    assert "Sección B" in secciones


def test_titulo_de_seccion_prependido_al_texto():
    texto = "## Memoria virtual\n\nLa memoria virtual permite abstraer la RAM."
    doc = Document(text=texto, source="doc.md", metadata={})
    chunks = split_document(doc)
    assert chunks[0].text.startswith("## Memoria virtual")


def test_encabezados_dentro_de_codigo_no_crean_secciones():
    texto = (
        "## Única sección\n\n"
        "Texto con un bloque de código:\n\n"
        "```bash\n"
        "# esto es un comentario, NO un encabezado\n"
        "echo hola\n"
        "```\n\n"
        "Más texto de la misma sección."
    )
    doc = Document(text=texto, source="doc.md", metadata={})
    chunks = split_document(doc)
    secciones = {c.metadata["section"] for c in chunks}
    assert secciones == {"Única sección"}


def test_seccion_grande_se_subdivide():
    parrafo = "Frase de relleno para forzar la subdivisión de la sección. " * 10
    texto = "## Sección enorme\n\n" + "\n\n".join([parrafo] * 8)
    doc = Document(text=texto, source="doc.md", metadata={})
    chunks = split_document(doc, chunk_size=1000, overlap=100)

    assert len(chunks) > 1
    # Todos los sub-chunks conservan la sección y el título prependido
    for c in chunks:
        assert c.metadata["section"] == "Sección enorme"
        assert c.text.startswith("## Sección enorme")


def test_texto_sin_encabezados_usa_chunker_plano():
    texto = "Un texto plano sin ningún encabezado.\n\nOtro párrafo normal."
    doc = Document(text=texto, source="doc.txt", metadata={})
    chunks = split_document(doc)
    # El chunker plano añade 'page', el markdown añade 'section'
    assert "page" in chunks[0].metadata
    assert "section" not in chunks[0].metadata