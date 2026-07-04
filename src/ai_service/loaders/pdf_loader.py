from pathlib import Path

from pypdf import PdfReader

from ai_service.models import Document


def load_pdf(path: str | Path) -> Document:
    """Carga un PDF y devuelve un Document con el texto de todas las páginas.

    Inserta marcadores [page:N] entre páginas para poder rastrear
    de qué página viene cada chunk más adelante.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero: {path}")

    reader = PdfReader(path)
    pages_text = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:  # Ignoramos páginas vacías (portadas en blanco, etc.)
            pages_text.append(f"[page:{i + 1}]\n{page_text}")

    full_text = "\n\n".join(pages_text)

    return Document(
        text=full_text,
        source=path.name,
        metadata={
            "type": "pdf",
            "path": str(path),
            "num_pages": len(reader.pages),
        },
    )