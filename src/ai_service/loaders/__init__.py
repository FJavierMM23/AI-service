from pathlib import Path

from ai_service.loaders.markdown_loader import load_markdown
from ai_service.loaders.markitdown_loader import load_with_markitdown
from ai_service.loaders.pdf_loader import load_pdf
from ai_service.models import Document

# Extensiones y su loader correspondiente.
# - Markdown/texto plano: lectura directa (no hay nada que convertir).
# - PDF: de momento pypdf; pendiente comparativa con markitdown (ver roadmap).
# - Formatos ofimáticos: MarkItDown los convierte a Markdown.
LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_markdown,
    ".pdf": load_pdf,
    ".docx": load_with_markitdown,
    ".pptx": load_with_markitdown,
    ".xlsx": load_with_markitdown,
    ".html": load_with_markitdown,
    ".htm": load_with_markitdown,
}


def load_document(path: str | Path) -> Document:
    """Carga cualquier documento soportado, eligiendo el loader por extensión."""
    path = Path(path)
    suffix = path.suffix.lower()

    loader = LOADERS.get(suffix)
    if loader is None:
        supported = ", ".join(sorted(LOADERS.keys()))
        raise ValueError(
            f"Extensión no soportada: '{suffix}'. Soportadas: {supported}"
        )

    return loader(path)


def load_directory(path: str | Path) -> list[Document]:
    """Carga todos los documentos soportados de un directorio (recursivo)."""
    path = Path(path)
    documents = []

    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix.lower() in LOADERS:
            documents.append(load_document(file))

    return documents