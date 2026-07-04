from pathlib import Path

from ai_service.loaders.markdown_loader import load_markdown
from ai_service.loaders.pdf_loader import load_pdf
from ai_service.models import Document

# Extensiones soportadas y su loader correspondiente
LOADERS = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_markdown,  # el texto plano se carga igual que el markdown
}


def load_document(path: str | Path) -> Document:
    """Carga cualquier documento soportado, eligiendo el loader por extensión."""
    path = Path(path)
    suffix = path.suffix.lower()

    loader = LOADERS.get(suffix)
    if loader is None:
        supported = ", ".join(LOADERS.keys())
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