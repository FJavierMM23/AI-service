"""Loader universal basado en MarkItDown.

Convierte formatos ofimáticos y PDF a Markdown antes de devolverlos
como Document. Al normalizar todo a Markdown, el chunker trabaja
siempre sobre texto estructurado.
"""
from pathlib import Path

from markitdown import MarkItDown

import re

from ai_service.models import Document

# Instancia única del conversor (su creación tiene coste; es reutilizable).
_converter = MarkItDown()

# Imágenes markdown con data-URI embebido: ![alt](data:image/...;base64,....)
_BASE64_IMAGE = re.compile(r"!\[[^\]]*\]\(data:image/[^)]+\)")


def load_with_markitdown(path: str | Path) -> Document:
    """Convierte un fichero a Markdown usando MarkItDown y devuelve un Document.

    Args:
        path: Ruta al fichero (pdf, docx, pptx, xlsx, html...).

    Returns:
        Document con el contenido en Markdown.

    Raises:
        FileNotFoundError: Si no existe el fichero.
        ValueError: Si la conversión no produce texto (fichero vacío,
                    corrupto o formato no convertible).
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero: {path}")

    result = _converter.convert(str(path))
    text = _clean_markdown(result.text_content or "")

    if not text:
        raise ValueError(
            f"MarkItDown no extrajo contenido de '{path.name}'. "
            "¿Fichero vacío, escaneado sin OCR o corrupto?"
        )

    return Document(
        text=text,
        source=path.name,
        metadata={
            "type": path.suffix.lower().lstrip("."),
            "path": str(path),
            "converted_by": "markitdown",
        },
    )

def _clean_markdown(text: str) -> str:
    """Elimina artefactos de conversión que no aportan al RAG."""
    text = _BASE64_IMAGE.sub("", text)
    return text.strip()