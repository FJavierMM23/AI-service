from pathlib import Path

from ai_service.models import Document


def load_markdown(path: str | Path) -> Document:
    """Carga un fichero Markdown o de texto plano y devuelve un Document."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero: {path}")

    text = path.read_text(encoding="utf-8")

    return Document(
        text=text,
        source=path.name,
        metadata={"type": "markdown", "path": str(path)},
    )