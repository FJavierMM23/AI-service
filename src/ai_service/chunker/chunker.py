"""Chunker de documentos.

Dos estrategias:
  - Markdown-aware: trocea respetando secciones (encabezados #, ##...).
    Cada chunk lleva el título de su sección prependido (mejora el embedding)
    y como metadato (mejora la citación).
  - Párrafos (fallback): la estrategia clásica para texto sin estructura.

split_document elige automáticamente según el documento tenga encabezados o no.
"""
import re

from ai_service.models import Chunk, Document

# Configuración por defecto (en caracteres)
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 150

# Marcador de página que inserta el pdf_loader
PAGE_MARKER = re.compile(r"\[page:(\d+)\]")

# Encabezado markdown: entre 1 y 6 almohadillas + espacio + título
HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Delimitador de bloque de código fenced
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def split_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Trocea un documento eligiendo la estrategia según su estructura.

    Si el texto contiene encabezados markdown, se trocea por secciones.
    Si no, se usa el chunker clásico por párrafos con solapamiento.
    """
    if _has_markdown_headers(document.text):
        return _split_markdown(document, chunk_size, overlap)
    return _split_plain(document, chunk_size, overlap)


# Estrategia markdown-aware

def _has_markdown_headers(text: str) -> bool:
    """True si el texto tiene al menos un encabezado fuera de bloques de código."""
    in_code = False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_code = not in_code
            continue
        if not in_code and HEADER_RE.match(line):
            return True
    return False


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Parte el texto en secciones (título, contenido) según los encabezados.

    - El contenido previo al primer encabezado forma una sección sin título.
    - Los encabezados dentro de bloques de código fenced se ignoran.
    - Cada sección abarca desde su encabezado hasta el siguiente (de cualquier nivel).
    """
    sections: list[tuple[str | None, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    in_code = False

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_code = not in_code
            current_lines.append(line)
            continue

        match = None if in_code else HEADER_RE.match(line)
        if match:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_title, content))
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    content = "\n".join(current_lines).strip()
    if content:
        sections.append((current_title, content))

    return sections


def _split_markdown(
    document: Document,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Trocea por secciones. Sin solapamiento entre secciones distintas."""
    sections = _split_into_sections(document.text)
    chunks: list[Chunk] = []

    for title, content in sections:
        # El título se prepende a cada chunk de la sección: da contexto
        # temático al embedding y facilita la lectura del chunk aislado.
        header_prefix = f"## {title}\n\n" if title else ""
        budget = chunk_size - len(header_prefix)

        if len(content) <= budget:
            parts = [content]
        else:
            parts = _pack_paragraphs(content, budget, overlap)

        for part in parts:
            chunks.append(
                Chunk(
                    text=f"{header_prefix}{part}".strip(),
                    source=document.source,
                    chunk_index=len(chunks),
                    metadata={"section": title or "(inicio)", **document.metadata},
                )
            )

    return chunks


# Estrategia clásica por párrafos (fallback para texto sin encabezados)

def _split_plain(
    document: Document,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """El chunker original: párrafos + solapamiento + marcadores de página."""
    paragraphs = _split_paragraphs(document.text)
    chunks: list[Chunk] = []
    current = ""
    current_page = 1

    for para in paragraphs:
        match = PAGE_MARKER.search(para)
        if match:
            current_page = int(match.group(1))
            para = PAGE_MARKER.sub("", para).strip()
            if not para:
                continue

        if len(para) > chunk_size:
            sub_parts = _split_by_sentences(para, chunk_size)
        else:
            sub_parts = [para]

        for part in sub_parts:
            if len(current) + len(part) + 2 <= chunk_size:
                current = f"{current}\n\n{part}" if current else part
            else:
                if current:
                    chunks.append(
                        _make_plain_chunk(current, document, len(chunks), current_page)
                    )
                tail = current[-overlap:] if overlap and current else ""
                current = f"{tail}\n\n{part}" if tail else part

    if current.strip():
        chunks.append(_make_plain_chunk(current, document, len(chunks), current_page))

    return chunks


def _make_plain_chunk(text: str, document: Document, index: int, page: int) -> Chunk:
    metadata = {"page": page, **document.metadata}
    return Chunk(
        text=text.strip(),
        source=document.source,
        chunk_index=index,
        metadata=metadata,
    )


# Utilidades compartidas

def _split_paragraphs(text: str) -> list[str]:
    """Parte el texto por líneas en blanco y descarta párrafos vacíos."""
    raw = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw if p.strip()]


def _pack_paragraphs(text: str, max_size: int, overlap: int) -> list[str]:
    """Agrupa los párrafos de un texto en trozos de hasta max_size,
    con solapamiento entre trozos consecutivos."""
    paragraphs = _split_paragraphs(text)
    parts: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > max_size:
            sub_parts = _split_by_sentences(para, max_size)
        else:
            sub_parts = [para]

        for piece in sub_parts:
            if len(current) + len(piece) + 2 <= max_size:
                current = f"{current}\n\n{piece}" if current else piece
            else:
                if current:
                    parts.append(current)
                tail = current[-overlap:] if overlap and current else ""
                current = f"{tail}\n\n{piece}" if tail else piece

    if current.strip():
        parts.append(current)

    return parts


def _split_by_sentences(text: str, max_size: int) -> list[str]:
    """Parte un texto largo por frases, agrupándolas hasta max_size."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                parts.append(current)
            while len(sentence) > max_size:
                parts.append(sentence[:max_size])
                sentence = sentence[max_size:]
            current = sentence

    if current:
        parts.append(current)

    return parts