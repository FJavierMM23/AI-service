import re

from ai_service.models import Chunk, Document

# Configuración por defecto (en caracteres)
DEFAULT_CHUNK_SIZE = 2000     # ~500 tokens
DEFAULT_OVERLAP = 300         # ~75 tokens

# Regex para detectar el marcador de página que insertó el pdf_loader
PAGE_MARKER = re.compile(r"\[page:(\d+)\]")


def split_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Trocea un documento en chunks con solapamiento."""
    paragraphs = _split_paragraphs(document.text)
    chunks: list[Chunk] = []
    current = ""
    current_page = 1

    for para in paragraphs:
        # Actualizamos la página actual si el párrafo trae marcador
        match = PAGE_MARKER.search(para)
        if match:
            current_page = int(match.group(1))
            para = PAGE_MARKER.sub("", para).strip()
            if not para:
                continue

        # Si el párrafo solo ya es demasiado grande, lo partimos por frases
        if len(para) > chunk_size:
            sub_parts = _split_by_sentences(para, chunk_size)
        else:
            sub_parts = [para]

        for part in sub_parts:
            # ¿Cabe en el chunk actual?
            if len(current) + len(part) + 2 <= chunk_size:
                current = f"{current}\n\n{part}" if current else part
            else:
                # Cerramos el chunk actual y empezamos otro con solapamiento
                if current:
                    chunks.append(_make_chunk(current, document, len(chunks), current_page))
                tail = current[-overlap:] if overlap and current else ""
                current = f"{tail}\n\n{part}" if tail else part

    # No olvidar el último chunk
    if current.strip():
        chunks.append(_make_chunk(current, document, len(chunks), current_page))

    return chunks


def _split_paragraphs(text: str) -> list[str]:
    """Parte el texto por líneas en blanco y descarta párrafos vacíos."""
    raw = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw if p.strip()]


def _split_by_sentences(text: str, max_size: int) -> list[str]:
    """Parte un texto largo por frases, agrupándolas hasta max_size."""
    # Split simple por final de frase (punto, interrogación, exclamación + espacio)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                parts.append(current)
            # Si una sola frase supera max_size, la cortamos a lo bruto
            while len(sentence) > max_size:
                parts.append(sentence[:max_size])
                sentence = sentence[max_size:]
            current = sentence

    if current:
        parts.append(current)

    return parts


def _make_chunk(text: str, document: Document, index: int, page: int) -> Chunk:
    metadata = {"page": page, **document.metadata}
    return Chunk(
        text=text.strip(),
        source=document.source,
        chunk_index=index,
        metadata=metadata,
    )