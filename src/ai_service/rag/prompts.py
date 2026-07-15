"""Templates de prompts para el pipeline RAG."""

from ai_service.models import SearchResult


SYSTEM_PROMPT = """Eres un asistente experto que responde preguntas basándose \
EXCLUSIVAMENTE en los documentos proporcionados como contexto.

Reglas estrictas:
1. Responde ÚNICAMENTE con información presente en el contexto. No uses tu \
conocimiento general.
2. Si el contexto no contiene información suficiente para responder, di \
claramente: "No encuentro información sobre esto en los documentos indexados." \
No intentes adivinar ni completar con conocimiento externo.
3. Cita siempre las fuentes que uses, con el formato [fuente, pág. N] al final \
de cada afirmación relevante.
4. Responde en el mismo idioma en el que está formulada la pregunta.
5. Sé conciso y directo. Si la respuesta tiene varios puntos, organízalos con \
claridad."""


NO_CONTEXT_ANSWER = (
    "No encuentro información sobre esto en los documentos indexados."
)


def build_context(results: list[SearchResult]) -> str:
    """Formatea los SearchResult en un bloque de contexto legible por el LLM.

    Cada chunk se presenta con su fuente y página para que el modelo
    pueda citarlas en la respuesta.
    """
    blocks = []
    for r in results:
        page = r.chunk.metadata.get("page")
        page_info = f", página {page}" if page is not None else ""
        header = f"[Fuente: {r.chunk.source}{page_info}]"
        blocks.append(f"{header}\n{r.chunk.text}")

    return "\n\n---\n\n".join(blocks)


def build_user_prompt(context: str, question: str) -> str:
    """Monta el prompt final de usuario: contexto + pregunta."""
    return f"""Contexto (documentos indexados):

{context}

---

Pregunta: {question}

Responde a la pregunta usando exclusivamente el contexto anterior."""