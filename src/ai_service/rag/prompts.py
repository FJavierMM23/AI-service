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
de cada afirmación relevante. Usa exactamente la sección o página que \
aparece en el contexto; no inventes números de página.
4. Responde en el mismo idioma en el que está formulada la pregunta.
5. Sé conciso y directo. Si la respuesta tiene varios puntos, organízalos con \
claridad."""


NO_CONTEXT_ANSWER = (
    "No encuentro información sobre esto en los documentos indexados."
)


def build_context(results: list[SearchResult]) -> str:
    """Formatea los SearchResult en un bloque de contexto legible por el LLM."""
    blocks = []
    for r in results:
        section = r.chunk.metadata.get("section")
        page = r.chunk.metadata.get("page")

        location = ""
        if section:
            location = f", sección: {section}"
        elif page is not None:
            location = f", página {page}"

        header = f"[Fuente: {r.chunk.source}{location}]"
        blocks.append(f"{header}\n{r.chunk.text}")

    return "\n\n---\n\n".join(blocks)


def build_user_prompt(context: str, question: str) -> str:
    """Monta el prompt final de usuario: contexto + pregunta."""
    return f"""Contexto (documentos indexados):

{context}

---

Pregunta: {question}

Responde a la pregunta usando exclusivamente el contexto anterior."""