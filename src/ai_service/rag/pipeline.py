"""Pipeline RAG: pregunta → retrieval → prompt → LLM → respuesta con fuentes."""

from ai_service.config import settings
from ai_service.llm import generate
from ai_service.models import RagAnswer
from ai_service.rag.prompts import (
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    build_context,
    build_user_prompt,
)
from ai_service.vectorstore import ChromaStore, get_store


def ask(
    question: str,
    top_k: int = 5,
    min_score: float = 0.3,
    store: ChromaStore | None = None,
) -> RagAnswer:
    """Responde una pregunta usando los documentos indexados.

    Flujo:
      1. Retrieval: buscar top_k chunks relevantes.
      2. Filtrar los que no lleguen a min_score (contexto irrelevante = ruido).
      3. Si no queda ningún chunk → respuesta directa "no hay información"
         SIN llamar al LLM (ahorra tiempo y evita alucinaciones).
      4. Montar contexto y prompt.
      5. Generar respuesta con el LLM.
      6. Devolver RagAnswer con texto + fuentes utilizadas.
    """
    if not question or not question.strip():
        raise ValueError("La pregunta no puede estar vacía.")

    if store is None:
        store = get_store()

    results = store.search(question, top_k=top_k)
    relevant = [r for r in results if r.score >= min_score]

    if not relevant:
        return RagAnswer(
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            question=question,
            model=settings.llm_model,
        )

    context = build_context(relevant)
    prompt = build_user_prompt(context, question)
    answer_text = generate(prompt, system=SYSTEM_PROMPT, temperature=0.1)

    return RagAnswer(
        answer=answer_text,
        sources=relevant,
        question=question,
        model=settings.llm_model,
    )