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
    top_k: int | None = None,
    min_score: float | None = None,
    store: ChromaStore | None = None,
    filters: dict[str, str | int | float | bool] | None = None,
    model: str | None = None,
) -> RagAnswer:
    """Responde una pregunta usando los documentos indexados.

    Flujo:
      1. Retrieval: buscar top_k chunks relevantes.
      2. Filtrar los que no lleguen a min_score (contexto irrelevante = ruido).
      3. Si no queda ningún chunk → respuesta directa "no hay información"
         SIN llamar al LLM (ahorra tiempo y evita alucinaciones).
      4. Montar contexto y prompt.
      5. Generar respuesta con el LLM (usando `model` si se especifica,
         o el modelo por defecto de settings en caso contrario).
      6. Devolver RagAnswer con texto + fuentes utilizadas + modelo usado.
    """
    if not question or not question.strip():
        raise ValueError("La pregunta no puede estar vacía.")

    # None = "usa el default de config" (una sola fuente de verdad: .env).
    if top_k is None:
        top_k = settings.default_top_k
    if min_score is None:
        min_score = settings.default_min_score

    # Mismo patrón: None = usar el modelo por defecto, sin tocar nada persistente.
    modelo_usado = model or settings.llm_model

    if store is None:
        store = get_store()

    results = store.search(question, top_k=top_k, where=filters)
    relevant = [r for r in results if r.score >= min_score]

    if not relevant:
        return RagAnswer(
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            question=question,
            model=modelo_usado,
        )

    context = build_context(relevant)
    prompt = build_user_prompt(context, question)
    answer_text = generate(prompt, system=SYSTEM_PROMPT, temperature=0.1, model=modelo_usado)

    return RagAnswer(
        answer=answer_text,
        sources=relevant,
        question=question,
        model=modelo_usado,
    )