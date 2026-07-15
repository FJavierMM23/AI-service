"""Cliente de generación de texto sobre Ollama.

Usa /api/chat en lugar de /api/generate porque permite separar las
instrucciones de sistema (comportamiento) del prompt de usuario (datos),
lo que mejora la adherencia del modelo a las instrucciones.
"""
import httpx

from ai_service.config import settings


class GenerationError(Exception):
    """Error al comunicarse con Ollama o al procesar la respuesta."""
    pass


# La generación es lenta: un contexto grande + respuesta larga puede
# tardar minutos en CPU. Timeout generoso.
_DEFAULT_TIMEOUT = 300.0


def generate(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.1,
) -> str:
    """Genera una respuesta de texto a partir de un prompt.

    Args:
        prompt: El prompt de usuario (contexto + pregunta ya montados).
        system: Instrucciones de sistema opcionales (comportamiento del modelo).
        temperature: Aleatoriedad de la generación. Para RAG se quiere BAJA
                     (0.0-0.2) porque buscamos fidelidad al contexto, no creatividad.

    Returns:
        El texto generado por el modelo.

    Raises:
        GenerationError: Si Ollama no responde o la respuesta es inválida.
        ValueError: Si prompt está vacío.
    """
    if not prompt or not prompt.strip():
        raise ValueError("El prompt no puede estar vacío.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.post(
                f"{settings.ollama_url}/api/chat",
                json={
                    "model": settings.llm_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        raise GenerationError(
            f"Ollama respondió con error HTTP {e.response.status_code}: "
            f"{e.response.text}"
        ) from e
    except httpx.HTTPError as e:
        raise GenerationError(
            f"No se pudo contactar con Ollama en {settings.ollama_url}: {e}"
        ) from e

    message = data.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise GenerationError(f"Respuesta inesperada de Ollama: {data!r}")

    return message["content"].strip()