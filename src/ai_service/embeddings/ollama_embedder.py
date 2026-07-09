"""Cliente de embeddings sobre Ollama.

Todas las funciones asumen que hay una instancia de Ollama corriendo
en settings.ollama_url con el modelo settings.embedding_model descargado.
"""
import httpx

from ai_service.config import settings


class EmbeddingError(Exception):
    """Error al comunicarse con Ollama o al procesar la respuesta."""
    pass

# Timeout de 120 segundos para dar tiempo a ollama en la primera llamada, que puede ser lenta si el modelo no está cargado.
_DEFAULT_TIMEOUT = 120.0


def embed_text(text: str) -> list[float]:
    """Genera el vector de embedding para un único texto.
    
    Args:
        text: Texto a vectorizar. No puede estar vacío.
    
    Returns:
        Lista de floats con la dimensión propia del modelo (768 para nomic-embed-text).
    
    Raises:
        EmbeddingError: Si Ollama no responde, el modelo no existe,
                        o la respuesta no tiene el formato esperado.
        ValueError: Si text está vacío o es solo whitespace.
    """
    if not text or not text.strip():
        raise ValueError("El texto a vectorizar no puede estar vacío.")
    
    vectores = embed_batch([text])
    return vectores[0]


def embed_batch(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = False,
) -> list[list[float]]:
    """Vectoriza una lista de textos en lotes.
    
    Ollama procesa un texto por request, pero agrupamos las llamadas
    en un solo cliente HTTP para reutilizar conexión y así reducir overhead.
    
    Args:
        texts: Lista de textos. Cada uno debe ser no-vacío.
        batch_size: Tamaño de lote (afecta a la barra de progreso, no a la API).
        show_progress: Si True, imprime progreso por consola cada batch.
    
    Returns:
        Lista de vectores, en el mismo orden que la entrada.
    
    Raises:
        EmbeddingError: Ídem embed_text.
        ValueError: Si algún texto de la lista está vacío.
    """
    if not texts:
        return []
    
    for i, t in enumerate(texts):
        if not t or not t.strip():
            raise ValueError(f"El texto en la posición {i} está vacío.")
    
    all_vectors: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        for batch_index in range(total_batches):
            start = batch_index * batch_size
            end = start + batch_size
            batch = texts[start:end]
            try:
                response = client.post(
                    f"{settings.ollama_url}/api/embed",
                    json={
                        "model": settings.embedding_model,
                        "input": batch,
                    },)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                raise EmbeddingError(
                    f"Ollama respondió con error HTTP {e.response.status_code}: {e.response.text}") from e
            except httpx.HTTPError as e:
                raise EmbeddingError(
                    f"No se pudo contactar con Ollama en {settings.ollama_url}: {e}") from e

            vectors = data.get("embeddings")
            if not isinstance(vectors, list) or len(vectors) != len(batch):
                raise EmbeddingError(
                    f"Respuesta inesperada de Ollama: se esperaban {len(batch)} embeddings, se recibió: {data!r}"
                )

            all_vectors.extend(vectors)

            if show_progress:
                done = min(end, len(texts))
                print(f" embeddings: {done} / {len(texts)}")
                
    return all_vectors


def embedding_dimension() -> int:
    """Devuelve la dimensión de los vectores que produce el modelo actual.
    
    Se implementa vectorizando un texto pequeño ("test") y midiendo la longitud
    del resultado. Se llama una única vez al inicializar el vector store.
    
    Raises:
        EmbeddingError: Si no puede comunicarse con Ollama.
    """
    vector = embed_text("dimension test")
    return len(vector)


def health_check() -> bool:
    """Verifica que el modelo de embeddings está disponible en Ollama.
    
    Consulta /api/tags y comprueba que settings.embedding_model está en la lista.
    No lanza excepciones: devuelve True/False para uso en checks de arranque.
    """
    try:
        response = httpx.get(f"{settings.ollama_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        models = response.json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return False

    installed_names = {m.get("name", "") for m in models}
    target = settings.embedding_model
    # Ollama a veces reporta el nombre con sufijo ":latest".
    return target in installed_names or f"{target}:latest" in installed_names