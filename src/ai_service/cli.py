import httpx
import typer

from ai_service.config import settings
from ai_service.loaders import load_document
from ai_service.ingestion import ingest_path
from ai_service.vectorstore import get_store
from ai_service.rag import ask as rag_ask

app = typer.Typer(help="AI-service CLI")


@app.command()
def health():
    """Verifica que Ollama está accesible y muestra los modelos disponibles."""
    try:
        response = httpx.get(f"{settings.ollama_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        models = response.json().get("models", [])
        typer.echo(f"✓ Ollama respondiendo en {settings.ollama_url}")
        typer.echo(f"Modelos disponibles ({len(models)}):")
        for m in models:
            typer.echo(f"  - {m['name']}")
    except Exception as e:
        typer.echo(f"✗ Error contactando con Ollama: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def hello(prompt: str = "Di 'hola mundo' en español."):
    """Envía un prompt a Ollama y muestra la respuesta."""
    typer.echo(f"→ Prompt: {prompt}")
    typer.echo("→ Esperando respuesta...\n")

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{settings.ollama_url}/api/generate",
            json={
                "model": settings.llm_model,
                "prompt": prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        typer.echo(data["response"])

@app.command()
def ingest(
    path: str,
    chunk_size: int = 1000,
    overlap: int = 150,
    replace: bool = True,
):
    """Carga documento(s) y los persiste vectorizados en la BD.

    Uso: ai-service ingest manuales/foo.pdf
         ai-service ingest manuales/ --chunk-size 1500
    """
    try:
        report = ingest_path(
            path=path,
            chunk_size=chunk_size,
            overlap=overlap,
            replace_existing=replace,
        )
    except FileNotFoundError as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\n✓ Ingesta completada en {report.elapsed_seconds:.1f}s")
    typer.echo(f"  Documentos: {report.documents_processed}")
    typer.echo(f"  Chunks:     {report.chunks_created}")
    for src in report.sources:
        typer.echo(f"    - {src}")


@app.command()
def search(
    query: str,
    top_k: int = 5,
    show_text: bool = False,
):
    """Busca los chunks más relevantes para una pregunta.

    Uso: ai-service search "cómo funciona la memoria virtual"
         ai-service search "..." --top-k 3 --show-text
    """
    store = get_store()
    results = store.search(query, top_k=top_k)

    typer.echo(f"\n🔍 Query: {query!r}")
    if not results:
        typer.echo("Sin resultados. ¿Has ingestado algún documento?")
        return

    typer.echo(f"Resultados ({len(results)}):\n")
    for i, r in enumerate(results, start=1):
        section = r.chunk.metadata.get("section")
        page = r.chunk.metadata.get("page")
        location = f"§ {section}" if section else f"pág.{page}" if page else ""
        preview = r.chunk.text[:120].replace("\n", " ")
        typer.echo(
            f"[{i}] score={r.score:.3f}  {r.chunk.source}  "
            f"{location}  chunk#{r.chunk.chunk_index}"
        )
        typer.echo(f"    {preview}...")
        if show_text:
            typer.echo(f"\n{r.chunk.text}\n{'-' * 60}")


@app.command("list-docs")
def list_docs():
    """Lista los documentos indexados con su número de chunks."""
    store = get_store()
    sources = store.list_sources()
    if not sources:
        typer.echo("La BD está vacía.")
        return

    typer.echo(f"\nDocumentos indexados ({len(sources)}):")
    for src, n in sorted(sources.items()):
        typer.echo(f"  {n:>4} chunks  {src}")
    typer.echo(f"\nTotal: {store.count()} chunks")


@app.command()
def stats():
    """Muestra estadísticas: nº total de chunks, modelo y ruta de la BD."""
    store = get_store()
    typer.echo("\n📊 Estadísticas del ai-service")
    typer.echo(f"  Chunks totales:    {store.count()}")
    typer.echo(f"  Documentos:        {len(store.list_sources())}")
    typer.echo(f"  Modelo embeddings: {settings.embedding_model}")
    typer.echo(f"  Modelo LLM:        {settings.llm_model}")
    typer.echo(f"  Ollama URL:        {settings.ollama_url}")
    typer.echo(f"  Ruta ChromaDB:     {settings.chroma_path}")


@app.command()
def delete(source: str):
    """Borra todos los chunks de un documento por nombre.

    Uso: ai-service delete manual_docker.pdf
    """
    store = get_store()
    n = store.delete_by_source(source)
    if n == 0:
        typer.echo(f"No había chunks del source '{source}'.")
    else:
        typer.echo(f"✓ Eliminados {n} chunks del source '{source}'.")


@app.command()
def reset(yes: bool = False):
    """Vacía completamente la BD vectorial (irreversible).

    Uso: ai-service reset --yes
    """
    if not yes:
        typer.echo("Esto BORRARÁ todos los chunks indexados.")
        typer.echo("Vuelve a ejecutar con --yes para confirmar.")
        raise typer.Exit(1)

    store = get_store()
    store.reset()
    typer.echo("✓ Base de datos vaciada.")


@app.command()
def ask(
    question: str,
    top_k: int | None = None,
    min_score: float | None = None,
    show_sources: bool = True,
):
    """Hace una pregunta y responde usando los documentos indexados.

    Uso: ai-service ask "¿qué es CAS y qué clases lo usan?"
    """
    typer.echo(f"\n🤔 Pregunta: {question}")
    typer.echo("⏳ Buscando y generando respuesta...\n")

    try:
        result = rag_ask(question, top_k=top_k, min_score=min_score)
    except ValueError as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1)

    typer.echo("💬 Respuesta:")
    typer.echo(result.answer)

    if show_sources and result.sources and "no encuentro información" not in result.answer.lower():
        typer.echo(f"\n📚 Fuentes ({len(result.sources)}):")
        for r in result.sources:
            section = r.chunk.metadata.get("section")
            page = r.chunk.metadata.get("page")
            location = f"sección: {section}" if section else f"pág. {page}"
            typer.echo(
                f"  - {r.chunk.source} ({location}, "
                f"chunk #{r.chunk.chunk_index}, score {r.score:.3f})"
            )


@app.command()
def preview(path: str, chars: int = 3000):
    """Convierte un documento y muestra el texto extraído (sin indexar)."""

    doc = load_document(path)
    typer.echo(f"📄 {doc.source} — {len(doc.text):,} caracteres")
    typer.echo(f"   Metadatos: {doc.metadata}")
    typer.echo("=" * 60)
    typer.echo(doc.text[:chars])
    if len(doc.text) > chars:
        typer.echo(f"\n[... {len(doc.text) - chars:,} caracteres más ...]")


if __name__ == "__main__":
    app()