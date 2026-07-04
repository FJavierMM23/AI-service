import httpx
import typer
from pathlib import Path

from ai_service.config import settings
from ai_service.loaders import load_document, load_directory
from ai_service.chunker import split_document

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
    chunk_size: int = 2000,
    overlap: int = 300,
    show_text: bool = False,
):
    """Carga un documento o directorio y muestra los chunks generados"""
    target = Path(path)

    if target.is_dir():
        documents = load_directory(target)
    else:
        documents = [load_document(target)]

    if not documents:
        typer.echo("No se encontraron documentos soportados.")
        raise typer.Exit(1)

    total_chunks = 0
    for doc in documents:
        chunks = split_document(doc, chunk_size=chunk_size, overlap=overlap)
        total_chunks += len(chunks)

        typer.echo(f"\n📄 {doc.source}")
        typer.echo(f"   Caracteres: {len(doc.text):,}")
        typer.echo(f"   Chunks: {len(chunks)}")

        for chunk in chunks:
            page = chunk.metadata.get("page", "?")
            preview = chunk.text[:80].replace("\n", " ")
            typer.echo(f"   [{chunk.chunk_index:>3}] pag.{page} ({len(chunk.text)} chars) {preview}...")
            if show_text:
                typer.echo(f"\n{chunk.text}\n{'-' * 60}")

    typer.echo(f"\n✓ Total: {len(documents)} documento(s), {total_chunks} chunk(s)")


if __name__ == "__main__":
    app()