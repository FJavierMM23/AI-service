import httpx
import typer
from ai_service.config import settings

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


if __name__ == "__main__":
    app()