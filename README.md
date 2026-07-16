# AI-Service

Microservicio RAG (Retrieval-Augmented Generation) 100% local: indexa tus documentos (PDF, Markdown) y responde preguntas sobre ellos en lenguaje natural, citando las fuentes. Sin APIs externas, sin datos saliendo de tu máquina.

Construido con **Python + FastAPI + ChromaDB + Ollama**.

## Características

- **Ingesta de documentos** PDF, Markdown y texto plano, con chunking por párrafos y solapamiento configurable.
- **Búsqueda semántica**: encuentra contenido por significado, no por palabras clave (embeddings con `bge-m3`).
- **Respuestas con grounding**: el LLM (`qwen2.5:7b`) responde únicamente con la información de tus documentos y cita fuente y página. Si la respuesta no está en los documentos, lo dice — no inventa.
- **API REST** completa con documentación Swagger automática.
- **CLI** para uso directo desde terminal.
- **100% local**: Ollama corre en tu máquina; ningún dato sale de ella.
- **Dockerizado** con healthcheck y persistencia, listo para integrarse en arquitecturas de microservicios.

## Arquitectura

```
                    ┌─────────────────────────────────────────┐
                    │              ai-service                 │
                    │                                         │
  HTTP / CLI ──────►│  api/ (FastAPI)      cli.py (Typer)     │
                    │        │                  │             │
                    │        ▼                  ▼             │
                    │  ┌──────────────────────────────┐       │
                    │  │  rag/pipeline   ingestion    │       │
                    │  └──────┬────────────┬──────────┘       │
                    │         │            │                  │
                    │   ┌─────▼─────┐ ┌────▼──────┐           │
                    │   │vectorstore│ │  loaders   │           │
                    │   │ (ChromaDB)│ │  chunker   │           │
                    │   └─────┬─────┘ └───────────┘           │
                    │         │                               │
                    │   ┌─────▼──────────────┐                │
                    │   │ embeddings / llm   │                │
                    │   └─────┬──────────────┘                │
                    └─────────┼───────────────────────────────┘
                              │ HTTP
                        ┌─────▼─────┐
                        │  Ollama   │  (bge-m3 + qwen2.5:7b)
                        └───────────┘
```

**Flujo de una pregunta** (`POST /query`):
1. La pregunta se vectoriza y se buscan los chunks más similares en ChromaDB (retrieval).
2. Se descartan los chunks por debajo de un score mínimo (defensa contra alucinaciones).
3. Los chunks relevantes se montan como contexto en un prompt con instrucciones de grounding.
4. El LLM redacta la respuesta usando exclusivamente ese contexto.
5. Se devuelve la respuesta junto con las fuentes utilizadas (documento, página, score).

## Requisitos

- Python ≥ 3.11
- [Ollama](https://ollama.com) corriendo en local con los modelos:
  ```bash
  ollama pull bge-m3           # embeddings
  ollama pull qwen2.5:7b       # generación
  ```
- [uv](https://docs.astral.sh/uv/) (recomendado) o pip
- Docker + Docker Compose (opcional, para el despliegue containerizado)

## Instalación (desarrollo local)

```bash
git clone https://github.com/FJavierMM23/AI-service.git
cd AI-service

# Entorno virtual e instalación
uv venv
source .venv/bin/activate        # bash/zsh
# source .venv/bin/activate.fish # fish

uv pip install -e ".[dev]"

# Configuración
cp .env.example .env             # edita si es necesario

# Verificar que Ollama está accesible
ai-service health
```

## Uso por CLI

```bash
# Indexar un documento o directorio completo
ai-service ingest manuales/
ai-service ingest manuales/mi_manual.pdf --chunk-size 1000 --overlap 150

# Preguntar
ai-service ask "¿qué es CAS y en qué clases lo usan?"

# Búsqueda semántica pura (sin LLM, solo retrieval)
ai-service search "memoria virtual" --top-k 5 --show-text

# Gestión
ai-service list-docs             # documentos indexados
ai-service stats                 # estadísticas del índice
ai-service delete manual.pdf     # borrar un documento del índice
ai-service reset --yes           # vaciar la base de datos
```

## Uso por API REST

Arrancar el servidor:

```bash
uvicorn ai_service.api.app:app --reload
```

Documentación interactiva (Swagger UI): **http://localhost:8000/docs**

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/query` | Pregunta → respuesta RAG con fuentes |
| `POST` | `/documents` | Subir e indexar un documento (multipart) |
| `GET` | `/documents` | Listar documentos indexados |
| `DELETE` | `/documents/{source}` | Borrar un documento del índice |
| `GET` | `/health` | Estado del servicio y de Ollama |

### Ejemplos

```bash
# Subir un documento
curl -X POST http://localhost:8000/documents \
  -F "file=@manuales/mi_manual.pdf"

# Preguntar
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿qué es CAS?", "top_k": 8, "min_score": 0.3}'
```

Respuesta de `/query`:

```json
{
  "answer": "CAS (Compare-And-Swap) es una instrucción de la CPU que...",
  "sources": [
    {
      "source": "estructuras_datos_concurrentes.md",
      "page": 1,
      "chunk_index": 1,
      "score": 0.842
    }
  ],
  "question": "¿qué es CAS?",
  "model": "qwen2.5:7b"
}
```

## Docker

```bash
docker compose build
docker compose up -d
docker compose ps        # esperar a STATUS "(healthy)"
```

El servicio queda en `http://localhost:8000` con:
- **Persistencia** en volúmenes Docker (`chroma_data`, `manuales_data`) — el índice sobrevive a reinicios.
- **Healthcheck** integrado (permite `depends_on: condition: service_healthy` desde otros servicios).
- **Conexión a Ollama del host** vía `host.docker.internal`.

> **Nota (Linux):** Ollama debe escuchar en todas las interfaces (`OLLAMA_HOST=0.0.0.0` vía override de systemd) y, si usas un firewall como ufw, permitir el tráfico desde las redes de Docker:
> ```bash
> sudo ufw allow from 172.16.0.0/12 to any port 11434 proto tcp
> ```

## Configuración

Variables de entorno (fichero `.env` en local, `environment:` en Docker Compose):

| Variable | Por defecto | Descripción |
|----------|-------------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | URL del servicio Ollama |
| `LLM_MODEL` | `qwen2.5:7b` | Modelo de generación |
| `EMBEDDING_MODEL` | `bge-m3` | Modelo de embeddings |
| `CHROMA_PATH` | `./chroma_db` | Directorio de persistencia de ChromaDB |
| `DOCUMENTS_PATH` | `./manuales` | Directorio de documentos subidos por la API |
| `API_HOST` | `0.0.0.0` | Host de la API |
| `API_PORT` | `8000` | Puerto de la API |

## Tests

```bash
pytest -v
```

Los tests de embeddings, vectorstore, RAG y API requieren Ollama corriendo con los modelos descargados; si no está disponible, se saltan automáticamente.

## Estructura del proyecto

```
src/ai_service/
├── api/              # API REST (FastAPI): app, schemas (DTOs) y routers
├── chunker/          # Troceado de documentos con solapamiento
├── embeddings/       # Cliente de embeddings sobre Ollama
├── llm/              # Cliente de generación sobre Ollama
├── loaders/          # Carga de PDF / Markdown / texto plano
├── rag/              # Pipeline RAG y templates de prompts
├── vectorstore/      # Wrapper sobre ChromaDB
├── ingestion.py      # Pipeline de ingesta (documento → chunks → BD)
├── models.py         # Modelos de dominio (Document, Chunk, SearchResult...)
├── config.py         # Configuración por variables de entorno
└── cli.py            # Interfaz de línea de comandos (Typer)
```

## Roadmap

- [ ] Chunker markdown-aware (troceado por encabezados de sección)
- [ ] Integración como microservicio en una arquitectura mayor con Spring Boot (gateway + gestión de manuales con BD relacional)
- [ ] CI con GitHub Actions
- [ ] Soporte OCR para PDFs escaneados

## Licencia

Proyecto personal de aprendizaje. Uso libre.
