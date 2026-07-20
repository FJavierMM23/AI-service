# AI-Service
 
Microservicio RAG (Retrieval-Augmented Generation) 100% local: indexa tus documentos (PDF, Markdown) y responde preguntas sobre ellos en lenguaje natural, citando las fuentes. Sin APIs externas, sin datos saliendo de tu máquina.
 
Construido con **Python + FastAPI + ChromaDB + Ollama**.
 
## Características
 
- **Ingesta de documentos** PDF, Markdown y texto plano, con chunking por párrafos y solapamiento configurable.
- **Búsqueda semántica**: encuentra contenido por significado, no por palabras clave (embeddings con `bge-m3`).
- **Respuestas con grounding**: el LLM (`gemma4:e4b-it-q4_K_M` por defecto, intercambiable) responde únicamente con la información de tus documentos y cita fuente y página. Si la respuesta no está en los documentos, lo dice — no inventa.
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
                    │   │vectorstore│ │  loaders  │           │
                    │   │ (ChromaDB)│ │  chunker  │           │
                    │   └─────┬─────┘ └───────────┘           │
                    │         │                               │
                    │   ┌─────▼──────────────┐                │
                    │   │ embeddings / llm   │                │
                    │   └─────┬──────────────┘                │
                    └─────────┼───────────────────────────────┘
                              │ HTTP
                        ┌─────▼─────┐
                        │  Ollama   │  (bge-m3 + gemma4:e4b-it-q4_K_M)
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
- [Ollama](https://ollama.com) con los modelos:
```bash
  ollama pull bge-m3           # embeddings
  ollama pull gemma4:e4b-it-q4_K_M    # generación (modelo por defecto)
```
  El modelo de generación es intercambiable — ver [Modelos LLM disponibles](#modelos-llm-disponibles).
- [uv](https://docs.astral.sh/uv/) (recomendado) o pip
- Docker + Docker Compose (opcional, para el despliegue containerizado)
### Aceleración por GPU (opcional pero muy recomendable)
 
Con GPU, la generación de respuestas es entre 3 y 5 veces más rápida que en CPU.
Verifica el estado con `ollama ps` tras hacer una consulta: la columna
`PROCESSOR` debe indicar `100% GPU`.
 
**NVIDIA (Windows / Linux):** funciona automáticamente si tienes los drivers
oficiales instalados (CUDA viene incluido en Ollama). No requiere configuración.
 
**Apple Silicon (macOS):** aceleración automática vía Metal. No requiere
configuración — es la plataforma con menos fricción.
 
**AMD en Windows:** las RX 6000/7000/9000 de escritorio están soportadas
automáticamente por el instalador oficial (backend ROCm incluido).
 
**AMD en Linux:** instala la variante ROCm (`ollama-rocm` en Arch/CachyOS) y
reinicia el servicio. **Solo si `ollama ps` sigue mostrando `100% CPU`**
(típico en RX 6600/6650/6700, no soportadas oficialmente por ROCm), añade un
override con `sudo systemctl edit ollama`:
 
```ini
[Service]
Environment="HSA_OVERRIDE_GFX_VERSION=10.3.0"
```
 
Para la serie RX 7000 con problemas de detección, el valor sería `11.0.0`.
Los logs de detección: `journalctl -u ollama | grep -iE "gpu|vram|gfx"`
 
### Instalación de Ollama por sistema operativo
 
**Linux (Arch/CachyOS):**
```bash
sudo pacman -S ollama          # o ollama-rocm para GPU AMD (ver sección GPU)
sudo systemctl enable --now ollama
```
 
**Linux (otras distros):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
 
**macOS:**
```bash
brew install ollama
brew services start ollama     # o abre la app Ollama descargada de ollama.com
```
 
**Windows:**
Descarga el instalador desde [ollama.com/download](https://ollama.com/download) y ejecútalo.
Ollama queda corriendo como aplicación en segundo plano.
 
### Instalación de uv por sistema operativo
 
**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
 
**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
 
## Instalación (desarrollo local)
 
```bash
git clone https://github.com/FJavierMM23/AI-service.git
cd AI-service
 
uv venv
```
 
Activar el entorno virtual:
 
```bash
# Linux / macOS (bash/zsh)
source .venv/bin/activate
 
# Linux / macOS (fish)
source .venv/bin/activate.fish
 
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
 
# Windows (cmd)
.venv\Scripts\activate.bat
```
 
Instalar y configurar:
 
```bash
uv pip install -e ".[dev]"
 
# Configuración (Linux/macOS)
cp .env.example .env
 
# Configuración (Windows PowerShell)
copy .env.example .env
 
# Verificar que Ollama está accesible
ai-service health
```
 
## Uso por CLI
 
```bash
# Indexar un documento o directorio completo
ai-service ingest manuales/
ai-service ingest manuales/mi_archivo.pdf --chunk-size 1000 --overlap 150 # También acepta (.md, .markdown, .txt, .docx, .pptx, .xlsx, .html, .htm)
 
# Preguntar
ai-service ask "pregunta o duda sobre los archivos que haya en /manuales"
 
# Búsqueda semántica pura (sin LLM, solo retrieval)
ai-service search "busqueda de los chunks que hay en /manuales" --top-k 5 --show-text
 
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
| `POST` | `/documents` | Subir e indexar un documento (multipart, metadatos opcionales) |
| `GET` | `/documents` | Listar documentos indexados |
| `DELETE` | `/documents/{source}` | Borrar un documento del índice |
| `GET` | `/models` | Modelos de generación disponibles en Ollama + el modelo por defecto |
| `GET` | `/health` | Estado del servicio y de Ollama |
 
### Ejemplos
 
```bash
# Subir un documento, con metadatos opcionales (JSON como string)
curl -X POST http://localhost:8000/documents \
  -F "file=@manuales/mi_manual.pdf" \
  -F 'metadata={"asignatura":"PC","tema":"semaforos"}'
 
# Preguntar, con filtro por metadatos y modelo opcionales
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "pregunta", "top_k": 8, "min_score": 0.3, "filters": {"asignatura": "PC"}, "model": "qwen2.5:7b"}'
 
# Ver qué modelos de generación tienes descargados
curl http://localhost:8000/models
```
 
Todos los campos de `/query` salvo `question` son opcionales: sin `top_k`/`min_score` se usan los defaults de `.env`, sin `filters` se busca en todo el índice, y sin `model` se usa el modelo por defecto (`LLM_MODEL`). `model` solo afecta a esa llamada — no persiste ni requiere reiniciar el servicio.
 
Respuesta de `/query`:
 
```json
{
  "answer": "respuesta",
  "sources": [
    {
      "source": "archivo con su extensión",
      "page": 1,
      "section": "título de la sección (si el documento tiene encabezados)",
      "chunk_index": 1,
      "score": 0.842
    }
  ],
  "question": "pregunta a la que responde",
  "model": "gemma4:e4b-it-q4_K_M"
}
```
 
Respuesta de `/models`:
 
```json
{
  "models": ["gemma4:e4b-it-q4_K_M", "qwen2.5:7b", "qwen3.5:4b"],
  "default": "gemma4:e4b-it-q4_K_M"
}
```
 
### Filtrado por metadatos
 
`ai-service` almacena metadatos arbitrarios por documento y permite filtrar por ellos en la búsqueda, sin necesidad de saber qué significan — el servicio se mantiene genérico. Los metadatos se adjuntan en la ingesta (campo `metadata`, un JSON en forma de string) y se guardan como campos escalares en cada chunk. El filtro (`filters` en `/query`) aplica una condición `where` nativa de ChromaDB: solo se recuperan los chunks que cumplen el criterio, **antes** de aplicar el umbral `min_score`.
 
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
> **Nota por sistema operativo:**
> - **Windows / macOS** (Docker Desktop): `host.docker.internal` funciona
>   directamente, sin configuración adicional.
> - **Linux**: Ollama debe escuchar en todas las interfaces
>   (`OLLAMA_HOST=0.0.0.0` vía override de systemd). Si usas un firewall
>   como ufw, permite el tráfico desde las redes de Docker:
>   ```bash
>   sudo ufw allow from 172.16.0.0/12 to any port 11434 proto tcp
>   ```
 
## Configuración
 
Variables de entorno (fichero `.env` en local, `environment:` en Docker Compose):
 
| Variable | Por defecto | Descripción |
|----------|-------------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | URL del servicio Ollama |
| `LLM_MODEL` | `gemma4:e4b-it-q4_K_M` | Modelo de generación (ver [Modelos LLM disponibles](#modelos-llm-disponibles)) |
| `EMBEDDING_MODEL` | `bge-m3` | Modelo de embeddings |
| `CHROMA_PATH` | `./chroma_db` | Directorio de persistencia de ChromaDB |
| `DOCUMENTS_PATH` | `./manuales` | Directorio de documentos subidos por la API |
| `API_HOST` | `0.0.0.0` | Host de la API |
| `API_PORT` | `8000` | Puerto de la API |
| `DEFAULT_TOP_K` | `5` | Nº de chunks que recupera el retrieval por consulta |
| `DEFAULT_MIN_SCORE` | `0.7` | Umbral de similitud (0–1); por debajo, el chunk se descarta como ruido |

### Modelos LLM disponibles
 
El modelo de generación es intercambiable: descárgalo con `ollama pull` y cambia `LLM_MODEL` en tu `.env` — no hay que tocar código. El de embeddings (`bge-m3`) es mejor no cambiarlo: rinde bien en español técnico y cambiarlo obliga a re-ingerir todos los documentos.
 
| Modelo (`LLM_MODEL`) | Perfil | Tamaño en disco |
|----------------------|--------|-----------------|
| `qwen3.5:0.8b-q4_K_M` | Buena velocidad pero menor calidad de respuesta que otros modelos más pesado. Para CPU o GPUs con muy poca VRAM (1-2 GB) / servidores muy modestos. La menor calidad. | 0,9 GB |
| `gemma4:e2b-it-q4_K_M` | El más ligero y rápido, con respuestas de calidad decente. Para CPU o GPUs con poca VRAM (4 GB) / servidores modestos. Menor calidad. | 3,1 GB |
| `gemma4:e4b-it-q4_K_M` **(por defecto)** | Modelo por defecto; buen equilibrio calidad/velocidad para este proyecto. Se recomiendan GPUs con más de 8 GB | 4,6 GB |
| `qwen2.5:7b` | Muy buen seguidor de instrucciones; grounding sólido. | 4,7 GB |
| `qwen3.5:9b` | Más capaz que qwen2.5, algo más pesado. | 5,1 GB |
| `phi4:14b-q4_K_M` | El más capaz en razonamiento, y el más pesado. Se recomiendan GPUs con más de 12 GB | 7,7 GB |
 
```bash
# Ejemplo: cambiar a un modelo más ligero (p. ej. para un servidor 24/7)
ollama pull gemma4:e2b-it
# y en .env:  LLM_MODEL=gemma4:e2b-it
```
 
> Los tamaños marcados son el peso en disco del modelo (aproximadamente lo que ocupa en VRAM al cargarlo); consulta el resto con `ollama list` o en [ollama.com/library](https://ollama.com/library). Si un modelo supera tu VRAM, Ollama reparte capas entre GPU y CPU: sigue funcionando, pero más lento. Verifica con `ollama ps` que la columna `PROCESSOR` indique `100% GPU`.
>
> **Grounding y tamaño de modelo:** los modelos pequeños (e2b/e4b) se distraen más con contexto de ruido, así que conviene un `DEFAULT_MIN_SCORE` algo más alto (~0.7–0.8). Los modelos grandes toleran mejor un umbral más bajo.


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
