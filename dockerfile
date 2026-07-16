# ---------- Etapa 1: builder ----------
# Instala las dependencias en un venv aislado que luego copiamos.
FROM python:3.12-slim AS builder

# uv: instalador rápido de dependencias
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copiamos SOLO el pyproject primero: si no cambia, Docker reutiliza
# la capa cacheada de dependencias aunque cambie el código fuente.
COPY pyproject.toml ./

# Creamos el venv e instalamos las dependencias (sin el código aún)
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache -r pyproject.toml

# Ahora sí, el código
COPY src/ ./src/

# Instalamos el paquete en sí (rápido: las deps ya están)
RUN VIRTUAL_ENV=/opt/venv uv pip install --no-cache --no-deps .


# ---------- Etapa 2: runtime ----------
# Imagen final mínima: solo el venv y lo imprescindible.
FROM python:3.12-slim

# Usuario no-root (buena práctica de seguridad en contenedores)
RUN useradd --create-home --shell /bin/bash aiservice

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Directorios de datos (se montarán como volúmenes)
RUN mkdir -p /data/chroma /data/manuales && \
    chown -R aiservice:aiservice /data

USER aiservice
WORKDIR /home/aiservice

# Configuración por defecto DENTRO del contenedor
# (sobreescribible desde docker-compose con environment:)
ENV CHROMA_PATH=/data/chroma \
    DOCUMENTS_PATH=/data/manuales \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

# Healthcheck usando httpx (ya instalado) — sin necesidad de curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health', timeout=4); exit(0 if r.status_code == 200 else 1)"

CMD ["uvicorn", "ai_service.api.app:app", "--host", "0.0.0.0", "--port", "8000"]