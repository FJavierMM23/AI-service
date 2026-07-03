from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"
    chroma_path: str = "./chroma_db"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()