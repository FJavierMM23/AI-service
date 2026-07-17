from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_url: str = "http://localhost:11434"
    llm_model: str = "gemma4:e4b-it-q4_K_M"
    embedding_model: str = "nomic-embed-text"
    chroma_path: str = "./chroma_db"
    documents_path: str = "./manuales"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    default_top_k: int = 5
    default_min_score: float = 0.7


settings = Settings()