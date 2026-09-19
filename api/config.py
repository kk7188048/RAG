from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    pinecone_api_key: str
    pinecone_index_name: str = "healthcare-rag"
    nvidia_api_key: str
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    llm_model: str = "deepseek-ai/deepseek-v4-flash-0731"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    tavily_api_key: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()
