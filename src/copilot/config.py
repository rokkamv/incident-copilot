from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["anthropic", "openai"]
    llm_model: str
    llm_api_key: str
    llm_base_url: str | None = None
    llm_effort: str | None = None
    pricing_file: str = "data/pricing.json"

    # Optional - only needed to run Stage 4 (embeddings/retrieval). Unlike llm_provider/
    # llm_model/llm_api_key, these have no effect on Stages 1-3, so Stage 1-3 code and
    # tests keep working with none of them set. Validated fail-fast inside
    # copilot.embeddings at the point of use instead.
    embedding_provider: Literal["voyage", "google"] | None = None
    embedding_model: str | None = None
    embedding_api_key: str | None = None
    embedding_dimensions: int = 1024

    database_url: str = "postgresql://incident_copilot:incident_copilot@localhost:5432/incident_copilot"
    redis_url: str = "redis://localhost:6379/0"


settings = Settings()  # pyright: ignore[reportCallIssue] - fields are filled from env/.env at runtime
