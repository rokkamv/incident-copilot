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
    database_url: str = "postgresql://copilot:copilot@localhost:5432/copilot"
    redis_url: str = "redis://localhost:6379/0"


settings = Settings()  # pyright: ignore[reportCallIssue] - fields are filled from env/.env at runtime
