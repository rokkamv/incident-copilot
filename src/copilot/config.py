from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    copilot_model: str = "claude-opus-5-5"
    database_url: str = "postgresql://copilot:copilot@localhost:5432/copilot"
    redis_url: str = "redis://localhost:6379/0"


settings = Settings()
