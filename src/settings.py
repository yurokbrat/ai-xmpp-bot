from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    BOT_JID: str
    BOT_PASSWORD: str
    BOT_NICK: str
    ADMIN_JID: str
    MUC_ROOM: str
    AI_DEFAULT_MODEL: str
    AI_CODE_MODEL: str
    OLLAMA_URL: str
    IS_DEBUG: bool = False
    LOGGING_LEVEL: str = "INFO"
    ENABLE_TYPING_EFFECT: bool = True


settings = Settings()  # type: ignore[call-arg]
