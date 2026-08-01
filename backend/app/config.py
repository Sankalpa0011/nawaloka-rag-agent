"""Central configuration, loaded from the environment / .env file.

Every tunable in the system lands here so that behaviour can be changed for a
demo (or disabled when a provider is down) without touching code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_HTML_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
PROMPTS_DIR = BACKEND_DIR / "app" / "prompts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM providers -------------------------------------------------------
    gemini_api_key: str = ""
    groq_api_key: str = ""
    llm_provider_order: str = "gemini,groq"

    llm_fast_model_gemini: str = "gemini-2.5-flash-lite"
    llm_strong_model_gemini: str = "gemini-2.5-flash"
    llm_fast_model_groq: str = "llama-3.1-8b-instant"
    llm_strong_model_groq: str = "llama-3.3-70b-versatile"

    gemini_rpm: int = 12
    groq_rpm: int = 28

    # --- Postgres ------------------------------------------------------------
    database_url: str = "postgresql://nawaloka:nawaloka_dev_pw@localhost:5433/nawaloka"
    database_ro_url: str = (
        "postgresql://chatbot_ro:chatbot_ro_dev_pw@localhost:5433/nawaloka"
    )

    # --- Qdrant --------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "nawaloka_web"
    qdrant_faq_collection: str = "faq_curated"
    qdrant_cache_collection: str = "faq_cache"

    # --- Retrieval -----------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    retrieval_prefetch_limit: int = 30
    retrieval_fusion_limit: int = 20
    retrieval_top_k: int = 5
    retrieval_score_floor: float = 0.15

    # --- Scraper -------------------------------------------------------------
    scrape_sitemap_url: str = "https://www.nawaloka.com/sitemap.xml"
    scrape_concurrency: int = 3
    scrape_delay_seconds: float = 1.0
    scrape_timeout_ms: int = 45_000
    scrape_min_chars: int = 300
    scrape_user_agent: str = (
        "NawalokaAgentBot/1.0 (technical assignment; contact: candidate) "
        "Chrome/131 Playwright"
    )

    # --- Feature flags -------------------------------------------------------
    enable_faq_fastpath: bool = True
    enable_semantic_cache: bool = True
    enable_memory: bool = True
    enable_rerank: bool = True
    enable_text2sql_fallback: bool = True

    # --- App -----------------------------------------------------------------
    log_level: str = "INFO"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:3000"

    @field_validator("llm_provider_order")
    @classmethod
    def _strip_order(cls, v: str) -> str:
        return ",".join(p.strip().lower() for p in v.split(",") if p.strip())

    @property
    def provider_order(self) -> list[str]:
        return [p for p in self.llm_provider_order.split(",") if p]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def configured_providers(self) -> list[str]:
        """Providers that are both requested and actually have a key."""
        keys = {"gemini": self.gemini_api_key, "groq": self.groq_api_key}
        return [p for p in self.provider_order if keys.get(p)]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
