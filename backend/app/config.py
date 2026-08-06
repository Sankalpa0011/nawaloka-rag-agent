"""Config, read from the environment / .env.

Every tunable value (thresholds, ceilings, feature flags) is a field on
`Settings` with its calibrated default, overridable via env vars (uppercase
form of the field name - see .env.example).
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

    # LLM providers 
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""

    llm_provider_order: str = "groq,gemini,openrouter"

    llm_fast_model_gemini: str = "gemini-3.1-flash-lite"
    llm_strong_model_gemini: str = "gemini-3.5-flash"
    llm_fast_model_groq: str = "llama-3.1-8b-instant"
    llm_strong_model_groq: str = "llama-3.3-70b-versatile"

    llm_fast_model_openrouter: str = "google/gemma-4-26b-a4b-it:free"
    llm_strong_model_openrouter: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    gemini_rpm: int = 15
    groq_rpm: int = 25
    openrouter_rpm: int = 20
    gemini_tpm: int = 250_000
    gemini_tpm_strong: int = 250_000
    groq_tpm: int = 6_000
    groq_tpm_strong: int = 12_000
    openrouter_tpm: int = 200_000
    openrouter_tpm_strong: int = 200_000

    # LLM router: failover, retries, budgeting 
    llm_max_attempts_per_provider: int = Field(
        default=2,
        description="Attempts against one provider before failing over to the next.",
    )
    llm_failover_wait_seconds: float = Field(
        default=1.5,
        description=(
            "How long to queue behind a throttled provider while another one is "
            "untried. Beyond this, failing over is faster than waiting. Ignored "
            "on the last remaining provider, which waits out its full timeout."
        ),
    )
    llm_output_reserve_tokens: int = Field(
        default=700,
        description=(
            "Output tokens charged against the TPM budget regardless of the "
            "caller's max_tokens. max_tokens is a safety ceiling, not a "
            "prediction: charging the ceiling made synthesis estimate 5,045 "
            "tokens against Groq's 6,000/min for a turn that really cost 2,125, "
            "and needlessly failed it over. reconcile() corrects this from the "
            "provider's own usage numbers after the call."
        ),
    )
    llm_default_max_tokens: int = Field(
        default=1024,
        description="max_tokens for callers that do not ask for a specific ceiling.",
    )
    llm_request_timeout_seconds: float = Field(
        default=45.0, description="Per-request timeout for a non-streaming completion."
    )
    llm_stream_timeout_seconds: float = Field(
        default=30.0,
        description="How long a streaming call waits for a slot in the request bucket.",
    )
    llm_default_tpm: int = Field(
        default=30_000,
        description=(
            "Token budget for a provider/model pair nobody declared a TPM for. "
            "Deliberately the smaller of the two typical allowances: pacing too "
            "slowly costs latency, pacing too fast costs 429s."
        ),
    )
    llm_max_retry_after_seconds: float = Field(
        default=30.0,
        description="Cap on a provider-supplied Retry-After header before backing off.",
    )
    llm_max_backoff_seconds: float = Field(
        default=20.0, description="Ceiling on the exponential retry backoff, before jitter."
    )
    llm_bucket_burst: int = Field(
        default=5,
        description=(
            "Request-bucket burst capacity, capped at the provider's RPM. Lets a "
            "short burst through before pacing kicks in."
        ),
    )
    llm_bucket_timeout_seconds: float = Field(
        default=30.0, description="Default wait for a request-bucket token before giving up."
    )
    llm_budget_timeout_seconds: float = Field(
        default=60.0, description="Default wait for token-budget headroom before giving up."
    )
    llm_wait_slice_seconds: float = Field(
        default=0.5,
        description=(
            "Longest single sleep while waiting on a bucket or budget. Capping it "
            "lets several waiters wake and re-check fairly."
        ),
    )
    llm_breaker_threshold: int = Field(
        default=3,
        description="Consecutive provider failures before the circuit breaker opens.",
    )
    llm_breaker_cooldown_seconds: float = Field(
        default=60.0, description="First cooldown after the circuit opens. Doubles per failed trial."
    )
    llm_breaker_max_cooldown_seconds: float = Field(
        default=900.0,
        description=(
            "Ceiling on the doubling cooldown. A provider that is out of quota for "
            "the day should stop being re-probed every minute - while it still "
            "looks eligible the router keeps its failover wait short, and the "
            "healthy provider never gets to wait out its own token budget."
        ),
    )
    approx_chars_per_token: int = Field(
        default=4,
        description=(
            "Characters per token, used for pre-call budget estimates and for "
            "chunk sizing. Rough on purpose - the real count comes back from the "
            "provider and reconciles the budget, and chunk boundaries are "
            "heuristic anyway. Holds for English; revisit for other scripts."
        ),
    )

    # Postgres 
    database_url: str = "postgresql://nawaloka:nawaloka_dev_pw@localhost:5433/nawaloka"
    database_ro_url: str = (
        "postgresql://chatbot_ro:chatbot_ro_dev_pw@localhost:5433/nawaloka"
    )
    db_max_rows: int = Field(
        default=200,
        description=(
            "Hard row cap on anything travelling to a prompt. Nothing the agent "
            "legitimately asks for on this dataset comes close; this stops a bad "
            "generated query dumping tens of thousands of rows into a prompt."
        ),
    )
    db_pool_size: int = Field(default=5, description="SQLAlchemy pool size, per engine.")
    db_max_overflow: int = Field(
        default=5, description="Extra connections the read-only pool may open under load."
    )

    # Qdrant 
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "nawaloka_web"
    qdrant_faq_collection: str = "faq_curated"
    qdrant_cache_collection: str = "faq_cache"
    qdrant_ontology_collection: str = "symptom_ontology"
    qdrant_timeout_seconds: int = Field(
        default=60, description="HTTP timeout on the shared Qdrant client."
    )
    qdrant_index_batch_size: int = Field(
        default=64, description="Points per upsert batch when indexing chunks."
    )
    qdrant_scroll_limit: int = Field(
        default=1000, description="Page size when scrolling a whole collection's payloads."
    )

    # Retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    retrieval_prefetch_limit: int = 30
    retrieval_fusion_limit: int = 12
    # 0 = auto (cpu_count - 1)
    onnx_threads: int = 0
    retrieval_top_k: int = 5
    retrieval_score_floor: float = 0.02
    rerank_doc_chars: int = Field(
        default=1200,
        description=(
            "How much chunk body the cross-encoder sees. ~1200 chars is about its "
            "512-token window once the breadcrumb is on the front."
        ),
    )

    # Routing 
    # Confidence bands for reconciling the classifier against deterministic signals.
    router_min_confidence: float = 0.50
    router_high_confidence: float = 0.80
    router_fallback_confidence: float = Field(
        default=0.50,
        description=(
            "Confidence recorded when routing deliberately skips the classifier "
            "(use_llm=False) and decides on signals alone."
        ),
    )
    router_error_fallback_confidence: float = Field(
        default=0.40,
        description=(
            "Confidence recorded when the classifier failed or hallucinated and "
            "signals had to stand in. Lower than the deliberate fallback: this one "
            "is a degraded path, not a chosen one."
        ),
    )
    router_max_tokens: int = Field(
        default=400, description="Output ceiling for the routing classifier call."
    )
    router_rationale_max_chars: int = Field(
        default=400, description="How much of the classifier's rationale is kept on the decision."
    )
    router_history_turns: int = Field(
        default=2, description="Prior turns shown to the classifier as conversational context."
    )
    router_history_answer_chars: int = Field(
        default=200, description="How much of each prior answer goes into that context block."
    )

    # Deterministic signals / gazetteer 
    signals_shared_word_cutoff: int = Field(
        default=2,
        description=(
            "How many different record names a word must appear in before it stops "
            "being distinctive and is dropped from the gazetteer. Two: a word "
            "shared by even two records cannot pick one of them. At 3 'general' and "
            "'surgery' survive and 'do you do laser eye surgery' matches General "
            "Surgery & Gastroenterology."
        ),
    )
    signals_word_min_len: int = Field(
        default=5, description="Shortest word worth indexing as a partial specialty-name match."
    )
    signals_doctor_word_min_len: int = Field(
        default=4, description="Shortest word worth indexing from a doctor's name."
    )
    signals_lab_word_min_len: int = Field(
        default=6, description="Shortest word worth indexing from a lab test name."
    )
    signals_alias_min_len: int = Field(
        default=3, description="Shortest practitioner alias kept ('gp', 'ent' and friends)."
    )
    signals_term_min_len: int = Field(
        default=3,
        description="Shortest gazetteer term matched against a query. Below this it fires on noise.",
    )

    # Symptom ontology 
    ontology_top_k: int = Field(
        default=4,
        description=(
            "Symptom nodes retrieved per lookup. A complaint rarely maps to more "
            "than two specialties and asking for more drags in weak matches. "
            "ONTOLOGY_DECISIVE_MARGIN was measured at this top-k - widening one "
            "means re-measuring the other."
        ),
    )
    ontology_match_floor: float = Field(
        default=0.65,
        description=(
            "Similarity floor for 'which clinic sees this complaint', where the "
            "caller has already decided it is a complaint. Sits between flat "
            "gibberish (~0.60) and the weakest real complaint (0.723)."
        ),
    )
    ontology_relative_margin: float = Field(
        default=0.10,
        description=(
            "How far below the top score a second specialty may sit and still be "
            "listed. Keeps 'chest pain' on Cardiology instead of also offering the "
            "clinics that matched on the word 'pain'."
        ),
    )
    ontology_complaint_floor: float = Field(
        default=0.80,
        description=(
            "Much higher bar for 'is this a symptom at all', because that guards "
            "the doctor-name field and a false positive refuses a real patient "
            "looking up a real doctor. At 0.55 every surname read as a symptom."
        ),
    )
    ontology_decisive_margin: float = Field(
        default=0.05,
        description=(
            "How far the winning specialty must stand clear of the nearest "
            "different one before its match is acted on. Measured over 29 probes: "
            "real complaints gap 0.075-0.398, non-complaints 0.001-0.039. Nothing "
            "lands in between, so the cutoff sits in the gap. This is what stops a "
            "dental question being answered with general physicians."
        ),
    )
    ontology_doctor_term_min_len: int = Field(
        default=4,
        description=(
            "Shortest gazetteer doctor term allowed to claim a phrase as a name "
            "rather than a symptom."
        ),
    )

    #  FAQ fast path and semantic cache 
    faq_curated_threshold: float = Field(
        default=0.75,
        description=(
            "Similarity floor for the curated FAQ tier. Calibrated on 13 probes - "
            "real hits land 0.755-0.987 and near-misses top out at 0.700. Started "
            "at 0.90 and it matched literally nothing."
        ),
    )
    faq_cache_threshold: float = Field(
        default=0.88,
        description=(
            "Higher bar than the curated tier: these answers were generated rather "
            "than reviewed, and the key is the earlier question itself so a real "
            "repeat scores very high anyway. Too strict costs a cache miss; too "
            "loose serves someone the wrong answer."
        ),
    )
    faq_cache_ttl_seconds: int = Field(
        default=3600, description="How long a cached answer stays servable."
    )

    #  SQL branch 
    sql_max_attempts: int = Field(
        default=3,
        description=(
            "Times the graph re-enters the SQL node on a failure. Three total "
            "attempts covers the failures actually seen (a bad argument, a rejected "
            "statement) and is cheap enough not to matter on a free tier."
        ),
    )
    sql_tool_choice_max_tokens: int = Field(
        default=400, description="Output ceiling for the tool-choice call."
    )
    sql_tool_reason_max_chars: int = Field(
        default=200, description="How much of the model's stated reason is kept on the trace."
    )
    sql_broaden_max_attempt: int = Field(
        default=1,
        description=(
            "Last attempt allowed to retry an empty-but-filtered result by "
            "broadening it. A second empty result is evidence the answer really is "
            "empty, and patients were already waiting 6-12s."
        ),
    )
    sql_name_similarity: float = Field(
        default=0.3,
        description=(
            "pg_trgm floor for fuzzy doctor-name matching. Copes with honorifics, "
            "typos and partial surnames without dragging in unrelated doctors."
        ),
    )
    sql_name_word_similarity: float = Field(
        default=0.45,
        description=(
            "pg_trgm floor for matching a single query word against the best-matching "
            "run inside a doctor's name (word_similarity). Higher than the whole-string "
            "floor because a lone word is weak evidence: 'gunasekera' scores 0.64 "
            "against Gunasekara but still 0.36 against Gunawardena, and only a floor "
            "between the two keeps broad matching from dragging in a different doctor."
        ),
    )
    sql_specialty_threshold: float = Field(
        default=0.80,
        description=(
            "Word-level score a specialty must reach to be resolved exactly rather "
            "than matched with a literal ILIKE. Whole-string similarity cannot do "
            "this job at all - a genuine typo ('rhumatology', 0.323) scores below a "
            "made-up specialty ('eye surgery', 0.600) - so matching is on words that "
            "appear in exactly one specialty name."
        ),
    )
    sql_prefix_match_score: float = Field(
        default=0.95,
        description=(
            "Score given to a prefix match ('neuro' for 'neurosurgery'), which is "
            "how people actually shorten these. Plain edit distance scores that pair "
            "0.59 and would reject it."
        ),
    )
    sql_prefix_min_len: int = Field(
        default=4,
        description="Shortest query word allowed to claim a prefix match. Stops 'ent' claiming 'dental'.",
    )
    sql_min_word_len: int = Field(
        default=3, description="Shortest word worth matching on when resolving a specialty."
    )
    sql_default_row_limit: int = Field(
        default=20, description="Default row limit offered to the model on the search tools."
    )
    sql_max_row_limit: int = Field(
        default=50, description="Hard ceiling on that limit, whatever the model asks for."
    )
    sql_doctor_schedule_limit: int = Field(
        default=5,
        description="Fuzzy name matches returned by get_doctor_schedule, best match first.",
    )
    sql_specialty_list_limit: int = Field(
        default=50, description="Row cap on the full specialty listing."
    )
    sql_suggest_specialty_limit: int = Field(
        default=20, description="Doctors returned for a complaint mapped to its clinic(s)."
    )

    #  Text-to-SQL 
    text2sql_max_tokens: int = Field(
        default=1200,
        description=(
            "Output ceiling for SQL generation. Generous on purpose: a truncated "
            "response is unparseable JSON, and 600 was measured cutting off "
            "mid-statement on multi-column SELECTs."
        ),
    )
    text2sql_relax_max_tokens: int = Field(
        default=400, description="Output ceiling for the broadened re-generation of an empty query."
    )
    text2sql_explanation_max_chars: int = Field(
        default=300, description="How much of the model's explanation is kept."
    )
    text2sql_max_rows_in_payload: int = Field(
        default=50, description="Rows carried on the trace payload for a generated query."
    )

    #  Synthesis 
    synthesis_max_tokens: int = Field(
        default=8000,
        description=(
            "Output ceiling for the answer call. The Gemini 3.x flash models reason "
            "internally and those tokens are charged against the same budget, so "
            "the visible answer only gets what the reasoning left behind: at 800 a "
            "list of 11 doctors stopped at the second one. Headroom is the only "
            "lever the installed SDK offers, and it is nearly free - the TPM budget "
            "is charged LLM_OUTPUT_RESERVE_TOKENS regardless of what is asked for."
        ),
    )
    synthesis_retry_max_tokens: int = Field(
        default=16000, description="Second attempt, for when even that ceiling was not enough."
    )
    synthesis_temperature: float = Field(
        default=0.2, description="Sampling temperature for the answer call."
    )
    synthesis_max_rows_in_prompt: int = Field(
        default=25, description="Rows per tool result shown to the answer model."
    )
    synthesis_max_doc_chars: int = Field(
        default=1100, description="How much of each retrieved chunk body goes into the prompt."
    )
    synthesis_history_turns: int = Field(
        default=2, description="Prior turns included in the answer prompt."
    )
    synthesis_history_answer_chars: int = Field(
        default=200, description="How much of each prior answer goes into that block."
    )
    synthesis_fallback_citations: int = Field(
        default=3,
        description=(
            "Citations shown when the answer used no [n] markers at all, so a "
            "sourced answer never renders sourceless."
        ),
    )

    # Contextualize (follow-up rewriting) 
    contextualize_max_tokens: int = Field(
        default=300, description="Output ceiling for the question-rewriting call."
    )
    contextualize_max_focus_chars: int = Field(
        default=1200,
        description=(
            "How much of the previous assistant turn the rewriter sees. At 160 this "
            "cut the referent off mid-list and bare follow-ups got attached to "
            "whatever the thread had talked about most, not to what was just said."
        ),
    )
    contextualize_history_turns: int = Field(
        default=5, description="Turns loaded from the thread when the caller supplied no history."
    )
    contextualize_min_rewrite_chars: int = Field(
        default=6,
        description="Absolute floor on a rewrite's length; shorter has thrown the question away.",
    )
    contextualize_min_rewrite_ratio: int = Field(
        default=3,
        description=(
            "A rewrite must also be at least 1/N the length of the original. Guards "
            "long questions, which the absolute floor above cannot."
        ),
    )

    # Conversation memory 
    memory_max_history_turns: int = Field(
        default=20, description="Turns returned by a history read when no limit is given."
    )
    memory_summary_history_turns: int = Field(
        default=10, description="Turns handed to the summariser when it runs."
    )
    memory_summary_after_turns: int = Field(
        default=6, description="Thread length before summarising is worth doing at all."
    )
    memory_summary_every_n_turns: int = Field(
        default=4,
        description="After that, how often the summary is regenerated. It is a background cost, not critical path.",
    )
    memory_max_summary_words: int = Field(
        default=120, description="Hard truncation on the rolling summary, so prompt size stays flat."
    )
    memory_summary_max_tokens: int = Field(
        default=400, description="Output ceiling for the summarising call."
    )
    memory_summary_answer_chars: int = Field(
        default=200, description="How much of each answer the summariser sees per turn."
    )
    memory_value_max_chars: int = Field(
        default=500, description="Length cap on a stored cross-session preference value."
    )

    #  Routing decision log 
    decision_log_max_query_chars: int = Field(
        default=2000,
        description="Query text kept per audit row. This is an audit trail, not a transcript store.",
    )
    decision_log_max_rationale_chars: int = Field(
        default=1000, description="Rationale text kept per audit row."
    )
    decision_log_recent_limit: int = Field(
        default=50, description="Default page size when reading the log back."
    )

    #  Feature flags 
    enable_faq_fastpath: bool = True
    enable_semantic_cache: bool = True
    enable_memory: bool = True
    enable_rerank: bool = True
    enable_text2sql_fallback: bool = True
    # A SQL turn that found nothing gets one pass through retrieval before synthesis.
    enable_sql_vector_fallback: bool = True
    # Routing audit trail (routing_decisions); written after the answer, so this is non-blocking.
    enable_decision_log: bool = True

    #  Scraper 
    scrape_sitemap_url: str = "https://www.nawaloka.com/sitemap.xml"
    scrape_concurrency: int = 3
    scrape_delay_seconds: float = 1.0
    scrape_timeout_ms: int = 45_000
    scrape_min_chars: int = 300
    scrape_user_agent: str = (
        "NawalokaAgentBot/1.0 (technical assignment; contact: candidate) "
        "Chrome/131 Playwright"
    )
    scrape_min_rendered_chars: int = Field(
        default=400,
        description=(
            "Rendered text below this means hydration produced no page-specific "
            "content - nav and footer alone - so the fetch is retried rather than "
            "banked as a successful but empty page."
        ),
    )
    scrape_max_attempts: int = Field(
        default=3, description="Render attempts per URL before recording the page as failed."
    )

    #  Extraction 
    extract_boilerplate_ratio: float = Field(
        default=0.35,
        description=(
            "A block appearing on more than this share of pages is nav/footer "
            "furniture and is dropped, without naming any CSS selector."
        ),
    )
    extract_min_block_chars: int = Field(
        default=3, description="Shortest text block kept from a page."
    )
    extract_catalogue_min_options: int = Field(
        default=25,
        description="A <select> with at least this many options is treated as a data catalogue.",
    )

    # Chunking 
    # Rough tokens - see APPROX_CHARS_PER_TOKEN.
    chunk_target_tokens: int = Field(
        default=450,
        description=(
            "Sections are packed greedily up to this. These pages are made of short "
            "cards, so heading-only splitting yields 20-150 token fragments that are "
            "too small for the reranker to judge. 450 holds a whole idea while "
            "keeping a hit precise."
        ),
    )
    chunk_max_tokens: int = Field(
        default=700, description="Above this a section is recursively split on paragraph boundaries."
    )
    chunk_min_tokens: int = Field(
        default=60, description="Below this it is a stray label and is never emitted alone."
    )
    chunk_overlap_tokens: int = Field(
        default=80,
        description=(
            "Tail carried forward across a split, so a fact spanning the boundary "
            "stays retrievable from both sides."
        ),
    )
    chunk_catalogue_batch: int = Field(
        default=40, description="Catalogue entries per emitted chunk."
    )

    #  Startup seeding 
    seed_on_startup: bool = Field(
        default=True,
        description=(
            "Build the Qdrant collections from the committed HTML cache when the "
            "API boots. Off if you seed out of band with "
            "`python -m app.ingest.ensure_seeded`."
        ),
    )
    seed_wait_timeout: float = Field(
        default=120.0, description="How long startup waits for Postgres and Qdrant to answer."
    )
    seed_poll_seconds: float = Field(
        default=2.0, description="Interval between those readiness probes."
    )

    #  App 
    log_level: str = "INFO"
    backend_port: int = 8001
    cors_origins: str = "http://localhost:3000"
    api_max_message_chars: int = Field(
        default=2000, description="Longest chat message the API will accept."
    )
    api_stream_words_per_chunk: int = Field(
        default=4,
        description=(
            "Words per SSE token event. Synthesis is one non-streaming call - the "
            "answer is post-checked for unsupported prices before anyone sees it - "
            "so this replays finished text at a readable pace."
        ),
    )
    api_stream_delay_seconds: float = Field(
        default=0.02, description="Pause between those events."
    )

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

    # providers that are both asked for and actually have a key set
    def configured_providers(self) -> list[str]:
        keys = {
            "gemini": self.gemini_api_key,
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
        }
        out = []
        for p in self.provider_order:
            if keys.get(p):
                out.append(p)
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
