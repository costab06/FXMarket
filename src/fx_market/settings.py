"""FXMarket service configuration."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Server ────────────────────────────────────────────────────────────────
    server_host: str = "0.0.0.0"
    server_port: int = Field(default=9001, validation_alias=AliasChoices("server_port", "PORT"))

    # ── OANDA v20 ─────────────────────────────────────────────────────────────
    oanda_api_key: str | None = None
    oanda_account_id: str | None = None
    oanda_env: str = "practice"          # "practice" | "live"

    # ── Interactive Brokers ───────────────────────────────────────────────────
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1

    # ── FIX session ───────────────────────────────────────────────────────────
    fix_config_path: str = "fix.cfg"

    # ── Enabled feeds and pairs ───────────────────────────────────────────────
    enabled_feeds: list[str] = Field(
        default=["oanda"],
        description="Active feed adapters: 'oanda', 'ib', 'fix'",
    )
    default_pairs: list[str] = Field(
        default=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
    )

    # ── Aggregator ────────────────────────────────────────────────────────────
    stale_seconds: float = 5.0

    # ── Smart Order Router ────────────────────────────────────────────────────
    sor_stale_seconds: float = 3.0
    sor_max_slippage_pips: float = 3.0
