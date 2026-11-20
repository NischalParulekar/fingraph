"""Service settings, read from environment variables (all optional).

Adapted from tableqa's service config.
"""

import os
from dataclasses import dataclass, field

from fingraph.db import DEFAULT_URL


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = DEFAULT_URL
    arms: list[str] = field(default_factory=lambda: ["single_agent", "graph", "graph_verifier"])
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    # A header the edge proxy sets to the client's address (e.g. CF-Connecting-IP).
    client_ip_header: str = ""

    max_question_chars: int = 400
    cache_ttl_s: int = 7 * 24 * 3600
    run_ttl_s: int = 7 * 24 * 3600

    # Token bucket per client address. Every arm calls the model, so every fresh
    # question costs the same; a cached one costs 1.
    rate_capacity: float = 20
    rate_refill_per_s: float = 0.1
    fresh_cost: float = 5

    # Hard ceiling on model spend per UTC day, across all clients.
    daily_spend_cap_usd: float = 0.5
    concurrency: int = 4
    ask_timeout_s: float = 120

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            redis_url=env("REDIS_URL", cls.redis_url),
            database_url=env("DATABASE_URL", cls.database_url),
            arms=_list("FINGRAPH_ARMS", "single_agent,graph,graph_verifier"),
            cors_origins=_list("CORS_ORIGINS", "http://localhost:5173"),
            client_ip_header=env("CLIENT_IP_HEADER", ""),
            daily_spend_cap_usd=float(env("DAILY_SPEND_CAP_USD", cls.daily_spend_cap_usd)),
            rate_capacity=float(env("RATE_CAPACITY", cls.rate_capacity)),
            rate_refill_per_s=float(env("RATE_REFILL_PER_S", cls.rate_refill_per_s)),
        )
