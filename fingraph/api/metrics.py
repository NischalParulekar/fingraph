"""Prometheus metrics in the app's own registry (as in tableqa), so tests can build many apps."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 90, 120)
OUTCOMES = (
    "answered",
    "abstained",
    "pending_review",
    "cached",
    "failed",
    "rate_limited",
    "spend_capped",
)
REVIEW_ACTIONS = ("approve", "edit", "reject")


class Metrics:
    def __init__(self):
        self.registry = r = CollectorRegistry()
        self.asks = Counter(
            "fingraph_ask_total",
            "Questions asked, by arm and outcome.",
            ["arm", "outcome"],
            registry=r,
        )
        self.latency = Histogram(
            "fingraph_ask_latency_seconds",
            "Time to answer a question, by arm and whether it came from the cache.",
            ["arm", "cache"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.llm_calls = Counter(
            "fingraph_llm_calls_total", "Model requests, by arm.", ["arm"], registry=r
        )
        self.tokens = Counter(
            "fingraph_llm_tokens_total", "Model tokens billed, by arm.", ["arm"], registry=r
        )
        self.cost = Counter(
            "fingraph_llm_cost_usd_total", "Model spend in US dollars, by arm.", ["arm"], registry=r
        )
        self.verifier = Counter(
            "fingraph_verifier_checks_total",
            "Verifier verdicts on drafts (Arm C), by result.",
            ["result"],
            registry=r,
        )
        self.reviews = Counter(
            "fingraph_reviews_total", "Human reviews, by action.", ["action"], registry=r
        )
        self.pending = Gauge("fingraph_reviews_pending", "Runs waiting for a reviewer.", registry=r)
        self.spend_today = Gauge(
            "fingraph_llm_spend_today_usd", "Model spend so far today (UTC).", registry=r
        )

    def start_at_zero(self, arms) -> None:
        """Create every labelled series at zero, so rate() sees its first increase."""
        for arm in arms:
            for outcome in OUTCOMES:
                self.asks.labels(arm, outcome)
            for cache in ("hit", "miss"):
                self.latency.labels(arm, cache)
            self.llm_calls.labels(arm)
            self.tokens.labels(arm)
            self.cost.labels(arm)
        for result in ("pass", "fail"):
            self.verifier.labels(result)
        for action in REVIEW_ACTIONS:
            self.reviews.labels(action)
