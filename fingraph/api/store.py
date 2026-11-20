"""What the service keeps in Redis: cached answers, finished runs, the thread index,
rate-limit buckets and the day's model spend. Keys are prefixed `fingraph:`; all expire.

The token bucket and the spend counter are tableqa's (tableqa/service/store.py).
Paused runs are not here: they live in Postgres, in the LangGraph checkpointer.
"""

import datetime
import hashlib
import json
import re
import time

from redis.asyncio import Redis

PREFIX = "fingraph:"

TOKEN_BUCKET = """
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1]) or capacity
local ts = tonumber(state[2]) or now
tokens = math.min(capacity, tokens + math.max(0, now - ts) * refill)
local allowed, wait = 0, 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  wait = (cost - tokens) / refill
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', KEYS[1], math.ceil(capacity / refill) + 1)
return {allowed, tostring(wait)}
"""


def normalize_question(question: str) -> str:
    """Questions that differ only in case, spacing or a trailing ? share a cache entry."""
    return re.sub(r"\s+", " ", question).strip().rstrip("?!. ").lower()


def answer_key(arm: str, context_uid: str, question: str) -> str:
    digest = hashlib.sha256(f"{context_uid}\n{normalize_question(question)}".encode()).hexdigest()
    return f"{PREFIX}answer:{arm}:{digest}"


class Store:
    def __init__(self, redis: Redis, cache_ttl_s: int, run_ttl_s: int):
        self.redis = redis
        self.cache_ttl_s = cache_ttl_s
        self.run_ttl_s = run_ttl_s
        self._bucket = redis.register_script(TOKEN_BUCKET)

    async def ping(self) -> bool:
        try:
            return bool(await self.redis.ping())
        except Exception:  # noqa: BLE001 - a health check reports, never raises
            return False

    async def _get(self, key: str) -> dict | None:
        body = await self.redis.get(key)
        return json.loads(body) if body else None

    async def _put(self, key: str, value: dict, ttl: int) -> None:
        await self.redis.set(key, json.dumps(value, ensure_ascii=False, default=str), ex=ttl)

    # -- answers and runs -----------------------------------------------------------

    async def get_answer(self, key: str) -> dict | None:
        return await self._get(key)

    async def put_answer(self, key: str, run: dict) -> None:
        await self._put(key, run, self.cache_ttl_s)

    async def get_run(self, thread_id: str) -> dict | None:
        """A run: its arm, question and report, and (once finished) its result."""
        return await self._get(f"{PREFIX}run:{thread_id}")

    async def put_run(self, thread_id: str, run: dict) -> None:
        await self._put(f"{PREFIX}run:{thread_id}", run, self.run_ttl_s)

    # -- rate limiting --------------------------------------------------------------

    async def take_tokens(
        self, client: str, cost: float, capacity: float, refill_per_s: float
    ) -> tuple[bool, float]:
        allowed, wait = await self._bucket(
            keys=[f"{PREFIX}bucket:{client}"], args=[capacity, refill_per_s, time.time(), cost]
        )
        return bool(int(allowed)), float(wait)

    # -- daily model spend ----------------------------------------------------------

    @staticmethod
    def _spend_key() -> str:
        return f"{PREFIX}spend:{datetime.datetime.now(datetime.UTC).date().isoformat()}"

    async def spend_today(self) -> float:
        return float(await self.redis.get(self._spend_key()) or 0)

    async def add_spend(self, usd: float) -> float:
        key = self._spend_key()
        total = float(await self.redis.incrbyfloat(key, usd))
        await self.redis.expire(key, 2 * 24 * 3600)
        return total
