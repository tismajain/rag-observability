"""Concurrency and per-user LLM budget controls."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import HTTPException, status

from src.auth.principal import Principal
from src.config.settings import settings

_global = asyncio.Semaphore(settings.rate_limit.max_concurrent_global)
_users: defaultdict[str, asyncio.Semaphore] = defaultdict(
    lambda: asyncio.Semaphore(settings.rate_limit.max_concurrent_per_user)
)
_daily_usage: defaultdict[tuple[str, str], int] = defaultdict(int)
_minute_usage: defaultdict[tuple[int, str, str], int] = defaultdict(int)
_lock = asyncio.Lock()


async def _redis_client() -> object:
    from redis.asyncio import from_url

    assert settings.rate_limit.redis_url is not None
    return from_url(  # type: ignore[no-untyped-call]
        settings.rate_limit.redis_url.get_secret_value(), decode_responses=True
    )


@asynccontextmanager
async def request_capacity(principal: Principal) -> AsyncIterator[None]:
    """Bound expensive work. Memory is allowed only by validated local config."""
    if settings.rate_limit.backend == "redis":
        client = await _redis_client()
        user_key = f"rag:concurrency:user:{principal.subject}"
        global_key = "rag:concurrency:global"
        acquire_script = """
        local g = redis.call('INCR', KEYS[1]); redis.call('EXPIRE', KEYS[1], ARGV[3])
        if g > tonumber(ARGV[1]) then redis.call('DECR', KEYS[1]); return 0 end
        local u = redis.call('INCR', KEYS[2]); redis.call('EXPIRE', KEYS[2], ARGV[3])
        if u > tonumber(ARGV[2]) then redis.call('DECR', KEYS[2]); redis.call('DECR', KEYS[1]); return 0 end
        return 1
        """
        acquired = await client.eval(  # type: ignore[attr-defined]
            acquire_script,
            2,
            global_key,
            user_key,
            settings.rate_limit.max_concurrent_global,
            settings.rate_limit.max_concurrent_per_user,
            300,
        )
        if not acquired:
            await client.aclose()  # type: ignore[attr-defined]
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Concurrent request limit exceeded",
            )
        try:
            yield
        finally:
            await client.eval(  # type: ignore[attr-defined]
                "for i=1,#KEYS do local v=redis.call('DECR',KEYS[i]); if v < 0 then redis.call('SET',KEYS[i],0) end end",
                2,
                global_key,
                user_key,
            )
            await client.aclose()  # type: ignore[attr-defined]
        return
    user_sem = _users[principal.subject]
    try:
        await asyncio.wait_for(_global.acquire(), timeout=0.05)
        try:
            await asyncio.wait_for(user_sem.acquire(), timeout=0.05)
        except Exception:
            _global.release()
            raise
    except TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Concurrent request limit exceeded",
        ) from None
    try:
        yield
    finally:
        user_sem.release()
        _global.release()


async def reserve_llm_cost(principal: Principal, estimated_units: int) -> None:
    if estimated_units < 0:
        raise ValueError("estimated_units must be non-negative")
    day = datetime.now(UTC).date().isoformat()
    key = (day, principal.subject)
    if settings.rate_limit.backend == "redis":
        client = await _redis_client()
        redis_key = f"rag:llm-budget:{day}:{principal.subject}"
        script = """
        local v=redis.call('INCRBY',KEYS[1],ARGV[1]); redis.call('EXPIRE',KEYS[1],172800)
        if v > tonumber(ARGV[2]) then redis.call('DECRBY',KEYS[1],ARGV[1]); return 0 end
        return 1
        """
        allowed = await client.eval(  # type: ignore[attr-defined]
            script,
            1,
            redis_key,
            estimated_units,
            settings.resources.daily_llm_cost_units_per_user,
        )
        await client.aclose()  # type: ignore[attr-defined]
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Daily model budget exceeded",
            )
        return
    async with _lock:
        updated = _daily_usage[key] + estimated_units
        if updated > settings.resources.daily_llm_cost_units_per_user:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Daily model budget exceeded",
            )
        _daily_usage[key] = updated


async def enforce_endpoint_rate(
    principal: Principal, *, endpoint: str, client_ip: str, limit: int
) -> None:
    """Enforce both per-user and per-IP fixed-window endpoint limits."""
    subjects = (f"user:{principal.subject}", f"ip:{client_ip}")
    window = int(time.time() // 60)
    if settings.rate_limit.backend == "redis":
        client = await _redis_client()
        script = """
        local v=redis.call('INCR',KEYS[1]); if v == 1 then redis.call('EXPIRE',KEYS[1],120) end
        if v > tonumber(ARGV[1]) then return 0 end; return 1
        """
        try:
            for subject in subjects:
                redis_rate_key = f"rag:rate:{window}:{endpoint}:{subject}"
                allowed = await client.eval(  # type: ignore[attr-defined]
                    script, 1, redis_rate_key, limit
                )
                if not allowed:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Endpoint rate limit exceeded",
                    )
        finally:
            await client.aclose()  # type: ignore[attr-defined]
        return
    async with _lock:
        for subject in subjects:
            usage_key = (window, endpoint, subject)
            if _minute_usage[usage_key] >= limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Endpoint rate limit exceeded",
                )
        for subject in subjects:
            _minute_usage[(window, endpoint, subject)] += 1
