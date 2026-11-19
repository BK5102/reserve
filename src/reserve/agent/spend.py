"""A hard spend cap for live agent runs (methodology M45).

`BudgetedLLM` wraps the real backend. Before every call it works out the most that call
could cost and refuses (`SpendCapReached`) if that would pass the cap; it also lowers the
call's `max_tokens` so the output alone cannot pass it. After every call it charges what
the response's `usage` says was used. Prices and the cap come from `configs/agent.yaml`.

The worst case of one call assumes a server-side fallback (the agent sends
`fallbacks: "default"`), so it charges the input and `max_tokens` twice: once at the main
model's price and once at the "unlisted model" price, which is set at least as high as any
listed model; input is priced at the cache-write rate where that is higher. Input tokens
are estimated as one token per two characters of the request JSON, which overestimates for
English and JSON. Replayed cassettes cost nothing and are never wrapped.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = "configs/agent.yaml"
CHARS_PER_TOKEN_ESTIMATE = 2.0
MIN_OUTPUT_TOKENS = 1024


class SpendCapReached(RuntimeError):
    """The next call could pass the configured spend cap, so it was not made."""


@dataclass(frozen=True)
class Prices:
    """US dollars per million tokens."""

    input: float
    output: float
    cache_write: float
    cache_read: float


@dataclass
class AgentSettings:
    model: str
    effort: str
    max_tokens: int
    max_tool_calls: int
    max_turns: int
    spend_cap_usd: float
    prices: dict[str, Prices]
    unlisted: Prices


def load_settings(path: str | Path = DEFAULT_CONFIG) -> AgentSettings:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    prices = {m: Prices(**p) for m, p in raw["prices_usd_per_mtok"].items()}
    unlisted = Prices(**raw["unlisted_model_prices"])
    cap = float(raw["spend_cap_usd"])
    if cap <= 0:
        raise ValueError("spend_cap_usd must be positive")
    if raw["model"] not in prices:
        raise ValueError(f"no price for the agent's model {raw['model']!r} in {path}")
    for name, p in prices.items():
        if p.input > unlisted.input or p.output > unlisted.output:
            raise ValueError(f"unlisted_model_prices must be at least {name}'s prices")
    return AgentSettings(
        model=raw["model"],
        effort=raw["effort"],
        max_tokens=int(raw["max_tokens"]),
        max_tool_calls=int(raw["max_tool_calls"]),
        max_turns=int(raw["max_turns"]),
        spend_cap_usd=cap,
        prices=prices,
        unlisted=unlisted,
    )


def usage_cost(usage: dict[str, Any], prices: Prices) -> float:
    """Dollars for one attempt's usage block."""
    tokens = (
        usage.get("input_tokens", 0) * prices.input
        + usage.get("output_tokens", 0) * prices.output
        + usage.get("cache_creation_input_tokens", 0) * prices.cache_write
        + usage.get("cache_read_input_tokens", 0) * prices.cache_read
    )
    return (tokens or 0) / 1e6


@dataclass
class SpendTracker:
    settings: AgentSettings
    spent_usd: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def price_of(self, model: str | None) -> Prices:
        return self.settings.prices.get(model or "", self.settings.unlisted)

    @property
    def remaining_usd(self) -> float:
        return self.settings.spend_cap_usd - self.spent_usd

    def _worst_input_rate(self) -> float:
        """Per-token input price of a call plus a fallback, counting cache writes (dearer)."""
        main, worst = self.price_of(self.settings.model), self.settings.unlisted
        return (max(main.input, main.cache_write) + max(worst.input, worst.cache_write)) / 1e6

    def allowed_max_tokens(self, request_chars: int) -> int:
        """The largest `max_tokens` whose worst case fits the remaining budget (0 if none)."""
        main, worst = self.price_of(self.settings.model), self.settings.unlisted
        est_input = math.ceil(request_chars / CHARS_PER_TOKEN_ESTIMATE)
        input_cost = est_input * self._worst_input_rate()
        per_output = (main.output + worst.output) / 1e6
        room = self.remaining_usd - input_cost
        allowed = min(self.settings.max_tokens, math.floor(room / per_output)) if room > 0 else 0
        return allowed if allowed >= MIN_OUTPUT_TOKENS else 0

    def charge(self, response: dict[str, Any], request_chars: int, max_tokens: int) -> float:
        """Add the cost of one response; every fallback attempt is priced at its own model."""
        usage = response.get("usage")
        if not usage:  # no usage reported: charge the worst case rather than nothing
            main, worst = self.price_of(self.settings.model), self.settings.unlisted
            est = math.ceil(request_chars / CHARS_PER_TOKEN_ESTIMATE)
            cost = est * self._worst_input_rate()
            cost += max_tokens * (main.output + worst.output) / 1e6
        elif usage.get("iterations"):
            cost = sum(usage_cost(it, self.price_of(it.get("model"))) for it in usage["iterations"])
        else:
            cost = usage_cost(usage, self.price_of(response.get("model")))
        self.spent_usd += cost
        self.calls.append({"model": response.get("model"), "usd": round(cost, 6)})
        return cost

    def summary(self) -> dict[str, Any]:
        return {
            "spend_cap_usd": self.settings.spend_cap_usd,
            "spent_usd_estimate": round(self.spent_usd, 4),
            "calls": self.calls,
            "note": "estimated from each response's usage and configs/agent.yaml prices",
        }


class BudgetedLLM:
    """A live backend behind the spend cap. The backend must have a `max_tokens` attribute."""

    def __init__(self, backend: Any, tracker: SpendTracker):
        self.backend = backend
        self.tracker = tracker

    def create(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        chars = len(json.dumps({"system": system, "messages": messages, "tools": tools}))
        allowed = self.tracker.allowed_max_tokens(chars)
        if allowed == 0:
            raise SpendCapReached(
                f"spend cap ${self.tracker.settings.spend_cap_usd:.2f} reached: "
                f"${self.tracker.spent_usd:.2f} spent, and the next call could cost more than "
                f"the ${self.tracker.remaining_usd:.2f} left"
            )
        self.backend.max_tokens = allowed
        response = self.backend.create(system=system, messages=messages, tools=tools)
        self.tracker.charge(response, chars, allowed)
        return response
