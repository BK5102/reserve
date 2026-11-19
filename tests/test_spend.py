"""The agent's hard spend cap and record mode (methodology M45). No test calls the API."""

import json
from pathlib import Path

import pytest

from reserve import cli
from reserve.agent import spend

SETTINGS = spend.load_settings("configs/agent.yaml")


class FakeBackend:
    """Stands in for AnthropicLLM. Like the API, it never returns more output than
    `max_tokens`, and its input count follows the request size (3 characters a token)."""

    def __init__(self, wanted_output=30_000, model="claude-opus-5-5"):
        self.wanted, self.model, self.max_tokens, self.seen = wanted_output, model, 16000, []

    def create(self, *, system, messages, tools):
        self.seen.append(self.max_tokens)
        chars = len(json.dumps({"system": system, "messages": messages, "tools": tools}))
        usage = {"input_tokens": chars // 3, "output_tokens": min(self.wanted, self.max_tokens)}
        return {"stop_reason": "end_turn", "content": [], "model": self.model, "usage": usage}


def test_shipped_settings_and_prices():
    assert SETTINGS.model == "claude-opus-5-5" and SETTINGS.spend_cap_usd == 10.0
    assert SETTINGS.prices["claude-opus-5-5"] == spend.Prices(4.0, 20.0, 5.0, 0.2)


def test_usage_cost_by_hand():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 2000,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 1000,
    }
    # 1000 x 4 + 2000 x 20 + 100 x 5 + 1000 x 0.2 = 44,700 dollars per million tokens
    assert spend.usage_cost(usage, SETTINGS.prices["claude-opus-5-5"]) == pytest.approx(0.0447)


def test_fallback_attempts_are_priced_at_their_own_model():
    tracker = spend.SpendTracker(SETTINGS)
    response = {
        "model": "some-fallback-model",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 0,
            "iterations": [
                {"type": "message", "model": "claude-opus-5-5", "input_tokens": 1_000_000},
                {"type": "fallback_message", "model": "unknown", "output_tokens": 1_000_000},
            ],
        },
    }
    # 1M input at the Opus 5.5 rate (4) plus 1M output at the unlisted rate (50).
    assert tracker.charge(response, 100, 1000) == pytest.approx(54.0)


def test_cap_lowers_max_tokens_then_stops_before_passing_the_cap():
    settings = spend.load_settings("configs/agent.yaml")
    settings.spend_cap_usd = 1.0
    tracker = spend.SpendTracker(settings)
    backend = FakeBackend()
    model = spend.BudgetedLLM(backend, tracker)
    message = [{"role": "user", "content": "q" * 30_000}]  # about 15,000 tokens estimated
    calls = 0
    with pytest.raises(spend.SpendCapReached, match="spend cap"):
        for _ in range(50):
            model.create(system="s", messages=message, tools=[])
            calls += 1
    assert tracker.spent_usd <= settings.spend_cap_usd
    assert calls >= 3
    assert backend.seen[0] < settings.max_tokens  # lowered on the first call already
    assert backend.seen == sorted(backend.seen, reverse=True)  # shrinks as money is spent
    assert min(backend.seen) >= spend.MIN_OUTPUT_TOKENS


def test_missing_usage_is_charged_at_the_worst_case():
    tracker = spend.SpendTracker(SETTINGS)
    cost = tracker.charge({"model": "claude-opus-5-5"}, request_chars=2000, max_tokens=1000)
    # Input at the dearer cache-write rates (5 and 12.5), output at 20 and 50.
    assert cost == pytest.approx((1000 * (5 + 12.5) + 1000 * (20 + 50)) / 1e6)


def test_bad_settings_are_refused(tmp_path):
    raw = Path("configs/agent.yaml").read_text(encoding="utf-8")
    path = tmp_path / "a.yaml"
    path.write_text(raw.replace("spend_cap_usd: 10.00", "spend_cap_usd: 0"))
    with pytest.raises(ValueError, match="positive"):
        spend.load_settings(path)
    path.write_text(raw.replace("{input: 10.00, output: 50.00", "{input: 1.00, output: 2.00"))
    with pytest.raises(ValueError, match="at least"):
        spend.load_settings(path)


def test_record_needs_a_key_and_never_mixes_with_replay(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["agent", "q", "--record"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    c = tmp_path / "c.json"
    c.write_text(json.dumps([]))
    assert cli.main(["agent", "q", "--record", str(c), "--cassette", str(c)]) == 2
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-used")
    assert cli.main(["agent", "q", "--record", str(c)]) == 1  # never overwrite a cassette
    assert "exists" in capsys.readouterr().err


def test_agent_settings_are_never_offered_as_an_experiment_config():
    from reserve.agent.tools import Toolbox

    names = Toolbox(configs_dir=Path("configs")).config_names()
    assert "agent" not in names and "synthetic_fit" not in names
    assert {"freddie_sample", "freddie_test", "synthetic_small"} <= set(names)
