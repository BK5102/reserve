"""The Runner: executes the model's tool calls through the Toolbox only, under a hard budget.

The model plans and calls tools (instructions in `planner.SYSTEM`); this loop executes
each call only through the `Toolbox` allow-list, returns denials and errors as tool
results the model can read, and stops at a fixed number of tool calls. It cannot score
the test period: the Toolbox does not offer it without `allow_test`, and the pipeline
refuses it without a committed lock (methodology M37).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from reserve.agent.llm import LLM
from reserve.agent.planner import SYSTEM, first_message
from reserve.agent.tools import Toolbox, ToolDenied


@dataclass
class AgentResult:
    question: str
    final_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    denied: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    stopped_by_budget: bool = False
    turns: int = 0


def run_agent(
    question: str, toolbox: Toolbox, llm: LLM, max_tool_calls: int = 10, max_turns: int = 12
) -> AgentResult:
    tools = toolbox.specs()
    messages: list[dict[str, Any]] = [first_message(question)]
    result = AgentResult(question=question, final_text="")
    while result.turns < max_turns:
        result.turns += 1
        response = llm.create(system=SYSTEM, messages=messages, tools=tools)
        # Pass the content back unchanged (thinking blocks included).
        messages.append({"role": "assistant", "content": response["content"]})
        if response["stop_reason"] == "pause_turn":
            continue
        uses = [b for b in response["content"] if b.get("type") == "tool_use"]
        if response["stop_reason"] != "tool_use" or not uses:
            result.final_text = "\n".join(
                b["text"] for b in response["content"] if b.get("type") == "text"
            )
            return result
        tool_results = []
        for block in uses:
            if len(result.tool_calls) >= max_tool_calls:
                result.stopped_by_budget = True
                content, is_error = "Tool-call budget used up. Write your final answer now.", True
            else:
                result.tool_calls.append({"tool": block["name"], "input": block["input"]})
                try:
                    content = json.dumps(toolbox.call(block["name"], dict(block["input"])))
                    is_error = False
                except ToolDenied as err:
                    result.denied.append({"tool": block["name"], "reason": str(err)})
                    content, is_error = f"Denied: {err}", True
                except Exception as err:  # noqa: BLE001 - any failure goes back to the model
                    message = f"{type(err).__name__}: {err}"
                    result.errors.append({"tool": block["name"], "error": message})
                    content, is_error = f"Error: {message}", True
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": content,
                    "is_error": is_error,
                }
            )
        # All results of one turn go back in a single user message.
        messages.append({"role": "user", "content": tool_results})
    result.stopped_by_budget = True
    return result
