"""What a client sees of a finished run, shared by the API and the recorded demo.

The answer, the evidence the arm gathered (table cells and paragraphs), its
calculations, and a verifier report: Arm C's own last verdict, or for Arms A and B the
same verifier run after the fact, so every arm's answer can be checked the same way.
"""

import dataclasses

from fingraph.arms.base import Prediction
from fingraph.data.tatqa import Example
from fingraph.eval.offline_verify import check, inputs_from_trace

MAX_EVIDENCE = 60


def report_dict(report) -> dict:
    return {
        "passed": report.passed,
        "supported": [dataclasses.asdict(c) for c in report.supported],
        "unsupported": [dataclasses.asdict(c) for c in report.unsupported],
        "scale_problems": report.scale_problems,
    }


def run_view(prediction: Prediction, example: Example) -> dict:
    record = {"answer": prediction.answer, "scale": prediction.scale, "trace": prediction.trace}
    state = prediction.trace[-1] if prediction.trace else {}
    reports = state.get("verifier_reports") if isinstance(state, dict) else None
    if reports:  # Arm C: its own last verdict
        last = reports[-1]
        verifier = {k: last[k] for k in ("passed", "supported", "unsupported", "scale_problems")}
        verifier["source"] = "arm"
    else:  # A and B: the same verifier, run after the fact
        verifier = {**report_dict(check(record, example)), "source": "offline"}
    evidence, calculations = inputs_from_trace(prediction.trace)
    return {
        "answer": prediction.answer,
        "scale": prediction.scale,
        "abstained": not prediction.answer,
        "error": prediction.error,
        "evidence": [dataclasses.asdict(e) for e in evidence[:MAX_EVIDENCE]],
        "calculations": [dataclasses.asdict(c) for c in calculations],
        "verifier": verifier,
        "llm_calls": prediction.llm_calls,
        "tokens": prediction.tokens,
        "cost_usd": prediction.cost_usd,
        "arm_latency_ms": prediction.latency_ms,
    }
