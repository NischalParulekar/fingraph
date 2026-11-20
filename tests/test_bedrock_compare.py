import json

from fingraph.eval import bedrock_compare


def write_run(directory, run, config, rows):
    stem = directory / f"graph_verifier-{run}"
    records = []
    for uid, em, answered, retries in rows:
        records.append(
            {
                "uid": uid,
                "em": em,
                "f1": em,
                "answered": answered,
                "confident_wrong": answered and em == 0,
                "cost_usd": 0.001,
                "latency_ms": 1000.0,
                "llm_calls": 1,
                "error": None if answered else "abstained",
                "trace": [
                    {
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "cache_read_tokens": 50,
                        "cache_write_tokens": 0,
                        **({"retries": retries} if run == "bedrock" else {}),
                    }
                ],
            }
        )
    with open(f"{stem}.predictions.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    summary = {"config": config, "complete": True}
    stem.with_suffix(".summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_report_compares_the_two_runs(tmp_path):
    api = {"model": "claude-haiku-5-5", "provider": "anthropic"}
    bedrock = {
        "model": "claude-haiku-5-5",
        "provider": "bedrock",
        "bedrock_model_id": "global.anthropic.claude-haiku-5-5",
        "aws_region": "ap-south-1",
    }
    rows = [(f"q{i}", 1, True, 0) for i in range(97)]
    write_run(
        tmp_path,
        "api",
        api,
        [*rows, ("q97", 1, True, 0), ("q98", 0, True, 0), ("q99", 0, False, 0)],
    )
    write_run(
        tmp_path,
        "bedrock",
        bedrock,
        [*rows, ("q97", 0, True, 2), ("q98", 1, True, 0), ("q99", 0, False, 0)],
    )

    report = bedrock_compare.build(tmp_path)
    assert report["flipped"] == 2
    assert report["em"]["difference"] == 0
    assert report["runs"]["bedrock"]["retries"] == 2
    assert report["runs"]["bedrock"]["retried_calls"] == 1
    text = bedrock_compare.markdown(report)
    assert "**EM: no measurable difference**" in text
    assert "Dev subset only" in text


def test_differences_beyond_the_noise_are_reported_with_a_direction():
    assert bedrock_compare.verdict(0.029) == "no measurable difference"
    assert bedrock_compare.verdict(-0.04) == "lower on Bedrock by 4.0 points"
