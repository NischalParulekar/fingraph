"""Freeze the question-ID splits every run uses.

    python -m fingraph.data.splits          # rebuild splits/*.json and report any change

- dev:     150 questions from the official dev split. Used freely while building.
- heldout: 500 questions from the official test split (gold answers are public).
           Run once per arm, after the arms are frozen. Never tuned on.

Both are stratified by answer_from x answer_type: each stratum gets its share of the
sample in proportion to its size in the source split (largest-remainder rounding), and
questions within a stratum are drawn with a fixed seed, then shuffled. The ID lists are
committed, so the samples do not depend on this code staying the same; rerunning it only
checks them.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from fingraph.data.tatqa import Example, load

SPLITS_DIR = Path(__file__).resolve().parents[2] / "splits"
SEED = 42
SPECS = {"dev": ("dev", 150), "heldout": ("test", 500)}  # name -> (official split, size)


def stratum(example: Example) -> str:
    return f"{example.answer_from}/{example.answer_type}"


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Split n across strata in proportion to their sizes (largest remainder)."""
    total = sum(sizes.values())
    if not 0 < n <= total:
        raise ValueError(f"cannot draw {n} of {total}")
    exact = {k: n * size / total for k, size in sizes.items()}
    counts = {k: int(v) for k, v in exact.items()}
    by_remainder = sorted(sizes, key=lambda k: (-(exact[k] - counts[k]), k))
    for k in by_remainder[: n - sum(counts.values())]:
        counts[k] += 1
    return counts


def sample(examples: dict[str, Example], n: int, seed: int = SEED) -> list[str]:
    """A stratified sample of question uids, in a fixed order."""
    groups: dict[str, list[str]] = defaultdict(list)
    for uid, example in examples.items():
        groups[stratum(example)].append(uid)
    counts = allocate({k: len(v) for k, v in groups.items()}, n)
    rng = random.Random(seed)
    chosen = []
    for key in sorted(groups):
        chosen += rng.sample(sorted(groups[key]), counts[key])
    rng.shuffle(chosen)  # so that the first k questions of a split are a mixed sample too
    return chosen


def load_split(name: str) -> list[Example]:
    """The examples of a frozen split, in the committed order."""
    official, _ = SPECS[name]
    examples = load(official)
    with open(SPLITS_DIR / f"{name}.json", encoding="utf-8") as f:
        return [examples[uid] for uid in json.load(f)["ids"]]


def main() -> None:
    SPLITS_DIR.mkdir(exist_ok=True)
    for name, (official, n) in SPECS.items():
        examples = load(official)
        ids = sample(examples, n)
        strata: dict[str, int] = defaultdict(int)
        for uid in ids:
            strata[stratum(examples[uid])] += 1
        record = {
            "source": f"TAT-QA official {official} split",
            "seed": SEED,
            "n": len(ids),
            "strata": dict(sorted(strata.items())),
            "ids": ids,
        }
        path = SPLITS_DIR / f"{name}.json"
        text = json.dumps(record, indent=1) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != text:
            print(f"{path}: CHANGED - the committed split differs from what this code builds")
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"{path}: {len(ids)} questions, {dict(sorted(strata.items()))}")


if __name__ == "__main__":
    main()
