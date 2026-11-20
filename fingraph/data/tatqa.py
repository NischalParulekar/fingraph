"""Load TAT-QA questions together with the table and paragraphs they are asked about.

    python -m fingraph.data.tatqa download

The dataset is not redistributed in this repo (the paper restricts it to non-commercial
research use). `download` fetches the official dev and test-gold files from the TAT-QA
repository at a pinned commit and checks their SHA-256, so every run reads the same bytes.
"""

import argparse
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from functools import cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "tatqa"
COMMIT = "870accc41953dcde885aabeb963d94aabdc0fbc3"
URL = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{commit}/dataset_raw/{name}"

# The official test split's gold answers are public, so the held-out set comes from it.
FILES = {
    "dev": (
        "tatqa_dataset_dev.json",
        "8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5",
    ),
    "test": (
        "tatqa_dataset_test_gold.json",
        "c4d08418359c1d76468dec420ee748a37f48c06b63cb8ec2766f19d5d314b597",
    ),
}


@dataclass(frozen=True)
class Paragraph:
    uid: str
    order: int  # 1-based position in the report; gold `rel_paragraphs` refer to it
    text: str


@dataclass(frozen=True)
class Example:
    uid: str
    question: str
    answer_type: str  # span | multi-span | arithmetic | count
    answer_from: str  # table | text | table-text
    scale: str  # "" | thousand | million | billion | percent
    context_uid: str
    table: tuple[tuple[str, ...], ...]  # the raw cell grid, header rows included
    paragraphs: tuple[Paragraph, ...]
    gold: dict  # the question's original record, as the official scorer expects it


def download(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, sha256 in FILES.values():
        path = data_dir / name
        if not (path.exists() and _sha256(path) == sha256):
            with urllib.request.urlopen(URL.format(commit=COMMIT, name=name)) as response:
                path.write_bytes(response.read())
        if _sha256(path) != sha256:
            raise RuntimeError(f"{name}: checksum mismatch, refusing to use it")
        print(f"{path} ok")


@cache
def load(split: str, data_dir: Path = DATA_DIR) -> dict[str, Example]:
    """All questions of an official split, keyed by question uid, in file order."""
    name, _ = FILES[split]
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `make data` first")
    with open(path, encoding="utf-8") as f:
        contexts = json.load(f)
    examples = {}
    for context in contexts:
        table = tuple(tuple(row) for row in context["table"]["table"])
        paragraphs = tuple(
            Paragraph(p["uid"], int(p["order"]), p["text"])
            for p in sorted(context["paragraphs"], key=lambda p: int(p["order"]))
        )
        for q in context["questions"]:
            examples[q["uid"]] = Example(
                uid=q["uid"],
                question=q["question"],
                answer_type=q["answer_type"],
                answer_from=q["answer_from"],
                scale=q["scale"],
                context_uid=context["table"]["uid"],
                table=table,
                paragraphs=paragraphs,
                gold=q,
            )
    return examples


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download"])
    parser.parse_args()
    download()


if __name__ == "__main__":
    main()
