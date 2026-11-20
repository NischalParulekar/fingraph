# Out of scope (for now)

Ideas that are not in the spec's "Must" list. They wait here until the Must items are done.

- Fine-tuning any model.
- Parsing raw annual-report PDFs (TAT-QA's tables and paragraphs are already extracted).
- Multi-document, cross-company questions.
- A polished UI (the demo UI stays plain React).

## Verifier, next version (after the held-out run; Arm C as measured stays frozen)

- Recognise "€m", "£m", "€'000" and similar unit marks (a real false reject, see README).
- Do not count a constant "expression" (e.g. "7") as a calculation that supports a number.
- Check a "percent" scale against the evidence behind the answer, not any "%" in the text.
- Parse quoted amounts with unit words ("$1.9 million") into numbers for calc_node.
