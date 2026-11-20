"""Hand labels for sampled held-out failures, and their tally.

    python -m fingraph.eval.error_labels       # rewrite error_labels.csv, print the tally

The labels below were assigned by Claude while reading the failure cards
(`python -m fingraph.eval.failure_cards results/heldout`), one bucket per failure: the
first thing that went wrong. They have not been reviewed by a person yet; the CSV says
so in its `reviewed` column.

Buckets (the spec's, plus answer_form and pipeline, which it has no bucket for):
- planning: misread the question or chose the wrong approach, item or source (e.g.
  computed when the answer is a quoted phrase, wrong formula, an extra or missing item);
- retrieval_miss: the paragraph that holds the answer was never retrieved;
- table_query: right approach, wrong cells read from the table;
- arithmetic: right inputs, wrong computation or rounding;
- scale: right number, wrong scale (including percent vs plain ratio);
- answer_form: right content in the wrong form: span boundaries, a phrase split into a
  list or a list merged, paraphrase, sign convention, "(3,753)" for -3,753 (the official
  scorer scores parentheses as 0), an equivalent value in another unit;
- pipeline: a fault in the graph's own code, not the model's (e.g. a quoted "$1.9
  million" that the value parser cannot read, so the calculation fails);
- verifier_false_reject: Arm C abstained although its draft answer was correct;
- gold_label: the gold answer looks wrong; each is also in benchmark_issues.md.
"""

import csv
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results" / "heldout" / "error_labels.csv"
BUCKETS = (
    "planning",
    "retrieval_miss",
    "table_query",
    "arithmetic",
    "scale",
    "answer_form",
    "pipeline",
    "verifier_false_reject",
    "gold_label",
)

# (uid, bucket, note) per arm, in failure_cards' sample order.
LABELS: dict[str, list[tuple[str, str, str]]] = {
    "single_agent": [
        (
            "ca3b22acafffb620d3deda515bbccf57",
            "answer_form",
            "table cells (12,586) thousand; gold quotes text $12.6 million",
        ),
        (
            "9f6491d767032159922ae7b2d7e025e3",
            "answer_form",
            "split into 3 items incl. an extra one; gold one span",
        ),
        ("ce73c7927786de6e65f628b4afdb76dd", "answer_form", "extra words 'exclusively to us'"),
        ("ba4d1603666b347b325b7f9f31573bc8", "answer_form", "one phrase split into a list"),
        (
            "ccfc74ecc489f611ae4f5685cfb354e0",
            "planning",
            "gave the table title; gold is the paragraph describing it",
        ),
        ("4e02f54a4ad0af10c8f2ccc939d9396a", "answer_form", "span too short"),
        (
            "332a37c0d6f010a0ee6810b907adb0ab",
            "table_query",
            "used the wrong rows (73.1, 59.2) for largest/smallest",
        ),
        ("fa64104d2b254454e60202e0c24ff9dc", "answer_form", "span too short"),
        ("169c8ca0af907215a2c57c0fe2f25053", "answer_form", "span missing leading 'as'"),
        (
            "22c6b672ffb5e3be9b0651f46898773e",
            "planning",
            "listed customer types; gold is the stated reason",
        ),
        (
            "cd28daf1b5dafb877634c4bcb1c468bd",
            "answer_form",
            "added 'Structured', cut the span short",
        ),
        ("c8a174f7806948a5f8e2db2446367e13", "answer_form", "span missing its subject"),
        (
            "b7cb9deee91596540a0e61f6ef10b4d8",
            "scale",
            "missed 'in thousands' stated in a paragraph",
        ),
        ("428359977c232b88b7cb609205de5e04", "answer_form", "span missing 'range from'"),
        (
            "a23b3dcb820703a1d9aea7dc33091afb",
            "answer_form",
            "sign: -26,490 vs gold 26,490 for 'difference'",
        ),
        (
            "41a751bd0e1105fa875a7ed4d0051fba",
            "gold_label",
            "gold 29,810-19,105; the table's total assets row is 48,915 = pred",
        ),
        (
            "d06c686c798b7f12bb3217764a542527",
            "gold_label",
            "(0.47+0.12)/2 = 0.295: half-up and half-even both give 0.30; gold 0.29",
        ),
        ("12b99bc9e7f7dc268193acf071ecfdaf", "answer_form", "span missing 'among the'"),
        (
            "243d029fde48f37c5b86a605a2fc6f72",
            "answer_form",
            "answered '30%'; gold is the whole sentence",
        ),
        (
            "bd2b6555ad74c76f74cb479b0215eded",
            "answer_form",
            "table cell (149,389) thousand; gold quotes text $149.4 million",
        ),
        ("f65bc5a189414ccb21ae21bbdd22245a", "planning", "gave two values for one figure"),
        ("fabe3b6a76158d5e91d4d3bab10fe51e", "scale", "count given scale thousand"),
        (
            "ccba5ee4e0e933f5b41b0148d493bc4c",
            "scale",
            "missed 'in thousands' stated in a paragraph",
        ),
        (
            "f2fe29ddabd4d939646a0c71704b60aa",
            "planning",
            "computed a difference; gold quotes the text",
        ),
        (
            "2870200c1799c44402bc9135e7b68daf",
            "gold_label",
            "gold span runs into an unrelated bullet; pred is the stated cause",
        ),
        ("8db7f46bf649c3befa9d8020ccad6874", "answer_form", "span missing 'due to' and its tail"),
        (
            "7f9ac26d62e28f2366b9453d13c78fc1",
            "arithmetic",
            "calculate gave 905.0078; answered 905.00, not 905.01",
        ),
        ("1c2cc847d7badaddece34c58473b71fd", "answer_form", "paraphrased item names"),
        (
            "99f77f6a84c3ba2f350bbffa9fcf3e64",
            "planning",
            "ambiguous: took the FY2018 comparative, gold the Sept 2018 raise; also scale",
        ),
        ("8be7331907754560e0edbc5d0ad52b27", "answer_form", "span includes its subject"),
        ("363e5b4e5f1d5e1fd5819d862a4340cf", "planning", "wrong part of the paragraph"),
        ("62f5c4cba4e328b1ffe8eaa6c329bd1e", "answer_form", "span missing 'The increase was'"),
        ("7822c08381f47d253e30e315b8c7cf03", "scale", "count given scale thousand"),
        ("81f47bab187b044b80a807f813271c58", "answer_form", "span includes 'The Company uses a'"),
        (
            "cf774f6bcde2d8681b99b375c9902ec3",
            "gold_label",
            "text: 65nm volume production began in 2006 (pred); 2011 is 28nm",
        ),
        (
            "dd5932654955fd5f6c14401cbfe7d8ad",
            "planning",
            "wrong formula: difference of totals, not of averages",
        ),
        ("8bdc5d0e036bee3e14bcf9f9a16b776f", "scale", "percent 32.21 vs gold ratio 0.32"),
        ("b610b2e2b8975ae32ecd9774c45d7979", "scale", "percent 1.83 vs gold ratio 0.02"),
        ("646fdf5774eafc1db2ca26899c6fcb60", "planning", "extra item 'Interest income'"),
        (
            "e798612f7b362bf8def3436937307191",
            "gold_label",
            "'percentage change' 40%->30% is -25%; gold uses the point difference -10",
        ),
        ("de48128dd966339965e26e5aff46bbdc", "answer_form", "same item twice; gold once"),
        (
            "3043971ee8b1d77cdbebedeec092a2d4",
            "scale",
            "missed 'in thousands' stated in a paragraph",
        ),
        ("3699a34071087425aeb759e2a26a4f23", "scale", "count given scale thousand"),
        (
            "32533d1bd67d1ad3e0d03442495e23f0",
            "planning",
            "computed -4.27; text states 'improved by 4.3%'",
        ),
        ("ea04ef99bd2e5a0ec0abda707cb3b52a", "answer_form", "span missing 'costs include'"),
        ("61037b64d7f42743484bd78063fa2a63", "answer_form", "one span split into two items"),
        (
            "0103b1e44ef0047d788e2cd47d5e7f8c",
            "retrieval_miss",
            "paragraph 7 (License and Collaboration...) never retrieved",
        ),
        (
            "e3096b1ac65f83346bcf04e5020d1a3f",
            "answer_form",
            "'(3,753)': the official scorer scores it 0, '-3,753' scores 1",
        ),
        (
            "3d0d640a8a63cef330ccffb57cf87ceb",
            "gold_label",
            "gold scale thousand on share counts the text gives in full (1,216,000)",
        ),
        (
            "cc6823f9305ced1af85ac2ce5f77f3df",
            "planning",
            "listed both values instead of their total",
        ),
    ],
    "graph": [
        ("8db7f46bf649c3befa9d8020ccad6874", "answer_form", "span: extra 'primarily', tail cut"),
        (
            "6c069ac5f02ac75e060313e1a2ca1cd8",
            "answer_form",
            "quoted 'Note 6'; gold is the whole sentence",
        ),
        ("75521a39d33b66cb6e29a592d8636027", "answer_form", "span missing 'relates mainly to'"),
        ("cd28daf1b5dafb877634c4bcb1c468bd", "answer_form", "span: extra 'structured', cut short"),
        (
            "4be20501905f6e3edc14f95ebf7933b3",
            "pipeline",
            "quote '$1.9 million' parses to no number, calc failed",
        ),
        ("fc58bb9998b44e3423e5f6aa7df7b028", "scale", "years given scale billion"),
        (
            "243d029fde48f37c5b86a605a2fc6f72",
            "answer_form",
            "answered '30%'; gold is the whole sentence",
        ),
        (
            "539fa070ca866c8cdeeee1fa3d56d0dd",
            "scale",
            "missed 'in thousands' stated in a paragraph",
        ),
        ("cf72b8288d2cdfcf9a59f8d879029243", "answer_form", "span missing 'uses'"),
        ("2071846240269f8a75b1c9c42f365aef", "scale", "text span given scale billion"),
        ("62f5c4cba4e328b1ffe8eaa6c329bd1e", "answer_form", "span missing 'The increase was'"),
        ("7d201138c55a318b114f8472e07d06da", "answer_form", "'fiscal 2019' for '2019'"),
        ("a26fe888a673a5d1d6f4d5f8c13aab32", "scale", "scale thousand, gold none"),
        (
            "a2cec9bc9e41a2d0e39353adf5e5d244",
            "table_query",
            "wrong row (12,002 / 1,726; gold 8,290 / 569)",
        ),
        ("5c54743f6de457f4d0af525fc5ba37ed", "answer_form", "sign: change of two negative costs"),
        (
            "7f9ac26d62e28f2366b9453d13c78fc1",
            "arithmetic",
            "calc gave 905.0078; synthesizer wrote 905.00",
        ),
        ("9aa04f2968dfabe3144285c4bf9277f9", "scale", "EPS given scale percent"),
        ("4e02f54a4ad0af10c8f2ccc939d9396a", "answer_form", "span too short"),
        (
            "61037b64d7f42743484bd78063fa2a63",
            "retrieval_miss",
            "lookup query retrieved paras 7,10,8; answer in 6",
        ),
        ("1419718d245e6762ab8e187b06212d5d", "answer_form", "items missing 'thereof'"),
        (
            "ccba5ee4e0e933f5b41b0148d493bc4c",
            "scale",
            "missed 'in thousands' stated in a paragraph",
        ),
        ("aed0cb4b76c2ebf21037804772519839", "planning", "computed 31.94; text states 32%"),
        ("79988a243da6bfe50860a248a57b5634", "scale", "scale million, gold none"),
        (
            "d57a456ed635b1a301fd19144837f69b",
            "answer_form",
            "table cells ($ -, $752 thousand); gold quotes text $0, $0.8 million",
        ),
        ("f65bc5a189414ccb21ae21bbdd22245a", "planning", "summed two rows; asked for one"),
        (
            "631f39ecd2dba43f94884cc0b27d729e",
            "planning",
            "wrong formula: a change, gold sums all four",
        ),
        ("43c792d0803c8cd37130be5e06db5ebc", "answer_form", "span missing 'result of'"),
        (
            "3043971ee8b1d77cdbebedeec092a2d4",
            "scale",
            "missed 'in thousands' stated in a paragraph",
        ),
        ("586b24199ab25b1747ae3573492f1ccb", "scale", "synthesizer set percent; planner said none"),
        (
            "02a6fbeb55b46d0ed218df0319ecffce",
            "answer_form",
            "two amounts; gold is the whole sentence",
        ),
        (
            "f005fbea468ed191b98d44c447b868ba",
            "gold_label",
            "gold '$19.1 million' with scale thousand counts the unit twice",
        ),
        ("ce73c7927786de6e65f628b4afdb76dd", "answer_form", "extra words 'exclusively to us'"),
        (
            "635d756eb2458f9cb07ad2fc9ef5ce7a",
            "planning",
            "empty plan; answer said it cannot be determined",
        ),
        (
            "9056a1ff4e2facdb93f139a8f7c60a65",
            "planning",
            "computed a difference; gold is the prior-quarter figure",
        ),
        ("2642dd42d14315d3ab62a4a1754d0bad", "scale", "share count given scale thousand"),
        (
            "332a37c0d6f010a0ee6810b907adb0ab",
            "planning",
            "took two figures from the text; answer is in the table",
        ),
        (
            "91add58b02eb761d380b13df7a61401a",
            "table_query",
            "read total deferred tax assets, not total assets",
        ),
        ("23fe1be9117b398035a9ead824b331a8", "scale", "scale thousand, gold none"),
        ("0b268aa13efe3e52409849e3cd1b711c", "answer_form", "same date twice"),
        ("d629fcff8592f6f2d8eaa97089abe3de", "answer_form", "span includes leading 'by'"),
        ("cc5c000b89c342f017727ebeced22c54", "table_query", "swapped the 2018 and 2019 columns"),
        (
            "4478c71ff25482f914636301d4c2413a",
            "planning",
            "looked in the text; the values are table cells",
        ),
        ("a7d1cbfe9fd8b97d8fdacad4dbe214e6", "scale", "dollar amount given scale percent"),
        ("b610b2e2b8975ae32ecd9774c45d7979", "scale", "percent 1.83 vs gold ratio 0.02"),
        (
            "2a9309359cd15e65d9937f9749279725",
            "answer_form",
            "extra 'option pricing model'; also scale billion",
        ),
        ("251074b99618ccfe6106b218f831ef94", "answer_form", "span missing its subject"),
        ("8bdc5d0e036bee3e14bcf9f9a16b776f", "scale", "percent 32.21 vs gold ratio 0.32"),
        ("c8a174f7806948a5f8e2db2446367e13", "answer_form", "span missing its subject"),
        (
            "487d16d4c65d212490edf8096840da2d",
            "answer_form",
            "span missing 'The cash flows comprise'",
        ),
        ("ea04ef99bd2e5a0ec0abda707cb3b52a", "answer_form", "span missing 'costs include'"),
    ],
    "graph_verifier": [
        (
            "d96471e34f661f8d062757ba3f712cd9",
            "gold_label",
            "asks operating over investing = 13,189/10,085 = 1.31 (pred); gold divides the other way",
        ),
        (
            "e798612f7b362bf8def3436937307191",
            "gold_label",
            "'percentage change' 40%->30% is -25%; gold uses the point difference -10",
        ),
        (
            "332a37c0d6f010a0ee6810b907adb0ab",
            "planning",
            "took two figures from the text; answer is in the table",
        ),
        (
            "02a6fbeb55b46d0ed218df0319ecffce",
            "answer_form",
            "two amounts; gold is the whole sentence",
        ),
        (
            "239ac320bf2943193d8bcf13605d6319",
            "table_query",
            "used the amount rows (131/1,381); gold the per-share rows (10/105)",
        ),
        (
            "2870200c1799c44402bc9135e7b68daf",
            "gold_label",
            "gold span runs into an unrelated bullet; pred is the stated cause",
        ),
        ("de48128dd966339965e26e5aff46bbdc", "answer_form", "same item twice; gold once"),
        ("063a3796dad19cc0c92c71424a600946", "answer_form", "first bullet only; gold spans all"),
        ("8672104278996c87094a74016c6e2ffb", "answer_form", "span too short"),
        ("12b99bc9e7f7dc268193acf071ecfdaf", "answer_form", "extra 'of companies'"),
        ("81f47bab187b044b80a807f813271c58", "answer_form", "span includes 'The Company uses a'"),
        (
            "9f6491d767032159922ae7b2d7e025e3",
            "answer_form",
            "split into 3 items incl. an extra one; gold one span",
        ),
        ("1d97637ce187754735e3ccd77409e237", "scale", "scale thousand, gold none"),
        (
            "a23b3dcb820703a1d9aea7dc33091afb",
            "answer_form",
            "sign: -26,490 vs gold 26,490 for 'difference'",
        ),
        (
            "cc6823f9305ced1af85ac2ce5f77f3df",
            "planning",
            "listed both values instead of their total",
        ),
        (
            "8ddab4f696534c10763a19b55229e415",
            "gold_label",
            "'percentage change' 2.7%->3.3% is 22.22%; gold uses the point difference 0.6",
        ),
        (
            "2464a21560316d1eff84a8647733adb3",
            "planning",
            "values are negative; gold compares absolute values, pred the signed ones",
        ),
        (
            "bd2b6555ad74c76f74cb479b0215eded",
            "scale",
            "'$149.4 million' with scale percent; verifier passed it on a '%' elsewhere in the text",
        ),
        ("6660bd6720b9c4d9150e048c2c9b63e0", "scale", "names given scale million"),
        (
            "61037b64d7f42743484bd78063fa2a63",
            "retrieval_miss",
            "lookup query never retrieved para 6; abstained after 3 tries",
        ),
        ("cc5c000b89c342f017727ebeced22c54", "table_query", "swapped the 2018 and 2019 columns"),
        (
            "cf774f6bcde2d8681b99b375c9902ec3",
            "gold_label",
            "text: 65nm volume production began in 2006 (pred); 2011 is 28nm",
        ),
        (
            "4c9ee3a62029fd94fd8978737cb3276a",
            "scale",
            "missed 'in thousands' stated in a paragraph",
        ),
        ("4e02f54a4ad0af10c8f2ccc939d9396a", "answer_form", "span too short"),
        ("a26fe888a673a5d1d6f4d5f8c13aab32", "scale", "scale thousand, gold none"),
        ("79988a243da6bfe50860a248a57b5634", "scale", "scale million, gold none"),
        (
            "75521a39d33b66cb6e29a592d8636027",
            "answer_form",
            "synthesizer dropped 'relates mainly to' from the quote",
        ),
        ("cd28daf1b5dafb877634c4bcb1c468bd", "answer_form", "span cut short"),
        ("4cac8704d20d5c746d09258b90c49b09", "table_query", "read 83,692; gold 66,128"),
        ("d2a8cd279ae55ad737ce418f4aeec0fe", "answer_form", "one phrase for a two-year list"),
        ("985e3306cb2075bc0807db442ac8e23a", "answer_form", "one phrase split into a list"),
        (
            "9aab1000cf8d1e140d933d7f1b5226c0",
            "answer_form",
            "computed 4,311 thousand; gold quotes text $4.3 million",
        ),
        ("4c8709f777694f5ea71b7c56a0080bfa", "scale", "missed 'in thousands'"),
        (
            "17fbe44670010dbf16d6229dbb59500b",
            "scale",
            "verifier rightly flagged scale million on '$41.8 million'; retries never fixed it; abstained",
        ),
        (
            "99f77f6a84c3ba2f350bbffa9fcf3e64",
            "planning",
            "ambiguous: took the FY2018 comparative, gold the Sept 2018 raise",
        ),
        ("0b268aa13efe3e52409849e3cd1b711c", "answer_form", "same date twice"),
        ("faf9f137376dc08ac692f904c7ab012d", "scale", "scale million, gold none"),
        ("0cdd3bd97106aad28eaa14ba87afbbfe", "answer_form", "span too short"),
        (
            "b0b1580d6fd9fc014e30a5f20a25df66",
            "scale",
            "count given scale billion; verifier accepted the planner's constant '7' as a calculation",
        ),
        ("995522509165f690157556078f2fdef7", "answer_form", "span includes leading 'by'"),
        (
            "f005fbea468ed191b98d44c447b868ba",
            "gold_label",
            "gold '$19.1 million' with scale thousand counts the unit twice",
        ),
        ("b610b2e2b8975ae32ecd9774c45d7979", "scale", "percent 1.83 vs gold ratio 0.02"),
        ("251074b99618ccfe6106b218f831ef94", "answer_form", "span missing its subject"),
        (
            "5da5c8d73c3c37c90a98aecb193f42df",
            "table_query",
            "picked the wrong row as the largest factor",
        ),
        ("602cf3cf43944b171a7af0db93773379", "answer_form", "span includes 'valued'"),
        (
            "0173550282fd1daf154239a105950031",
            "answer_form",
            "quoted 'Note 15'; gold is the longer phrase",
        ),
        ("8db7f46bf649c3befa9d8020ccad6874", "answer_form", "span cut short"),
        ("9dfa7018eae65bf2917063da6c671668", "table_query", "read 36.4; gold 32.8"),
        ("10e3df9f6f165a446ad57ee5b9a1a1ad", "scale", "paragraph quote given scale thousand"),
        ("8be7331907754560e0edbc5d0ad52b27", "answer_form", "span includes its subject"),
    ],
}


def rows(records_by_arm: dict[str, dict[str, dict]]) -> list[dict]:
    out = []
    for arm, labels in LABELS.items():
        for uid, bucket, note in labels:
            assert bucket in BUCKETS, bucket
            r = records_by_arm[arm][uid]
            assert r["em"] != 1, (arm, uid)
            out.append(
                {
                    "arm": arm,
                    "uid": uid,
                    "answer_from": r["answer_from"],
                    "answer_type": r["answer_type"],
                    "bucket": bucket,
                    "note": note,
                    "labelled_by": "claude",
                    "reviewed": "no",
                }
            )
    return out


def main() -> None:
    directory = OUT.parent
    records = {}
    for arm in LABELS:
        with open(directory / f"{arm}.predictions.jsonl", encoding="utf-8") as f:
            records[arm] = {r["uid"]: r for r in map(json.loads, f)}
    table = rows(records)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    tally = {arm: Counter(r["bucket"] for r in table if r["arm"] == arm) for arm in LABELS}
    print(json.dumps({arm: {b: t[b] for b in BUCKETS} for arm, t in tally.items()}, indent=2))


if __name__ == "__main__":
    main()
