# Official TAT-QA scorer (vendored, unmodified)

Copied byte for byte from https://github.com/NExTplusplus/TAT-QA at commit
`870accc41953dcde885aabeb963d94aabdc0fbc3` (MIT licence, see `LICENSE`).
`tests/test_scorer.py` checks these hashes so a local edit cannot slip in.

| File | SHA-256 |
|---|---|
| tatqa_metric.py | `2aeeac479f89f8c76300af1cc0e8d098eb86af84bc386b38b6ab4af484a6dea8` |
| tatqa_utils.py | `a84bb2f960737cf0a53733637a674cc4b20ef030a2be6a4b21dc2c4356f415ec` |
| tatqa_eval.py | `a4110626c477e7ed88f42b5edeb8449eaa9a4370740a627631b4a462bcf43762` |

The files import each other by bare module name (`from tatqa_utils import *`), so
`fingraph/eval/scorer.py` puts this directory on `sys.path` instead of editing them.
