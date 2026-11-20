"""The `query_table` tool: SQL over one TAT-QA table, cell coordinates kept.

TAT-QA tables are small grids of report text with one or more header rows, a row-label
column and cells such as "1,320", "(19)" or "18.6%". Guessing clean headers and column
types from that is error-prone, so the grid is loaded as it is:

    t(row INTEGER, c0 VARCHAR, ..., cK VARCHAR, n0 DOUBLE, ..., nK DOUBLE)

`row` is the 0-based row of the grid and `cJ` the text of column J, so every value a
query returns can be traced to a cell (row, J). `nJ` is the same cell read as a number
(NULL if it is not one). Header rows are ordinary rows; the model reads them like any
other.
"""

import json
from dataclasses import dataclass

from fingraph.tools.numbers import parse_financial
from fingraph.tools.sandbox import TABLE_NAME, Sandbox, SandboxError

MAX_RESULT_ROWS = 50
MAX_ERROR_CHARS = 600


@dataclass(frozen=True)
class ToolResult:
    text: str  # what the model sees
    is_error: bool


class TableTool:
    def __init__(self, grid: tuple[tuple[str, ...], ...]):
        self.n_rows = len(grid)
        self.n_cols = max((len(r) for r in grid), default=0)
        cells = [list(r) + [""] * (self.n_cols - len(r)) for r in grid]
        columns = ["row"] + [f"c{j}" for j in range(self.n_cols)]
        columns += [f"n{j}" for j in range(self.n_cols)]
        types = ["INTEGER"] + ["VARCHAR"] * self.n_cols + ["DOUBLE"] * self.n_cols
        rows = [
            (i, *[c or None for c in row], *[parse_financial(c) for c in row])
            for i, row in enumerate(cells)
        ]
        self._sandbox = Sandbox(columns, types, rows, max_rows=MAX_RESULT_ROWS)

    def describe(self) -> str:
        last = self.n_cols - 1
        return (
            f"Table `{TABLE_NAME}` has {self.n_rows} rows and {self.n_cols} columns. Its "
            f"columns are `row` (0-based row number), `c0`..`c{last}` (each cell's text as "
            f"printed, NULL if empty) and `n0`..`n{last}` (the same cell read as a number, "
            "NULL if it is not one; parentheses mean negative, and $, commas and % are "
            "removed). The first rows are usually headers (periods, units, column titles) "
            "and `c0` usually holds the row labels."
        )

    def query(self, sql: str) -> ToolResult:
        try:
            result = self._sandbox.run(sql)
        except SandboxError as e:
            return ToolResult(f"{type(e).__name__}: {e}"[:MAX_ERROR_CHARS], True)
        payload = {"columns": list(result.columns), "rows": [list(r) for r in result.rows]}
        if result.truncated:
            payload["truncated"] = True
        return ToolResult(json.dumps(payload, ensure_ascii=False, default=str), False)

    def close(self) -> None:
        self._sandbox.close()
