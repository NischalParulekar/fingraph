// Which parts of the report to mark for one answer.
//
// - "read": a table cell the arm read by its coordinates (the graph arms keep them), or a
//   passage it quoted from a paragraph;
// - "match": a table cell whose number equals a number in the answer or in one of the
//   arm's calculations (how the single agent's SQL results are traced back to cells);
// - "retrieved": a whole paragraph the arm's text search returned.

export function parseNumber(text) {
  if (text == null) return null;
  let s = String(text).trim();
  let negative = false;
  if (s.startsWith('(') && s.endsWith(')')) {
    negative = true;
    s = s.slice(1, -1);
  }
  s = s.replace(/[$€£¥%,\s]/g, '');
  if (s.startsWith('-')) {
    negative = !negative;
    s = s.slice(1);
  }
  if (!/^\d*\.?\d+$/.test(s)) return null;
  const value = Number(s);
  return negative ? -value : value;
}

const same = (a, b) => Math.round(Math.abs(a) * 100) === Math.round(Math.abs(b) * 100);

// Evidence comes as {text, location} from a finished run, or as the graph's own items
// ({source, row, column, paragraph, text}) in a pending review.
function locate(item) {
  if (item.source === 'table') return { kind: 'cell', row: item.row, column: item.column };
  if (item.source === 'text') return { kind: 'paragraph', order: item.paragraph };
  const cell = /^table row (\d+), column c(\d+)$/.exec(item.location || '');
  if (cell) return { kind: 'cell', row: Number(cell[1]), column: Number(cell[2]) };
  const para = /^paragraph (\d+)$/.exec(item.location || '');
  if (para) return { kind: 'paragraph', order: Number(para[1]) };
  return null;
}

export function marks(report, run) {
  const cells = new Map(); // "row,column" -> "read" | "match"
  const paragraphs = new Map(); // order -> { retrieved: bool, quotes: [text] }
  if (!report || !run) return { cells, paragraphs };

  const evidence = run.evidence || run.review?.evidence || [];
  for (const item of evidence) {
    const where = locate(item);
    if (!where || !item.text) continue;
    if (where.kind === 'cell') {
      cells.set(`${where.row},${where.column}`, 'read');
    } else {
      const para = report.paragraphs.find((p) => p.order === where.order);
      if (!para) continue;
      const entry = paragraphs.get(where.order) || { retrieved: false, quotes: [] };
      if (item.text.trim().length >= para.text.trim().length * 0.9) entry.retrieved = true;
      else entry.quotes.push(item.text);
      paragraphs.set(where.order, entry);
    }
  }

  const answer = run.answer?.length ? run.answer : run.review?.answer || [];
  const targets = [];
  for (const text of answer) {
    for (const token of String(text).match(/\(?-?[$€£]?\d[\d,]*\.?\d*\)?%?/g) || []) {
      const n = parseNumber(token);
      if (n !== null) targets.push(n);
    }
  }
  for (const calc of run.calculations || []) {
    // The graph arms write names ("current_2019"); only literal numbers count.
    const literals = calc.expression.replace(/[A-Za-z_][\w]*/g, ' ');
    for (const token of literals.match(/\d[\d,]*\.?\d*/g) || []) {
      const n = parseNumber(token);
      if (n !== null) targets.push(n);
    }
  }
  report.table.forEach((row, r) =>
    row.forEach((cell, c) => {
      const key = `${r},${c}`;
      if (cells.has(key)) return;
      const n = parseNumber(cell);
      if (n !== null && n !== 0 && targets.some((t) => same(t, n))) cells.set(key, 'match');
    }),
  );
  return { cells, paragraphs };
}
