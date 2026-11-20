import React from 'react';

// The report the question is about: its table and paragraphs, with the evidence for the
// current answer marked.

function Quoted({ text, quotes }) {
  if (!quotes.length) return text;
  // Mark each quote where it first appears (case-insensitive), left to right.
  const lower = text.toLowerCase();
  const spans = [];
  for (const q of quotes) {
    const at = lower.indexOf(q.toLowerCase().trim());
    if (at >= 0) spans.push([at, at + q.trim().length]);
  }
  spans.sort((a, b) => a[0] - b[0]);
  const parts = [];
  let cursor = 0;
  for (const [start, end] of spans) {
    if (start < cursor) continue;
    parts.push(text.slice(cursor, start));
    parts.push(
      <mark key={start} className="read">
        {text.slice(start, end)}
      </mark>,
    );
    cursor = end;
  }
  parts.push(text.slice(cursor));
  return parts;
}

export default function Report({ report, marked }) {
  if (!report) return <section className="report" aria-busy="true" />;
  const width = Math.max(...report.table.map((row) => row.length));
  return (
    <section className="report" aria-label="The report">
      <div className="table-wrap">
        <table>
          <tbody>
            {report.table.map((row, r) => (
              <tr key={r}>
                {Array.from({ length: width }, (_, c) => {
                  const kind = marked.cells.get(`${r},${c}`);
                  const text = row[c] ?? '';
                  const numeric = /\d/.test(text) && c > 0;
                  return (
                    <td
                      key={c}
                      className={[kind, numeric ? 'num' : '', c === 0 ? 'label' : '']
                        .filter(Boolean)
                        .join(' ')}
                      title={kind === 'read' ? 'Read by the arm' : kind === 'match' ? 'Matches a number in the answer or a calculation' : undefined}
                    >
                      {text}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="paragraphs">
        {report.paragraphs.map((p) => {
          const entry = marked.paragraphs.get(p.order);
          return (
            <p key={p.order} className={entry?.retrieved ? 'retrieved' : undefined}>
              <span className="para-number" aria-label={`Paragraph ${p.order}`}>
                {p.order}
              </span>
              <Quoted text={p.text} quotes={entry?.quotes || []} />
            </p>
          );
        })}
      </div>

      <ul className="legend" aria-label="What the marks mean">
        <li>
          <span className="swatch read" /> read or quoted by the arm
        </li>
        <li>
          <span className="swatch match" /> matches a number in its answer or calculation
        </li>
        <li>
          <span className="swatch retrieved" /> paragraph its text search returned
        </li>
      </ul>
    </section>
  );
}
