import React, { useState } from 'react';
import { RECORDED } from './api';

// One arm's answer: the answer itself, what the verifier made of it, the calculations,
// a reviewer's controls when the run is paused, and the gold answer to compare with.

const SCALE_WORDS = { thousand: 'thousand', million: 'million', billion: 'billion', percent: '%' };

export function formatAnswer(answer, scale) {
  const text = answer.join('; ');
  if (!scale) return text;
  return scale === 'percent' ? `${text}%` : `${text} ${SCALE_WORDS[scale]}`;
}

function goldText(gold) {
  const items = Array.isArray(gold.answer) ? gold.answer.map(String) : [String(gold.answer)];
  return formatAnswer(items, gold.scale);
}

function Claims({ report }) {
  const claims = [
    ...report.supported.map((c) => ({ ...c, ok: true })),
    ...report.unsupported.map((c) => ({ ...c, ok: false })),
  ];
  return (
    <ul className="claims">
      {claims.map((c, i) => (
        <li key={i} className={c.ok ? 'ok' : 'bad'}>
          <span className="claim-mark" aria-hidden="true">
            {c.ok ? '✓' : '✗'}
          </span>
          <span>
            <strong>{c.value || '(no answer)'}</strong>
            {c.ok ? ` found in ${c.support}` : ' not found in the evidence'}
          </span>
        </li>
      ))}
      {report.scale_problems.map((p, i) => (
        <li key={`s${i}`} className="bad">
          <span className="claim-mark" aria-hidden="true">✗</span>
          <span>Scale {p}</span>
        </li>
      ))}
    </ul>
  );
}

function Verdict({ report, source }) {
  return (
    <div className={`verdict ${report.passed ? 'ok' : 'bad'}`}>
      <p className="verdict-line">
        {report.passed ? 'Every part of this answer is in the evidence.' : 'Part of this answer is not in the evidence.'}
      </p>
      <p className="note">
        {source === 'arm'
          ? 'Checked by the verifier inside this arm.'
          : 'Checked afterwards with the same verifier Arm C uses; this arm does not use it.'}
      </p>
      <Claims report={report} />
    </div>
  );
}

function Review({ run, onDecide, busy }) {
  const pending = run.review;
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(pending.answer.join('; '));
  return (
    <div className="review" role="region" aria-label="Review">
      <p className="review-title">Waiting for a reviewer</p>
      <p>
        After its retries, this draft still failed the evidence check, so the run paused instead of
        answering. Approve it, correct it, or reject it to leave the question unanswered.
      </p>
      <p className="draft">{formatAnswer(pending.answer, pending.scale)}</p>
      <Claims report={pending.verifier_report} />
      {editing ? (
        <form
          className="edit"
          onSubmit={(e) => {
            e.preventDefault();
            const answer = value.split(';').map((s) => s.trim()).filter(Boolean);
            if (answer.length) onDecide({ action: 'edit', answer, scale: pending.scale });
          }}
        >
          <label htmlFor="edit-answer">Corrected answer (separate items with ;), scale stays “{pending.scale || 'none'}”</label>
          <input id="edit-answer" value={value} onChange={(e) => setValue(e.target.value)} />
          <div className="actions">
            <button type="submit" disabled={busy}>Save the correction</button>
            <button type="button" className="quiet" onClick={() => setEditing(false)}>
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <div className="actions">
          <button disabled={busy} onClick={() => onDecide({ action: 'approve' })}>
            Approve the draft
          </button>
          <button disabled={busy} className="quiet" onClick={() => setEditing(true)}>
            Correct it
          </button>
          <button disabled={busy} className="quiet" onClick={() => onDecide({ action: 'reject' })}>
            Reject it
          </button>
        </div>
      )}
    </div>
  );
}

function RecordedDraft({ draft }) {
  return (
    <div className="review recorded">
      <p className="review-title">No answer: the draft failed the evidence check</p>
      <p>
        In the evaluation, a run that still fails after two retries abstains. In the live API the same run
        pauses here for a reviewer to approve, correct or reject.
      </p>
      <p className="draft">{formatAnswer(draft.answer, draft.scale)}</p>
      <Claims report={draft.verifier_report} />
    </div>
  );
}

export default function Answer({ arm, state, gold, onDecide }) {
  if (!state) {
    return (
      <div className="answer empty">
        <p>Ask {arm.name.toLowerCase()} to answer this question. Its evidence will be marked in the report.</p>
      </div>
    );
  }
  if (state.status === 'loading') {
    return (
      <div className="answer" aria-busy="true">
        <p className="working">{arm.name} is working…</p>
      </div>
    );
  }
  if (state.status === 'error') {
    return (
      <div className="answer">
        <p className="error" role="alert">{state.message}</p>
      </div>
    );
  }

  const run = state.data;
  const pending = run.status === 'pending_review';
  return (
    <div className="answer">
      {pending ? (
        <Review run={run} onDecide={onDecide} busy={state.deciding} />
      ) : run.abstained ? (
        run.draft ? (
          <RecordedDraft draft={run.draft} />
        ) : (
          <p className="no-answer">
            No answer{run.review?.action === 'reject' ? ': the reviewer rejected the draft.' : '.'}
          </p>
        )
      ) : (
        <>
          <p className="answer-value">{formatAnswer(run.answer, run.scale)}</p>
          {run.review && (
            <p className="note">
              {run.review.action === 'approve' ? 'Approved by a reviewer.' : 'Corrected by a reviewer.'}
            </p>
          )}
        </>
      )}

      <div className="gold">
        <span>TAT-QA’s answer: {goldText(gold)}</span>
        {RECORDED && !pending && (
          <span className={run.exact_match ? 'ok' : 'bad'}>
            {run.exact_match ? '✓ exact match' : '✗ not an exact match'} (official scorer)
          </span>
        )}
      </div>

      {!pending && run.verifier && !run.abstained && <Verdict report={run.verifier} source={run.verifier.source} />}

      {!pending && run.calculations?.length > 0 && (
        <div className="calcs">
          <p className="section-title">Calculated with the calculator tool</p>
          {run.calculations.map((c, i) => (
            <p key={i} className="calc">
              {c.expression} = {Number(c.result.toFixed(6))}
            </p>
          ))}
        </div>
      )}

      <p className="meta">
        {[
          `${run.llm_calls ?? 0} model calls`,
          `$${(run.cost_usd ?? 0).toFixed(5)}`,
          `${((run.latency_ms ?? run.arm_latency_ms ?? 0) / 1000).toFixed(1)} s${run.cached ? ' when first answered' : ''}`,
          run.cached && 'now from the cache',
          run.recorded && 'recorded run',
        ]
          .filter(Boolean)
          .join(', ')}
      </p>
    </div>
  );
}
