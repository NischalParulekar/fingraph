import React, { useEffect, useMemo, useState } from 'react';
import Answer from './Answer';
import { OTHER_URL, RECORDED, ask, getArms, getQuestions, getReport, resume } from './api';
import { marks } from './evidence';
import Report from './Report';

const REPO = 'https://github.com/NischalParulekar/fingraph';
const SOURCE = { table: 'table', text: 'text', 'table-text': 'table and text' };

export default function App() {
  const [arms, setArms] = useState([]);
  const [questions, setQuestions] = useState([]);
  const [uid, setUid] = useState('');
  const [arm, setArm] = useState('graph_verifier');
  const [report, setReport] = useState(null);
  const [runs, setRuns] = useState({}); // arm -> {status, data | message, deciding}
  const [notice, setNotice] = useState('');

  useEffect(() => {
    Promise.all([getArms(), getQuestions()])
      .then(([armList, questionList]) => {
        setArms(armList);
        setQuestions(questionList);
        if (questionList.length) setUid(questionList[0].uid);
        if (!armList.some((a) => a.arm === 'graph_verifier') && armList.length) setArm(armList[0].arm);
      })
      .catch((e) => setNotice(e.message));
  }, []);

  useEffect(() => {
    if (!uid) return;
    setReport(null);
    setRuns({});
    getReport(uid)
      .then(setReport)
      .catch((e) => setNotice(e.message));
  }, [uid]);

  const current = arms.find((a) => a.arm === arm);
  const state = runs[arm];
  const marked = useMemo(
    () => marks(report, state?.status === 'done' ? state.data : null),
    [report, state],
  );
  const question = questions.find((q) => q.uid === uid);

  async function submit(event) {
    event.preventDefault();
    if (!uid || state?.status === 'loading') return;
    setNotice('');
    setRuns((now) => ({ ...now, [arm]: { status: 'loading' } }));
    try {
      const data = await ask(uid, arm);
      setRuns((now) => ({ ...now, [arm]: { status: 'done', data } }));
    } catch (e) {
      const wait = e.retryAfter ? ` Try again in ${e.retryAfter} s.` : '';
      setRuns((now) => ({ ...now, [arm]: { status: 'error', message: e.message + wait } }));
    }
  }

  async function decide(decision) {
    const run = runs[arm].data;
    setRuns((now) => ({ ...now, [arm]: { ...now[arm], deciding: true } }));
    try {
      const data = await resume(run.thread_id, decision);
      setRuns((now) => ({ ...now, [arm]: { status: 'done', data } }));
    } catch (e) {
      setRuns((now) => ({ ...now, [arm]: { ...now[arm], deciding: false } }));
      setNotice(e.message);
    }
  }

  return (
    <div className="page">
      <header className="masthead">
        <div>
          <h1>FinGraph</h1>
          <p className="lede">
            Ask a question about a company’s financial report and see the answer, the cells and
            sentences it rests on, and whether that evidence holds up.
          </p>
        </div>
        <p className="mode">
          {RECORDED ? 'Recorded runs, no live model' : 'Live API'}
          {OTHER_URL && (
            <>
              {' '}
              <a href={OTHER_URL}>{RECORDED ? 'Use the live API' : 'See the recorded runs'}</a>
            </>
          )}
        </p>
      </header>

      <form className="controls" onSubmit={submit}>
        <label className="field question-field">
          <span>Question</span>
          <select value={uid} onChange={(e) => setUid(e.target.value)}>
            {questions.map((q) => (
              <option key={q.uid} value={q.uid}>
                {q.question}
              </option>
            ))}
          </select>
        </label>
        <fieldset className="field arms">
          <legend>Answered by</legend>
          <div className="segmented">
            {arms.map((a) => (
              <label key={a.arm} className={a.arm === arm ? 'on' : undefined}>
                <input
                  type="radio"
                  name="arm"
                  value={a.arm}
                  checked={a.arm === arm}
                  onChange={() => setArm(a.arm)}
                />
                <span className="arm-letter">{a.short}</span> {a.name}
                {runs[a.arm]?.status === 'done' && <span className="dot" aria-label="answered" />}
              </label>
            ))}
          </div>
        </fieldset>
        <button type="submit" disabled={!uid || state?.status === 'loading'}>
          {state?.status === 'done' ? 'Ask again' : 'Ask'}
        </button>
      </form>

      {notice && (
        <p className="notice" role="alert">
          {notice}
        </p>
      )}

      {question && (
        <p className="question-meta">
          The answer is in the {SOURCE[question.answer_from]}; it is a{' '}
          {question.answer_type === 'arithmetic' ? 'calculation' : question.answer_type === 'count' ? 'count' : question.answer_type === 'multi-span' ? 'list of phrases' : 'phrase'}.
        </p>
      )}

      <main className="workspace">
        <Report report={report} marked={marked} />
        <aside className="answer-column" aria-live="polite">
          {current && report && <Answer arm={current} state={state} gold={report.gold} onDecide={decide} />}
        </aside>
      </main>

      <footer className="footer">
        <p>
          Three ways to answer the same question with the same model and tools: one tool-using agent (A), a
          LangGraph pipeline of planner, table, text and calculator steps (B), and that pipeline with an
          evidence verifier that retries, then pauses for a person (C). Questions and reports from TAT-QA.{' '}
          <a href={REPO}>Code, results and method</a>.
        </p>
      </footer>
    </div>
  );
}
