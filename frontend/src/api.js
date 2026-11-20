// Thin client for the FinGraph API. VITE_API_URL is empty in development (the dev
// server proxies to the API) and the API's address in a deployed build. With
// VITE_RECORDED=1 there is no API at all: answers come from demo.json, recorded from the
// committed dev runs (scripts/build_demo.py).
const BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');
// `npm run dev:recorded` (vite --mode recorded) does the same locally.
export const RECORDED =
  import.meta.env.VITE_RECORDED === '1' || import.meta.env.MODE === 'recorded';
export const OTHER_URL = import.meta.env.VITE_OTHER_URL || '';

const SHORT = { single_agent: 'A', graph: 'B', graph_verifier: 'C' };

export class ApiError extends Error {
  constructor(message, status, retryAfter) {
    super(message);
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(BASE + path, options);
  } catch {
    throw new ApiError('The API could not be reached. Check that it is running.', 0);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail =
      typeof body.detail === 'string' ? body.detail : `The request failed (${response.status}).`;
    throw new ApiError(detail, response.status, Number(response.headers.get('retry-after')) || null);
  }
  return body;
}

let bundle = null;
async function demo() {
  if (!bundle) {
    const r = await fetch(`${import.meta.env.BASE_URL}demo.json`);
    if (!r.ok) throw new ApiError('The recorded demo could not be loaded.', r.status);
    bundle = await r.json();
  }
  return bundle;
}

export async function getArms() {
  if (RECORDED) return (await demo()).arms;
  const arms = await request('/arms');
  return arms.map((a) => {
    const name = a.name.replace(/^[ABC]: /, '');
    return { arm: a.arm, name: name[0].toUpperCase() + name.slice(1), short: SHORT[a.arm] };
  });
}

export async function getQuestions() {
  if (RECORDED) return (await demo()).questions;
  return request('/questions');
}

export async function getReport(uid) {
  if (RECORDED) return (await demo()).reports[uid];
  return request(`/reports/${encodeURIComponent(uid)}`);
}

export async function ask(uid, arm) {
  if (!RECORDED) {
    return request('/ask', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ uid, arm }),
    });
  }
  const run = (await demo()).answers[uid]?.[arm];
  if (!run) throw new ApiError('This question has no recorded answer for that arm.', 404);
  // A short pause, so a recorded answer does not look instant; its real time is shown.
  await new Promise((resolve) => setTimeout(resolve, 300));
  return { ...run, recorded: true };
}

export function resume(threadId, decision) {
  if (RECORDED) {
    return Promise.reject(new ApiError('Reviews need the live API; this page replays recorded runs.', 0));
  }
  return request(`/runs/${encodeURIComponent(threadId)}/resume`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(decision),
  });
}
