import { API_BASE } from '../utils/constants';

/**
 * NeuroLens API Service
 * Handles all HTTP communication with the backend.
 */

export async function analyzeText(text, sessionId = null, explain = true) {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, session_id: sessionId, explain }),
  });
  if (!response.ok) throw new Error(`Analysis failed: ${response.statusText}`);
  return response.json();
}

export async function batchAnalyze(texts, sessionId = null) {
  const response = await fetch(`${API_BASE}/batch-analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts, session_id: sessionId }),
  });
  if (!response.ok) throw new Error(`Batch analysis failed: ${response.statusText}`);
  return response.json();
}

export async function getMetrics() {
  const response = await fetch(`${API_BASE}/metrics`);
  if (!response.ok) throw new Error(`Failed to fetch metrics: ${response.statusText}`);
  return response.json();
}

export async function getTimeline(sessionId) {
  const response = await fetch(`${API_BASE}/timeline`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!response.ok) throw new Error(`Failed to fetch timeline: ${response.statusText}`);
  return response.json();
}

export async function triggerTraining(config) {
  const response = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!response.ok) throw new Error(`Training failed: ${response.statusText}`);
  return response.json();
}
