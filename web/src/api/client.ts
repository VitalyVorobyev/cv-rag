import type {
  AnswerRequest,
  ChunkResponse,
  HealthResponse,
  PaperDetailResponse,
  PaperListResponse,
  RouteInfo,
  SearchResponse,
  StatsResponse,
} from './types';

const BASE = '/api';

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  return fetchJson('/health');
}

export async function fetchStats(topVenues = 10): Promise<StatsResponse> {
  return fetchJson(`/stats?top_venues=${topVenues}`);
}

export async function fetchPapers(
  offset = 0,
  limit = 20,
  search = '',
): Promise<PaperListResponse> {
  const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
  if (search) params.set('search', search);
  return fetchJson(`/papers?${params}`);
}

export async function fetchPaper(arxivId: string): Promise<PaperDetailResponse> {
  return fetchJson(`/papers/${encodeURIComponent(arxivId)}`);
}

export async function searchChunks(
  query: string,
  topK = 8,
): Promise<SearchResponse> {
  return fetchJson('/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  });
}

// SSE event types from the answer endpoint
export interface AnswerSSECallbacks {
  onRoute: (route: RouteInfo) => void;
  onSources: (sources: ChunkResponse[]) => void;
  onToken: (token: string) => void;
  onRepair: (repaired: string) => void;
  onDone: (response: {
    answer: string;
    sources: ChunkResponse[];
    route: RouteInfo;
    citation_valid: boolean;
    citation_reason: string;
    elapsed_ms: number;
  }) => void;
  onError: (error: { message: string; [key: string]: unknown }) => void;
}

export function streamAnswer(
  request: AnswerRequest,
  callbacks: AnswerSSECallbacks,
): AbortController {
  const controller = new AbortController();

  fetch(`${BASE}/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const text = await res.text();
        callbacks.onError({ message: `API error ${res.status}: ${text}` });
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        callbacks.onError({ message: 'No response body' });
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        let currentEvent = '';
        let currentData = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
            currentData = '';
          } else if (line.startsWith('data: ')) {
            currentData += (currentData ? '\n' : '') + line.slice(6);
          } else if (line === '' && currentEvent) {
            dispatchSSE(currentEvent, currentData, callbacks);
            currentEvent = '';
            currentData = '';
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        callbacks.onError({ message: String(err) });
      }
    });

  return controller;
}

function dispatchSSE(
  event: string,
  data: string,
  callbacks: AnswerSSECallbacks,
) {
  try {
    switch (event) {
      case 'route':
        callbacks.onRoute(JSON.parse(data));
        break;
      case 'sources':
        callbacks.onSources(JSON.parse(data));
        break;
      case 'token':
        callbacks.onToken(data);
        break;
      case 'repair':
        callbacks.onRepair(data);
        break;
      case 'done':
        callbacks.onDone(JSON.parse(data));
        break;
      case 'error':
        callbacks.onError(JSON.parse(data));
        break;
    }
  } catch {
    callbacks.onError({ message: `Failed to parse SSE event: ${event}` });
  }
}
