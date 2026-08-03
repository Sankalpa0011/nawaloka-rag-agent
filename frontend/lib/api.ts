// Talks to the FastAPI backend.
//
// The browser's built-in EventSource only does GET, and the chat endpoint needs
// a POST body, so the SSE stream is parsed by hand off fetch().

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

export type Citation = {
  index: number;
  label: string;
  url: string;
  source: string;
};

export type TraceEvent = Record<string, unknown> & { node?: string };

export type AgentEvent =
  | { type: "start"; thread_id: string }
  | { type: "rewrite"; original: string; resolved: string }
  | { type: "guardrail"; verdict: string; reason: string }
  | { type: "cache"; status: string; tier?: string; score?: number; reason?: string }
  | { type: "route"; route: string; confidence: number; rationale: string }
  | { type: "tool_call"; tool: string; row_count: number; sql: string; notes: string[] }
  | { type: "sources"; documents: { citation: string; url: string; rerank_score: number }[] }
  | { type: "token"; text: string }
  | { type: "trace"; trace: TraceEvent[] }
  | { type: "done"; answer: string; route: string; elapsed_ms: number; citations: Citation[] }
  | { type: "error"; message: string; detail?: string };

export async function* streamChat(
  message: string,
  threadId: string | null,
  signal?: AbortSignal
): AsyncGenerator<AgentEvent> {
  const response = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, thread_id: threadId }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`backend returned ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventName = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    // The last element may be a partial line; keep it for the next chunk.
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        const raw = line.slice(5).trim();
        if (!raw) continue;
        try {
          yield { type: eventName, ...JSON.parse(raw) } as AgentEvent;
        } catch {
          // A malformed frame should not kill the whole stream.
          console.warn("could not parse SSE frame", raw.slice(0, 120));
        }
      }
    }
  }
}

export async function checkHealth(): Promise<{ status: string } | null> {
  try {
    const r = await fetch(`${API}/health`);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}
