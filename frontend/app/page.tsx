"use client";

import { useEffect, useRef, useState } from "react";
import { checkHealth, streamChat, type AgentEvent, type Citation } from "@/lib/api";

// Everything the trace panel shows for one answer. Collected from the SSE
// events as they arrive, so the panel reflects what actually happened rather
// than a summary written after the fact.
type Details = {
  route?: string;
  confidence?: number;
  rationale?: string;
  cache?: { status: string; tier?: string; score?: number; reason?: string };
  rewrite?: { original: string; resolved: string };
  guardrail?: { verdict: string; reason: string };
  tools: { tool: string; row_count: number; sql: string; notes: string[] }[];
  sources: { citation: string; url: string; rerank_score: number }[];
  steps: string[];
  elapsed_ms?: number;
};

type Message = {
  role: "user" | "assistant";
  text: string;
  citations?: Citation[];
  details?: Details;
};

const EXAMPLES = [
  "What is Dr. Maiya Gunasekara's consultation fee?",
  "Which doctors are available on Tuesday?",
  "What services does the Heart Centre offer?",
  "Which health package includes a Pap Smear?",
];

function emptyDetails(): Details {
  return { tools: [], sources: [], steps: [] };
}

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const [error, setError] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    checkHealth().then((h) => setOnline(h?.status === "ok"));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  async function send(question: string) {
    const text = question.trim();
    if (!text || busy) return;

    setInput("");
    setError("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", text }]);

    const details = emptyDetails();
    let answer = "";
    let citations: Citation[] = [];
    let started = false;

    const pushAssistant = () => {
      setMessages((m) => {
        const next = [...m];
        const last = next[next.length - 1];
        const payload: Message = {
          role: "assistant",
          text: answer,
          citations,
          details: { ...details },
        };
        if (last?.role === "assistant") next[next.length - 1] = payload;
        else next.push(payload);
        return next;
      });
    };

    try {
      for await (const ev of streamChat(text, threadId)) {
        const e = ev as AgentEvent & Record<string, any>;
        switch (e.type) {
          case "start":
            setThreadId(e.thread_id);
            break;
          case "rewrite":
            details.rewrite = { original: e.original, resolved: e.resolved };
            details.steps.push("resolved follow-up");
            break;
          case "guardrail":
            details.guardrail = { verdict: e.verdict, reason: e.reason };
            break;
          case "cache":
            details.cache = e;
            break;
          case "route":
            details.route = e.route;
            details.confidence = e.confidence;
            details.rationale = e.rationale;
            details.steps.push(`routed to ${e.route}`);
            break;
          case "tool_call":
            details.tools.push({
              tool: e.tool,
              row_count: e.row_count,
              sql: e.sql,
              notes: e.notes ?? [],
            });
            details.steps.push(`${e.tool} -> ${e.row_count} rows`);
            break;
          case "sources":
            details.sources = e.documents ?? [];
            details.steps.push(`retrieved ${details.sources.length} passages`);
            break;
          case "token":
            answer += e.text;
            if (!started) {
              started = true;
              setMessages((m) => [...m, { role: "assistant", text: "" }]);
            }
            pushAssistant();
            break;
          case "done":
            citations = e.citations ?? [];
            details.elapsed_ms = e.elapsed_ms;
            if (!details.route) details.route = e.route;
            if (!started) setMessages((m) => [...m, { role: "assistant", text: "" }]);
            answer = e.answer || answer;
            pushAssistant();
            break;
          case "error":
            setError(e.message);
            break;
        }
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? `Could not reach the assistant (${err.message}).`
          : "Could not reach the assistant."
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header>
        <span className="status">
          <span
            className={`dot ${online === null ? "" : online ? "ok" : "down"}`}
          />
          {online === null ? "checking" : online ? "connected" : "backend offline"}
        </span>
        <h1>Nawaloka Hospitals Assistant</h1>
        <p>Doctors, channeling times, lab tests, health packages and hospital services.</p>
      </header>

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            Ask about a doctor&apos;s fee, when a consultant channels, what a lab
            test costs, or what one of our centres offers.
            <ul className="examples">
              {EXAMPLES.map((q) => (
                <li key={q}>
                  <button onClick={() => send(q)}>{q}</button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="who">{m.role === "user" ? "You" : "Assistant"}</div>
            <div className="bubble">{m.text || "…"}</div>
            {m.role === "assistant" && m.details && (
              <MessageMeta message={m} />
            )}
          </div>
        ))}

        {busy && <div className="thinking">Thinking…</div>}
        {error && <div className="error">{error}</div>}
        <div ref={bottomRef} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question…"
          disabled={busy}
          autoFocus
        />
        <button type="submit" disabled={busy || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

function MessageMeta({ message }: { message: Message }) {
  const d = message.details!;
  const cacheHit = d.cache?.status === "hit";
  const alert = d.route === "EMERGENCY" || d.route === "REFUSE";

  return (
    <>
      <div className="badges">
        {d.route && (
          <span className={`badge route ${alert ? "alert" : ""}`}>{d.route}</span>
        )}
        {cacheHit && <span className="badge fast">instant answer</span>}
        {typeof d.confidence === "number" && (
          <span className="badge">confidence {d.confidence.toFixed(2)}</span>
        )}
        {typeof d.elapsed_ms === "number" && (
          <span className="badge">{d.elapsed_ms} ms</span>
        )}
      </div>

      {message.citations && message.citations.length > 0 && (
        <div className="sources">
          Sources:{" "}
          {message.citations.map((c, i) => (
            <span key={c.index}>
              {i > 0 && ", "}
              <a href={c.url} target="_blank" rel="noreferrer">
                [{c.index}] {c.label}
              </a>
            </span>
          ))}
        </div>
      )}

      <details className="trace">
        <summary>How this answer was produced</summary>
        <div className="trace-body">
          {d.steps.length > 0 && (
            <div className="trace-row steps">{d.steps.join(" → ")}</div>
          )}

          {d.rewrite && (
            <div className="trace-row">
              <div className="label">Follow-up resolved</div>
              <div>
                <code>{d.rewrite.original}</code> → <code>{d.rewrite.resolved}</code>
              </div>
            </div>
          )}

          {d.rationale && (
            <div className="trace-row">
              <div className="label">Why this route</div>
              <div>{d.rationale}</div>
            </div>
          )}

          {d.cache && (
            <div className="trace-row">
              <div className="label">Fast path</div>
              <div>
                {d.cache.status}
                {d.cache.tier ? ` (${d.cache.tier}` : ""}
                {d.cache.score ? `, score ${d.cache.score.toFixed(2)})` : d.cache.tier ? ")" : ""}
                {d.cache.reason ? ` — ${d.cache.reason}` : ""}
              </div>
            </div>
          )}

          {d.guardrail && (
            <div className="trace-row">
              <div className="label">Safety</div>
              <div>
                {d.guardrail.verdict} — {d.guardrail.reason}
              </div>
            </div>
          )}

          {d.tools.map((t, i) => (
            <div className="trace-row" key={i}>
              <div className="label">
                Database · {t.tool} · {t.row_count} row{t.row_count === 1 ? "" : "s"}
              </div>
              {t.sql && <pre className="sql">{t.sql}</pre>}
              {t.notes.map((n, j) => (
                <div className="note" key={j}>
                  {n}
                </div>
              ))}
            </div>
          ))}

          {d.sources.length > 0 && (
            <div className="trace-row">
              <div className="label">Retrieved passages</div>
              {d.sources.map((s, i) => (
                <div key={i}>
                  {s.citation}{" "}
                  <span className="steps">({s.rerank_score?.toFixed(3)})</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </details>
    </>
  );
}
