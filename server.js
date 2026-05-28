import express from "express";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());

const API_KEY = process.env.GEMINI_API_KEY;

// Цепочка моделей по приоритету. Переопределяется через env MODEL_CHAIN.
const MODEL_CHAIN = (process.env.MODEL_CHAIN
  ? process.env.MODEL_CHAIN.split(",").map((s) => s.trim())
  : ["gemini-2.5-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"]
).filter(Boolean);

// Сколько доп. попыток на одну модель перед фолбеком и пауза между ними.
const RETRY_COUNT = Number(process.env.RETRY_COUNT ?? 1);
const RETRY_DELAY_MS = Number(process.env.RETRY_DELAY_MS ?? 600);

// Серверное состояние.
const downModels = new Set();
const requestLog = []; // последние запросы, новые сверху
const LOG_CAP = 50;
const stats = { served: 0, failed: 0 }; // только «реальные» запросы (с фолбеком)

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function geminiStream(model, prompt, signal) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse`;
  return fetch(url, {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json", "x-goog-api-key": API_KEY },
    body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
  });
}

// Разбор SSE-потока Gemini: вызывает onToken(text) на каждый кусок.
async function consumeStream(res, onToken) {
  const decoder = new TextDecoder();
  let buffer = "";
  let got = false;
  for await (const chunk of res.body) {
    buffer += decoder.decode(chunk, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;
      try {
        const json = JSON.parse(payload);
        const text =
          json?.candidates?.[0]?.content?.parts
            ?.map((p) => p.text)
            .filter(Boolean)
            .join("") || "";
        if (text) {
          got = true;
          onToken(text);
        }
      } catch {}
    }
  }
  return got;
}

app.get("/api/models", (_req, res) => {
  res.json({
    chain: MODEL_CHAIN.map((m) => ({ model: m, down: downModels.has(m) })),
    retryCount: RETRY_COUNT,
    retryDelayMs: RETRY_DELAY_MS,
  });
});

app.post("/api/toggle", (req, res) => {
  const { model } = req.body || {};
  if (!MODEL_CHAIN.includes(model)) {
    return res.status(400).json({ error: "unknown model" });
  }
  downModels.has(model) ? downModels.delete(model) : downModels.add(model);
  res.json({ model, down: downModels.has(model) });
});

app.get("/api/log", (_req, res) => res.json({ log: requestLog }));

app.get("/api/stats", (_req, res) => {
  res.json({
    served: stats.served,
    failed: stats.failed,
    modelsTotal: MODEL_CHAIN.length,
    modelsDown: MODEL_CHAIN.filter((m) => downModels.has(m)).length,
  });
});

// Поднять все модели (сброс хаоса).
app.post("/api/reset", (_req, res) => {
  downModels.clear();
  res.json({ ok: true });
});

// Стриминговый чат: SSE-события attempt / token / served / done / failed.
app.post("/api/chat", async (req, res) => {
  const prompt = (req.body?.prompt || "").toString().trim();

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);

  if (!prompt) {
    send({ type: "failed", error: "Empty prompt" });
    return res.end();
  }
  if (!API_KEY) {
    send({ type: "failed", error: "GEMINI_API_KEY is not set on the server" });
    return res.end();
  }

  // noFallback = режим «без фолбека»: только первая модель, без ретраев.
  const noFallback = !!req.body?.noFallback;
  const chain = noFallback ? MODEL_CHAIN.slice(0, 1) : MODEL_CHAIN;
  const retries = noFallback ? 0 : RETRY_COUNT;

  const attempts = [];
  let answer = "";
  let servedBy = null;

  outer: for (const model of chain) {
    if (downModels.has(model)) {
      const a = { model, status: "down", ms: 0, reason: "Forced offline (manual toggle)" };
      attempts.push(a);
      send({ type: "attempt", ...a });
      continue;
    }

    for (let tryNo = 0; tryNo <= retries; tryNo++) {
      const started = Date.now();
      const retry = tryNo > 0 ? { n: tryNo, of: retries } : null;
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 30000);
        const r = await geminiStream(model, prompt, controller.signal);

        if (!r.ok) {
          clearTimeout(timeout);
          let detail = "";
          try {
            detail = (await r.json())?.error?.message || "";
          } catch {}
          const a = {
            model, status: "error", ms: Date.now() - started, retry,
            reason: `HTTP ${r.status}${detail ? ": " + detail.slice(0, 140) : ""}`,
          };
          attempts.push(a);
          send({ type: "attempt", ...a });
          if (tryNo < retries) {
            send({ type: "retry", model, delayMs: RETRY_DELAY_MS, next: tryNo + 1, of: retries });
            await sleep(RETRY_DELAY_MS);
            continue;
          }
          continue outer; // фолбек на следующую модель
        }

        // Поток открылся успешно — стримим токены и фиксируем модель.
        send({ type: "served", model });
        servedBy = model;
        const onToken = (t) => {
          answer += t;
          send({ type: "token", text: t });
        };
        await consumeStream(r, onToken);
        clearTimeout(timeout);

        const a = { model, status: "ok", ms: Date.now() - started, retry };
        attempts.push(a);
        send({ type: "attempt", ...a });
        send({ type: "done", servedBy: model, fallbacks: attempts.filter((x) => x.status !== "ok").length });
        if (!noFallback) logRequest({ prompt, servedBy: model, attempts, ok: true });
        return res.end();
      } catch (e) {
        const a = {
          model, status: "error", ms: Date.now() - started, retry,
          reason: e.name === "AbortError" ? "Timeout (30s)" : String(e.message || e),
        };
        attempts.push(a);
        send({ type: "attempt", ...a });
        if (tryNo < retries) {
          send({ type: "retry", model, delayMs: RETRY_DELAY_MS, next: tryNo + 1, of: retries });
          await sleep(RETRY_DELAY_MS);
          continue;
        }
        continue outer;
      }
    }
  }

  send({ type: "failed", error: "All models in the chain are unavailable" });
  if (!noFallback) logRequest({ prompt, servedBy: null, attempts, ok: false });
  res.end();
});

function logRequest({ prompt, servedBy, attempts, ok }) {
  ok ? stats.served++ : stats.failed++;
  requestLog.unshift({
    ts: new Date().toISOString(),
    prompt: prompt.slice(0, 120),
    servedBy,
    ok,
    fallbacks: attempts.filter((a) => a.status !== "ok").length,
    attempts: attempts.map((a) => ({ model: a.model, status: a.status, ms: a.ms })),
  });
  if (requestLog.length > LOG_CAP) requestLog.length = LOG_CAP;
}

app.use(express.static(join(__dirname, "public")));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Gemini fallback demo → http://localhost:${PORT}`);
  console.log("Chain:", MODEL_CHAIN.join(" → "), `| retries: ${RETRY_COUNT}, delay: ${RETRY_DELAY_MS}ms`);
});
