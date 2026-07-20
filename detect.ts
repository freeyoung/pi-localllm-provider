// ─── Backend detection chain ───────────────────────────────────────
//
// Tries backend-specific endpoints first (richer metadata: context window,
// reasoning/vision capability, max response tokens), falling back to the
// generic OpenAI-compatible /v1/models endpoint when none match.
//
// Field names below were taken directly from each backend's source:
//   - oMLX / LM Studio: github.com/monroewilliams/pi-local's model-picker.ts
//   - MTPLX: github.com/youssofal/MTPLX's mtplx/server/openai.py (/health, /v1/models)
//   - llama.cpp: ggml-org/llama.cpp's tools/server/README.md (/props, /v1/models)
//   - Ollama: ollama/ollama's docs/api.md (/api/tags, /api/show, /api/ps) and
//     types/model/capability.go (capability string constants)
//   - vLLM: vllm-project/vllm's vllm/entrypoints/openai/models/serving.py
//     (/v1/models ModelCard) and vllm/entrypoints/serve/instrumentator/basic.py
//     (/version). vLLM's ModelCard only ever carries {id, max_model_len} — no
//     reasoning/vision signal exists anywhere in its OpenAI-compatible API, so
//     detectVllm exists purely to label the backend correctly; it extracts the
//     same max_model_len the generic OpenAI probe already reads.

export type ApiType = "mtplx" | "omlx" | "lmstudio" | "llamacpp" | "ollama" | "vllm" | "openai";

export interface DiscoveredModel {
  id: string;
  name: string;
  contextWindow: number;
  maxTokens: number;
  reasoning: boolean;
  input: ("text" | "image")[];
  // Not every backend can report these — undefined means "unknown", not "no".
  loaded?: boolean;
  sizeBytes?: number;
  quantization?: string;
}

export interface DetectResult {
  apiType: ApiType;
  models: DiscoveredModel[];
  // Set only when models is empty *because something went wrong* (auth
  // failure, timeout, unreachable) — distinct from a server that responded
  // fine and genuinely has zero models loaded, which leaves this undefined.
  error?: string;
}

type ProbeFailureReason = "unauthorized" | "forbidden" | "timeout" | "network-error" | "http-error";

interface ProbeDiagnostic {
  url: string;
  reason: ProbeFailureReason;
  status?: number;
}

// Fallback timeout for the individual detectXxx functions when called
// standalone (e.g. directly in tests) without a signal. detectModels always
// supplies its own chain-wide signal (see below), so this is a safety net
// that production code never actually hits.
const STANDALONE_PROBE_TIMEOUT_MS = 5000;

function recordFailure(diagnostics: ProbeDiagnostic[] | undefined, url: string, status: number): void {
  diagnostics?.push({
    url,
    status,
    reason: status === 401 ? "unauthorized" : status === 403 ? "forbidden" : "http-error",
  });
}

function recordException(diagnostics: ProbeDiagnostic[] | undefined, url: string, err: unknown): void {
  diagnostics?.push({
    url,
    reason: err instanceof Error && err.name === "AbortError" ? "timeout" : "network-error",
  });
}

async function fetchJson<T>(
  url: string,
  apiKey: string,
  signal?: AbortSignal,
  diagnostics?: ProbeDiagnostic[],
): Promise<T | null> {
  try {
    const headers: Record<string, string> = {};
    if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;
    const res = await fetch(url, {
      headers,
      signal: signal ?? AbortSignal.timeout(STANDALONE_PROBE_TIMEOUT_MS),
    });
    if (!res.ok) {
      recordFailure(diagnostics, url, res.status);
      return null;
    }
    return (await res.json()) as T;
  } catch (err) {
    recordException(diagnostics, url, err);
    return null;
  }
}

async function postJson<T>(
  url: string,
  apiKey: string,
  body: unknown,
  signal?: AbortSignal,
  diagnostics?: ProbeDiagnostic[],
): Promise<T | null> {
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;
    const res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: signal ?? AbortSignal.timeout(STANDALONE_PROBE_TIMEOUT_MS),
    });
    if (!res.ok) {
      recordFailure(diagnostics, url, res.status);
      return null;
    }
    return (await res.json()) as T;
  } catch (err) {
    recordException(diagnostics, url, err);
    return null;
  }
}

function capTokens(contextWindow: number): number {
  return Math.min(Math.floor(contextWindow / 2), 8192);
}

// ─── MTPLX ──────────────────────────────────────────────────────────
// No /v1/models/status or /api/v1/models — only a single active model,
// described richly by /health (context_window, max_response_tokens,
// enable_thinking, vision.enabled). No loaded/unloaded distinction exists
// (there's only ever the one model /health describes), so loaded is left
// undefined — same as the generic OpenAI probe.

interface MtplxHealth {
  model?: string;
  context_window?: number;
  max_response_tokens?: number;
  reasoning?: string;
  enable_thinking?: boolean;
  vision?: { enabled?: boolean };
}

export async function detectMtplx(
  root: string,
  apiKey: string,
  signal?: AbortSignal,
  diagnostics?: ProbeDiagnostic[],
): Promise<DetectResult | null> {
  const health = await fetchJson<MtplxHealth>(`${root}/health`, apiKey, signal, diagnostics);
  if (!health?.model || typeof health.context_window !== "number") return null;

  const reasoning = health.enable_thinking === true || health.reasoning === "on";
  const vision = health.vision?.enabled === true;

  return {
    apiType: "mtplx",
    models: [
      {
        id: health.model,
        name: health.model.split("/").pop() ?? health.model,
        contextWindow: health.context_window,
        maxTokens: health.max_response_tokens ?? capTokens(health.context_window),
        reasoning,
        input: vision ? ["text", "image"] : ["text"],
      },
    ],
  };
}

// ─── oMLX ───────────────────────────────────────────────────────────

interface OmlxModelsStatus {
  models?: Array<{
    id: string;
    display_name?: string | null;
    model_alias?: string | null;
    max_context_window?: number;
    max_tokens?: number;
    thinking_default?: boolean | null;
    model_type?: string | null;
    loaded?: boolean;
    estimated_size?: number;
  }>;
}

export async function detectOmlx(
  root: string,
  apiKey: string,
  signal?: AbortSignal,
  diagnostics?: ProbeDiagnostic[],
): Promise<DetectResult | null> {
  const res = await fetchJson<OmlxModelsStatus>(`${root}/v1/models/status`, apiKey, signal, diagnostics);
  if (!res?.models?.length) return null;

  const models: DiscoveredModel[] = [];
  for (const m of res.models) {
    if (!m.id || !m.model_type) continue;
    const type = m.model_type.toLowerCase();
    if (type !== "llm" && type !== "vlm") continue;
    const contextWindow = m.max_context_window ?? 32768;
    models.push({
      id: m.id,
      name: m.model_alias || m.display_name || m.id,
      contextWindow,
      maxTokens: m.max_tokens ?? capTokens(contextWindow),
      reasoning: m.thinking_default === true,
      input: type === "vlm" ? ["text", "image"] : ["text"],
      loaded: m.loaded === true,
      sizeBytes: m.estimated_size,
    });
  }
  return models.length > 0 ? { apiType: "omlx", models } : null;
}

// ─── LM Studio ──────────────────────────────────────────────────────

interface LmStudioModels {
  models?: Array<{
    key: string;
    display_name?: string;
    type?: string;
    max_context_length?: number;
    capabilities?: { vision?: boolean; reasoning?: unknown };
    loaded_instances?: unknown[];
    size_bytes?: number;
    quantization?: { name: string };
  }>;
}

export async function detectLmStudio(
  root: string,
  apiKey: string,
  signal?: AbortSignal,
  diagnostics?: ProbeDiagnostic[],
): Promise<DetectResult | null> {
  const res = await fetchJson<LmStudioModels>(`${root}/api/v1/models`, apiKey, signal, diagnostics);
  if (!res?.models?.length) return null;

  const models: DiscoveredModel[] = [];
  for (const m of res.models) {
    const type = (m.type ?? "").toLowerCase();
    if (type !== "llm" && type !== "vlm") continue;
    const contextWindow = m.max_context_length ?? 32768;
    models.push({
      id: m.key,
      name: m.display_name || m.key,
      contextWindow,
      maxTokens: capTokens(contextWindow),
      reasoning: !!m.capabilities?.reasoning,
      input: m.capabilities?.vision || type === "vlm" ? ["text", "image"] : ["text"],
      loaded: (m.loaded_instances?.length ?? 0) > 0,
      sizeBytes: m.size_bytes,
      quantization: m.quantization?.name,
    });
  }
  return models.length > 0 ? { apiType: "lmstudio", models } : null;
}

// ─── llama.cpp server (llama-server) ───────────────────────────────
// /props has the runtime-configured context (n_ctx) and vision support;
// /v1/models has the id (respects --alias), the model's trained max
// context (n_ctx_train) as a fallback ceiling, and file size. No
// loaded/unloaded distinction exists — one server process, one model —
// so loaded is left undefined, same as the generic OpenAI probe.

interface LlamaCppProps {
  default_generation_settings?: { n_ctx?: number };
  model_path?: string;
  modalities?: { vision?: boolean };
}

interface LlamaCppModels {
  data?: Array<{ id: string; meta?: { n_ctx_train?: number; size?: number } | null }>;
}

export async function detectLlamaCpp(
  root: string,
  apiKey: string,
  signal?: AbortSignal,
  diagnostics?: ProbeDiagnostic[],
): Promise<DetectResult | null> {
  const props = await fetchJson<LlamaCppProps>(`${root}/props`, apiKey, signal, diagnostics);
  if (typeof props?.default_generation_settings?.n_ctx !== "number" || !props.model_path) {
    return null;
  }

  const modelsRes = await fetchJson<LlamaCppModels>(`${root}/v1/models`, apiKey, signal, diagnostics);
  const entry = modelsRes?.data?.[0];

  const contextWindow =
    props.default_generation_settings.n_ctx || entry?.meta?.n_ctx_train || 32768;
  const id = entry?.id ?? props.model_path;

  return {
    apiType: "llamacpp",
    models: [
      {
        id,
        name: id.split(/[\\/]/).pop() ?? id,
        contextWindow,
        maxTokens: capTokens(contextWindow),
        reasoning: false,
        input: props.modalities?.vision ? ["text", "image"] : ["text"],
        sizeBytes: entry?.meta?.size,
      },
    ],
  };
}

// ─── Ollama (native API) ────────────────────────────────────────────
// Ollama's OpenAI-compat /v1/models carries no context/capability info at
// all, so this always beats the generic fallback. /api/tags lists locally
// pulled models (with size + quantization_level already inline); /api/show
// per model has model_info["<arch>.context_length"] (arch comes from
// model_info["general.architecture"] — see ollama/ollama's cmd/cmd.go) and
// a capabilities array ("thinking", "vision", ...); /api/ps lists which of
// those models are actually loaded into memory right now.

interface OllamaTags {
  models?: Array<{
    name: string;
    model: string;
    size?: number;
    details?: { quantization_level?: string };
  }>;
}

interface OllamaShow {
  model_info?: Record<string, unknown>;
  capabilities?: string[];
}

interface OllamaPs {
  models?: Array<{ model: string }>;
}

export async function detectOllama(
  root: string,
  apiKey: string,
  signal?: AbortSignal,
  diagnostics?: ProbeDiagnostic[],
): Promise<DetectResult | null> {
  const tags = await fetchJson<OllamaTags>(`${root}/api/tags`, apiKey, signal, diagnostics);
  if (!tags?.models?.length) return null;

  const [shows, ps] = await Promise.all([
    Promise.all(
      tags.models.map((m) =>
        postJson<OllamaShow>(`${root}/api/show`, apiKey, { model: m.model }, signal, diagnostics),
      ),
    ),
    fetchJson<OllamaPs>(`${root}/api/ps`, apiKey, signal, diagnostics),
  ]);
  const runningModels = new Set(ps?.models?.map((m) => m.model) ?? []);

  const models: DiscoveredModel[] = tags.models.map((m, i) => {
    const show = shows[i];
    const info = show?.model_info;
    const arch = typeof info?.["general.architecture"] === "string" ? info["general.architecture"] : undefined;
    const rawContextWindow = arch ? info?.[`${arch}.context_length`] : undefined;
    const contextWindow = typeof rawContextWindow === "number" ? rawContextWindow : 32768;
    const capabilities = show?.capabilities ?? [];

    return {
      id: m.model,
      name: m.name,
      contextWindow,
      maxTokens: capTokens(contextWindow),
      reasoning: capabilities.includes("thinking"),
      input: capabilities.includes("vision") ? ["text", "image"] : ["text"],
      loaded: runningModels.has(m.model),
      sizeBytes: m.size,
      quantization: m.details?.quantization_level,
    };
  });

  return { apiType: "ollama", models };
}

// ─── vLLM ────────────────────────────────────────────────────────────
// vLLM's ModelCard only ever carries {id, max_model_len} — the same field
// the generic OpenAI probe already reads — so this exists to label the
// backend correctly, not to extract anything new. /version is vLLM-specific
// but its shape ({version: string}) is generic enough that a false-positive
// match elsewhere is plausible, so this only claims "vllm" when /v1/models
// *also* has max_model_len on at least one entry (OpenAI's real API never
// has this field). No loaded/unloaded distinction exists — one process,
// one model — so loaded is left undefined, same as the generic OpenAI probe.

interface VllmVersion {
  version?: string;
}

interface VllmModels {
  data?: Array<{ id: string; max_model_len?: number }>;
}

export async function detectVllm(
  root: string,
  baseUrl: string,
  apiKey: string,
  signal?: AbortSignal,
  diagnostics?: ProbeDiagnostic[],
): Promise<DetectResult | null> {
  const version = await fetchJson<VllmVersion>(`${root}/version`, apiKey, signal, diagnostics);
  if (typeof version?.version !== "string") return null;

  const modelsRes = await fetchJson<VllmModels>(`${baseUrl}/models`, apiKey, signal, diagnostics);
  const entries = (modelsRes?.data ?? []).filter(
    (m): m is { id: string; max_model_len: number } => typeof m.max_model_len === "number",
  );
  if (entries.length === 0) return null;

  return {
    apiType: "vllm",
    models: entries.map((m) => ({
      id: m.id,
      name: m.id.split("/").pop() ?? m.id,
      contextWindow: m.max_model_len,
      maxTokens: capTokens(m.max_model_len),
      reasoning: false,
      input: ["text"],
    })),
  };
}

// ─── Generic OpenAI-compatible fallback ────────────────────────────

interface OpenAIModels {
  data?: Array<{ id: string; max_model_len?: number; context_window?: number }>;
}

export async function detectOpenAI(
  baseUrl: string,
  apiKey: string,
  signal?: AbortSignal,
  diagnostics?: ProbeDiagnostic[],
): Promise<DetectResult> {
  const res = await fetchJson<OpenAIModels>(`${baseUrl}/models`, apiKey, signal, diagnostics);
  if (!res?.data?.length) return { apiType: "openai", models: [] };

  return {
    apiType: "openai",
    models: res.data.map((m) => {
      const contextWindow = m.max_model_len ?? m.context_window ?? 32768;
      return {
        id: m.id,
        name: m.id.split("/").pop() ?? m.id,
        contextWindow,
        maxTokens: capTokens(contextWindow),
        reasoning: false,
        input: ["text"] as ("text" | "image")[],
      };
    }),
  };
}

// ─── Failure summarization ─────────────────────────────────────────
// Only called when the whole chain ends with zero models. Distinguishes
// "something actually went wrong" from "server's fine, just has nothing
// loaded" — and among the former, gives a specific enough reason to act on
// (bad key vs. unreachable) without over-claiming on ambiguous evidence
// (e.g. a handful of expected 404s from non-matching backend probes mixed
// with one unrelated 500 falls through to undefined — no confident enough
// signal to name a cause, so callers fall back to a generic message).

function summarizeFailure(diagnostics: ProbeDiagnostic[]): string | undefined {
  const authFailure = diagnostics.find((d) => d.reason === "unauthorized" || d.reason === "forbidden");
  if (authFailure) {
    return `Authentication failed (HTTP ${authFailure.status}) — check the API key.`;
  }
  if (diagnostics.length > 0 && diagnostics.every((d) => d.reason === "timeout")) {
    return "Timed out waiting for a response — check the server is running and reachable.";
  }
  if (diagnostics.length > 0 && diagnostics.every((d) => d.reason === "timeout" || d.reason === "network-error")) {
    return "Could not connect to the server — check the URL and that it's running.";
  }
  return undefined;
}

// ─── Chain ──────────────────────────────────────────────────────────
// baseUrl is expected to end with /v1 (per this extension's convention);
// backend-specific probes strip it since their endpoints live at the root.
//
// All probes share a single signal covering the whole chain, not one
// timeout per probe: an unreachable server (packets silently dropped,
// as opposed to a fast connection-refused) would otherwise pay the
// per-probe timeout up to 7 times sequentially before falling through
// to "cannot reach server". Once the shared deadline passes, every
// remaining probe's fetch() rejects immediately (an already-aborted
// signal never waits), so the whole chain is bounded by CHAIN_TIMEOUT_MS
// wall-clock regardless of how many probes it tries.

const CHAIN_TIMEOUT_MS = 8000;

export async function detectModels(
  baseUrl: string,
  apiKey: string,
  signal?: AbortSignal,
): Promise<DetectResult> {
  const chainSignal = signal
    ? AbortSignal.any([signal, AbortSignal.timeout(CHAIN_TIMEOUT_MS)])
    : AbortSignal.timeout(CHAIN_TIMEOUT_MS);
  const root = baseUrl.replace(/\/v1$/, "");
  const diagnostics: ProbeDiagnostic[] = [];

  const result =
    (await detectMtplx(root, apiKey, chainSignal, diagnostics)) ??
    (await detectOmlx(root, apiKey, chainSignal, diagnostics)) ??
    (await detectLmStudio(root, apiKey, chainSignal, diagnostics)) ??
    (await detectLlamaCpp(root, apiKey, chainSignal, diagnostics)) ??
    (await detectOllama(root, apiKey, chainSignal, diagnostics)) ??
    (await detectVllm(root, baseUrl, apiKey, chainSignal, diagnostics)) ??
    (await detectOpenAI(baseUrl, apiKey, chainSignal, diagnostics));

  if (result.models.length === 0) {
    const error = summarizeFailure(diagnostics);
    if (error) return { ...result, error };
  }
  return result;
}
