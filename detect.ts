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

export type ApiType =
  | "mtplx"
  | "omlx"
  | "lmstudio"
  | "llamacpp"
  | "ollama"
  | "sglang"
  | "vllm"
  | "ds4"
  | "ninfer"
  | "openai";

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
  // Only set by a detector that has read the backend's source and confirmed
  // the convention is really implemented. Left undefined everywhere else, so
  // registration keeps its safe defaults — see the compat note in index.ts.
  compat?: { supportsReasoningEffort?: boolean; supportsDeveloperRole?: boolean };
  // Rewrites Pi's thinking levels into the strings this backend actually
  // accepts. Also gates which levels Pi offers at all: pi-ai's
  // getSupportedThinkingLevels drops any level mapped to null, and hides
  // "xhigh"/"max" entirely unless they have an entry here.
  thinkingLevelMap?: Partial<Record<ThinkingLevel, string | null>>;
  // Merged into every request by Pi. Only set where a detector knows the
  // model family's published recommendation and the server would otherwise
  // fall back to a generation_config default that suits it badly.
  samplingParams?: { temperature?: number };
}

export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

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

// The fallback for backends that don't report an output limit of their own.
//
// This is a real generation cap, not just metadata: Pi resolves every turn's
// max_tokens as `options.maxTokens ?? model.maxTokens` (pi-ai's
// buildBaseOptions), clamped to the context left after the prompt.
//
// The ceiling is much higher for a reasoning model because reasoning and the
// answer come out of the same budget, and no local backend offers a separate
// thinking cap worth relying on — SGLang's OpenAI surface silently ignores
// max_thinking_tokens, and Pi's thinking_token_budget is vLLM-only. Set this
// too low and a model that thinks its way to the limit spends the entire
// budget reasoning and returns no answer at all, which then reads as a
// truncated turn and gets retried into the same wall. The cost of the
// opposite mistake is only a slower failure, so the generous side wins.
export function capTokens(contextWindow: number, reasoning = false): number {
  return Math.min(Math.floor(contextWindow / 2), reasoning ? 65536 : 8192);
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
        maxTokens: health.max_response_tokens ?? capTokens(health.context_window, reasoning),
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
      maxTokens: m.max_tokens ?? capTokens(contextWindow, m.thinking_default === true),
      reasoning: m.thinking_default === true,
      input: type === "vlm" ? ["text", "image"] : ["text"],
      loaded: m.loaded === true,
      sizeBytes: firstNumber(m.estimated_size),
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
      maxTokens: capTokens(contextWindow, !!m.capabilities?.reasoning),
      reasoning: !!m.capabilities?.reasoning,
      input: m.capabilities?.vision || type === "vlm" ? ["text", "image"] : ["text"],
      loaded: (m.loaded_instances?.length ?? 0) > 0,
      sizeBytes: firstNumber(m.size_bytes),
      quantization: m.quantization?.name,
    });
  }
  return models.length > 0 ? { apiType: "lmstudio", models } : null;
}

// ─── llama.cpp server (llama-server) ───────────────────────────────
// /props has the runtime-configured context (n_ctx) and vision support;
// /v1/models has the id (respects
// --alias), the model's trained max context (n_ctx_train) as a fallback
// ceiling, and file size. No loaded/unloaded distinction exists — one
// server process, one model — so loaded is left undefined, same as the
// generic OpenAI probe.
//
// Nothing in /props says whether the model thinks. The template
// capabilities it reports (chat_template_caps) cover tools, system role
// and reasoning_effort, but the "does this template have a thinking
// switch" answer that llama-server computes for itself at startup
// (common_chat_templates_support_enable_thinking) never reaches the API.
// So it is measured, through POST /apply-template — the server's own
// template renderer, which runs the full chat-completions request parse
// (chat_template_kwargs, reasoning_effort, everything) and returns the
// prompt it would have generated from, without generating. See
// probeLlamaCppThinking for what gets rendered and compared.

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

  // Not consulted: default_generation_settings.params.reasoning_format.
  // It reads "none" on a server that is splitting reasoning_content out of
  // every reply — /props builds that block from a default-constructed
  // task_params and only copies the sampling settings in, so the parser
  // fields never reflect the flags the server was started with.
  const thinking = await probeLlamaCppThinking(root, apiKey, signal);

  return {
    apiType: "llamacpp",
    models: [
      {
        id,
        name: id.split(/[\\/]/).pop() ?? id,
        contextWindow,
        maxTokens: capTokens(contextWindow, thinking.reasoning),
        reasoning: thinking.reasoning,
        input: props.modalities?.vision ? ["text", "image"] : ["text"],
        sizeBytes: firstNumber(entry?.meta?.size),
        ...(thinking.compat ? { compat: thinking.compat } : {}),
        ...(thinking.thinkingLevelMap ? { thinkingLevelMap: thinking.thinkingLevelMap } : {}),
      },
    ],
  };
}

// ─── llama.cpp thinking probe ───────────────────────────────────────
// Three renders of the same one-line conversation through /apply-template
// answer the question /props can't:
//
//   - with chat_template_kwargs.enable_thinking = true
//   - with chat_template_kwargs.enable_thinking = false
//   - with neither, i.e. whatever the server defaults to
//
// A template with a thinking switch renders the two forced cases
// differently (Qwen3 closes an empty <think></think> block into the
// generation prompt when thinking is off; GPT-OSS changes its system
// preamble). Identical output means the switch isn't there, and the model
// is registered as non-reasoning. When the switch exists, the unforced
// render says which side the server sits on — `--reasoning off` makes it
// match the disabled case — and only a server that thinks by default is
// registered as reasoning, since that is the state every request from Pi
// will actually run in: Pi can send reasoning_effort, but not
// chat_template_kwargs, so it has no way to turn thinking on against a
// server that was started with it off.
//
// The reasoning_effort sweep runs through the same endpoint, and is free
// where SGLang's costs a token per accepted tier: /apply-template stops
// after rendering. llama-server handles "none" itself (it flips
// enable_thinking off and never shows the template the value), and hands
// every other tier to the template as the reasoning_effort variable. A
// template that validates the value raises, and llama-server turns a Jinja
// raise into a 500 — not a 400 — with the raise message in the error body.
// The status alone can't tell that apart from a broken server, so the
// body is read: only a 500 that quotes a template exception counts as
// "rejected". Anything else is unanswered, and one unanswered tier
// discards the sweep, for the same reason as in probeRequestCompat.
//
// A template that never reads reasoning_effort accepts every tier and
// applies none of them; that shows up as an identity map, exactly as it
// does on SGLang, and the "none" entry is still real because llama-server
// implements that one without the template's help.

const TEMPLATE_RAISE = /Jinja Exception|Error executing/;

function classifyTemplateReply(reply: ProbeReply | null): "accepted" | "rejected" | "unanswered" {
  if (!reply) return "unanswered";
  if (reply.status === 200) return "accepted";
  if (reply.status === 400 || reply.status === 422) return "rejected";
  if (reply.status === 500 && TEMPLATE_RAISE.test(reply.body)) return "rejected";
  return "unanswered";
}

function renderedPrompt(reply: ProbeReply | null): string | undefined {
  if (reply?.status !== 200) return undefined;
  try {
    const prompt = (JSON.parse(reply.body) as { prompt?: unknown }).prompt;
    return typeof prompt === "string" ? prompt : undefined;
  } catch {
    return undefined;
  }
}

export async function probeLlamaCppThinking(
  root: string,
  apiKey: string,
  signal?: AbortSignal,
): Promise<{ reasoning: boolean } & RequestCompat> {
  const url = `${root}/apply-template`;
  const user = { role: "user", content: "hi" };
  const render = (extra: Record<string, unknown>) =>
    probe(url, apiKey, { messages: [user], ...extra }, signal);

  const [byDefault, forcedOn, forcedOff] = (
    await Promise.all([
      render({}),
      render({ chat_template_kwargs: { enable_thinking: true } }),
      render({ chat_template_kwargs: { enable_thinking: false } }),
    ])
  ).map(renderedPrompt);

  // A render that didn't come back is not evidence of anything; the safe
  // default (no thinking, no compat) is what a missing answer degrades to.
  if (byDefault === undefined || forcedOn === undefined || forcedOff === undefined) {
    return { reasoning: false };
  }
  if (forcedOn === forcedOff || byDefault !== forcedOn) return { reasoning: false };

  const [tierReplies, developerReply] = await Promise.all([
    Promise.all(THINKING_TIERS.map((tier) => render({ reasoning_effort: tier }))),
    render({ messages: [{ role: "developer", content: "You are terse." }, user] }),
  ]);
  const verdicts = tierReplies.map(classifyTemplateReply);
  if (verdicts.some((v) => v === "unanswered")) return { reasoning: true };

  const accepted = THINKING_TIERS.filter((_, i) => verdicts[i] === "accepted");
  return {
    reasoning: true,
    ...compatFromSweep(accepted, classifyTemplateReply(developerReply) === "accepted"),
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
      maxTokens: capTokens(contextWindow, capabilities.includes("thinking")),
      reasoning: capabilities.includes("thinking"),
      input: capabilities.includes("vision") ? ["text", "image"] : ["text"],
      loaded: runningModels.has(m.model),
      sizeBytes: firstNumber(m.size),
      quantization: m.details?.quantization_level,
    };
  });

  return { apiType: "ollama", models };
}

// ─── SGLang ──────────────────────────────────────────────────────────
// Must be probed *before* Ollama, not after: SGLang ships an Ollama
// compatibility shim, so its /api/tags and /api/show answer 200 with
// Ollama-shaped payloads and detectOllama would otherwise claim it. The
// shim is lossy in three ways, which is what makes the mislabel worth
// preventing rather than just cosmetic:
//   - /api/show reports capabilities: ["completion"] no matter what the
//     model can do, so vision and reasoning both come back false.
//   - /api/ps isn't implemented (404), so every model looks not-loaded
//     when SGLang in fact holds one resident model for the process's life.
//   - the shim's ids are the raw --model-path, same as /v1/models.
// Two SGLang-only endpoints carry the real answers:
//   - /model_info: has_image_understanding (the vision signal) plus
//     is_generation, which separates a chat server from an embedding one.
//   - /server_info: reasoning_parser, non-null exactly when SGLang was
//     started with a parser that splits reasoning out of the response —
//     which is the only sense in which "this server does reasoning" is
//     true or false, independent of the weights.
// The context window comes from /v1/models' max_model_len, the same field
// vLLM publishes; /server_info's context_length is the CLI override and
// is null unless it was passed explicitly, so it can't be relied on.
//
// compat and thinkingLevelMap are measured, not assumed — see the probe
// below for why they can't be hard-coded the way ds4's are.

// ─── Request-compat probe ───────────────────────────────────────────
// Shared by SGLang and ninfer, because the problem is shared: neither
// backend's accepted reasoning_effort values can be read off its source,
// because they aren't the backend's. SGLang validates the field
// against all seven OpenAI tiers and then hands the string straight to the
// model's chat template (serving_chat.py builds extra_template_kwargs
// ["reasoning_effort"]), and it is the template that accepts or rejects it.
// The vocabularies genuinely differ per model: a Qwen3.8 template answers
// 400 with "Supported types are xhigh (default), medium, and low", while
// SGLang's own Kimi K3 path recognises low/high/max instead. Hard-coding
// either set would 400 every request on a server running the other.
//
// So the tiers are measured: one throwaway completion per tier, capped at a
// single token. A rejected tier fails during template rendering, before any
// generation, so the cost of the whole sweep is one token per *accepted*
// tier. The same trick settles the developer role, which is a second
// template-level question SGLang's API layer can't answer — it accepts the
// role in its own schema (protocol.py's _GenericMessageRole lists it) and
// still lets a template that only knows "system" reject the message.
//
// Anything short of a clean HTTP answer leaves compat unset, so a server
// that is slow, busy, or unreachable degrades to the safe defaults rather
// than to a guess.

const THINKING_TIERS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"] as const;

// Pi's levels in the same order, so "nearest" can be measured by index.
// "off" is handled separately: it means stop reasoning, so it may only ever
// map to "none" — never to whatever tier happens to sit closest to it.
const PI_LEVELS: readonly ThinkingLevel[] = [
  "off",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
];

/** A probe's answer: the status, and whatever the server said about it. */
interface ProbeReply {
  status: number;
  body: string;
}

async function probe(
  url: string,
  apiKey: string,
  payload: unknown,
  signal?: AbortSignal,
): Promise<ProbeReply | null> {
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;
    const res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: signal ?? AbortSignal.timeout(STANDALONE_PROBE_TIMEOUT_MS),
    });
    // Always drained, so the socket can be reused by the probes still in
    // flight. Some probes only want the status; ninfer's context probe reads
    // the ceiling out of the rejection message.
    const body = await res.text().catch(() => "");
    return { status: res.status, body };
  } catch {
    return null;
  }
}

async function probeStatus(
  url: string,
  apiKey: string,
  payload: unknown,
  signal?: AbortSignal,
): Promise<number | null> {
  return (await probe(url, apiKey, payload, signal))?.status ?? null;
}

// Picks the accepted tier closest to `level`, preferring the weaker one when
// two are equally close — a request for more thinking than the model offers
// should land under the ceiling rather than over it. "none" is excluded: it
// disables reasoning, so it must never be the nearest match for a level that
// asked for some.
export function nearestTier(level: ThinkingLevel, accepted: readonly string[]): string | undefined {
  const wanted = PI_LEVELS.indexOf(level);
  if (wanted < 0) return undefined;
  const candidates = THINKING_TIERS.map((tier, i) => ({ tier, i })).filter(
    (c) => c.tier !== "none" && accepted.includes(c.tier),
  );
  let best: { tier: string; i: number } | undefined;
  for (const c of candidates) {
    if (!best || Math.abs(c.i - wanted) < Math.abs(best.i - wanted)) best = c;
  }
  return best?.tier;
}

export interface RequestCompat {
  compat?: DiscoveredModel["compat"];
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
}

export async function probeRequestCompat(
  baseUrl: string,
  apiKey: string,
  modelId: string,
  signal?: AbortSignal,
): Promise<RequestCompat> {
  const url = `${baseUrl}/chat/completions`;
  const base = { model: modelId, max_tokens: 1, stream: false };
  const user = { role: "user", content: "hi" };

  const [tierStatuses, developerStatus] = await Promise.all([
    Promise.all(
      THINKING_TIERS.map((tier) =>
        probeStatus(url, apiKey, { ...base, messages: [user], reasoning_effort: tier }, signal),
      ),
    ),
    probeStatus(
      url,
      apiKey,
      { ...base, messages: [{ role: "developer", content: "You are terse." }, user] },
      signal,
    ),
  ]);

  // Only a 200 or a request-rejected status is the server actually answering
  // the question. A timeout, a 429, a 5xx from an overloaded worker — those
  // are the question going unheard, and scoring them as "tier unsupported"
  // is the one failure mode that does lasting damage: it writes a map that
  // looks measured, gets saved, and then 400s on every real request until
  // the next Refresh happens to catch the server in a better mood. One
  // unanswered tier poisons the whole sweep, so the bar is all-or-nothing.
  const isDefinitive = (s: number | null) => s === 200 || s === 400 || s === 422;
  if (!tierStatuses.every(isDefinitive)) return {};

  const accepted = THINKING_TIERS.filter((_, i) => tierStatuses[i] === 200);
  // The developer probe is held to a lower bar because its unknown state
  // is already the safe one: false just means Pi keeps sending "system".
  return compatFromSweep(accepted, developerStatus === 200);
}

// Turns a finished sweep — the tiers the server accepted, and whether it
// took the developer role — into what gets registered. Shared by every
// backend that measures its levels, so they all map the same way.
function compatFromSweep(accepted: readonly string[], developerAccepted: boolean): RequestCompat {
  const result: RequestCompat = {};
  if (accepted.length > 0) {
    result.compat = { supportsReasoningEffort: true, supportsDeveloperRole: developerAccepted };

    const map: Partial<Record<ThinkingLevel, string>> = {};
    // Only claim "off" when the server really has an off switch; otherwise
    // leave it unmapped, which makes Pi omit the field and fall back to
    // whatever the model does by default.
    if (accepted.includes("none")) map.off = "none";
    for (const level of PI_LEVELS) {
      if (level === "off") continue;
      const tier = nearestTier(level, accepted);
      if (tier) map[level] = tier;
    }
    if (Object.keys(map).length > 0) result.thinkingLevelMap = map;
  } else if (developerAccepted) {
    result.compat = { supportsDeveloperRole: true };
  }
  return result;
}

// Qwen publishes 0.6 as the sampling temperature for its thinking mode, but a
// checkpoint's generation_config commonly ships 1.0, and that is what SGLang
// serves when the request names no temperature of its own — as Pi's do not.
// At 1.0 a thinking model in an agent loop is prone to re-planning the same
// task over and over until it exhausts the token budget, so the published
// value is applied for the family it belongs to. Any other family keeps its
// server-side default: this is a known recommendation, not a house style, and
// guessing one for an unfamiliar model would be worse than leaving it alone.
// Overridable per model from the TUI.
const QWEN_THINKING_TEMPERATURE = 0.6;

function samplingParamsFor(modelType: string | undefined, reasoning: boolean) {
  if (!reasoning || !modelType) return undefined;
  return /^qwen/i.test(modelType) ? { temperature: QWEN_THINKING_TEMPERATURE } : undefined;
}

interface SglangModelInfo {
  model_path?: string;
  is_generation?: boolean;
  has_image_understanding?: boolean;
  model_type?: string;
}

interface SglangServerInfo {
  reasoning_parser?: string | null;
}

interface SglangModels {
  data?: Array<{ id: string; max_model_len?: number }>;
}

// SGLang renamed /get_model_info and /get_server_info to /model_info and
// /server_info, and now logs a deprecation warning naming the replacement
// on every call to the old pair. Both still answer, so nothing is broken
// yet — but the old names are documented as going away, and a detector
// that knew only those would stop recognising SGLang at all on the release
// that removes them, silently falling through to the generic OpenAI path.
// Ask for the new name first and keep the old one as a fallback, so a
// server on either side of the rename is detected identically. The cost is
// one extra 404 per detection against a server that is not SGLang, which
// the chain already spends several of.
async function fetchSglangJson<T>(
  root: string,
  name: string,
  apiKey: string,
  signal?: AbortSignal,
  diagnostics?: ProbeDiagnostic[],
): Promise<T | null> {
  const current = await fetchJson<T>(`${root}/${name}`, apiKey, signal, diagnostics);
  if (current) return current;
  return fetchJson<T>(`${root}/get_${name}`, apiKey, signal, diagnostics);
}

export async function detectSglang(
  root: string,
  baseUrl: string,
  apiKey: string,
  signal?: AbortSignal,
  diagnostics?: ProbeDiagnostic[],
): Promise<DetectResult | null> {
  const info = await fetchSglangJson<SglangModelInfo>(root, "model_info", apiKey, signal, diagnostics);
  // is_generation false means an embedding-only server: it has no chat
  // endpoint to register, so let it fall through rather than claiming it.
  if (!info?.model_path || info.is_generation !== true) return null;

  const [serverInfo, modelsRes] = await Promise.all([
    fetchSglangJson<SglangServerInfo>(root, "server_info", apiKey, signal, diagnostics),
    fetchJson<SglangModels>(`${baseUrl}/models`, apiKey, signal, diagnostics),
  ]);

  const reasoning = typeof serverInfo?.reasoning_parser === "string" && serverInfo.reasoning_parser !== "";
  const vision = info.has_image_understanding === true;
  // One resident model per process, so /v1/models is a single entry in
  // practice — mapped anyway so a multi-model build doesn't lose entries.
  const entries = modelsRes?.data?.length ? modelsRes.data : [{ id: info.model_path }];

  // Only worth measuring when a parser is configured: without one SGLang
  // never separates reasoning from the answer, so the levels have nothing
  // to steer even where the template would accept them. One sweep serves
  // every entry, since they all resolve to the same resident model.
  const requestCompat = reasoning
    ? await probeRequestCompat(baseUrl, apiKey, entries[0].id, signal)
    : {};
  const samplingParams = samplingParamsFor(info.model_type, reasoning);

  return {
    apiType: "sglang",
    models: entries.map((m) => {
      const contextWindow = firstNumber((m as { max_model_len?: number }).max_model_len) ?? 32768;
      return {
        id: m.id,
        name: m.id.split(/[\\/]/).filter(Boolean).pop() ?? m.id,
        contextWindow,
        maxTokens: capTokens(contextWindow, reasoning),
        reasoning,
        input: (vision ? ["text", "image"] : ["text"]) as ("text" | "image")[],
        ...requestCompat,
        ...(samplingParams ? { samplingParams } : {}),
      };
    }),
  };
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
//
// vLLM has no public API for vision/reasoning capability (its debug-only
// /server_info endpoint would carry it, but that's undocumented, gated
// behind VLLM_SERVER_DEV_MODE=1, and known to crash on some setups — not
// something to build detection on), so both always default to false/text.
// Use ✎ Edit model capabilities in the TUI to override by hand.

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
// Model cards in the wild carry the context window under four different
// names, so all four are read before falling back to the 32768 default:
//   - max_model_len          vLLM
//   - context_window         various MLX-based servers
//   - context_length         OpenRouter-style cards (top level), used by
//     any proxy or router that mirrors OpenRouter's /v1/models shape
//   - top_provider.context_length   same cards, per-provider override —
//     this is the one that reflects the *serving* limit, so it wins
//
// OpenRouter-style cards also carry a human-readable `name`, a declared
// output ceiling, the accepted input modalities, and the list of request
// parameters the server honours. Reading them is what separates this from
// a bare id list; none of the fields is required, and each falls back to
// the old behaviour when absent.

interface OpenAIModelCard {
  id: string;
  name?: string;
  owned_by?: string;
  max_model_len?: number;
  context_window?: number;
  context_length?: number;
  top_provider?: { context_length?: number; max_completion_tokens?: number };
  architecture?: { input_modalities?: string[] };
  supported_parameters?: string[];
}

interface OpenAIModels {
  data?: OpenAIModelCard[];
}

function firstNumber(...values: unknown[]): number | undefined {
  for (const v of values) if (typeof v === "number" && v > 0) return v;
  return undefined;
}

// ─── ds4 (antirez/ds4, "DwarfStar") ────────────────────────────────
// ds4 serves nothing outside /v1, publishes no Server header, and has no
// /health, /props or /version — so it cannot be probed like the backends
// above. Its one reliable fingerprint is inside the model card the generic
// probe already fetches, which is why this is a branch of detectOpenAI
// rather than another link in the chain: ds4_server.c hard-codes
// `"owned_by":"ds4.c"` on every card, with no conditional around it.
//
// Everything else on those cards has to be read from the source rather than
// taken at face value:
//   - The ids are aliases, not distinct models. ds4_server.c emits either
//     {deepseek-v4-flash, deepseek-v4-pro} or, for a GLM-DSA engine,
//     {glm-5.2, glm-5.2-chat, glm-5.2-reasoner} — but every entry reports
//     the same loaded GGUF, so name/context/max are identical across them.
//     Matching on the ids would therefore break on the next engine added;
//     only owned_by is stable. The aliases are all registered anyway, since
//     the server accepts each one as a model parameter.
//   - supported_parameters is a hard-coded constant, not a capability
//     report — it lists reasoning_effort regardless of which GGUF is
//     loaded. Reasoning is set here because every engine ds4 serves is a
//     reasoning model, not because that array says so.
//   - There is no image handling anywhere in ds4_server.c, so text-only.
//
// Both compat conventions are confirmed against the source: reasoning_effort
// is parsed on the chat path, and the developer role is accepted wherever
// system is. Turning them on is what makes Pi's thinking levels reach the
// server at all — but on its own that is not enough to make them *mean*
// anything, which is what thinkingLevelMap below is for.
//
// ds4 collapses effort down to three modes (ds4_server.c's
// think_mode_from_enabled): NONE, HIGH, and MAX, with everything between
// "minimal" and "xhigh" landing on HIGH. Two of Pi's seven levels therefore
// need rewriting, and the rest are already correct as-is:
//
//   - "off" is the one that actually matters. pi-ai turns level "off" into
//     an *absent* reasoning_effort (openai-completions.js: `clampedReasoning
//     === "off" ? undefined : ...`) unless thinkingLevelMap.off names a
//     string to send instead. An absent reasoning_effort makes ds4 fall back
//     to its own defaults — thinking_enabled = true, effort = HIGH — so
//     without this mapping, selecting "off" in Pi leaves the server thinking
//     at full strength. Sending "none" is what genuinely disables it.
//   - "max" is a real, distinct mode in ds4, but pi-ai hides "xhigh"/"max"
//     from the level list unless thinkingLevelMap has an entry for them, so
//     it is unreachable until named here.
//   - "xhigh" is deliberately left out: ds4 folds it into HIGH, making it an
//     exact duplicate of "high". Offering it would imply a distinction the
//     server does not make.
//   - "minimal"/"low"/"medium"/"high" pass through unmapped. ds4 accepts all
//     four strings and treats them as HIGH, which is as close as it gets.

// ─── ninfer (Neroued/ninfer) ────────────────────────────────────────
// Identified the same way ds4 is, and for the same reason: it publishes no
// endpoint of its own to probe. Its five GET routes are /health,
// /v1/models, /v1/models/{id} and two Responses lookups, and /health answers
// a bare {"status":"ok"} — which is also why detectMtplx, the only other
// probe that reads /health, passes over it. The marker is in the model card
// src/serve/openai_schema.cpp hard-codes: `"owned_by": "ninfer"`.
//
// Where ds4's card at least carries a context window, ninfer's carries
// nothing but {id, object, created, owned_by}. Everything else this
// extension needs is a startup flag with no runtime reader:
//
//   --max-context   the context ceiling, default 8192
//   --vision        media input, off unless passed
//   the artifact's chat_template.jinja, which decides the effort tiers
//
// So they are measured. Each probe below is chosen to cost nothing on the
// GPU; the notes on each say why that holds and what still needs checking
// against a running server.

const NINFER_OWNER = "ninfer";

export function isNinferCards(cards: OpenAIModelCard[]): boolean {
  return cards.length > 0 && cards.every((m) => m.owned_by === NINFER_OWNER);
}

/**
 * What ninfer uses when --max-context is omitted.
 *
 * The fallback if the probe cannot answer, and a far better guess than this
 * file's generic 32768: registering four times the real ceiling would let Pi
 * fill a context that the server then refuses.
 */
const NINFER_DEFAULT_CONTEXT = 8192;

/**
 * How far to overshoot when asking the server for its ceiling.
 *
 * The prompt has to exceed --max-context for the rejection to happen at all,
 * and the rejection is the only thing that names the number. Overshooting
 * costs nothing: engine.cpp raises ContextLengthExceeded straight after the
 * chat template and tokenizer, before any prefill. Falling short is what
 * costs — the request would be accepted and actually run.
 *
 * 300k clears every context these artifacts ship with, the largest being
 * 262144. A server configured beyond that accepts the prompt instead, and
 * then its own reported prompt_tokens becomes a floor — less precise than
 * the rejection, but true, and it costs nothing extra since that prefill
 * has already been paid for by the time the answer comes back.
 *
 * The ratio is measured: "x " repeated 25,000 times came back as 25,052
 * prompt tokens on a Qwen3.8 artifact, so two characters per token holds.
 * The whole detection, this 600 KB body included, took 367 ms.
 */
const NINFER_CONTEXT_PROBE_TOKENS = 300_000;
const NINFER_PROBE_CHARS_PER_TOKEN = 2;

/**
 * Read the ceiling out of a rejection.
 *
 * engine.cpp builds it as "prepared prompt has N tokens, exceeding Engine
 * max_context M", and the HTTP layer passes exception.what() through as the
 * error message. Parsing prose is brittle by nature, so a miss returns
 * undefined and the caller falls back rather than inventing a number.
 */
export function parseNinferContext(body: string): number | undefined {
  const match = /max_context\s+(\d+)/.exec(body);
  if (!match) return undefined;
  const value = Number(match[1]);
  return Number.isFinite(value) && value > 0 ? value : undefined;
}

/**
 * The prompt size a server accepted, which is a floor on its ceiling.
 *
 * Only reached when the overshoot was not enough — a context larger than
 * NINFER_CONTEXT_PROBE_TOKENS. Understating it would be worse than useless
 * there: the prefill has already run, and falling back to 8192 would throw
 * away the one number that outing bought.
 */
function acceptedPromptTokens(reply: ProbeReply): number | undefined {
  if (reply.status !== 200) return undefined;
  try {
    const tokens = (JSON.parse(reply.body) as { usage?: { prompt_tokens?: number } }).usage
      ?.prompt_tokens;
    return typeof tokens === "number" && tokens > 0 ? tokens : undefined;
  } catch {
    return undefined;
  }
}

export interface NinferProbe {
  contextWindow: number;
  /** Undefined when the vision probe could not be answered either way. */
  vision?: boolean;
}

/**
 * Ask a ninfer server the two things its model card leaves out.
 *
 * Both probes are shaped to do no GPU work:
 *
 *   - the context probe is rejected during prompt preparation, before
 *     prefill, and names the ceiling as it goes;
 *   - the vision probe is a token count, which the docs say runs "without
 *     running GPU generation" and which media requests fail with 400
 *     `vision_disabled` when --vision was omitted.
 */
export async function probeNinfer(
  baseUrl: string,
  apiKey: string,
  modelId: string,
  signal?: AbortSignal,
): Promise<NinferProbe> {
  const oversized = "x ".repeat(
    Math.ceil((NINFER_CONTEXT_PROBE_TOKENS * NINFER_PROBE_CHARS_PER_TOKEN) / 2),
  );

  const [context, vision] = await Promise.all([
    probe(
      `${baseUrl}/chat/completions`,
      apiKey,
      { model: modelId, max_tokens: 1, stream: false, messages: [{ role: "user", content: oversized }] },
      signal,
    ),
    // A 1x1 transparent PNG: the smallest thing that is still an image.
    // A server without --vision answers 400 with code `vision_disabled` and
    // the message "Vision is disabled for this server"; the code is what
    // gets matched, since the prose is the part free to change.
    probe(
      `${baseUrl}/responses/input_tokens`,
      apiKey,
      {
        model: modelId,
        input: [
          {
            role: "user",
            content: [
              {
                type: "input_image",
                image_url:
                  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
              },
            ],
          },
        ],
      },
      signal,
    ),
  ]);

  return {
    contextWindow:
      (context && parseNinferContext(context.body)) ??
      (context && acceptedPromptTokens(context)) ??
      NINFER_DEFAULT_CONTEXT,
    // Only a definite answer counts. A network failure, or a rejection for
    // some reason other than vision being off, leaves this unknown so the
    // caller can fall back rather than declare a vision model text-only.
    vision:
      vision === null
        ? undefined
        : vision.status === 200
          ? true
          : /vision_disabled/.test(vision.body)
            ? false
            : undefined,
  };
}

/**
 * Build the models a ninfer server serves.
 *
 * Its card lists exactly one entry — one resident artifact per process — but
 * it is mapped rather than indexed, so a build that ever lists more does not
 * silently lose them.
 */
async function ninferModels(
  cards: OpenAIModelCard[],
  baseUrl: string,
  apiKey: string,
  signal?: AbortSignal,
): Promise<DiscoveredModel[]> {
  const [measured, compat] = await Promise.all([
    probeNinfer(baseUrl, apiKey, cards[0].id, signal),
    // The same probe SGLang uses, because the situation is the same: the
    // accepted tiers come from the artifact's chat template, and an
    // unsupported one is rejected — 400 `reasoning_effort_not_supported`
    // here. docs/serving.md says an effort-capable template exposes low,
    // medium and xhigh, which is the set Qwen3.8 exposes under SGLang too;
    // measuring rather than hard-coding is what keeps that from being an
    // assumption about every artifact ninfer will ever load.
    probeRequestCompat(baseUrl, apiKey, cards[0].id, signal),
  ]);

  // Thinking is what the effort tiers steer, so a template that exposes any
  // of them is a reasoning model. Reasoning text comes back on a separate
  // `reasoning_content` field, which Pi reads without further help.
  const reasoning = Boolean(compat.thinkingLevelMap);

  return cards.map((card) => ({
    id: card.id,
    name: card.name || (card.id.split(/[\\/]/).filter(Boolean).pop() ?? card.id),
    contextWindow: measured.contextWindow,
    maxTokens: capTokens(measured.contextWindow, reasoning),
    reasoning,
    input: (measured.vision ? ["text", "image"] : ["text"]) as ("text" | "image")[],
    ...compat,
  }));
}

const DS4_OWNER = "ds4.c";

const DS4_THINKING_LEVELS: Partial<Record<ThinkingLevel, string | null>> = {
  off: "none",
  max: "max",
};

export function isDs4Cards(cards: OpenAIModelCard[]): boolean {
  return cards.length > 0 && cards.every((m) => m.owned_by === DS4_OWNER);
}

export async function detectOpenAI(
  baseUrl: string,
  apiKey: string,
  signal?: AbortSignal,
  diagnostics?: ProbeDiagnostic[],
): Promise<DetectResult> {
  const res = await fetchJson<OpenAIModels>(`${baseUrl}/models`, apiKey, signal, diagnostics);
  if (!res?.data?.length) return { apiType: "openai", models: [] };

  // Checked before ds4 only because both are owned_by markers on the same
  // payload; neither can match the other's value, so the order is arbitrary.
  if (isNinferCards(res.data)) {
    return { apiType: "ninfer", models: await ninferModels(res.data, baseUrl, apiKey, signal) };
  }

  if (isDs4Cards(res.data)) {
    return {
      apiType: "ds4",
      models: res.data.map((m) => {
        const contextWindow = firstNumber(m.top_provider?.context_length, m.context_length) ?? 32768;
        const declaredMax = firstNumber(m.top_provider?.max_completion_tokens);
        return {
          id: m.id,
          name: m.name || m.id,
          contextWindow,
          maxTokens:
            declaredMax !== undefined && declaredMax < contextWindow
              ? declaredMax
              : capTokens(contextWindow, true),
          reasoning: true,
          input: ["text"] as ("text" | "image")[],
          compat: { supportsReasoningEffort: true, supportsDeveloperRole: true },
          thinkingLevelMap: DS4_THINKING_LEVELS,
        };
      }),
    };
  }

  return {
    apiType: "openai",
    models: res.data.map((m) => {
      const contextWindow =
        firstNumber(
          m.max_model_len,
          m.top_provider?.context_length,
          m.context_window,
          m.context_length,
        ) ?? 32768;

      const params = m.supported_parameters ?? [];
      const modalities = m.architecture?.input_modalities ?? [];
      const reasoning = params.includes("reasoning_effort") || params.includes("include_reasoning");

      // A declared output ceiling is only trusted when it is a real
      // restriction. Cards that repeat the whole context window here mean
      // "no separate output limit", and taking that literally would make
      // every request reserve the entire window for the response.
      const declaredMax = firstNumber(m.top_provider?.max_completion_tokens);
      const maxTokens =
        declaredMax !== undefined && declaredMax < contextWindow
          ? declaredMax
          : capTokens(contextWindow, reasoning);

      return {
        id: m.id,
        name: m.name || (m.id.split("/").pop() ?? m.id),
        contextWindow,
        maxTokens,
        reasoning,
        input: (modalities.includes("image") ? ["text", "image"] : ["text"]) as ("text" | "image")[],
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
    // Ahead of Ollama on purpose — SGLang answers /api/tags through a
    // compatibility shim and would otherwise be claimed as Ollama.
    (await detectSglang(root, baseUrl, apiKey, chainSignal, diagnostics)) ??
    (await detectOllama(root, apiKey, chainSignal, diagnostics)) ??
    (await detectVllm(root, baseUrl, apiKey, chainSignal, diagnostics)) ??
    (await detectOpenAI(baseUrl, apiKey, chainSignal, diagnostics));

  if (result.models.length === 0) {
    const error = summarizeFailure(diagnostics);
    if (error) return { ...result, error };
  }
  return result;
}
