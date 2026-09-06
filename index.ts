import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import {
  capTokens,
  detectModels,
  type ApiType,
  type DetectResult,
  type DiscoveredModel,
  type ThinkingLevel,
} from "./detect.ts";
import { deleteFromKeychain, isDirectApiKey, keychainCommand, storeInKeychain } from "./keychain.ts";

// ─── Types ────────────────────────────────────────────────────────

interface LLMModel {
  id: string;
  name: string;
  contextWindow: number;
  maxTokens: number;
  // A manually entered value (via ✎ Override max context length) that takes
  // precedence over `contextWindow`. `contextWindow` itself keeps the value
  // the server reported, so the override can be cleared later and survives a
  // ↺ Refresh re-attached to the freshly detected model. Only backends that
  // can't report a context window (e.g. llama-swap proxies) need it.
  contextWindowOverride?: number;
  reasoning: boolean;
  input: ("text" | "image")[];
  // Not every backend can report these — undefined means "unknown", not "no".
  loaded?: boolean;
  sizeBytes?: number;
  quantization?: string;
  compat?: { supportsReasoningEffort?: boolean; supportsDeveloperRole?: boolean };
  thinkingLevelMap?: Partial<Record<ThinkingLevel, string | null>>;
  samplingParams?: { temperature?: number };
}

interface LLMServer {
  id: string;         // stable random ID, used as provider ID suffix
  name: string;       // user-facing display name
  baseUrl: string;    // always ends with /v1
  apiKey: string;     // "" if not required
  apiType: ApiType;   // backend detected at last add/refresh
  models: LLMModel[];
}

interface LocalLLMSettings {
  servers: LLMServer[];
}

// ─── settings.json persistence ────────────────────────────────────

const SETTINGS_FILE = path.join(os.homedir(), ".pi", "agent", "settings.json");
const SETTINGS_KEY = "localllm";

function readSettings(): LocalLLMSettings {
  try {
    if (!fs.existsSync(SETTINGS_FILE)) return { servers: [] };
    const all = JSON.parse(fs.readFileSync(SETTINGS_FILE, "utf8")) as Record<string, unknown>;
    return (all[SETTINGS_KEY] as LocalLLMSettings | undefined) ?? { servers: [] };
  } catch {
    return { servers: [] };
  }
}

function writeSettings(settings: LocalLLMSettings): void {
  let all: Record<string, unknown> = {};
  try {
    if (fs.existsSync(SETTINGS_FILE)) {
      all = JSON.parse(fs.readFileSync(SETTINGS_FILE, "utf8")) as Record<string, unknown>;
    }
  } catch {}
  all[SETTINGS_KEY] = settings;
  fs.writeFileSync(SETTINGS_FILE, JSON.stringify(all, null, 2), "utf8");
}

// ─── Helpers ──────────────────────────────────────────────────────

function generateId(): string {
  return Math.random().toString(36).slice(2, 8);
}

function toProviderId(server: LLMServer): string {
  const slug = server.name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 32);
  return `localllm-${slug || server.id}`;
}

export function normalizeBaseUrl(raw: string): string {
  let stripped = raw.trim().replace(/\/+$/, "");
  if (!/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(stripped)) {
    stripped = `http://${stripped}`;
  }
  return stripped.endsWith("/v1") ? stripped : `${stripped}/v1`;
}

export function apiTypeLabel(apiType: ApiType): string {
  switch (apiType) {
    case "mtplx": return "MTPLX";
    case "omlx": return "oMLX";
    case "lmstudio": return "LM Studio";
    case "llamacpp": return "llama.cpp";
    case "ollama": return "Ollama";
    case "sglang": return "SGLang";
    case "vllm": return "vLLM";
    case "ds4": return "ds4";
    case "ninfer": return "ninfer";
    case "openai": return "OpenAI-compatible";
  }
}

// The two numbers Pi actually uses for a model: the context window and the
// output cap, with a manual contextWindowOverride winning over whatever the
// server reported (see LLMModel.contextWindowOverride). maxTokens only
// derives from the window when an override is active — otherwise the
// detector's own value (which may be a backend-reported limit, not capTokens)
// stands.
function effectiveContextWindow(m: LLMModel): number {
  return m.contextWindowOverride ?? m.contextWindow;
}
function effectiveMaxTokens(m: LLMModel): number {
  return m.contextWindowOverride !== undefined
    ? capTokens(m.contextWindowOverride, m.reasoning)
    : m.maxTokens;
}

// Re-binds any user context-window overrides onto freshly detected models by
// matching model id. `from` is the previously stored models (the ones that may
// carry overrides), `detected` the result of a ↺ Refresh. The detected
// contextWindow is kept untouched so the override remains clearable and always
// re-attaches to the freshest value (a server that later reports a window does
// nothing to the stored override).
export function applyContextOverrides(
  from: LLMModel[],
  detected: DiscoveredModel[],
): LLMModel[] {
  return detected.map((d) => {
    const prev = from.find((e) => e.id === d.id);
    return prev?.contextWindowOverride !== undefined
      ? { ...d, contextWindowOverride: prev.contextWindowOverride }
      : d;
  });
}

function formatK(n: number): string {
  return n >= 1024 ? `${Math.round(n / 1024)}k` : `${n}`;
}

function formatBytes(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)}G`;
}

// Compares model ID sets, ignoring order — used to tell "the server got new
// models" apart from "the same model's metadata got refreshed". Only the
// former means whatever model the user had selected may no longer exist,
// which is the moment a switch prompt actually makes sense.
export function modelIdsChanged(before: LLMModel[], after: LLMModel[]): boolean {
  if (before.length !== after.length) return true;
  const beforeIds = new Set(before.map((m) => m.id));
  return after.some((m) => !beforeIds.has(m.id));
}

// "✓ " when known-loaded, "○ " when known-not-loaded, "" when the backend
// doesn't report loaded state at all (mtplx/llamacpp/vllm/generic OpenAI —
// see detect.ts for why).
function loadedIcon(loaded: boolean | undefined): string {
  if (loaded === true) return "✓ ";
  if (loaded === false) return "○ ";
  return "";
}

// Only backends that can actually distinguish loaded/unloaded models
// (oMLX, LM Studio, Ollama) ever set this — see detect.ts.
export function modelsHeading(models: LLMModel[]): string {
  const reportsLoaded = models.some((m) => m.loaded !== undefined);
  return reportsLoaded
    ? "Models:  (✓ = loaded in memory, ○ = will be loaded on first message)"
    : "Models:";
}

export function formatModelLine(m: LLMModel): string {
  const caps = [m.reasoning ? "reasoning" : null, m.input.includes("image") ? "vision" : null].filter(
    (c): c is string => c !== null,
  );
  // A trailing "*" marks a manually overridden context window, so it's clear
  // from the listing which value came from the server and which a human set.
  const ctxMarker = m.contextWindowOverride !== undefined ? "*" : "";
  const parts = [`ctx ${formatK(effectiveContextWindow(m))}${ctxMarker}`, `max ${formatK(effectiveMaxTokens(m))}`];
  // A zero size means "not reported", not a zero-byte model — SGLang's Ollama
  // shim sends size: 0 for the model it is actively serving. Detectors drop it
  // now, but servers configured before that still have the 0 on disk.
  if (typeof m.sizeBytes === "number" && m.sizeBytes > 0) parts.push(formatBytes(m.sizeBytes));
  if (m.quantization) parts.push(m.quantization);
  parts.push(...caps);
  return `  • ${loadedIcon(m.loaded)}${m.name}  (${parts.join(", ")})`;
}

// ─── Provider registration ────────────────────────────────────────

function registerServer(pi: ExtensionAPI, server: LLMServer): void {
  pi.registerProvider(toProviderId(server), {
    name: server.name,
    baseUrl: server.baseUrl,
    apiKey: server.apiKey || "no-key",
    api: "openai-completions",
    models: server.models.map((m) => ({
      id: m.id,
      name: m.name,
      reasoning: m.reasoning,
      input: m.input,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: effectiveContextWindow(m),
      maxTokens: effectiveMaxTokens(m),
      // `reasoning: true` here only means "this model can produce reasoning
      // output" (what our detectors observe) — it says nothing about whether
      // the server speaks the rest of OpenAI's o1-style reasoning-model
      // conventions, which none of our detectors check. pi-ai defaults both
      // of these to true for any non-hosted URL once reasoning is true:
      //   - supportsReasoningEffort: attaches a reasoning_effort request
      //     param to every message; strict backends (e.g. vLLM without a
      //     --reasoning-parser) 400 on the unrecognized field.
      //   - supportsDeveloperRole: sends the system prompt with role
      //     "developer" instead of "system"; a model's chat template that
      //     only handles "system" then rejects the request entirely
      //     ("Unexpected message role").
      // Disabling both keeps `reasoning` read-only: response parsing
      // (reasoning_content, etc.) still works if the backend sends it, but
      // nothing about the outgoing request changes because of it.
      //
      // A detector may opt a model back in via `compat`, but only on
      // evidence: ds4 by having had its source read; SGLang, ninfer and
      // llama.cpp by measuring the server directly, since their accepted
      // values belong to the loaded chat template rather than to the backend
      // (see detect.ts).
      // The defaults stay off for everything else, including hand-edited
      // models and servers configured before this field existed.
      //
      // Note what turning supportsReasoningEffort *on* actually changes: with
      // it off, no reasoning_effort is ever sent and the server keeps using
      // whatever it defaults to internally, no matter what Pi's status bar
      // says. Switching it on hands that decision to Pi — which is the point,
      // but it means a server that was quietly thinking at its own default
      // now follows the session's thinking level. Where those two disagree,
      // enabling this looks like a regression ("it stopped thinking") even
      // though it is the first time the setting was ever connected. A
      // detector that sets this should also supply thinkingLevelMap, so every
      // level Pi offers maps onto something the backend really distinguishes.
      compat: {
        supportsReasoningEffort: m.compat?.supportsReasoningEffort ?? false,
        supportsDeveloperRole: m.compat?.supportsDeveloperRole ?? false,
      },
      ...(m.thinkingLevelMap ? { thinkingLevelMap: m.thinkingLevelMap } : {}),
      // Left off entirely when unset, so the server keeps applying whatever
      // its own generation_config says rather than a value invented here.
      ...(m.samplingParams ? { samplingParams: m.samplingParams } : {}),
    })),
  });
}

function unregisterServer(pi: ExtensionAPI, server: LLMServer): void {
  pi.unregisterProvider(toProviderId(server));
}

async function removeServer(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
  server: LLMServer,
): Promise<boolean> {
  const ok = await ctx.ui.confirm(
    `Remove "${server.name}"?`,
    "Unregisters the provider and deletes its configuration.",
  );
  if (!ok) return false;
  unregisterServer(pi, server);
  if (os.platform() === "darwin") {
    await deleteFromKeychain(server.id);
  }
  const s = readSettings();
  s.servers = s.servers.filter((sv) => sv.id !== server.id);
  writeSettings(s);
  ctx.ui.notify(`${server.name} removed.`, "info");
  return true;
}

// ─── Setup wizard ─────────────────────────────────────────────────

async function runWizard(
  ctx: ExtensionCommandContext,
  existing?: LLMServer,
): Promise<LLMServer | null> {
  const name = await ctx.ui.input(
    'Step 1/3 - Server name (e.g. "My vLLM", "Ollama", "LM Studio")',
    existing?.name ?? "",
  );
  if (!name?.trim()) return null;

  const urlInput = await ctx.ui.input(
    "Step 2/3 - Base URL",
    existing ? existing.baseUrl.replace(/\/v1$/, "") : "http://localhost:8000",
  );
  if (!urlInput?.trim()) return null;
  const baseUrl = normalizeBaseUrl(urlInput);
  const id = existing?.id ?? generateId();

  const apiKeyInput = await ctx.ui.input(
    "Step 3/3 - API key (leave blank if not required)",
    existing?.apiKey ?? "",
  );
  let apiKey = apiKeyInput?.trim() ?? "";

  if (os.platform() === "darwin" && isDirectApiKey(apiKey)) {
    const store = await ctx.ui.confirm(
      "Store API key in macOS Keychain?",
      "Keeps the raw key out of settings.json — it'll be referenced via a !security command instead.",
    );
    if (store) {
      try {
        await storeInKeychain(id, apiKey);
        apiKey = keychainCommand(id);
        ctx.ui.notify("API key stored in Keychain.", "info");
      } catch (err: unknown) {
        ctx.ui.notify(
          `Failed to store in Keychain, keeping key in settings.json: ${err instanceof Error ? err.message : String(err)}`,
          "warning",
        );
      }
    }
  }

  ctx.ui.notify(`Connecting to ${baseUrl} ...`, "info");
  let result: DetectResult;
  try {
    result = await detectModels(baseUrl, apiKey, ctx.signal);
  } catch (err: unknown) {
    ctx.ui.notify(
      `Cannot reach server: ${err instanceof Error ? err.message : String(err)}`,
      "error",
    );
    return null;
  }

  if (result.models.length === 0) {
    ctx.ui.notify(result.error ?? "Server responded but has no loaded models.", "error");
    return null;
  }

  let selectedApiModels = result.models;
  if (result.models.length > 1) {
    const allOption = `All (${result.models.length} models)`;
    const modelOptions = [allOption, ...result.models.map((m) => m.id)];
    const picked = await ctx.ui.select(
      `${result.models.length} models found via ${apiTypeLabel(result.apiType)} - which to enable?`,
      modelOptions,
    );
    if (!picked) return null;
    if (picked !== allOption) {
      selectedApiModels = result.models.filter((m) => m.id === picked);
    }
  }

  return {
    id,
    name: name.trim(),
    baseUrl,
    apiKey,
    apiType: result.apiType,
    models: selectedApiModels,
  };
}

// ─── Manual capability & context override ─────────────────────────
// Some backends can't be asked whether a model supports vision/reasoning
// (see detect.ts's vLLM note), and some can't report a context window at
// all (llama-swap and other OpenAI-compat proxies, which always come back
// as whatever 32768 fallback detect.ts invented) — this lets a user fix
// all four by hand from the TUI instead of editing settings.json directly.
//
// The vision/reasoning/temperature tags are like any hand edit: they stick
// until the next ↺ Refresh overwrites them with fresh detected values. The
// context override is different — it is stored separately
// (contextWindowOverride) and re-attached to the freshly detected model by
// id in applyContextOverrides, so it survives refresh.

async function editModelCapabilities(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
  serverId: string,
): Promise<void> {
  const server = readSettings().servers.find((s) => s.id === serverId);
  if (!server || server.models.length === 0) return;

  let modelId = server.models[0].id;
  if (server.models.length > 1) {
    const picked = await ctx.ui.select(
      "Which model?",
      server.models.map((m) => m.id),
    );
    if (!picked) return;
    modelId = picked;
  }

  while (true) {
    const current = readSettings()
      .servers.find((s) => s.id === serverId)
      ?.models.find((m) => m.id === modelId);
    if (!current) return;

    const vision = current.input.includes("image");
    const temp = current.samplingParams?.temperature;
    const overridden = current.contextWindowOverride !== undefined;
    const OPT_VISION = `Vision: ${vision ? "on" : "off"}  (tap to turn ${vision ? "off" : "on"})`;
    const OPT_REASONING = `Reasoning: ${current.reasoning ? "on" : "off"}  (tap to turn ${current.reasoning ? "off" : "on"})`;
    const OPT_CONTEXT = `Max context: ${formatK(effectiveContextWindow(current))}${overridden ? " (manual)" : ""}  (tap to change)`;
    const OPT_TEMP = `Temperature: ${temp ?? "server default"}  (tap to change)`;
    const OPT_DONE = "✓ Done";

    const picked = await ctx.ui.select(
      `${current.name} - manual overrides\nVision / reasoning / temperature reset on ↺ Refresh; the context override persists.`,
      [OPT_VISION, OPT_REASONING, OPT_CONTEXT, OPT_TEMP, OPT_DONE],
    );
    if (!picked || picked === OPT_DONE) break;

    // Asked for before the settings are re-read, so a cancelled prompt
    // leaves the stored value untouched rather than clearing it.
    let nextTemp: { temperature?: number } | undefined;
    if (picked === OPT_TEMP) {
      const raw = await ctx.ui.input(
        "Temperature (0 - 2, or empty to let the server decide)",
        temp === undefined ? "" : String(temp),
      );
      if (raw === undefined) continue;
      const trimmed = raw.trim();
      if (trimmed === "") {
        nextTemp = undefined;
      } else {
        const parsed = Number(trimmed);
        if (!Number.isFinite(parsed) || parsed < 0 || parsed > 2) {
          ctx.ui.notify("Temperature must be a number between 0 and 2.", "error");
          continue;
        }
        nextTemp = { temperature: parsed };
      }
    }

    let nextCtxOverride: number | undefined;
    let clearCtxOverride = false;
    if (picked === OPT_CONTEXT) {
      const raw = await ctx.ui.input(
        `Max context length for "${current.name}" (tokens)\n` +
          `Current: ${effectiveContextWindow(current).toLocaleString()}  (${overridden ? "manual override" : "server-reported " + current.contextWindow.toLocaleString()})\n` +
          "Empty input removes the override and restores the server-reported value.",
        overridden ? String(current.contextWindowOverride) : "",
      );
      if (raw === undefined) continue;
      const trimmed = raw.trim();
      if (trimmed === "") {
        clearCtxOverride = true;
      } else {
        const parsed = Number(trimmed);
        if (!Number.isInteger(parsed) || parsed < 1) {
          ctx.ui.notify("Context length must be a positive integer.", "error");
          continue;
        }
        nextCtxOverride = parsed;
      }
    }

    const s = readSettings();
    const sv = s.servers.find((sv) => sv.id === serverId);
    if (!sv) return;
    sv.models = sv.models.map((m) => {
      if (m.id !== modelId) return m;
      if (picked === OPT_VISION) {
        return { ...m, input: vision ? (["text"] as const) : (["text", "image"] as const) };
      }
      if (picked === OPT_CONTEXT) {
        if (clearCtxOverride) {
          const { contextWindowOverride: _drop, ...rest } = m;
          return rest;
        }
        return { ...m, contextWindowOverride: nextCtxOverride };
      }
      if (picked === OPT_TEMP) {
        const { samplingParams: _dropped, ...rest } = m;
        return nextTemp ? { ...rest, samplingParams: nextTemp } : rest;
      }
      return { ...m, reasoning: !m.reasoning };
    });
    writeSettings(s);
  }

  const s = readSettings();
  const sv = s.servers.find((sv) => sv.id === serverId);
  if (!sv) return;
  unregisterServer(pi, server);
  registerServer(pi, sv);
  ctx.ui.notify("Capabilities updated.", "info");
}

// ─── Server sub-menu ──────────────────────────────────────────────

const OPT_REFRESH = "↺ Refresh model list from server";
const OPT_CAPS    = "✎ Edit model capabilities (vision / reasoning / context / temperature)";
const OPT_EDIT    = "✎ Reconfigure (name / URL / key)";
const OPT_REMOVE  = "✕ Remove this server";
const OPT_BACK    = "← Back";

async function showServerMenu(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
  serverId: string,
): Promise<void> {
  while (true) {
    const server = readSettings().servers.find((s) => s.id === serverId);
    if (!server) return;

    const modelSummary =
      server.models.length === 0
        ? "  (no models)"
        : server.models.map(formatModelLine).join("\n");
    const backend = apiTypeLabel(server.apiType);

    const picked = await ctx.ui.select(
      `${server.name}  [${backend}]\nURL: ${server.baseUrl}\n${modelsHeading(server.models)}\n${modelSummary}`,
      [OPT_REFRESH, ...(server.models.length > 0 ? [OPT_CAPS] : []), OPT_EDIT, OPT_REMOVE, OPT_BACK],
    );

    if (!picked || picked === OPT_BACK) return;

    if (picked === OPT_REFRESH) {
      ctx.ui.notify(`Refreshing ${server.name} ...`, "info");
      let result: DetectResult;
      try {
        result = await detectModels(server.baseUrl, server.apiKey, ctx.signal);
      } catch (err: unknown) {
        ctx.ui.notify(
          `Failed: ${err instanceof Error ? err.message : String(err)}`,
          "error",
        );
        continue;
      }

      // A real failure (bad key, timeout, unreachable) must not wipe out an
      // already-known-good model list — only overwrite on success or on a
      // genuine "server's fine, zero models loaded" response.
      if (result.models.length === 0 && result.error) {
        ctx.ui.notify(`Refresh failed, keeping existing configuration: ${result.error}`, "error");
        continue;
      }

      const updated: LLMServer = {
        ...server,
        apiType: result.apiType,
        models: applyContextOverrides(server.models, result.models),
      };
      const modelsChanged = modelIdsChanged(server.models, updated.models);
      const s = readSettings();
      s.servers = s.servers.map((sv) => (sv.id === serverId ? updated : sv));
      writeSettings(s);
      unregisterServer(pi, server);
      registerServer(pi, updated);
      ctx.ui.notify(
        `${server.name} (${apiTypeLabel(result.apiType)}): ${result.models.length} model(s) - ${result.models.map((m) => m.name).join(", ")}`,
        "info",
      );

      // Same metadata refreshed for the same model(s) isn't worth a prompt.
      // But if the server now serves different models entirely, whatever
      // was previously selected may no longer exist — that's the case a
      // switch prompt is for, same as Add/Reconfigure.
      if (modelsChanged && updated.models.length === 1) {
        await offerModelSwitch(pi, ctx, updated, updated.models[0]);
      }
      continue;
    }

    if (picked === OPT_CAPS) {
      await editModelCapabilities(pi, ctx, serverId);
      continue;
    }

    if (picked === OPT_EDIT) {
      const updatedRaw = await runWizard(ctx, server);
      if (!updatedRaw) continue;
      // Reconfigure re-detects from the server, so re-attach any manual
      // context-window overrides the same way a ↺ Refresh does.
      const updated: LLMServer = {
        ...updatedRaw,
        models: applyContextOverrides(server.models, updatedRaw.models),
      };
      const s = readSettings();
      s.servers = s.servers.map((sv) => (sv.id === serverId ? updated : sv));
      writeSettings(s);
      unregisterServer(pi, server);
      registerServer(pi, updated);
      ctx.ui.notify(
        `${updated.name} updated.` + (updated.models.length > 1 ? " Switch models with /model." : ""),
        "info",
      );
      if (updated.models.length === 1) {
        await offerModelSwitch(pi, ctx, updated, updated.models[0]);
      }
      return;
    }

    if (picked === OPT_REMOVE) {
      if (await removeServer(pi, ctx, server)) return;
      continue;
    }
  }
}

// ─── Switch-on-add ────────────────────────────────────────────────

async function offerModelSwitch(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
  server: LLMServer,
  model: LLMModel,
): Promise<void> {
  const switchNow = await ctx.ui.confirm(
    `Switch to ${model.name} now?`,
    `Makes it the active model for this session. You can always change it later with /model.`,
  );
  if (!switchNow) return;

  const resolved = ctx.modelRegistry.find(toProviderId(server), model.id);
  if (!resolved) {
    ctx.ui.notify("Couldn't find the newly registered model — switch with /model instead.", "warning");
    return;
  }

  const ok = await pi.setModel(resolved);
  if (!ok) {
    ctx.ui.notify("Couldn't switch — no API key available for this model.", "warning");
  }
}

// ─── Main menu ────────────────────────────────────────────────────

const OPT_ADD = "＋ Add server";

async function showMainMenu(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
): Promise<void> {
  while (true) {
    const { servers } = readSettings();

    // Server labels are used both as display text and as index keys.
    // Each label encodes name + URL so the indexOf lookup is unambiguous
    // even when two servers share the same name.
    const serverLabels = servers.map(
      (s) =>
        `${s.name}  [${apiTypeLabel(s.apiType)}]  (${s.baseUrl})  ${s.models.length} model(s)`,
    );

    const picked = await ctx.ui.select(
      servers.length === 0
        ? "LocalLLM - no servers configured"
        : `LocalLLM - ${servers.length} server(s)`,
      [...serverLabels, OPT_ADD],
    );

    if (!picked) return;

    if (picked === OPT_ADD) {
      const server = await runWizard(ctx);
      if (!server) continue;
      const s = readSettings();
      s.servers.push(server);
      writeSettings(s);
      registerServer(pi, server);
      ctx.ui.notify(
        `${server.name} added - ${server.models.length} model(s): ${server.models.map((m) => m.name).join(", ")}.` +
          (server.models.length > 1 ? " Switch with /model." : ""),
        "info",
      );

      // Only offer to switch when exactly one model is in play — either it
      // was the server's only model, or the user picked one specifically in
      // the wizard. With several enabled at once (the "All" option) there's
      // no clear single model to switch to, so this is skipped in favor of
      // the existing /model flow.
      if (server.models.length === 1) {
        await offerModelSwitch(pi, ctx, server, server.models[0]);
      }
      continue;
    }

    const idx = serverLabels.indexOf(picked);
    if (idx >= 0) {
      await showServerMenu(pi, ctx, servers[idx].id);
    }
  }
}

// ─── Extension entry point ────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  for (const server of readSettings().servers) {
    registerServer(pi, server);
  }

  pi.registerCommand("localllm", {
    description: "Manage LocalLLM providers - wizard-based setup for any OpenAI-compatible local server",
    async handler(_args, ctx) {
      await showMainMenu(pi, ctx);
    },
  });
}
