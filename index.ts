import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { detectModels, type ApiType, type DetectResult } from "./detect.ts";
import { deleteFromKeychain, isDirectApiKey, keychainCommand, storeInKeychain } from "./keychain.ts";

// ─── Types ────────────────────────────────────────────────────────

interface LLMModel {
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

function apiTypeLabel(apiType: ApiType): string {
  switch (apiType) {
    case "mtplx": return "MTPLX";
    case "omlx": return "oMLX";
    case "lmstudio": return "LM Studio";
    case "llamacpp": return "llama.cpp";
    case "ollama": return "Ollama";
    case "vllm": return "vLLM";
    case "openai": return "OpenAI-compatible";
  }
}

function formatK(n: number): string {
  return n >= 1024 ? `${Math.round(n / 1024)}k` : `${n}`;
}

function formatBytes(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)}G`;
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
  const parts = [`ctx ${formatK(m.contextWindow)}`, `max ${formatK(m.maxTokens)}`];
  if (typeof m.sizeBytes === "number") parts.push(formatBytes(m.sizeBytes));
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
      contextWindow: m.contextWindow,
      maxTokens: m.maxTokens,
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
      compat: { supportsReasoningEffort: false, supportsDeveloperRole: false },
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

// ─── Manual capability override ────────────────────────────────────
// Some backends can't be asked whether a model supports vision/reasoning
// (see detect.ts's vLLM note) — this lets a user fix the tags by hand from
// the TUI instead of editing settings.json directly. Like any hand edit,
// it sticks until the next ↺ Refresh overwrites it with fresh detected
// values.

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
    const OPT_VISION = `Vision: ${vision ? "on" : "off"}  (tap to turn ${vision ? "off" : "on"})`;
    const OPT_REASONING = `Reasoning: ${current.reasoning ? "on" : "off"}  (tap to turn ${current.reasoning ? "off" : "on"})`;
    const OPT_DONE = "✓ Done";

    const picked = await ctx.ui.select(
      `${current.name} - manual capability override\nOverwritten by the next ↺ Refresh.`,
      [OPT_VISION, OPT_REASONING, OPT_DONE],
    );
    if (!picked || picked === OPT_DONE) break;

    const s = readSettings();
    const sv = s.servers.find((sv) => sv.id === serverId);
    if (!sv) return;
    sv.models = sv.models.map((m) => {
      if (m.id !== modelId) return m;
      if (picked === OPT_VISION) {
        return { ...m, input: vision ? (["text"] as const) : (["text", "image"] as const) };
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
const OPT_CAPS    = "✎ Edit model capabilities (vision / reasoning)";
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

      const updated: LLMServer = { ...server, apiType: result.apiType, models: result.models };
      const s = readSettings();
      s.servers = s.servers.map((sv) => (sv.id === serverId ? updated : sv));
      writeSettings(s);
      unregisterServer(pi, server);
      registerServer(pi, updated);
      ctx.ui.notify(
        `${server.name} (${apiTypeLabel(result.apiType)}): ${result.models.length} model(s) - ${result.models.map((m) => m.name).join(", ")}`,
        "info",
      );
      continue;
    }

    if (picked === OPT_CAPS) {
      await editModelCapabilities(pi, ctx, serverId);
      continue;
    }

    if (picked === OPT_EDIT) {
      const updated = await runWizard(ctx, server);
      if (!updated) continue;
      const s = readSettings();
      s.servers = s.servers.map((sv) => (sv.id === serverId ? updated : sv));
      writeSettings(s);
      unregisterServer(pi, server);
      registerServer(pi, updated);
      ctx.ui.notify(`${updated.name} updated. Switch models with /model.`, "info");
      return;
    }

    if (picked === OPT_REMOVE) {
      if (await removeServer(pi, ctx, server)) return;
      continue;
    }
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
        `${server.name} added - ${server.models.length} model(s): ${server.models.map((m) => m.name).join(", ")}. Switch with /model.`,
        "info",
      );
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
