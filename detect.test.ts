import { afterEach, describe, expect, it, vi } from "vitest";
import {
  detectLlamaCpp,
  detectLmStudio,
  detectModels,
  detectMtplx,
  detectOllama,
  detectOmlx,
  detectOpenAI,
  detectSglang,
  detectVllm,
  isNinferCards,
  nearestTier,
  parseNinferContext,
  probeNinfer,
  probeRequestCompat,
} from "./detect.ts";

function mockFetch(routes: Record<string, unknown>) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const key = String(url);
      if (!(key in routes)) {
        return { ok: false, json: async () => ({}) } as unknown as Response;
      }
      return { ok: true, json: async () => routes[key] } as unknown as Response;
    }),
  );
}

// Every fetch call fails with the same HTTP status — simulates a bad API
// key (every endpoint 401s/403s alike) or a server that 404s everything.
function mockFetchAllStatus(status: number) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({ ok: false, status, json: async () => ({}) }) as unknown as Response),
  );
}

// Every fetch call rejects — simulates a timeout (AbortError) or a network-
// level failure (connection refused, DNS failure, etc).
function mockFetchAllReject(err: Error) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => {
      throw err;
    }),
  );
}

// Ollama's /api/show is a POST to the same URL for every model — the only
// thing that varies is the request body, so route by (url, body.model)
// instead of by URL alone.
function mockOllama(
  tagsResponse: unknown,
  showResponsesByModel: Record<string, unknown>,
  psResponse: unknown = { models: [] },
) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const u = String(url);
      if (u.endsWith("/api/tags")) {
        return { ok: true, json: async () => tagsResponse } as unknown as Response;
      }
      if (u.endsWith("/api/ps")) {
        return { ok: true, json: async () => psResponse } as unknown as Response;
      }
      if (u.endsWith("/api/show")) {
        const body = init?.body ? JSON.parse(String(init.body)) : {};
        const resp = showResponsesByModel[body.model];
        if (!resp) return { ok: false, json: async () => ({}) } as unknown as Response;
        return { ok: true, json: async () => resp } as unknown as Response;
      }
      return { ok: false, json: async () => ({}) } as unknown as Response;
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("detectMtplx", () => {
  it("returns null when /health lacks model or context_window", async () => {
    mockFetch({ "http://x/health": {} });
    expect(await detectMtplx("http://x", "")).toBeNull();
  });

  it("returns null on network failure / non-200", async () => {
    mockFetch({});
    expect(await detectMtplx("http://x", "")).toBeNull();
  });

  it("parses a full /health response", async () => {
    mockFetch({
      "http://x/health": {
        model: "org/Model-7B",
        context_window: 65536,
        max_response_tokens: 4096,
        enable_thinking: true,
        vision: { enabled: true },
      },
    });
    expect(await detectMtplx("http://x", "")).toEqual({
      apiType: "mtplx",
      models: [
        {
          id: "org/Model-7B",
          name: "Model-7B",
          contextWindow: 65536,
          maxTokens: 4096,
          reasoning: true,
          input: ["text", "image"],
        },
      ],
    });
  });

  it("falls back to capped maxTokens when max_response_tokens is absent", async () => {
    mockFetch({ "http://x/health": { model: "m", context_window: 4000 } });
    const result = await detectMtplx("http://x", "");
    expect(result?.models[0].maxTokens).toBe(2000);
  });

  it("treats reasoning:'on' the same as enable_thinking:true", async () => {
    mockFetch({ "http://x/health": { model: "m", context_window: 4000, reasoning: "on" } });
    const result = await detectMtplx("http://x", "");
    expect(result?.models[0].reasoning).toBe(true);
  });

  it("defaults to text-only input when vision is absent or disabled", async () => {
    mockFetch({ "http://x/health": { model: "m", context_window: 4000 } });
    const result = await detectMtplx("http://x", "");
    expect(result?.models[0].input).toEqual(["text"]);
  });
});

describe("detectOmlx", () => {
  it("returns null when the server has no models", async () => {
    mockFetch({ "http://x/v1/models/status": { models: [] } });
    expect(await detectOmlx("http://x", "")).toBeNull();
  });

  it("filters out non llm/vlm model types", async () => {
    mockFetch({
      "http://x/v1/models/status": {
        models: [
          { id: "a", model_type: "llm", max_context_window: 8192 },
          { id: "b", model_type: "embedding", max_context_window: 8192 },
          { id: "c" }, // missing model_type entirely
        ],
      },
    });
    const result = await detectOmlx("http://x", "");
    expect(result?.models.map((m) => m.id)).toEqual(["a"]);
  });

  it("prefers model_alias, falls back to display_name then id", async () => {
    mockFetch({
      "http://x/v1/models/status": {
        models: [{ id: "id1", display_name: "Display", model_type: "llm", max_context_window: 4096 }],
      },
    });
    const result = await detectOmlx("http://x", "");
    expect(result?.models[0].name).toBe("Display");
  });

  it("marks vlm models as vision-capable and reads thinking_default", async () => {
    mockFetch({
      "http://x/v1/models/status": {
        models: [
          {
            id: "id1",
            model_alias: "Alias",
            model_type: "vlm",
            max_context_window: 4096,
            thinking_default: true,
          },
        ],
      },
    });
    const result = await detectOmlx("http://x", "");
    expect(result?.models[0]).toMatchObject({
      name: "Alias",
      input: ["text", "image"],
      reasoning: true,
    });
  });

  it("reads loaded status and estimated_size", async () => {
    mockFetch({
      "http://x/v1/models/status": {
        models: [
          {
            id: "id1",
            model_type: "llm",
            max_context_window: 4096,
            loaded: true,
            estimated_size: 4912898304,
          },
        ],
      },
    });
    const result = await detectOmlx("http://x", "");
    expect(result?.models[0]).toMatchObject({ loaded: true, sizeBytes: 4912898304 });
  });

  it("treats a missing loaded field as not loaded", async () => {
    mockFetch({
      "http://x/v1/models/status": {
        models: [{ id: "id1", model_type: "llm", max_context_window: 4096 }],
      },
    });
    const result = await detectOmlx("http://x", "");
    expect(result?.models[0].loaded).toBe(false);
  });
});

describe("detectLmStudio", () => {
  it("returns null when the server has no models", async () => {
    mockFetch({ "http://x/api/v1/models": { models: [] } });
    expect(await detectLmStudio("http://x", "")).toBeNull();
  });

  it("filters by type and reads capabilities", async () => {
    mockFetch({
      "http://x/api/v1/models": {
        models: [
          {
            key: "k1",
            type: "llm",
            max_context_length: 32768,
            capabilities: { reasoning: { allowed_options: ["low"] } },
          },
          { key: "k2", type: "embedding" },
        ],
      },
    });
    const result = await detectLmStudio("http://x", "");
    expect(result?.models).toHaveLength(1);
    expect(result?.models[0]).toMatchObject({ id: "k1", contextWindow: 32768, reasoning: true });
  });

  it("marks vision-capable models via capabilities.vision or vlm type", async () => {
    mockFetch({
      "http://x/api/v1/models": {
        models: [{ key: "k1", type: "vlm", capabilities: { vision: true } }],
      },
    });
    const result = await detectLmStudio("http://x", "");
    expect(result?.models[0].input).toEqual(["text", "image"]);
  });

  it("reads loaded status, size, and quantization", async () => {
    mockFetch({
      "http://x/api/v1/models": {
        models: [
          {
            key: "k1",
            type: "llm",
            max_context_length: 4096,
            loaded_instances: [{}],
            size_bytes: 4912898304,
            quantization: { name: "Q4_K_M" },
          },
        ],
      },
    });
    const result = await detectLmStudio("http://x", "");
    expect(result?.models[0]).toMatchObject({
      loaded: true,
      sizeBytes: 4912898304,
      quantization: "Q4_K_M",
    });
  });

  it("treats an empty loaded_instances array as not loaded", async () => {
    mockFetch({
      "http://x/api/v1/models": {
        models: [{ key: "k1", type: "llm", loaded_instances: [] }],
      },
    });
    const result = await detectLmStudio("http://x", "");
    expect(result?.models[0].loaded).toBe(false);
  });
});

describe("detectLlamaCpp", () => {
  it("returns null when /props lacks n_ctx or model_path", async () => {
    mockFetch({ "http://x/props": { default_generation_settings: {} } });
    expect(await detectLlamaCpp("http://x", "")).toBeNull();
  });

  it("combines /props (context, vision) with /v1/models (alias-aware id)", async () => {
    mockFetch({
      "http://x/props": {
        default_generation_settings: { n_ctx: 8192 },
        model_path: "../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        modalities: { vision: true },
      },
      "http://x/v1/models": {
        data: [{ id: "gpt-4o-mini", meta: { n_ctx_train: 131072 } }],
      },
    });
    const result = await detectLlamaCpp("http://x", "");
    expect(result).toEqual({
      apiType: "llamacpp",
      models: [
        {
          id: "gpt-4o-mini",
          name: "gpt-4o-mini",
          contextWindow: 8192,
          maxTokens: 4096,
          reasoning: false,
          input: ["text", "image"],
        },
      ],
    });
  });

  it("derives a name from the model file basename when no --alias is set", async () => {
    mockFetch({
      "http://x/props": {
        default_generation_settings: { n_ctx: 4096 },
        model_path: "../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
      },
    });
    const result = await detectLlamaCpp("http://x", "");
    expect(result?.models[0]).toMatchObject({
      id: "../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
      name: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    });
  });

  it("falls back to n_ctx_train when /props doesn't report a running context", async () => {
    mockFetch({
      "http://x/props": {
        default_generation_settings: { n_ctx: 0 },
        model_path: "/models/m.gguf",
      },
      "http://x/v1/models": { data: [{ id: "/models/m.gguf", meta: { n_ctx_train: 131072 } }] },
    });
    const result = await detectLlamaCpp("http://x", "");
    expect(result?.models[0].contextWindow).toBe(131072);
  });

  it("reads file size from /v1/models meta", async () => {
    mockFetch({
      "http://x/props": { default_generation_settings: { n_ctx: 4096 }, model_path: "/models/m.gguf" },
      "http://x/v1/models": { data: [{ id: "/models/m.gguf", meta: { size: 4912898304 } }] },
    });
    const result = await detectLlamaCpp("http://x", "");
    expect(result?.models[0].sizeBytes).toBe(4912898304);
  });
});

// A llama-server whose /apply-template answers. Each render is resolved
// from the request body: the two forced enable_thinking cases and the
// unforced one return the prompt named in `prompts`; a reasoning_effort
// tier returns the status in `tiers` (default 200), where a 500 carries
// the Jinja raise message the server really emits; "__developer__" is the
// role probe. A missing `prompts` leaves /apply-template unrouted.
interface LlamaCppTemplate {
  prompts?: { byDefault: string; on: string; off: string };
  tiers?: Record<string, number | "network-error">;
  developer?: number;
}

const TEMPLATE_RAISE_BODY =
  '{"error":{"code":500,"message":"\\nError executing statement at position 1234: Jinja Exception: Unexpected reasoning effort high.","type":"server_error"}}';

function mockLlamaCpp(template: LlamaCppTemplate, props: Record<string, unknown> = {}) {
  const routes: Record<string, unknown> = {
    "http://x/props": {
      default_generation_settings: { n_ctx: 32768 },
      model_path: "/models/Qwen3.8-27B-Q4_K_M.gguf",
      ...props,
    },
    "http://x/v1/models": { data: [{ id: "/models/Qwen3.8-27B-Q4_K_M.gguf" }] },
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const u = String(url);
      if (u.endsWith("/apply-template") && template.prompts) {
        const body = init?.body ? JSON.parse(String(init.body)) : {};
        const reply = (status: number, text: string) =>
          ({ ok: status === 200, status, text: async () => text, json: async () => ({}) }) as unknown as Response;
        if ((body.messages ?? []).some((m: { role: string }) => m.role === "developer")) {
          return reply(template.developer ?? 200, "");
        }
        if (typeof body.reasoning_effort === "string") {
          const status = template.tiers?.[body.reasoning_effort] ?? 200;
          if (status === "network-error") throw new Error("connection refused");
          return reply(status, status === 500 ? TEMPLATE_RAISE_BODY : "");
        }
        const enable = body.chat_template_kwargs?.enable_thinking;
        const prompt =
          enable === true ? template.prompts.on : enable === false ? template.prompts.off : template.prompts.byDefault;
        return reply(200, JSON.stringify({ prompt }));
      }
      if (u in routes) return { ok: true, status: 200, json: async () => routes[u] } as unknown as Response;
      return { ok: false, status: 404, json: async () => ({}) } as unknown as Response;
    }),
  );
}

// Qwen3's generation prompt, with and without the empty think block the
// template closes when thinking is disabled.
const QWEN3_THINKING = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
const QWEN3_NO_THINKING = `${QWEN3_THINKING}<think>\n\n</think>\n\n`;

// The tiers a Qwen3.8 template accepts, reached through llama-server this
// time: "none" is handled by the server before the template sees it, and
// a rejected tier is a Jinja raise, which llama-server reports as a 500.
const QWEN38_TEMPLATE_TIERS = { minimal: 500, high: 500, max: 500 };

describe("detectLlamaCpp thinking", () => {
  it("registers a thinking template as reasoning and measures its levels", async () => {
    mockLlamaCpp({
      prompts: { byDefault: QWEN3_THINKING, on: QWEN3_THINKING, off: QWEN3_NO_THINKING },
      tiers: QWEN38_TEMPLATE_TIERS,
    });
    const result = await detectLlamaCpp("http://x", "");
    expect(result?.models[0]).toMatchObject({
      reasoning: true,
      maxTokens: 16384,
      compat: { supportsReasoningEffort: true, supportsDeveloperRole: true },
      thinkingLevelMap: {
        off: "none",
        minimal: "low",
        low: "low",
        medium: "medium",
        high: "medium",
        xhigh: "xhigh",
        max: "xhigh",
      },
    });
  });

  it("treats a template with no thinking switch as non-reasoning", async () => {
    mockLlamaCpp({
      prompts: { byDefault: QWEN3_THINKING, on: QWEN3_THINKING, off: QWEN3_THINKING },
    });
    const result = await detectLlamaCpp("http://x", "");
    expect(result?.models[0]).toMatchObject({ reasoning: false, maxTokens: 8192 });
    expect(result?.models[0]).not.toHaveProperty("compat");
    expect(result?.models[0]).not.toHaveProperty("thinkingLevelMap");
  });

  // `--reasoning off`: the switch exists, but the server sits on the off
  // side of it, and Pi has no field that could move it back.
  it("treats a server started with thinking off as non-reasoning", async () => {
    mockLlamaCpp({
      prompts: { byDefault: QWEN3_NO_THINKING, on: QWEN3_THINKING, off: QWEN3_NO_THINKING },
    });
    const result = await detectLlamaCpp("http://x", "");
    expect(result?.models[0].reasoning).toBe(false);
    expect(fetch).not.toHaveBeenCalledWith(
      "http://x/apply-template",
      expect.objectContaining({ body: expect.stringContaining("reasoning_effort") }),
    );
  });

  // /props reports reasoning_format "none" on a server that is splitting
  // reasoning_content out of every reply (measured on b10692), so that
  // field must not gate anything.
  it("ignores the reasoning_format /props reports", async () => {
    mockLlamaCpp(
      { prompts: { byDefault: QWEN3_THINKING, on: QWEN3_THINKING, off: QWEN3_NO_THINKING } },
      { default_generation_settings: { n_ctx: 32768, params: { reasoning_format: "none" } } },
    );
    const result = await detectLlamaCpp("http://x", "");
    expect(result?.models[0].reasoning).toBe(true);
  });

  it("keeps the safe defaults on a build without /apply-template", async () => {
    mockLlamaCpp({});
    const result = await detectLlamaCpp("http://x", "");
    expect(result?.models[0]).toMatchObject({ reasoning: false });
    expect(result?.models[0]).not.toHaveProperty("compat");
  });

  // A 500 that does not quote a template exception is the server failing,
  // not the template answering — same all-or-nothing bar as the SGLang sweep.
  it("keeps reasoning but claims no levels when a tier fails for another reason", async () => {
    mockLlamaCpp({
      prompts: { byDefault: QWEN3_THINKING, on: QWEN3_THINKING, off: QWEN3_NO_THINKING },
      tiers: { ...QWEN38_TEMPLATE_TIERS, medium: "network-error" },
    });
    const result = await detectLlamaCpp("http://x", "");
    expect(result?.models[0].reasoning).toBe(true);
    expect(result?.models[0]).not.toHaveProperty("compat");
    expect(result?.models[0]).not.toHaveProperty("thinkingLevelMap");
  });

  it("records a developer role the template rejects", async () => {
    mockLlamaCpp({
      prompts: { byDefault: QWEN3_THINKING, on: QWEN3_THINKING, off: QWEN3_NO_THINKING },
      developer: 500,
    });
    const result = await detectLlamaCpp("http://x", "");
    expect(result?.models[0].compat).toEqual({ supportsReasoningEffort: true, supportsDeveloperRole: false });
  });
});

// SGLang's real payloads, trimmed to the fields the detector reads. The
// /api/* entries are its Ollama compatibility shim, present so the chain
// tests exercise the ordering that shim makes necessary.
function mockSglang(overrides: { modelInfo?: unknown; serverInfo?: unknown; models?: unknown } = {}) {
  mockFetch({
    "http://x/get_model_info": overrides.modelInfo ?? {
      model_path: "/models/Qwen3.8-27B-NVFP4",
      is_generation: true,
      has_image_understanding: true,
    },
    "http://x/get_server_info": overrides.serverInfo ?? { reasoning_parser: "qwen3", context_length: null },
    "http://x/v1/models": overrides.models ?? {
      data: [{ id: "/models/Qwen3.8-27B-NVFP4", owned_by: "sglang", max_model_len: 262144 }],
    },
    "http://x/api/tags": {
      models: [
        {
          name: "/models/Qwen3.8-27B-NVFP4",
          model: "/models/Qwen3.8-27B-NVFP4",
          details: { format: "sglang" },
        },
      ],
    },
    "http://x/api/show": { model_info: {}, capabilities: ["completion"] },
  });
}

// Mirrors a real SGLang probe sweep: GETs answer from `routes`, and each
// POST /chat/completions is resolved by looking up its reasoning_effort (or
// "__developer__" for the role probe) in `statuses`.
function mockSglangProbe(statuses: Record<string, number | "network-error">, routes: Record<string, unknown> = {}) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const u = String(url);
      if (u.endsWith("/chat/completions")) {
        const body = init?.body ? JSON.parse(String(init.body)) : {};
        const isDeveloper = (body.messages ?? []).some((m: { role: string }) => m.role === "developer");
        const key = isDeveloper ? "__developer__" : String(body.reasoning_effort);
        const status = statuses[key] ?? 400;
        if (status === "network-error") throw new Error("connection refused");
        return { ok: status === 200, status, text: async () => "", json: async () => ({}) } as unknown as Response;
      }
      if (u in routes) return { ok: true, status: 200, json: async () => routes[u] } as unknown as Response;
      return { ok: false, status: 404, json: async () => ({}) } as unknown as Response;
    }),
  );
}

// The vocabulary measured from a real Qwen3.8-27B server, whose template
// answers 400 with "Supported types are xhigh (default), medium, and low".
const QWEN38_STATUSES = {
  none: 200,
  minimal: 400,
  low: 200,
  medium: 200,
  high: 400,
  xhigh: 200,
  max: 400,
  __developer__: 400,
};

describe("nearestTier", () => {
  const accepted = ["none", "low", "medium", "xhigh"];

  it("passes through a level the model accepts verbatim", () => {
    expect(nearestTier("low", accepted)).toBe("low");
    expect(nearestTier("xhigh", accepted)).toBe("xhigh");
  });

  it("lifts a level below the model's floor up to it", () => {
    expect(nearestTier("minimal", accepted)).toBe("low");
  });

  it("drops a level above the model's ceiling down to it", () => {
    expect(nearestTier("max", accepted)).toBe("xhigh");
  });

  // "high" sits one step from both "medium" and "xhigh"; asking for more
  // thinking than exists should land under the ceiling, not over it.
  it("breaks a tie toward the weaker tier", () => {
    expect(nearestTier("high", accepted)).toBe("medium");
  });

  // "none" means stop reasoning — it must never win on proximity alone.
  it("never resolves a thinking level to none", () => {
    expect(nearestTier("minimal", ["none", "xhigh"])).toBe("xhigh");
  });

  it("returns undefined when the model accepts no thinking tier", () => {
    expect(nearestTier("medium", ["none"])).toBeUndefined();
  });
});

describe("probeRequestCompat", () => {
  it("reproduces the map hand-verified against a Qwen3.8 server", async () => {
    mockSglangProbe(QWEN38_STATUSES);
    const result = await probeRequestCompat("http://x/v1", "", "m");
    expect(result).toEqual({
      compat: { supportsReasoningEffort: true, supportsDeveloperRole: false },
      thinkingLevelMap: {
        off: "none",
        minimal: "low",
        low: "low",
        medium: "medium",
        high: "medium",
        xhigh: "xhigh",
        max: "xhigh",
      },
    });
  });

  it("records the developer role when the template accepts it", async () => {
    mockSglangProbe({ ...QWEN38_STATUSES, __developer__: 200 });
    const result = await probeRequestCompat("http://x/v1", "", "m");
    expect(result.compat?.supportsDeveloperRole).toBe(true);
  });

  // Without an accepted "none" there is no off switch, so leaving the level
  // unmapped is what makes Pi omit the field and take the model's default.
  it("leaves off unmapped when the model cannot disable reasoning", async () => {
    mockSglangProbe({ ...QWEN38_STATUSES, none: 400 });
    const result = await probeRequestCompat("http://x/v1", "", "m");
    expect(result.thinkingLevelMap).not.toHaveProperty("off");
    expect(result.thinkingLevelMap?.medium).toBe("medium");
  });

  it("claims nothing when the server rejects every tier", async () => {
    mockSglangProbe({ __developer__: 400 });
    expect(await probeRequestCompat("http://x/v1", "", "m")).toEqual({});
  });

  // The damaging case: a busy server 503s one tier, and scoring that as
  // "unsupported" would save a map that 400s on every later request.
  it("claims nothing when a single tier fails for a reason other than rejection", async () => {
    mockSglangProbe({ ...QWEN38_STATUSES, medium: 503 });
    expect(await probeRequestCompat("http://x/v1", "", "m")).toEqual({});
  });

  it("treats a rate-limited tier as unanswered, not as unsupported", async () => {
    mockSglangProbe({ ...QWEN38_STATUSES, xhigh: 429 });
    expect(await probeRequestCompat("http://x/v1", "", "m")).toEqual({});
  });

  it("still trusts a sweep when only the developer probe fails", async () => {
    mockSglangProbe({ ...QWEN38_STATUSES, __developer__: "network-error" });
    const result = await probeRequestCompat("http://x/v1", "", "m");
    expect(result.compat).toEqual({ supportsReasoningEffort: true, supportsDeveloperRole: false });
    expect(result.thinkingLevelMap?.high).toBe("medium");
  });

  // A server that never answers is not evidence against the conventions.
  it("claims nothing when no probe gets a reply at all", async () => {
    mockSglangProbe({
      none: "network-error",
      minimal: "network-error",
      low: "network-error",
      medium: "network-error",
      high: "network-error",
      xhigh: "network-error",
      max: "network-error",
      __developer__: "network-error",
    });
    expect(await probeRequestCompat("http://x/v1", "", "m")).toEqual({});
  });
});

describe("detectSglang", () => {
  it("reads context from /v1/models and capabilities from the SGLang endpoints", async () => {
    mockSglang();
    const result = await detectSglang("http://x", "http://x/v1", "");
    expect(result?.apiType).toBe("sglang");
    expect(result?.models).toHaveLength(1);
    expect(result?.models[0]).toMatchObject({
      id: "/models/Qwen3.8-27B-NVFP4",
      name: "Qwen3.8-27B-NVFP4",
      contextWindow: 262144,
      maxTokens: 65536,
      reasoning: true,
      input: ["text", "image"],
    });
  });

  // A Qwen checkpoint's generation_config commonly ships temperature 1.0,
  // which SGLang applies when the request names none — and Pi's don't. Qwen
  // publishes 0.6 for thinking mode, and 1.0 makes a thinking model in an
  // agent loop re-plan until it exhausts the budget.
  it("applies Qwen's published thinking temperature", async () => {
    mockSglang({
      modelInfo: {
        model_path: "/models/m",
        is_generation: true,
        model_type: "qwen3_5",
        has_image_understanding: true,
      },
    });
    const result = await detectSglang("http://x", "http://x/v1", "");
    expect(result?.models[0].samplingParams).toEqual({ temperature: 0.6 });
  });

  // A recommendation for one family is not a house style for every model.
  it("leaves an unfamiliar family on the server's own default", async () => {
    mockSglang({
      modelInfo: { model_path: "/models/m", is_generation: true, model_type: "llama" },
    });
    const result = await detectSglang("http://x", "http://x/v1", "");
    expect(result?.models[0].samplingParams).toBeUndefined();
  });

  it("leaves a non-reasoning Qwen alone", async () => {
    mockSglang({
      modelInfo: { model_path: "/models/m", is_generation: true, model_type: "qwen3_5" },
      serverInfo: { reasoning_parser: null },
    });
    const result = await detectSglang("http://x", "http://x/v1", "");
    expect(result?.models[0].samplingParams).toBeUndefined();
  });

  it("attaches the measured request compat to every model", async () => {
    mockSglangProbe(QWEN38_STATUSES, {
      "http://x/get_model_info": {
        model_path: "/models/m",
        is_generation: true,
        has_image_understanding: true,
      },
      "http://x/get_server_info": { reasoning_parser: "qwen3" },
      "http://x/v1/models": { data: [{ id: "/models/m", max_model_len: 262144 }] },
    });
    const result = await detectSglang("http://x", "http://x/v1", "");
    expect(result?.models[0].compat).toEqual({
      supportsReasoningEffort: true,
      supportsDeveloperRole: false,
    });
    expect(result?.models[0].thinkingLevelMap?.high).toBe("medium");
  });

  // reasoning_parser is the whole signal: without one, SGLang never splits
  // reasoning out of the response, whatever the weights can do — and there
  // is then nothing worth spending a probe sweep on.
  it("reports no reasoning, and probes nothing, without a reasoning_parser", async () => {
    mockSglang({ serverInfo: { reasoning_parser: null } });
    const result = await detectSglang("http://x", "http://x/v1", "");
    expect(result?.models[0].reasoning).toBe(false);
    expect(result?.models[0].compat).toBeUndefined();
    const calls = (fetch as unknown as { mock: { calls: unknown[][] } }).mock.calls;
    expect(calls.some((c) => String(c[0]).endsWith("/chat/completions"))).toBe(false);
  });

  it("reports text-only when the model has no image understanding", async () => {
    mockSglang({
      modelInfo: { model_path: "/models/m", is_generation: true, has_image_understanding: false },
    });
    const result = await detectSglang("http://x", "http://x/v1", "");
    expect(result?.models[0].input).toEqual(["text"]);
  });

  // An embedding-only server has no chat endpoint worth registering.
  it("declines an embedding-only server so the chain keeps looking", async () => {
    mockSglang({ modelInfo: { model_path: "/models/e", is_generation: false } });
    expect(await detectSglang("http://x", "http://x/v1", "")).toBeNull();
  });

  // SGLang renamed /get_model_info and /get_server_info; the old pair still
  // answers but logs a deprecation warning naming the replacement. Both
  // spellings have to detect identically, because a server can be on either
  // side of the rename and the release that drops the old names would
  // otherwise make SGLang undetectable rather than merely degraded. The
  // surrounding tests all mock the legacy names, so those cover the fallback.
  it("detects a server that serves only the renamed endpoints", async () => {
    mockFetch({
      "http://x/model_info": {
        model_path: "/models/m",
        is_generation: true,
        has_image_understanding: true,
      },
      "http://x/server_info": { reasoning_parser: null },
      "http://x/v1/models": { data: [{ id: "/models/m", max_model_len: 262144 }] },
    });
    const result = await detectSglang("http://x", "http://x/v1", "");
    expect(result?.apiType).toBe("sglang");
    expect(result?.models[0].contextWindow).toBe(262144);
    expect(result?.models[0].input).toEqual(["text", "image"]);
  });

  it("prefers the renamed endpoints when a server answers both", async () => {
    mockFetch({
      "http://x/model_info": { model_path: "/models/new", is_generation: true },
      "http://x/get_model_info": { model_path: "/models/legacy", is_generation: true },
      "http://x/server_info": { reasoning_parser: null },
      "http://x/get_server_info": { reasoning_parser: "qwen3" },
      "http://x/v1/models": { data: [] },
    });
    const result = await detectSglang("http://x", "http://x/v1", "");
    // Both fields come from the pair that was actually read: the id falls
    // back to model_path when /v1/models is empty, and reasoning follows
    // reasoning_parser. Legacy values for either would mean the old
    // endpoint won.
    expect(result?.models[0].id).toBe("/models/new");
    expect(result?.models[0].reasoning).toBe(false);
  });

  it("returns null when /get_model_info is absent", async () => {
    mockFetch({ "http://x/v1/models": { data: [{ id: "m", max_model_len: 4096 }] } });
    expect(await detectSglang("http://x", "http://x/v1", "")).toBeNull();
  });

  // The regression this detector exists for: SGLang's shim answers /api/tags
  // and /api/show, so a chain that probed Ollama first would claim it — and
  // silently lose vision, reasoning, and the resident-model state.
  it("wins over Ollama in the chain despite the compatibility shim", async () => {
    mockSglang();
    const result = await detectModels("http://x/v1", "");
    expect(result.apiType).toBe("sglang");
    expect(result.models[0].input).toEqual(["text", "image"]);
  });

  // The shim must still not cost real Ollama servers their own detection.
  it("leaves a real Ollama server to the Ollama probe", async () => {
    mockOllama(
      { models: [{ name: "llama3:8b", model: "llama3:8b" }] },
      {
        "llama3:8b": {
          model_info: { "general.architecture": "llama", "llama.context_length": 8192 },
          capabilities: ["completion", "thinking"],
        },
      },
    );
    const result = await detectModels("http://x/v1", "");
    expect(result.apiType).toBe("ollama");
  });
});

describe("detectOllama", () => {
  it("returns null when there are no local models", async () => {
    mockOllama({ models: [] }, {});
    expect(await detectOllama("http://x", "")).toBeNull();
  });

  it("reads context_length via the architecture-prefixed key and thinking/vision capabilities", async () => {
    mockOllama(
      { models: [{ name: "deepseek-r1:latest", model: "deepseek-r1:latest" }] },
      {
        "deepseek-r1:latest": {
          model_info: { "general.architecture": "qwen2", "qwen2.context_length": 32768 },
          capabilities: ["completion", "thinking"],
        },
      },
    );
    const result = await detectOllama("http://x", "");
    expect(result).toEqual({
      apiType: "ollama",
      models: [
        {
          id: "deepseek-r1:latest",
          name: "deepseek-r1:latest",
          contextWindow: 32768,
          maxTokens: 16384,
          reasoning: true,
          input: ["text"],
          loaded: false,
        },
      ],
    });
  });

  it("marks vision-capable models via the vision capability", async () => {
    mockOllama(
      { models: [{ name: "llava:latest", model: "llava:latest" }] },
      { "llava:latest": { model_info: {}, capabilities: ["completion", "vision"] } },
    );
    const result = await detectOllama("http://x", "");
    expect(result?.models[0].input).toEqual(["text", "image"]);
  });

  it("queries /api/show independently per model", async () => {
    mockOllama(
      {
        models: [
          { name: "a:latest", model: "a:latest" },
          { name: "b:latest", model: "b:latest" },
        ],
      },
      {
        "a:latest": { model_info: { "general.architecture": "llama", "llama.context_length": 8192 }, capabilities: [] },
        "b:latest": { model_info: { "general.architecture": "llama", "llama.context_length": 4096 }, capabilities: ["thinking"] },
      },
    );
    const result = await detectOllama("http://x", "");
    expect(result?.models.map((m) => [m.id, m.contextWindow, m.reasoning])).toEqual([
      ["a:latest", 8192, false],
      ["b:latest", 4096, true],
    ]);
  });

  it("defaults to 32768 when /api/show fails or lacks context_length", async () => {
    mockOllama({ models: [{ name: "a:latest", model: "a:latest" }] }, {});
    const result = await detectOllama("http://x", "");
    expect(result?.models[0].contextWindow).toBe(32768);
  });

  it("reads size and quantization directly from /api/tags, and loaded state from /api/ps", async () => {
    mockOllama(
      {
        models: [
          { name: "a:latest", model: "a:latest", size: 4683075271, details: { quantization_level: "Q4_K_M" } },
          { name: "b:latest", model: "b:latest", size: 2019393189, details: { quantization_level: "Q8_0" } },
        ],
      },
      {},
      { models: [{ model: "a:latest" }] },
    );
    const result = await detectOllama("http://x", "");
    expect(result?.models).toEqual([
      expect.objectContaining({
        id: "a:latest",
        sizeBytes: 4683075271,
        quantization: "Q4_K_M",
        loaded: true,
      }),
      expect.objectContaining({
        id: "b:latest",
        sizeBytes: 2019393189,
        quantization: "Q8_0",
        loaded: false,
      }),
    ]);
  });
});

describe("detectVllm", () => {
  it("returns null when /version doesn't respond", async () => {
    mockFetch({ "http://x/v1/models": { data: [{ id: "m1", max_model_len: 4096 }] } });
    expect(await detectVllm("http://x", "http://x/v1", "")).toBeNull();
  });

  it("returns null when /version responds but no model has max_model_len", async () => {
    mockFetch({
      "http://x/version": { version: "0.6.3" },
      "http://x/v1/models": { data: [{ id: "m1" }] },
    });
    expect(await detectVllm("http://x", "http://x/v1", "")).toBeNull();
  });

  it("requires both /version and a max_model_len-bearing /v1/models entry", async () => {
    mockFetch({
      "http://x/version": { version: "0.6.3" },
      "http://x/v1/models": { data: [{ id: "org/Model-7B", max_model_len: 32768 }] },
    });
    const result = await detectVllm("http://x", "http://x/v1", "");
    expect(result).toEqual({
      apiType: "vllm",
      models: [
        {
          id: "org/Model-7B",
          name: "Model-7B",
          contextWindow: 32768,
          maxTokens: 8192,
          reasoning: false,
          input: ["text"],
        },
      ],
    });
  });

});

describe("detectOpenAI", () => {
  it("reads max_model_len, then context_window, then defaults to 32768", async () => {
    mockFetch({
      "http://x/v1/models": {
        data: [{ id: "m1", max_model_len: 16384 }, { id: "m2", context_window: 8192 }, { id: "m3" }],
      },
    });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result.models.map((m) => m.contextWindow)).toEqual([16384, 8192, 32768]);
  });

  it("reads OpenRouter-style context_length and top_provider.context_length", async () => {
    mockFetch({
      "http://x/v1/models": {
        data: [
          { id: "m1", context_length: 131072 },
          { id: "m2", context_length: 8192, top_provider: { context_length: 131072 } },
        ],
      },
    });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result.models.map((m) => m.contextWindow)).toEqual([131072, 131072]);
  });

  it("reads name, reasoning and vision from an OpenRouter-style card", async () => {
    mockFetch({
      "http://x/v1/models": {
        data: [
          {
            id: "deepseek-v4-flash",
            name: "DeepSeek V4 Flash",
            context_length: 131072,
            top_provider: { context_length: 131072, max_completion_tokens: 131072 },
            architecture: { input_modalities: ["text", "image"] },
            supported_parameters: ["tools", "temperature", "reasoning_effort"],
          },
        ],
      },
    });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result.models[0]).toMatchObject({
      id: "deepseek-v4-flash",
      name: "DeepSeek V4 Flash",
      contextWindow: 131072,
      reasoning: true,
      input: ["text", "image"],
    });
  });

  it("ignores a max_completion_tokens that just repeats the context window", async () => {
    mockFetch({
      "http://x/v1/models": {
        data: [
          { id: "m1", context_length: 131072, top_provider: { max_completion_tokens: 131072 } },
          { id: "m2", context_length: 131072, top_provider: { max_completion_tokens: 4096 } },
        ],
      },
    });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result.models.map((m) => m.maxTokens)).toEqual([8192, 4096]);
  });

  it("returns an empty model list when the response has no data", async () => {
    mockFetch({ "http://x/v1/models": {} });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result).toEqual({ apiType: "openai", models: [] });
  });

  // maxTokens is never sent as a request cap, so a bigger number truncates
  // nothing; it exists so Pi can tell a length-stopped reasoning response
  // apart from one that simply ran out of its own budget.
  it("gives a reasoning model more headroom than a plain one", async () => {
    mockFetch({
      "http://x/v1/models": {
        data: [
          { id: "plain", context_length: 262144 },
          { id: "thinker", context_length: 262144, supported_parameters: ["reasoning_effort"] },
        ],
      },
    });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result.models.map((m) => m.maxTokens)).toEqual([8192, 65536]);
  });

  // Half the context still wins when the window is the tighter constraint.
  it("keeps the half-context floor below either ceiling", async () => {
    mockFetch({
      "http://x/v1/models": {
        data: [{ id: "small", context_length: 8192, supported_parameters: ["reasoning_effort"] }],
      },
    });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result.models[0].maxTokens).toBe(4096);
  });

  it("reads portal-style card with snake_case and camelCase flags, tokens, and reasoning effort levels", async () => {
    mockFetch({
      "http://x/v1/models": {
        data: [
          {
            id: "portal/snake",
            max_input_tokens: 1048576,
            max_output_tokens: 32768,
            supports_reasoning: true,
            supports_vision: false,
            reasoning_effort_levels: ["none", "low", "medium", "high"],
          },
          {
            id: "portal/camel",
            maxInputTokens: 262144,
            maxOutputTokens: 16384,
            supportsReasoning: true,
            supportsVision: true,
            effortLevels: ["minimal", "low", "medium", "high", "max"],
          },
        ],
      },
    });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result.models[0]).toMatchObject({
      id: "portal/snake",
      contextWindow: 1048576,
      maxTokens: 32768,
      reasoning: true,
      input: ["text"],
      compat: { supportsReasoningEffort: true },
      thinkingLevelMap: {
        off: "none",
        minimal: "low",
        low: "low",
        medium: "medium",
        high: "high",
        xhigh: "high",
        max: "high",
      },
    });
    expect(result.models[1]).toMatchObject({
      id: "portal/camel",
      contextWindow: 262144,
      maxTokens: 16384,
      reasoning: true,
      input: ["text", "image"],
      compat: { supportsReasoningEffort: true },
      thinkingLevelMap: {
        minimal: "minimal",
        low: "low",
        medium: "medium",
        high: "high",
        xhigh: "high",
        max: "max",
      },
    });
  });
});

// A card as ds4_server.c actually emits it — same shape for every alias,
// since they all report the one loaded GGUF.
function ds4Card(id: string, ctx = 131072, maxCompletion = 131072) {
  return {
    id,
    object: "model",
    created: 1767225600,
    owned_by: "ds4.c",
    name: "DeepSeek V4 Flash",
    context_length: ctx,
    top_provider: { context_length: ctx, max_completion_tokens: maxCompletion, is_moderated: false },
    supported_parameters: ["tools", "tool_choice", "max_tokens", "temperature", "reasoning_effort"],
  };
}

describe("ds4", () => {
  it("identifies ds4 by owned_by and registers every alias", async () => {
    mockFetch({
      "http://x/v1/models": { data: [ds4Card("deepseek-v4-flash"), ds4Card("deepseek-v4-pro")] },
    });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result.apiType).toBe("ds4");
    expect(result.models.map((m) => m.id)).toEqual(["deepseek-v4-flash", "deepseek-v4-pro"]);
    expect(result.models[0]).toMatchObject({
      name: "DeepSeek V4 Flash",
      contextWindow: 131072,
      maxTokens: 65536,
      reasoning: true,
      input: ["text"],
      compat: { supportsReasoningEffort: true, supportsDeveloperRole: true },
    });
  });

  // "off" must send an explicit "none": pi-ai drops an unmapped "off" level
  // instead of sending anything, and ds4 defaults an absent reasoning_effort
  // to thinking-on-at-HIGH. "max" must be named or pi-ai never offers it.
  it("maps the two thinking levels ds4 would otherwise get wrong", async () => {
    mockFetch({ "http://x/v1/models": { data: [ds4Card("deepseek-v4-flash")] } });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result.models[0].thinkingLevelMap).toEqual({ off: "none", max: "max" });
  });

  // ds4 folds xhigh into HIGH, making it identical to high — and pi-ai only
  // offers xhigh when it is mapped, so leaving it out is what hides it.
  it("leaves xhigh unmapped so Pi does not offer a duplicate of high", async () => {
    mockFetch({ "http://x/v1/models": { data: [ds4Card("deepseek-v4-flash")] } });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result.models[0].thinkingLevelMap).not.toHaveProperty("xhigh");
  });

  // ds4_server.c picks the id set from the loaded engine, so matching on ids
  // instead of owned_by would miss any engine added later.
  it("identifies a GLM-DSA engine's alias set too", async () => {
    mockFetch({
      "http://x/v1/models": {
        data: [ds4Card("glm-5.2"), ds4Card("glm-5.2-chat"), ds4Card("glm-5.2-reasoner")],
      },
    });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result.apiType).toBe("ds4");
    expect(result.models).toHaveLength(3);
  });

  it("honours --default-tokens when it restricts output below the context", async () => {
    mockFetch({ "http://x/v1/models": { data: [ds4Card("deepseek-v4-flash", 131072, 4096)] } });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result.models[0].maxTokens).toBe(4096);
  });

  it("stays generic when only some cards carry the ds4 owner", async () => {
    mockFetch({
      "http://x/v1/models": { data: [ds4Card("deepseek-v4-flash"), { id: "other", owned_by: "acme" }] },
    });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result.apiType).toBe("openai");
    expect(result.models.every((m) => m.compat === undefined)).toBe(true);
  });

  it("reaches ds4 through the full detection chain", async () => {
    mockFetch({ "http://x/v1/models": { data: [ds4Card("deepseek-v4-flash")] } });
    const result = await detectModels("http://x/v1", "");
    expect(result.apiType).toBe("ds4");
  });
});

describe("detectModels chain", () => {
  it("prefers MTPLX when its /health responds with valid data", async () => {
    mockFetch({ "http://x/health": { model: "m", context_window: 4096 } });
    const result = await detectModels("http://x/v1", "");
    expect(result.apiType).toBe("mtplx");
  });

  it("falls through to oMLX when MTPLX is absent", async () => {
    mockFetch({
      "http://x/v1/models/status": {
        models: [{ id: "a", model_type: "llm", max_context_window: 4096 }],
      },
    });
    const result = await detectModels("http://x/v1", "");
    expect(result.apiType).toBe("omlx");
  });

  it("falls through to LM Studio when MTPLX and oMLX are absent", async () => {
    mockFetch({
      "http://x/api/v1/models": {
        models: [{ key: "k1", type: "llm", max_context_length: 4096 }],
      },
    });
    const result = await detectModels("http://x/v1", "");
    expect(result.apiType).toBe("lmstudio");
  });

  it("falls through to llama.cpp when MTPLX/oMLX/LM Studio are absent", async () => {
    mockFetch({
      "http://x/props": {
        default_generation_settings: { n_ctx: 4096 },
        model_path: "/models/m.gguf",
      },
    });
    const result = await detectModels("http://x/v1", "");
    expect(result.apiType).toBe("llamacpp");
  });

  it("falls through to Ollama when the above are all absent", async () => {
    mockFetch({
      "http://x/api/tags": { models: [{ name: "a:latest", model: "a:latest" }] },
    });
    const result = await detectModels("http://x/v1", "");
    expect(result.apiType).toBe("ollama");
  });

  it("falls through to vLLM when only /version + max_model_len are present", async () => {
    mockFetch({
      "http://x/version": { version: "0.6.3" },
      "http://x/v1/models": { data: [{ id: "m1", max_model_len: 4096 }] },
    });
    const result = await detectModels("http://x/v1", "");
    expect(result.apiType).toBe("vllm");
  });

  it("falls all the way through to the generic OpenAI probe", async () => {
    mockFetch({ "http://x/v1/models": { data: [{ id: "m1" }] } });
    const result = await detectModels("http://x/v1", "");
    expect(result.apiType).toBe("openai");
  });

  it("shares one AbortSignal across every probe instead of a fresh one each", async () => {
    // Regression test for the timeout-stacking bug: before this, each probe
    // defaulted to its own 5s AbortSignal.timeout(), so an unreachable
    // server paid that timeout up to 7x sequentially. Every fetch() call in
    // one detectModels() run should now receive the exact same signal
    // instance, so the whole chain shares one deadline.
    const seenSignals: unknown[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init?: RequestInit) => {
        seenSignals.push(init?.signal);
        return { ok: false, json: async () => ({}) } as unknown as Response;
      }),
    );

    await detectModels("http://x/v1", "");

    // mtplx, omlx, lmstudio, llamacpp, ollama, vllm, openai all failing
    // means every probe ran — at least 7 fetch calls.
    expect(seenSignals.length).toBeGreaterThanOrEqual(7);
    expect(new Set(seenSignals).size).toBe(1);
    expect(seenSignals[0]).toBeInstanceOf(AbortSignal);
  });

  it("combines an externally-provided signal with the chain deadline rather than replacing it", async () => {
    const seenSignals: unknown[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init?: RequestInit) => {
        seenSignals.push(init?.signal);
        return { ok: false, json: async () => ({}) } as unknown as Response;
      }),
    );
    const external = new AbortController().signal;

    await detectModels("http://x/v1", "", external);

    expect(new Set(seenSignals).size).toBe(1);
    // The combined signal must not just be the external one passed straight
    // through — it needs its own chain-timeout component too.
    expect(seenSignals[0]).not.toBe(external);
  });
});

describe("detectModels error summarization", () => {
  it("reports an auth failure when every probe returns 401", async () => {
    mockFetchAllStatus(401);
    const result = await detectModels("http://x/v1", "wrong-key");
    expect(result.models).toEqual([]);
    expect(result.error).toBe("Authentication failed (HTTP 401) — check the API key.");
  });

  it("reports an auth failure when every probe returns 403", async () => {
    mockFetchAllStatus(403);
    const result = await detectModels("http://x/v1", "");
    expect(result.error).toBe("Authentication failed (HTTP 403) — check the API key.");
  });

  it("reports a timeout when every probe aborts", async () => {
    const abortError = new DOMException("The operation was aborted.", "AbortError");
    mockFetchAllReject(abortError);
    const result = await detectModels("http://x/v1", "");
    expect(result.error).toBe(
      "Timed out waiting for a response — check the server is running and reachable.",
    );
  });

  it("reports a connection failure when every probe throws a network error", async () => {
    mockFetchAllReject(new TypeError("fetch failed"));
    const result = await detectModels("http://x/v1", "");
    expect(result.error).toBe("Could not connect to the server — check the URL and that it's running.");
  });

  it("leaves error undefined when the server genuinely has zero models (no auth/timeout signal)", async () => {
    // The 6 non-matching backend probes 404 (expected, harmless) and the
    // final generic OpenAI probe succeeds with an empty model list — this
    // is a real "nothing loaded" response, not a failure, so no error
    // should be synthesized from the incidental 404 noise.
    mockFetch({ "http://x/v1/models": { data: [] } });
    const result = await detectModels("http://x/v1", "");
    expect(result.models).toEqual([]);
    expect(result.error).toBeUndefined();
  });
});

// ─── ninfer ─────────────────────────────────────────────────────────
// Written from the source rather than from a running server, so each of
// these pins an assumption that live testing has to confirm. The exact
// strings come from Neroued/ninfer at master: src/serve/openai_schema.cpp
// for the card, src/runtime/engine/engine.cpp for the rejection wording.

function ninferCard(id = "qwen3.8-27b") {
  return { id, object: "model", created: 1786000000, owned_by: "ninfer" };
}

/**
 * Routes the probes ninfer's detector fires. `context` and `vision` are the
 * replies those two endpoints give; everything else answers 400.
 */
function mockNinfer(
  cards: unknown[],
  replies: { context?: { status: number; body: string }; vision?: { status: number; body: string } } = {},
  tiers: Record<string, number> = { none: 200, low: 200, medium: 200, xhigh: 200 },
) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const u = String(url);
      const reply = (status: number, body: string) =>
        ({ ok: status === 200, status, text: async () => body, json: async () => ({}) }) as unknown as Response;

      if (u.endsWith("/v1/models")) {
        return { ok: true, status: 200, json: async () => ({ object: "list", data: cards }) } as unknown as Response;
      }
      if (u.endsWith("/responses/input_tokens")) {
        const v = replies.vision ?? { status: 400, body: '{"error":{"code":"vision_disabled"}}' };
        return reply(v.status, v.body);
      }
      if (u.endsWith("/chat/completions")) {
        const body = init?.body ? JSON.parse(String(init.body)) : {};
        // The compat probe names a tier; the context probe sends a long prompt.
        if (typeof body.reasoning_effort === "string") {
          return reply(tiers[body.reasoning_effort] ?? 400, "");
        }
        if (body.messages?.some((m: { role: string }) => m.role === "developer")) {
          return reply(400, '{"error":{"message":"Unexpected message role"}}');
        }
        const c = replies.context ?? {
          status: 400,
          body: '{"error":{"message":"prepared prompt has 300004 tokens, exceeding Engine max_context 16384"}}',
        };
        return reply(c.status, c.body);
      }
      return reply(404, "");
    }),
  );
}

describe("parseNinferContext", () => {
  // The exact body a live ninfer 0.x returned, not a reconstruction.
  it("reads the ceiling out of a real rejection", () => {
    const body =
      '{"error":{"code":"context_length_exceeded","message":"prepared prompt has 25052 tokens, ' +
      'exceeding Engine max_context 16384","param":"messages","type":"invalid_request_error"}}';
    expect(parseNinferContext(body)).toBe(16384);
  });

  // engine.cpp: "prepared prompt has N tokens, exceeding Engine max_context M"
  it("reads the ceiling out of the rejection", () => {
    expect(
      parseNinferContext("prepared prompt has 300004 tokens, exceeding Engine max_context 16384"),
    ).toBe(16384);
  });

  it("survives the message arriving wrapped in JSON", () => {
    expect(parseNinferContext('{"error":{"message":"... Engine max_context 8192"}}')).toBe(8192);
  });

  // Parsing prose is brittle by nature, so a miss has to fall through to the
  // caller's fallback rather than invent a number.
  it("gives up rather than guess", () => {
    expect(parseNinferContext("context length exceeded")).toBeUndefined();
    expect(parseNinferContext("")).toBeUndefined();
    expect(parseNinferContext("max_context none")).toBeUndefined();
  });
});

describe("isNinferCards", () => {
  it("identifies the owner the schema hard-codes", () => {
    expect(isNinferCards([ninferCard()])).toBe(true);
  });

  it("stays out of the way of a card that is not ninfer's", () => {
    expect(isNinferCards([{ id: "m", owned_by: "sglang" }])).toBe(false);
    expect(isNinferCards([{ id: "m" }])).toBe(false);
    expect(isNinferCards([])).toBe(false);
  });

  it("declines a mixed list, as a gateway aggregating backends would send", () => {
    expect(isNinferCards([ninferCard(), { id: "other", owned_by: "vllm" }])).toBe(false);
  });
});

describe("probeNinfer", () => {
  it("learns the context ceiling from the rejection", async () => {
    mockNinfer([ninferCard()]);
    expect((await probeNinfer("http://x/v1", "", "m")).contextWindow).toBe(16384);
  });

  // Registering four times the real ceiling would let Pi fill a context the
  // server then refuses, so an unreadable answer falls back to ninfer's own
  // default rather than this file's generic one.
  it("falls back to ninfer's default when the rejection says nothing useful", async () => {
    mockNinfer([ninferCard()], { context: { status: 400, body: "context_length_exceeded" } });
    expect((await probeNinfer("http://x/v1", "", "m")).contextWindow).toBe(8192);
  });

  // Only reached on a context larger than the overshoot. The prefill has
  // already run by then, so throwing away the number it bought and falling
  // back to 8192 would be the worst of both.
  it("takes an accepted prompt as a floor when the overshoot was not enough", async () => {
    mockNinfer([ninferCard()], {
      context: { status: 200, body: '{"usage":{"prompt_tokens":300004,"completion_tokens":1}}' },
    });
    expect((await probeNinfer("http://x/v1", "", "m")).contextWindow).toBe(300004);
  });

  it("falls back when an accepted probe reports no usable prompt size", async () => {
    mockNinfer([ninferCard()], { context: { status: 200, body: "not json" } });
    expect((await probeNinfer("http://x/v1", "", "m")).contextWindow).toBe(8192);
  });

  it("falls back when the probe never gets an answer", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("connection refused"); }));
    expect((await probeNinfer("http://x/v1", "", "m")).contextWindow).toBe(8192);
  });

  it("reads vision off the token-count rejection", async () => {
    mockNinfer([ninferCard()]);
    expect((await probeNinfer("http://x/v1", "", "m")).vision).toBe(false);
  });

  it("reads vision on when the token count is accepted", async () => {
    mockNinfer([ninferCard()], { vision: { status: 200, body: '{"input_tokens":12}' } });
    expect((await probeNinfer("http://x/v1", "", "m")).vision).toBe(true);
  });

  // A rejection for some other reason says nothing about vision, and
  // declaring a vision model text-only is the worse of the two mistakes.
  it("leaves vision unknown when the rejection is about something else", async () => {
    mockNinfer([ninferCard()], { vision: { status: 400, body: '{"error":{"code":"invalid_request"}}' } });
    expect((await probeNinfer("http://x/v1", "", "m")).vision).toBeUndefined();
  });
});

describe("ninfer through the chain", () => {
  it("is detected, measured, and mapped", async () => {
    mockNinfer([ninferCard()]);
    const result = await detectModels("http://x/v1", "");

    expect(result.apiType).toBe("ninfer");
    expect(result.models[0]).toMatchObject({
      id: "qwen3.8-27b",
      contextWindow: 16384,
      reasoning: true,
      input: ["text"],
    });
  });

  // docs/serving.md: an effort-capable template exposes low, medium and
  // xhigh; minimal, high and max are parsed and then rejected. That is the
  // same set Qwen3.8 exposes under SGLang, which is why the probe is shared
  // rather than either backend's tiers being written down.
  it("maps Pi's levels onto the tiers the artifact accepts", async () => {
    mockNinfer([ninferCard()]);
    const result = await detectModels("http://x/v1", "");

    expect(result.models[0].thinkingLevelMap).toEqual({
      off: "none",
      minimal: "low",
      low: "low",
      medium: "medium",
      high: "medium",
      xhigh: "xhigh",
      max: "xhigh",
    });
  });

  it("reports a text-only artifact as text-only", async () => {
    mockNinfer([ninferCard()]);
    const result = await detectModels("http://x/v1", "");
    expect(result.models[0].input).toEqual(["text"]);
  });

  it("reports vision when the server was started with it", async () => {
    mockNinfer([ninferCard()], { vision: { status: 200, body: '{"input_tokens":12}' } });
    const result = await detectModels("http://x/v1", "");
    expect(result.models[0].input).toEqual(["text", "image"]);
  });

  // A template with no effort tiers is not a reasoning model, and nothing
  // about reasoning should be claimed for it.
  it("claims no reasoning when the template exposes no tiers", async () => {
    mockNinfer([ninferCard()], {}, {});
    const result = await detectModels("http://x/v1", "");
    expect(result.models[0].reasoning).toBe(false);
    expect(result.models[0].thinkingLevelMap).toBeUndefined();
  });
});
