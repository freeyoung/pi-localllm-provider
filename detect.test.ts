import { afterEach, describe, expect, it, vi } from "vitest";
import {
  detectLlamaCpp,
  detectLmStudio,
  detectModels,
  detectMtplx,
  detectOllama,
  detectOmlx,
  detectOpenAI,
  detectVllm,
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
          maxTokens: 8192,
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

  it("returns an empty model list when the response has no data", async () => {
    mockFetch({ "http://x/v1/models": {} });
    const result = await detectOpenAI("http://x/v1", "");
    expect(result).toEqual({ apiType: "openai", models: [] });
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
