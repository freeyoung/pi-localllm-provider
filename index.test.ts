import { describe, expect, it } from "vitest";
import {
  applyContextOverrides,
  formatModelLine,
  modelIdsChanged,
  modelsHeading,
  normalizeBaseUrl,
} from "./index.ts";

describe("normalizeBaseUrl", () => {
  it("appends /v1 when missing", () => {
    expect(normalizeBaseUrl("http://localhost:8000")).toBe("http://localhost:8000/v1");
  });

  it("leaves an existing /v1 suffix alone", () => {
    expect(normalizeBaseUrl("http://localhost:8000/v1")).toBe("http://localhost:8000/v1");
  });

  it("strips trailing slashes before checking the suffix", () => {
    expect(normalizeBaseUrl("http://localhost:8000/v1/")).toBe("http://localhost:8000/v1");
    expect(normalizeBaseUrl("http://localhost:8000/")).toBe("http://localhost:8000/v1");
  });

  it("defaults to http:// for a bare host:port with no scheme", () => {
    expect(normalizeBaseUrl("localhost:11434")).toBe("http://localhost:11434/v1");
    expect(normalizeBaseUrl("192.168.1.50:8000")).toBe("http://192.168.1.50:8000/v1");
  });

  it("preserves an explicit https:// scheme", () => {
    expect(normalizeBaseUrl("https://my.server.com")).toBe("https://my.server.com/v1");
  });

  it("trims surrounding whitespace", () => {
    expect(normalizeBaseUrl("  localhost:11434  ")).toBe("http://localhost:11434/v1");
  });
});

describe("formatModelLine", () => {
  it("formats context window and max tokens in k, with no capability tags", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "some-model",
        contextWindow: 65536,
        maxTokens: 8192,
        reasoning: false,
        input: ["text"],
      }),
    ).toBe("  • some-model  (ctx 64k, max 8k)");
  });

  it("appends reasoning and vision tags when present", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "vision-model",
        contextWindow: 65536,
        maxTokens: 8192,
        reasoning: true,
        input: ["text", "image"],
      }),
    ).toBe("  • vision-model  (ctx 64k, max 8k, reasoning, vision)");
  });

  it("shows sub-1024 windows without a k suffix", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "tiny",
        contextWindow: 512,
        maxTokens: 256,
        reasoning: false,
        input: ["text"],
      }),
    ).toBe("  • tiny  (ctx 512, max 256)");
  });

  it("prefixes a checkmark when loaded is true", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "m",
        contextWindow: 4096,
        maxTokens: 2048,
        reasoning: false,
        input: ["text"],
        loaded: true,
      }),
    ).toBe("  • ✓ m  (ctx 4k, max 2k)");
  });

  it("prefixes a hollow circle when loaded is false", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "m",
        contextWindow: 4096,
        maxTokens: 2048,
        reasoning: false,
        input: ["text"],
        loaded: false,
      }),
    ).toBe("  • ○ m  (ctx 4k, max 2k)");
  });

  it("omits the loaded prefix entirely when loaded is unknown", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "m",
        contextWindow: 4096,
        maxTokens: 2048,
        reasoning: false,
        input: ["text"],
      }),
    ).toBe("  • m  (ctx 4k, max 2k)");
  });

  it("shows size and quantization when present, in order before capability tags", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "m",
        contextWindow: 4096,
        maxTokens: 2048,
        reasoning: true,
        input: ["text", "image"],
        sizeBytes: 4912898304,
        quantization: "Q4_K_M",
      }),
    ).toBe("  • m  (ctx 4k, max 2k, 4.6G, Q4_K_M, reasoning, vision)");
  });

  it("uses an overridden context window and marks it with *, deriving max from it", () => {
    const m = {
      id: "m1",
      name: "m",
      contextWindow: 32768, // what the server reported (e.g. llama-swap's fallback)
      maxTokens: 8192,
      contextWindowOverride: 131072, // set by hand
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
    };
    expect(formatModelLine(m)).toBe("  • m  (ctx 128k*, max 8k)");
  });

  it("derives a larger max for a reasoning model with an override", () => {
    const m = {
      id: "m1",
      name: "m",
      contextWindow: 32768,
      maxTokens: 8192,
      contextWindowOverride: 131072,
      reasoning: true,
      input: ["text"] as ("text" | "image")[],
    };
    // capTokens(131072, true) = min(65536, 65536) = 65536
    expect(formatModelLine(m)).toBe("  • m  (ctx 128k*, max 64k, reasoning)");
  });
});

describe("modelIdsChanged", () => {
  const baseModel = {
    id: "m1",
    name: "m1",
    contextWindow: 4096,
    maxTokens: 2048,
    reasoning: false,
    input: ["text"] as ("text" | "image")[],
  };

  it("is false when the same single model is refreshed with new metadata", () => {
    expect(modelIdsChanged([baseModel], [{ ...baseModel, contextWindow: 8192 }])).toBe(false);
  });

  it("is false when the same set of models comes back in a different order", () => {
    const a = { ...baseModel, id: "a" };
    const b = { ...baseModel, id: "b" };
    expect(modelIdsChanged([a, b], [b, a])).toBe(false);
  });

  it("is true when the model count changes", () => {
    const a = { ...baseModel, id: "a" };
    const b = { ...baseModel, id: "b" };
    expect(modelIdsChanged([a], [a, b])).toBe(true);
  });

  it("is true when a same-count refresh swaps in a different model id", () => {
    const a = { ...baseModel, id: "a" };
    const c = { ...baseModel, id: "c" };
    expect(modelIdsChanged([a], [c])).toBe(true);
  });

  it("is false for two empty lists", () => {
    expect(modelIdsChanged([], [])).toBe(false);
  });
});

describe("applyContextOverrides", () => {
  it("re-attaches a stored override to the freshly detected model by id", () => {
    const stored = {
      id: "m1",
      name: "old-name",
      contextWindow: 32768,
      contextWindowOverride: 131072,
      maxTokens: 8192,
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
    };
    const detected = {
      id: "m1",
      name: "fresh-name",
      contextWindow: 65536,
      maxTokens: 16384,
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
    };
    const [out] = applyContextOverrides([stored], [detected]);
    // Detected context/max are kept; only the override is re-attached.
    expect(out.contextWindow).toBe(65536);
    expect(out.maxTokens).toBe(16384);
    expect(out.contextWindowOverride).toBe(131072);
    expect(out.name).toBe("fresh-name");
  });

  it("passes a detected model through untouched when no override exists", () => {
    const stored = {
      id: "m1",
      name: "m",
      contextWindow: 32768,
      maxTokens: 8192,
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
    };
    const detected = {
      id: "m1",
      name: "m",
      contextWindow: 65536,
      maxTokens: 16384,
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
    };
    const [out] = applyContextOverrides([stored], [detected]);
    expect(out.contextWindowOverride).toBeUndefined();
    expect(out).toEqual(detected);
  });

  it("only re-attaches overrides to models that carry one, matching by id", () => {
    const storedA = {
      id: "a",
      name: "a",
      contextWindow: 32768,
      contextWindowOverride: 200000,
      maxTokens: 8192,
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
    };
    const storedB = {
      id: "b",
      name: "b",
      contextWindow: 32768,
      maxTokens: 8192,
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
    };
    const detectedA = {
      id: "a",
      name: "a",
      contextWindow: 32768,
      maxTokens: 8192,
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
    };
    const detectedC = {
      id: "c",
      name: "c",
      contextWindow: 32768,
      maxTokens: 8192,
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
    };
    const out = applyContextOverrides([storedA, storedB], [detectedA, detectedC]);
    expect(out[0].contextWindowOverride).toBe(200000);
    // "c" is new and "b" isn't served anymore — neither gets an override.
    expect(out[1].contextWindowOverride).toBeUndefined();
  });
});

describe("modelsHeading", () => {
  const baseModel = {
    id: "m1",
    name: "m1",
    contextWindow: 4096,
    maxTokens: 2048,
    reasoning: false,
    input: ["text"] as ("text" | "image")[],
  };

  it("adds the loaded-state legend when at least one model reports it", () => {
    expect(modelsHeading([{ ...baseModel, loaded: true }])).toBe(
      "Models:  (✓ = loaded in memory, ○ = will be loaded on first message)",
    );
    expect(modelsHeading([{ ...baseModel }, { ...baseModel, loaded: false }])).toBe(
      "Models:  (✓ = loaded in memory, ○ = will be loaded on first message)",
    );
  });

  it("omits the legend when no model reports loaded state", () => {
    expect(modelsHeading([baseModel])).toBe("Models:");
    expect(modelsHeading([])).toBe("Models:");
  });
});
