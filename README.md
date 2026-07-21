# pi-localllm-provider

A Pi extension for wizard-based setup of local LLM servers — MTPLX, oMLX, LM Studio, llama.cpp, Ollama, vLLM, or anything else with an OpenAI-compatible API.

- **One command, one place** — `/localllm` is a single TUI menu for adding, inspecting, and managing every local server's integration with Pi — no subcommands, no hand-editing `settings.json`.
- **Reads the server, doesn't guess** — context window, reasoning, vision, size, quantization: pulled from real backend APIs across 7 detection paths, not typed into a config file and hoped correct.

## Quick start

```bash
pi install npm:pi-localllm-provider
```

Start Pi, run `/localllm`, select **＋ Add server**, follow the wizard.

Hacking on the extension itself? Skip npm and point Pi straight at your checkout:

```bash
mkdir -p ~/.pi/agent/extensions/pi-localllm-provider
cp -r /path/to/pi-localllm-provider/* ~/.pi/agent/extensions/pi-localllm-provider/
```

## Usage

`/localllm` opens a TUI menu — everything is managed from there, no subcommands.

```
LocalLLM — 2 server(s)
──────────────────────────────────────────────────
  Mac Studio  [oMLX]    (http://192.168.1.50:8000/v1)  2 model(s)
  Ollama      [Ollama]  (http://localhost:11434/v1)    5 model(s)
  ＋ Add server
```

Selecting a server opens its sub-menu with detected metadata per model — no need to open `settings.json`:

```
Mac Studio  [oMLX]
URL: http://192.168.1.50:8000/v1
Models:  (✓ = loaded in memory, ○ = will be loaded on first message)
  • ✓ Qwen2.5-Coder-7B-Instruct  (ctx 32k, max 8k, 4.6G)
  • ○ Qwen2.5-Coder-32B-Instruct  (ctx 32k, max 8k, 18.2G, reasoning, vision)
──────────────────────────────────────────────────
  ↺ Refresh model list from server
  ✎ Edit model capabilities (vision / reasoning)
  ✎ Reconfigure (name / URL / key)
  ✕ Remove this server
  ← Back
```

`[oMLX]` is the detected backend, persisted with the server. `✓`/`○` is server-side memory state, unrelated to which model Pi has selected via `/model`.

## Wizard flow

1. **Server name** — any label
2. **Base URL** — `http://host:8000`, `.../v1`, or a bare `host:8000` (scheme defaults to `http://`)
3. **API key** — blank if unauthenticated. A plain key on macOS offers Keychain storage — see [API key storage](#api-key-storage)

Then the [detection chain](#backend-detection) runs and you pick which discovered models to enable. If that's a single model, you'll be asked whether to switch to it right away; otherwise, switch between them with `/model`.

## Multiple servers

Each server registers as its own Pi provider — add as many as you like.

## FAQ

**I added a server, but Pi's still talking to the old model — what gives?**
**＋ Add server** and **✎ Reconfigure** both ask, but only when you end up with a single enabled model. **↺ Refresh** normally doesn't — it's just resyncing metadata on a server you're likely already using — unless the refreshed model IDs actually differ from before (the server started serving something else entirely), in which case whatever you had selected may no longer exist, and it asks too. If you enabled several models, or said no to the prompt, run `/model` and pick one yourself.

**The context window looks wrong. Can I fix it?**
It comes from whichever backend endpoint got detected (defaulting to 32,768 if nothing usable came back). Easiest fix is at the server/backend config, then **↺ Refresh** to pick up the corrected value.

**Does this work with Ollama?**
Yes, and properly — it talks to Ollama's native API, not just its OpenAI-compatible shim, so context window, reasoning, vision, size, quantization, and loaded state all get detected automatically. Point it at `http://localhost:11434`, with or without `/v1`.

**A server's unreachable — how long am I stuck waiting?**
About 8 seconds, total. The whole detection chain shares one deadline; it's not ~8s per backend probed along the way.

**If my API key is wrong, or something times out, will I actually be told?**
Yes — you'll see a real message like `"Authentication failed (HTTP 401)..."` or `"Timed out..."`/`"Could not connect..."`, not a vague "no models found." And on **↺ Refresh**, a failure like that never overwrites your existing config: a server that's briefly down won't cost you a model list that was working fine a minute ago.

**Do I have to redo any of this after restarting Pi?**
No — every configured server re-registers automatically on startup.

## Backend detection

`Add`/`Refresh` probes a fallback chain, richest metadata first. First match wins; anything else falls through to the generic probe.

| Backend | Detection | Extra metadata |
|---------|-----------|-----------------|
| MTPLX | `GET /health` | context window, max tokens, reasoning, vision |
| oMLX | `GET /v1/models/status` | + loaded state, size |
| LM Studio | `GET /api/v1/models` | + loaded state, size, quantization |
| llama.cpp (`llama-server`) | `GET /props` + `/v1/models` | context window (`n_ctx`, falls back to `n_ctx_train`), vision, size, `--alias` id |
| Ollama (native API) | `/api/tags` + `/api/show` per model + `/api/ps` | context window, reasoning, vision, size, quantization, loaded state |
| vLLM | `GET /version` + `/v1/models` | context window only — see note |
| OpenAI-compatible | `GET /v1/models` | context window if `max_model_len`/`context_window` present |

Only oMLX, LM Studio, and Ollama report loaded state — MTPLX, llama.cpp, and vLLM each serve exactly one model, so there's no loaded/unloaded distinction to make.

vLLM's `/v1/models` never carries reasoning or vision data; its detector exists only to label the backend `[vLLM]` correctly, not to unlock extra metadata.

**Known limitation — vLLM vision/reasoning.** Nothing in vLLM's public API says whether the served model supports images or reasoning, so both always come back `false`/text-only for `[vLLM]` servers, even for VLMs. (vLLM does have an internal `/server_info` debug endpoint that carries this, gated behind a `VLLM_SERVER_DEV_MODE=1` env var — but it's undocumented, dumps your full server config on request, and its system-info collection is known to crash on some setups, so this extension deliberately doesn't probe it.) If a tag is wrong for your model, use **✎ Edit model capabilities** in the server's sub-menu to flip vision/reasoning by hand — same effect as editing `settings.json` directly, just without leaving Pi. It survives until the next **↺ Refresh**, which overwrites it with whatever the server reports.

Flipping `reasoning` to `true` (auto-detected or by hand) only changes how *responses* are parsed. This extension always disables Pi's OpenAI o1-style reasoning-model conventions — the `reasoning_effort` request param and `developer`-role system prompts — for every model it registers, since none of the detection paths above actually confirm the server speaks either convention. So toggling reasoning on is safe to try even against a server that doesn't really support it: nothing about the outgoing request changes because of it.

## Configuration

Stored under the `localllm` key in `~/.pi/agent/settings.json`:

```json
{
  "localllm": {
    "servers": [
      {
        "id": "a3f7k2",
        "name": "Mac Studio",
        "baseUrl": "http://192.168.1.50:8000/v1",
        "apiKey": "",
        "apiType": "omlx",
        "models": [
          {
            "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "name": "Qwen2.5-Coder-7B-Instruct",
            "contextWindow": 32768,
            "maxTokens": 8192,
            "reasoning": false,
            "input": ["text"]
          }
        ]
      }
    ]
  }
}
```

Hand edits stick until the next **↺ Refresh**, which overwrites every model field with fresh live values — the server is always the source of truth. Useful for correcting a field the server misreports, or for dropping a model locally without changing anything on the server.

## API key storage

`apiKey` accepts any Pi-resolvable form:

| Form | Example |
|------|---------|
| Plain | `sk-...` (stored in the clear) |
| Env var | `$MY_API_KEY` |
| Shell command | `!security find-generic-password ...` |
| macOS Keychain | offered automatically for a plain key — see below |
| Empty | no auth |

**macOS Keychain:** typing a plain key in the wizard offers to store it via `security add-generic-password` (using `execFile`, not a shell string, so the key can't be interpreted as shell syntax) and replaces `apiKey` with a `!security find-generic-password ...` reference — the raw key never touches disk. Keyed by the server's internal `id` (survives **✎ Reconfigure**), deleted on **✕ Remove**. A key already starting with `!` or `$` skips the prompt.

Configured servers also show up in Pi's `/login` → **Use an API key**, if you'd rather use `auth.json` instead.

## License

[MIT](LICENSE)
