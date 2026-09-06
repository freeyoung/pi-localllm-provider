# pi-localllm-provider

A Pi extension for wizard-based setup of local LLM servers — MTPLX, oMLX, LM Studio, llama.cpp, Ollama, vLLM, SGLang, ds4, ninfer, or anything else with an OpenAI-compatible API.

- **One command, one place** — `/localllm` is a single TUI menu for adding, inspecting, and managing every local server's integration with Pi — no subcommands, no hand-editing `settings.json`.
- **Reads the server, doesn't guess** — context window, reasoning, vision, size, quantization: pulled from each backend's own API by a detector written for it, and probed directly where a backend publishes nothing, not typed into a config file and hoped correct.

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
  Mac Studio  [oMLX]    (http://mac-studio.lan:8000/v1)  2 model(s)
  Ollama      [Ollama]  (http://localhost:11434/v1)    5 model(s)
  ＋ Add server
```

Selecting a server opens its sub-menu with detected metadata per model — no need to open `settings.json`:

```
Mac Studio  [oMLX]
URL: http://mac-studio.lan:8000/v1
Models:  (✓ = loaded in memory, ○ = will be loaded on first message)
  • ✓ Qwen2.5-Coder-7B-Instruct  (ctx 32k, max 8k, 4.6G)
  • ○ Qwen2.5-Coder-32B-Instruct  (ctx 32k, max 8k, 18.2G, reasoning, vision)
──────────────────────────────────────────────────
  ↺ Refresh model list from server
  ✎ Edit model capabilities (vision / reasoning / context / temperature)
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

**The server doesn't report a context window at all (e.g. llama-swap), so Pi always assumes the 32,768 default?**
Use **✎ Edit model capabilities** in the server's sub-menu and pick **Max context** to set the real value by hand. Unlike the other fields there, this one **survives ↺ Refresh**: the override is stored separately and re-applied to the freshly detected model by id, so you can refresh as often as you like without losing it. It shows in the model list as a `*` next to the context window (e.g. `ctx 128k*`). Empty input removes the override and falls back to the server-reported value.

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
| llama.cpp (`llama-server`) | `GET /props` + `/v1/models` + `POST /apply-template` | context window (`n_ctx`, falls back to `n_ctx_train`), vision, size, `--alias` id, reasoning and request compat — both measured, see note |
| SGLang | `GET /model_info` + `/server_info` + `/v1/models` | context window, reasoning, vision, measured request compat — see note |
| Ollama (native API) | `/api/tags` + `/api/show` per model + `/api/ps` | context window, reasoning, vision, size, quantization, loaded state |
| vLLM | `GET /version` + `/v1/models` | context window only — see note |
| ds4 | `GET /v1/models` with `owned_by: "ds4.c"` | context window, max tokens, reasoning, request compat — see note |
| ninfer | `GET /v1/models` with `owned_by: "ninfer"` | context window, reasoning, vision, request compat — all measured, see note |
| OpenAI-compatible | `GET /v1/models` | context window from `max_model_len`, `top_provider.context_length`, `context_window` or `context_length`; name, reasoning and vision from an OpenRouter-style card |

Only oMLX, LM Studio, and Ollama report loaded state — MTPLX, llama.cpp, vLLM, SGLang, ninfer, and ds4 each serve exactly one model, so there's no loaded/unloaded distinction to make.

**SGLang is probed before Ollama, and the order matters.** SGLang ships an Ollama compatibility shim, so its `/api/tags` and `/api/show` answer with Ollama-shaped payloads — enough for the Ollama probe to claim it if it got there first. The shim is lossy in three ways that all read as a working setup, which is what makes the mislabel worth preventing:

- `/api/show` reports `capabilities: ["completion"]` whatever the model can actually do, so **vision and reasoning both come back false**.
- `/api/ps` isn't implemented, so every model shows as `○ will be loaded on first message` when SGLang in fact holds one model resident for the life of the process.
- Model ids come through as the raw `--model-path`, so a name like `/models/Org/Model-NVFP4` ends up in the picker.

Probing SGLang first replaces all three with the real answers, from two SGLang-only endpoints: `/model_info` for `has_image_understanding` (the vision signal) and `is_generation` (which tells a chat server from an embedding-only one, so the latter falls through instead of being registered), and `/server_info` for `reasoning_parser`. That last one is the whole reasoning signal, and it's a property of how the server was launched, not of the weights — SGLang only splits reasoning out of a response when it was started with a parser, so a model that can reason reports `reasoning: false` here until the server is given `--reasoning-parser`. The context window comes from `/v1/models`' `max_model_len`; `/server_info`'s `context_length` is the CLI override and stays `null` unless it was passed explicitly. Both endpoints were called `/get_model_info` and `/get_server_info` before SGLang renamed them; the old names still answer but log a deprecation warning, so they are tried only as a fallback and a server on either side of the rename detects the same.

#### Why SGLang's request compat is measured instead of declared

ds4's `compat` could be hard-coded because its accepted values live in its own source. SGLang's don't — they aren't SGLang's to define. It validates `reasoning_effort` against all seven OpenAI tiers and then hands the string to the model's chat template, and the *template* decides. The vocabularies really do differ: a Qwen3.8 template answers

```
400  Unexpected reasoning effort high.
     Supported types are xhigh (default), medium, and low.
```

while SGLang's own Kimi K3 path recognises `low`/`high`/`max` instead. Hard-coding either set would 400 every request on a server running the other. The `developer` role splits the same way — SGLang's schema accepts the role, and a template that only knows `system` still rejects the message with `Unexpected message role.`

So when a `reasoning_parser` is configured, `Add`/`Refresh` measures both: one throwaway completion per tier, capped at a single token, plus one for the developer role. A rejected tier fails during template rendering, before any generation, so the sweep costs one token per *accepted* tier — four, on the Qwen3.8 server above. Nothing is measured when no parser is configured, since the levels would have nothing to steer.

Levels the model rejects are then mapped to the nearest one it accepts, ties going to the weaker tier so a request for more thinking than exists lands under the ceiling rather than over it. `off` is special-cased: it may only ever map to `none`, never to whatever tier happens to sit nearest, and it stays unmapped on a model with no off switch — which makes Pi omit the field and take the model's default. For Qwen3.8-27B that yields:

| Pi level | Sent as | |
|----------|---------|---|
| `off` | `none` | |
| `minimal` | `low` | lifted to the floor |
| `low` | `low` | |
| `medium` | `medium` | |
| `high` | `medium` | tie, resolved downward |
| `xhigh` | `xhigh` | |
| `max` | `xhigh` | dropped to the ceiling |

Without that map, three of Pi's seven levels would 400. If the server can't be reached or answers nothing, no `compat` is recorded at all — an unanswered probe isn't evidence against a convention, so it degrades to the same safe defaults every other backend uses.

**llama.cpp reports no reasoning flag, so it is measured too.** `/props` says a lot — context, modalities, the chat template's source and its `chat_template_caps` — but not the one thing that decides `reasoning`: whether that template has a thinking switch. `llama-server` works this out for itself at startup (`common_chat_templates_support_enable_thinking`) and never puts the answer on the API. What it does put on the API is the renderer itself: `POST /apply-template` runs the full chat-completions request parse, `chat_template_kwargs` and `reasoning_effort` included, and returns the prompt it would have generated from, without generating. So the detector renders one throwaway user turn three times:

- with `enable_thinking: true` forced through `chat_template_kwargs`
- with `enable_thinking: false` forced the same way
- with neither, which is whatever the server was started with

If the two forced renders come out identical, the template has no switch and the model registers as non-reasoning. If they differ, the unforced render shows which side the server sits on: a `--reasoning off` server renders like the disabled case and registers as non-reasoning too, because that is the state every request from Pi will run in — Pi can send `reasoning_effort`, but not `chat_template_kwargs`, so nothing it sends could turn thinking back on. Only a server that thinks by default registers as reasoning. One field that looks like it should help is deliberately ignored: `/props` reports `default_generation_settings.params.reasoning_format` as `none` on a server that is splitting `reasoning_content` out of every reply, because that block is built from default parser settings rather than the server's flags.

The tier sweep reuses the same endpoint, which makes it free where SGLang's costs a token per accepted tier. `llama-server` handles `"none"` itself — it flips `enable_thinking` off and never shows the template the value — and hands every other tier to the template as `reasoning_effort`, where a Qwen3.8 template rejects the ones it doesn't know exactly as it does under SGLang. One difference matters: a template `raise` comes back as a **500**, not a 400, with the raise message in the error body. A bare status can't tell that from a broken server, so the detector reads the body and counts only a 500 quoting a `Jinja Exception` as a rejection. Anything else is an unanswered tier, and one of those discards the sweep, for the same reason as on SGLang. A template that never reads `reasoning_effort` accepts every tier and acts on none of them; that registers as an identity map, and `off → "none"` still works on it, since that entry is implemented by the server rather than the template.

vLLM's `/v1/models` never carries reasoning or vision data; its detector exists only to label the backend `[vLLM]` correctly, not to unlock extra metadata.

**Known limitation — vLLM vision/reasoning.** Nothing in vLLM's public API says whether the served model supports images or reasoning, so both always come back `false`/text-only for `[vLLM]` servers, even for VLMs. (vLLM does have an internal `/server_info` debug endpoint that carries this, gated behind a `VLLM_SERVER_DEV_MODE=1` env var — but it's undocumented, dumps your full server config on request, and its system-info collection is known to crash on some setups, so this extension deliberately doesn't probe it.) If a tag is wrong for your model, use **✎ Edit model capabilities** in the server's sub-menu to flip vision/reasoning by hand — same effect as editing `settings.json` directly, just without leaving Pi. It survives until the next **↺ Refresh**, which overwrites it with whatever the server reports.

**ds4** ([antirez/ds4](https://github.com/antirez/ds4), "DwarfStar") serves nothing outside `/v1` — no `/health`, `/props` or `/version`, and no `Server` header — so it can't be probed the way the backends above are. It's identified instead by the one thing `ds4_server.c` hard-codes onto every model card, `"owned_by":"ds4.c"`, which the generic `/v1/models` request already fetches. Three things on those cards are read from ds4's source rather than taken at face value:

- **The ids are aliases, not separate models.** ds4 emits either `deepseek-v4-flash` + `deepseek-v4-pro`, or `glm-5.2` + `glm-5.2-chat` + `glm-5.2-reasoner` for a GLM-DSA engine — but every entry reports the same GGUF passed with `-m`, so their name, context and max tokens are identical. All of them are registered, since the server accepts each as a `model` parameter, which is why one server can show several same-named entries. Detection keys on `owned_by` alone: matching the ids would break on the next engine ds4 adds.
- **`supported_parameters` is a constant**, not a capability report — it lists `reasoning_effort` no matter which GGUF is loaded. Reasoning is set for ds4 because every engine it serves is a reasoning model, not because that array says so.
- **`top_provider.max_completion_tokens` is `min(--default-tokens, --ctx)`.** When it equals the context window it means "no separate output limit", so the usual max-tokens cap applies instead of reserving the whole window for one response.

Flipping `reasoning` to `true` (auto-detected or by hand) only changes how *responses* are parsed. This extension disables Pi's OpenAI o1-style reasoning-model conventions — the `reasoning_effort` request param and `developer`-role system prompts — by default for every model it registers, since most detection paths above can't confirm the server speaks either convention. So toggling reasoning on is safe to try even against a server that doesn't really support it: nothing about the outgoing request changes because of it.

The exceptions are the backends that could prove it. ds4, because both conventions were confirmed in its source: it parses `reasoning_effort` on the chat path, and accepts the `developer` role wherever it accepts `system`. SGLang, ninfer and llama.cpp, because each is measured against the loaded template, as described above. Those enable both conventions on evidence, which is what lets Pi's thinking levels actually reach the server. Hand-edited models and servers configured before this existed keep the safe defaults.

### ninfer publishes nothing, so everything is measured

Written from [Neroued/ninfer](https://github.com/Neroued/ninfer) at `master` and then checked against a running one. Every number below was measured; the file citations are where each probe came from.

ninfer is identified the way ds4 is, by a hard-coded owner on the model card — `"owned_by": "ninfer"` in `src/serve/openai_schema.cpp`. It has no endpoint of its own to probe: its five GET routes are `/health`, `/v1/models`, `/v1/models/{id}` and two Responses lookups, and `/health` answers a bare `{"status":"ok"}`, which is also why the MTPLX probe — the only other one that reads `/health` — passes over it.

Where ds4's card at least carries a context window, ninfer's carries nothing beyond `{id, object, created, owned_by}`. Everything else is a startup flag with no runtime reader: `--max-context` (default **8192**), `--vision` (off unless passed), and the artifact's own `chat_template.jinja`, which decides the reasoning tiers. Registering this extension's generic 32768 fallback would claim four times the real ceiling on a default server and let Pi fill a context it would then be refused.

So three things are measured, each shaped to cost nothing on the GPU:

- **Context window.** A deliberately oversized prompt is rejected during prompt preparation, before any prefill, and the rejection names the ceiling: `src/runtime/engine/engine.cpp` builds `"prepared prompt has N tokens, exceeding Engine max_context M"`. Overshooting is free; falling short is not, because a prompt that fits would be accepted and actually run — so if one is accepted anyway, its reported `prompt_tokens` is kept as a floor rather than discarded. The probe aims 300k tokens past the largest context these artifacts ship with, and two characters per token is measured, not assumed: `"x "` × 25,000 came back as 25,052 prompt tokens. Where nothing can be parsed at all, the fallback is ninfer's own 8192 rather than this file's 32768.
- **Vision.** A token count carrying a 1×1 PNG. `docs/serving.md` says media requests *and token-count requests* fail with 400 `vision_disabled` when `--vision` was omitted, and that the count endpoints run "without running GPU generation" — so the answer costs nothing whichever way it goes. A rejection for any other reason leaves vision unknown rather than declaring a vision model text-only.
- **Reasoning tiers.** The same probe SGLang uses, because the situation is the same: the accepted values come from the loaded artifact's chat template, and an unsupported one is rejected — 400 `reasoning_effort_not_supported` here. A Qwen3.8 artifact accepts `none`, `low`, `medium` and `xhigh` and rejects `minimal`, `high` and `max` — the same set that artifact exposes under SGLang, reached through a completely different server. Measuring rather than writing it down is what keeps that from becoming an assumption about every artifact ninfer will ever load.

The `developer` role works, so it is enabled rather than left at the safe default. Reasoning comes back on a separate `reasoning_content` field, which Pi reads without help. And the Chat Completions `usage` object carries only `prompt_tokens`, `completion_tokens` and `total_tokens` — no `completion_tokens_details.reasoning_tokens` — so anything wanting a thinking/answer split has to infer it from the stream.

The whole detection, 600 KB probe body included, takes under 400 ms.

### Thinking levels on ds4

Worth understanding before reading anything into the status bar, because the interesting cases are the ones where Pi and the server disagree.

While `supportsReasoningEffort` is off — the default everywhere it has not been confirmed or measured — Pi never sends `reasoning_effort` at all, so the server keeps using whatever it defaults to internally and the level shown in Pi has no effect on it whatsoever. Turning it on is what connects the two for the first time. That can look like a regression: a server quietly running at its own default now follows the session instead. It isn't one — it's the first time the setting was ever wired up.

ds4 recognises only three modes internally (`think_mode_from_enabled` in `ds4_server.c`) — off, high, and max — so Pi's seven levels don't map one-to-one. Two of them would land somewhere surprising without help:

- **`off` would not turn thinking off.** Pi converts an `off` level into an *omitted* `reasoning_effort` rather than a value, and ds4 reads an omitted one as "use my defaults", which are thinking-enabled at high. The `[ds4]` detector maps `off` to the literal `"none"` — the only string that genuinely disables it.
- **`max` would be unreachable.** Pi hides `xhigh` and `max` unless a model explicitly declares them, so ds4's real max mode wouldn't appear in the level list. The detector declares `max`, but deliberately not `xhigh`: ds4 folds `xhigh` into high, so offering it would imply a distinction that doesn't exist.

That leaves this mapping, with `minimal`/`low`/`medium`/`high` passing through unchanged — ds4 accepts all four and treats them alike:

| Pi level | Sent as | ds4 mode |
|----------|---------|----------|
| `off` | `"none"` | no reasoning |
| `minimal`, `low`, `medium`, `high` | unchanged | high |
| `xhigh` | not offered | — |
| `max` | `"max"` | max |

**If the status bar is stuck on `thinking off`,** that's Pi's session state, not a detection problem. Pi carries the current thinking level across model switches whenever the new model supports thinking, and `off` is a legal level for a reasoning model, so nothing raises it back. A model that reported `reasoning: false` earlier in the session — which is how ds4 looked before it was detected properly — pins the level to `off`, and it stays there through the next **↺ Refresh**. Pi deliberately doesn't persist that particular `off` to settings, so set the level once (`pi --thinking high`, or Pi's thinking selector) and it sticks from then on.

## Configuration

Stored under the `localllm` key in `~/.pi/agent/settings.json`:

```json
{
  "localllm": {
    "servers": [
      {
        "id": "a3f7k2",
        "name": "Mac Studio",
        "baseUrl": "http://mac-studio.lan:8000/v1",
        "apiKey": "",
        "apiType": "omlx",
        "models": [
          {
            "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "name": "Qwen2.5-Coder-7B-Instruct",
            "contextWindow": 32768,
            "contextWindowOverride": 131072,
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

### Overriding the context window manually

The one exception to "refresh overwrites everything" is the field above: `contextWindowOverride`. A server that never reports a window — like an OpenAI-compatible **llama-swap** proxy — lands on this extension's 32,768 fallback, and no config change on the backend can fix that because nothing it publishes carries the number. Set it by hand with **✎ Edit model capabilities → Max context** in the server's sub-menu (one override per model, since model ids are unique). The override is stored *next to* `contextWindow`, which keeps the server-reported value, so:

- **↺ Refresh** and **✎ Reconfigure** re-attach the override to the re-detected model by id — it is never wiped.
- Clearing the override (empty input) drops the field and falls back to the server-reported window.
- While an override is active, the model's max-output cap is re-derived from it with the same half-window rule as everywhere else (`capTokens`), so a raised window raises the cap too unless the reasoning cap applies.

### Why `maxTokens` is so much larger for a reasoning model

Pi resolves each turn's `max_tokens` as `options.maxTokens ?? model.maxTokens`, clamped to the context left after the prompt — so this is a real generation cap, and **reasoning and the answer come out of the same budget**.

That matters because no local backend offers a thinking cap worth relying on. Pi can send `thinking_token_budget`, but that is vLLM's field and gated behind a compat flag; SGLang's OpenAI surface accepts `max_thinking_tokens` and then silently ignores it. So a reasoning model that thinks its way to the ceiling spends the *entire* budget reasoning and returns no answer at all — which then reads as a truncated turn, gets fed back with a truncation notice, and re-plans into the same wall. Setting the number too low doesn't prevent that; it just reaches it sooner.

Hence the fallback where a backend reports no limit of its own:

| | Ceiling |
|---|---------|
| Reasoning model | `min(context ÷ 2, 65536)` |
| Everything else | `min(context ÷ 2, 8192)` |

Backends that *do* report a real limit — MTPLX's `max_response_tokens`, oMLX's `max_tokens`, an OpenRouter-style `max_completion_tokens` below the context window — are believed over both.

### Sampling

Nothing is sent unless a detector has a reason for it, so a server keeps applying its own `generation_config`. The one exception is Qwen on SGLang: Qwen publishes **0.6** for thinking mode, checkpoints commonly ship `1.0` in `generation_config`, and Pi names no temperature of its own — so `1.0` is what actually gets served. At that temperature a thinking model in an agent loop is prone to re-planning the same task until it runs out of budget. Detecting `model_type: qwen*` with a reasoning parser configured therefore sets `temperature: 0.6`, and no other family is guessed at.

Any model's temperature can be set by hand from **✎ Edit model capabilities**, including back to "server default" by clearing it — like the other overrides there, until the next **↺ Refresh**.

A related sharp edge: a reported size of `0` means "not reported", not a zero-byte model. SGLang's Ollama shim sends `size: 0` for the very model it is serving. Detectors drop a zero rather than storing it, and the model list ignores one that a pre-existing config still carries, so neither shows up as `0.0G`.

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
