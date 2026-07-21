# voicezero

**A production-pattern AI voice agent at $0.00 per minute. One Python file. No platform fees.**

[![ci](https://github.com/shauryalowkeygotaura/voicezero/actions/workflows/ci.yml/badge.svg)](https://github.com/shauryalowkeygotaura/voicezero/actions/workflows/ci.yml)
[![cost](https://img.shields.io/badge/cost-%240.00%2Fmin-brightgreen)](#the-math)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-black)](LICENSE)

Voice AI platforms charge per minute to run a loop you can run yourself:

```
mic ──► STT ──► LLM (+ tools) ──► TTS ──► speaker
```

voicezero is that exact loop, extracted from a real production outbound sales agent that was burning ~$0.14/min on a hosted stack, rebuilt on components that cost nothing:

| Stage | Component | Cost |
|---|---|---|
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) local, or Groq `whisper-large-v3-turbo` | $0.00 |
| LLM | [Groq](https://console.groq.com) free tier (Llama 3.3 70B and friends) | $0.00 |
| TTS | [edge-tts](https://github.com/rany2/edge-tts) neural voices, **auto-switching English ↔ Hindi per reply** | $0.00 |
| Audio I/O | sounddevice mic + native playback | $0.00 |

> **Speaks Hinglish, natively.** The agent detects the language of each reply and switches voice on the fly, English lines in an English voice, Hindi lines in a Hindi voice, turn by turn in the same call. It transcribes Hindi and English too. All on free voices, zero config. This is the thing most hosted platforms charge extra for or do badly. [How it works ↓](#switching-english--hindi-mid-call)

It is not a toy: it has tool calling, personas, template variables, on-the-fly Hindi/English voice switching, prompt-injection sanitization, dead-air protection, model and key rotation, post-call structured summaries, and a JSONL call log.

## The math

Approximate per-minute pricing for hosted voice agent platforms (ballpark, mid 2026, varies by model and voice choices):

| Platform | Per minute | 1,000 min/month |
|---|---|---|
| Bland.ai | ~$0.09 | ~$90 |
| Retell AI | ~$0.07 to $0.30 | $70+ |
| VAPI | ~$0.05 to $0.33 | $50+ |
| ElevenLabs Agents | ~$0.08 to $0.10 | $80+ |
| **voicezero** | **$0.00** | **$0** |

Hosted platforms are genuinely great for zero-setup telephony. But for demos, prototypes, kiosks, desktop assistants, internal tools, and for actually understanding how voice agents work, you should not be paying rent on a while-loop.

## 60-second start

```bash
git clone https://github.com/shauryalowkeygotaura/voicezero
cd voicezero
pip install -r requirements.txt
# ffmpeg must be on PATH: https://ffmpeg.org/download.html

# Step 1: prove the audio stack, no API key needed
python agent.py --loopback

# Step 2: grab a free key at https://console.groq.com/keys
cp .env.example .env   # paste your key in

# Step 3: full pipeline check, headless, no mic needed
python agent.py --selftest

# Step 4: talk to it
python agent.py --voice
```

First run downloads the whisper model (~75 MB for `base`). Set `WHISPER_MODEL=tiny` for faster, lighter turns.

## What a call looks like

```
AGENT: Thanks for calling Brightsmile Dental, this is Maya. How can I help you today?
YOU:   Hi, I need a checkup sometime this week.
AGENT: Of course! Can I get your name, and is there a day and time that works best for you?
YOU:   I'm Rohan, Thursday afternoon would be great.
AGENT: One moment, let me get that booked for you.
AGENT: All set, Rohan, you're booked for a checkup Thursday at three pm. Anything else?
YOU:   No that's all, thanks!
AGENT: Wonderful, see you Thursday, Rohan. Have a great day!

-- call ended: booked | booked (log: call_log.jsonl)
```

The `book_appointment` tool fired mid-call, the agent spoke a fixed line while it ran (no dead air), and a separate cheap model wrote a structured summary into the call log afterwards.

And the same agent, following the caller from Hindi into English mid-call (the `outbound_sales` persona):

```
AGENT: Namaste! Main Riya bol rahi hoon Star Telecom se.   <- Hindi voice
YOU:   Haan boliye, kya baat hai?
AGENT: Aapka plan expire ho raha hai, ek behtar plan hai.  <- Hindi voice
YOU:   So what does the new one actually cost?             (caller just switches)
AGENT: It is three forty nine a month, with more data      <- English voice
       and a streaming bundle.
YOU:   Okay, go ahead and send me the details.
AGENT: Sure, I am sending them over on WhatsApp right now.  <- English voice

-- call ended: interested | interested (log: call_log.jsonl)
```

The caller never says "speak English". They just do, and the agent's next reply comes back in an English voice. The persona is told to mirror the caller's language, so the LLM replies in English; voicezero then detects that reply is English and routes it to the English voice, all on its own, turn by turn. See [Switching English ↔ Hindi](#switching-english--hindi-mid-call).

## Modes

| Mode | What it does | Needs |
|---|---|---|
| `--loopback` | TTS a line, STT it back. Proves audio works. | nothing |
| `--selftest` | Full headless check: TTS, STT, LLM, tools, summary. | Groq key |
| `--text` | Chat with the agent in the console. | Groq key |
| `--voice` | Full mic and speakers conversation. | Groq key + mic |

## Personas

An agent is one JSON file. Three ship in [`personas/`](personas/):

- **receptionist** (default): books appointments for a dental clinic
- **outbound_sales**: Hinglish telecom upsell caller, permission-first, with a do-not-call tool
- **interview_coach**: runs a spoken mock interview, then gives a verdict

```bash
python agent.py --voice --persona personas/interview_coach.json --var role="backend engineer"
python agent.py --text  --persona personas/outbound_sales.json --var lead_name=Rohan --var current_plan="Rs 299"
```

A persona is:

```json
{
  "name": "receptionist",
  "voice": "en-US-AriaNeural",
  "first_message": "Thanks for calling, how can I help?",
  "system_prompt": "You are ... {{caller_name}} ...",
  "tools": [
    {
      "name": "book_appointment",
      "description": "Book once you have name, reason, and time.",
      "parameters": { "type": "object", "properties": { "...": {} } },
      "speech_line": "One moment, let me get that booked."
    }
  ]
}
```

Notes on the schema:

- `{{variables}}` in the prompt are filled from `--var key=value`. Values are sanitized (template and structure characters stripped, length capped) before substitution, so caller data stays data.
- `end_call` is built in and appended to every persona automatically. You never define it.
- `speech_line` is voicezero's own field, not part of the JSON Schema sent to the LLM. If a turn is tool-calls-only, the agent speaks this fixed line so the listener never hears silence. This is the classic silent-turn bug that hosted platforms solve with tool messages; same fix, locally.
- `voice` is any [edge-tts voice](https://gist.github.com/BettyJJ/17cbaa1de96235a7f5773b8690a20462) — the single-voice shorthand and the guaranteed fallback. For a bilingual agent, use the `tts` block instead (see [Better voices](#better-voices-optional)).
- `tts` is a per-language voice map that switches the voice on the fly (below). `stt_lang` sets the caller's language for transcription (`en` / `hi` / `auto`) — a Hinglish persona should use `hi`. Optional `stt_backend` overrides the STT backend for this agent only (see [Better transcription](#better-transcription-stt)).

## Better voices (optional)

edge-tts is genuinely good and stays the default: zero config, zero extra dependencies, no key, and it already has [neural voices in dozens of languages](https://gist.github.com/BettyJJ/17cbaa1de96235a7f5773b8690a20462) including `hi-IN` Hindi ones. You never need anything below.

### Switching English ↔ Hindi mid-call

Indian callers code-switch constantly. voicezero routes **each agent reply to the best voice for that reply's language** — English replies get an English voice, Hindi/Hinglish replies get a Hindi voice — switching turn by turn as the conversation flows. It's whole-utterance routing (one consistent voice per reply, on purpose: swapping engines mid-sentence would flip the timbre and sound broken; a Hindi-capable voice reads the English words inside a Hinglish sentence just fine).

Give a persona a `tts` block, one entry per language:

```json
"tts": {
  "en": { "provider": "edge", "edge_voice": "en-IN-NeerjaNeural" },
  "hi": { "provider": "edge", "edge_voice": "hi-IN-SwaraNeural" }
},
"stt_lang": "hi"
```

That's the shipped `outbound_sales` persona: **free edge voices, zero setup, but now the Hindi lines actually come out of a Hindi voice** instead of an English one straining through Devanagari. Language detection is a tiny dependency-free heuristic (Devanagari + a romanized-Hindi lexicon) tuned to *not* misfire on English — no model, no API. Per-entry fields: `provider` (default `edge`), `voice` (the provider's own voice name), `edge_voice` (the guaranteed fallback for that language), `rate`.

### Upgrading a language to a better engine

Swap any language's `provider` for a higher-quality backend. It mirrors the LLM model chain: if the backend's package/model/key isn't there, that reply **silently falls back to `edge_voice`** for the same language. A turn is never lost to an un-installed extra.

| Provider | Where it runs | Cost | English | Hindi | Enable |
|---|---|---|---|---|---|
| **edge** (default) | Microsoft cloud endpoint | $0.00 | great | good (`hi-IN` voices) | nothing — it's the default |
| **kokoro** | fully local CPU, offline | $0.00 forever | excellent | decent (experimental `hf_`/`hm_` voices) | `pip install kokoro-onnx` |
| **sarvam** | Sarvam cloud REST | free trial, then paid | good | best-in-class | set `SARVAM_API_KEY` |

```json
"tts": {
  "en": { "provider": "kokoro", "voice": "af_heart",  "edge_voice": "en-US-AriaNeural" },
  "hi": { "provider": "sarvam", "voice": "anushka",   "edge_voice": "hi-IN-SwaraNeural" }
}
```

- **kokoro** ([kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0) is the best *free* upgrade: entirely on CPU, offline once cached, $0 forever. First use auto-downloads a ~310 MB model to `~/.cache/voicezero` (override with `VOICEZERO_CACHE`).
- **sarvam** ([Bulbul](https://www.sarvam.ai/)) has the best Hindi / Indian-language voices — the natural pick for the `hi` slot of a Hinglish agent. Pure REST (no package), opt-in via `SARVAM_API_KEY`, and the *same key* also unlocks the Saarika STT backend below.

Optional deps live in `requirements-optional.txt` — install only what you want. The whole layer is [`tts_providers.py`](tts_providers.py), ~200 readable lines; adding a provider is one function plus a dict entry.

## Better transcription (STT)

The listener half matters as much as the voice. voicezero has a three-rung STT ladder, and `STT_BACKEND=auto` (the default) picks the best **available**, falling through on failure exactly like the LLM key rotation:

| Backend | Model | Where it runs | Cost | Best for |
|---|---|---|---|---|
| **local** | faster-whisper (`small`) | your CPU, offline | $0.00 | private / offline dev; audio never leaves the device |
| **groq** | `whisper-large-v3-turbo` | Groq cloud | ~free (pennies) | **the quality default** — far better Hindi/accents than local `small` |
| **sarvam** | Saarika v2 | Sarvam cloud | free trial, then paid | best-in-class Indian / code-switched speech |

`auto` resolves to `sarvam` (if `SARVAM_API_KEY`) → `groq` (if a Groq key) → `local`. Because a live call already needs a Groq key, **it transcribes on `whisper-large-v3-turbo` automatically** — the single biggest fix if your STT feels weak — while `--loopback` (no key) stays fully local.

Two things that quietly wreck Hinglish STT, both fixed here:
- **Forcing the wrong language.** Pinning Whisper to `en` mangles Hindi. STT language is now per-persona (`stt_lang`): the Hinglish persona uses `hi`, English personas use `en`, or set `auto` to detect per turn.
- **A too-small local model.** `small` is the CPU sweet spot but has a ceiling; the `groq`/`sarvam` rungs blow past it for a fraction of a cent.

Want STT fully on-device (no audio leaving your machine)? Set `STT_BACKEND=local` globally, or override **per voice agent** with a persona `stt_backend` field (`"auto"` / `"local"` / `"groq"` / `"sarvam"`; empty uses the global default). So one agent can stay fully private while another uses cloud STT, no env juggling.

## Wiring real tools

Tools currently log to `call_log.jsonl` and return `{"status": "queued"}`, the async fire-and-forget pattern hosted platforms use. To make a tool actually do something, edit one method in `agent.py`:

```python
def _handle_tool(self, name: str, args: dict) -> dict:
    self.events.append({"tool": name, "args": args, "t": time.time()})
    if name == "end_call":
        self.ended = True
        self.end_reason = args.get("reason", "")
    elif name == "book_appointment":
        calendar.create_event(args)        # your code here
    return {"status": "queued"}
```

This is the entire "webhook server" you would otherwise be deploying.

## How it stays free at scale

Groq's free tier gives each **model** its own daily token pool per account. voicezero exploits that two ways:

1. **Model rotation**: every LLM call walks a chain (`gpt-oss-120b`, `llama-3.3-70b`, `llama-3.1-8b`). Rate limited on one pool? The call silently lands on the next. One key gets roughly triple the headroom.
2. **Key rotation**: set `GROQ_API_KEYS=key1,key2` (one key per Groq account; keys on the same account share quota) and the chain multiplies again.

The post-call summary deliberately runs on `llama-3.1-8b-instant`: extraction is easy, and it draws from a pool the conversation never touches, so summaries keep working even when the big model's budget is spent.

## Honest limitations

- **Turn-based, no barge-in.** You cannot interrupt the agent mid-sentence. Hosted platforms do this better today.
- **Latency is 1.5 to 4 seconds per turn** on a normal CPU (STT + LLM + TTS, sequential). Usable for demos and internal tools; not yet indistinguishable from a human. `WHISPER_MODEL=tiny` and lower `max_tokens` help.
- **Energy-based VAD**, not semantic. A noisy room can self-trigger the mic (raise the threshold in `record_utterance`).
- **STT accuracy is a tradeoff.** Fully-local `small` on CPU is private and free but has a ceiling on Hindi/accented speech; the default `auto` backend sends live-call audio to Groq's `whisper-large-v3-turbo` for a big accuracy jump. Set `STT_BACKEND=local` to keep audio on-device. See [Better transcription](#better-transcription-stt).
- **No telephony out of the box.** This is mic and speakers. See roadmap.
- **edge-tts** (the default) uses the same endpoint as Microsoft Edge's read-aloud. Perfect for development and demos. For a better voice, especially in Hindi, drop in `kokoro` (free, local) or `sarvam` (paid, best Hindi) — see [Better voices](#better-voices-optional); for commercial-grade English, Azure TTS uses the identical voices, paid. All are one config field.

## Roadmap

- [ ] Streaming TTS playback (speak while generating)
- [ ] Barge-in (interrupt the agent mid-sentence)
- [ ] Telephony transport (pipecat + Twilio/Exotel SIP: you pay only carrier rates, roughly $0.007/min, still no platform fee)
- [ ] Browser/WebRTC demo
- [ ] Persona pack (support agent, language tutor, order taker)

PRs welcome. The codebase is intentionally one file so you can read all of it in one sitting.

## FAQ

**Is this actually $0.00?**
Yes, within Groq's free-tier daily limits, which are generous for development and demos (and multiply with the rotation tricks above). LLM and cloud STT run on Groq's free tier; TTS is free (edge-tts) or local (kokoro). Fully-local STT (`STT_BACKEND=local`) is $0 compute. At commercial volume you would move the LLM to a paid tier and TTS to Azure/sarvam, and you would still pay cents per hour, not per minute.

**Why not just use VAPI/Retell/Bland?**
If you need phone numbers today with zero setup, use them, they are good products. voicezero exists because most voice agent work (prototyping, demos, learning, internal tools) does not need a phone number, and because understanding the loop makes you better at using any platform.

**Does it speak languages other than English?**
Yes, and it switches on the fly. Each agent reply is routed to the best voice for its language, so a bilingual persona speaks English in an English voice and Hindi in a Hindi voice, turn by turn. The included `outbound_sales` persona runs Hinglish calls this way — English lines on an Indian-English voice, Hindi lines on a Hindi voice — using free edge voices out of the box. See [Better voices](#better-voices-optional).

**Where do conversations go?**
Nowhere except `call_log.jsonl` on your machine (gitignored). TTS text goes to Microsoft (edge) or stays local (kokoro); LLM text goes to Groq. STT audio: with the default `auto`/`groq`/`sarvam` backends the utterance is sent to that cloud API for accuracy — set `STT_BACKEND=local` to keep STT 100% on your CPU.

---

Extracted from a production outbound sales agent built for real clients. If this saved you a per-minute bill, **star the repo**, it genuinely helps.

[![Star History Chart](https://api.star-history.com/svg?repos=shauryalowkeygotaura/voicezero&type=Date)](https://star-history.com/#shauryalowkeygotaura/voicezero&Date)

MIT licensed. Built by [Shaurya Vardhan Shandilya](https://github.com/shauryalowkeygotaura).
