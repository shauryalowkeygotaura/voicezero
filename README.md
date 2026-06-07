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
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper), local CPU | $0.00 |
| LLM | [Groq](https://console.groq.com) free tier (Llama 3.3 70B and friends) | $0.00 |
| TTS | [edge-tts](https://github.com/rany2/edge-tts), Microsoft neural voices | $0.00 |
| Audio I/O | sounddevice mic + native playback | $0.00 |

It is not a toy: it has tool calling, personas, template variables, prompt-injection sanitization, dead-air protection, model and key rotation, post-call structured summaries, and a JSONL call log.

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
- `voice` is any [edge-tts voice](https://gist.github.com/BettyJJ/17cbaa1de96235a7f5773b8690a20462). There are hundreds, in dozens of languages.

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
- **No telephony out of the box.** This is mic and speakers. See roadmap.
- **edge-tts** uses the same endpoint as Microsoft Edge's read-aloud. Perfect for development and demos; for commercial production swap in Azure TTS (identical voices, paid) or any TTS you like, it is one function.

## Roadmap

- [ ] Streaming TTS playback (speak while generating)
- [ ] Barge-in (interrupt the agent mid-sentence)
- [ ] Telephony transport (pipecat + Twilio/Exotel SIP: you pay only carrier rates, roughly $0.007/min, still no platform fee)
- [ ] Browser/WebRTC demo
- [ ] Persona pack (support agent, language tutor, order taker)

PRs welcome. The codebase is intentionally one file so you can read all of it in one sitting.

## FAQ

**Is this actually $0.00?**
Yes, within Groq's free-tier daily limits, which are generous for development and demos (and multiply with the rotation tricks above). STT is local compute and TTS is free. At commercial volume you would move the LLM to a paid tier and TTS to Azure, and you would still pay cents per hour, not per minute.

**Why not just use VAPI/Retell/Bland?**
If you need phone numbers today with zero setup, use them, they are good products. voicezero exists because most voice agent work (prototyping, demos, learning, internal tools) does not need a phone number, and because understanding the loop makes you better at using any platform.

**Does it speak languages other than English?**
Yes. The LLM mirrors the caller's language and edge-tts has voices in dozens of languages. The included `outbound_sales` persona conducts calls in Hinglish with an Indian English voice.

**Where do conversations go?**
Nowhere except `call_log.jsonl` on your machine (gitignored). Audio goes to Microsoft (TTS) and Groq (LLM text); STT never leaves your CPU.

---

Extracted from a production outbound sales agent built for real clients. If this saved you a per-minute bill, **star the repo**, it genuinely helps.

[![Star History Chart](https://api.star-history.com/svg?repos=shauryalowkeygotaura/voicezero&type=Date)](https://star-history.com/#shauryalowkeygotaura/voicezero&Date)

MIT licensed. Built by [Shaurya Vardhan Shandilya](https://github.com/shauryalowkeygotaura).
