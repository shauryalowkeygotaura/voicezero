"""
voicezero: a production-pattern voice agent at $0.00 per minute.

Voice AI platforms charge $0.05 to $0.30 per minute to run this exact loop:

    mic -> STT -> LLM (+ tools) -> TTS -> speaker

voicezero runs the same loop locally with free components:

    STT   faster-whisper, local CPU                          free
    LLM   Groq free tier (llama 3.3 70b and friends)         free
    TTS   edge-tts, Microsoft neural voices                  free
    I/O   sounddevice mic capture + native playback          free

Modes:
  --text       type to the agent in the console (no audio path)
  --voice      full mic + speakers conversation
  --selftest   headless check: TTS a line -> STT it back -> one LLM turn ->
               TTS the reply. Proves the whole stack with no microphone.
  --loopback   TTS -> STT round trip only. Needs no API key at all.

Usage:
  python agent.py --selftest
  python agent.py --voice --persona personas/receptionist.json
  python agent.py --text  --persona personas/outbound_sales.json --var lead_name=Rohan

Requires GROQ_API_KEY (or GROQ_API_KEYS, comma separated) in the environment,
and ffmpeg on PATH. See README.md for the 60 second setup.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PERSONA = HERE / "personas" / "receptionist.json"
CALL_LOG = HERE / "call_log.jsonl"

STT_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny/base/small/medium
SAMPLE_RATE = 16000

# Each model has its OWN per-org daily token pool on Groq, so chaining models
# multiplies free headroom on a single key. GROQ_API_KEYS (comma separated,
# one key per Groq ACCOUNT; same-account keys share quota) multiplies further.
MODEL_CHAIN_EXTRA = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


# ── env + LLM: (model x key) rotation ────────────────────────────────────────

def _load_dotenv() -> None:
    """Tiny .env loader, no dependency: KEY=VALUE lines, real env always wins."""
    env_path = HERE / ".env"
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except OSError:
        pass


def _groq_keys() -> list[str]:
    multi = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",")]
    keys = [k for k in multi if k] or [os.environ.get("GROQ_API_KEY", "").strip()]
    return [k for k in keys if k]


def chat_with_rotation(preferred_model: str, **kwargs):
    """(model x key) failover, model-major. Returns (model, response).
    Rate limited on one pool? The call silently lands on the next."""
    import groq
    from groq import Groq

    chain = [preferred_model] + [m for m in MODEL_CHAIN_EXTRA if m != preferred_model]
    last = None
    for model in chain:
        for key in _groq_keys():
            try:
                return model, Groq(api_key=key).chat.completions.create(model=model, **kwargs)
            except (groq.RateLimitError, groq.AuthenticationError) as e:
                last = e
                continue
    raise last or RuntimeError("no usable GROQ key/model combination")


# ── persona loading ──────────────────────────────────────────────────────────

END_CALL_TOOL = {
    "type": "function",
    "function": {
        "name": "end_call",
        "description": "Hang up the call. Use after your warm closing line.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "one short phrase, e.g. 'booked', 'not interested'"}
        }},
    },
}


def _sanitize_value(v: str) -> str:
    """Caller-supplied values are DATA, never template syntax or instructions.
    Strip structure characters and cap length before substitution."""
    v = str(v)
    v = re.sub(r"[{}%#\[\]<>`]", "", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v[:80]


def fill_vars(text: str, variables: dict) -> str:
    """{{var}} substitution with sanitized values; unknown vars become blank."""
    for k, v in variables.items():
        text = text.replace("{{" + k + "}}", _sanitize_value(v))
    return re.sub(r"\{\{[^}]+\}\}", "", text)


def load_persona(path: Path, variables: dict) -> dict:
    if not path.is_file():
        raise SystemExit(f"Persona file not found: {path}")
    try:
        p = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"Could not load persona {path}: {e}")
    for field in ("name", "system_prompt", "first_message"):
        if not p.get(field):
            raise SystemExit(f"Persona missing required field: {field}")
    for t in p.get("tools", []):
        if not (t.get("name") and t.get("description")):
            raise SystemExit(f"Persona tool missing 'name' or 'description': {t}")
    tools = [
        {"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t.get("parameters", {"type": "object", "properties": {}})}}
        for t in p.get("tools", [])
    ]
    tools.append(END_CALL_TOOL)
    # name -> fixed spoken line for tool-calls-only turns, so the listener
    # never gets dead air while a tool fires (the classic silent-turn bug).
    tool_speech = {t["name"]: t["speech_line"] for t in p.get("tools", []) if t.get("speech_line")}
    return {
        "name": p["name"],
        "system_prompt": fill_vars(p["system_prompt"], variables),
        "first_message": fill_vars(p["first_message"], variables),
        "voice": p.get("voice", "en-US-AriaNeural"),
        "rate": p.get("rate", "+5%"),
        "model": p.get("model", "llama-3.3-70b-versatile"),
        "temperature": p.get("temperature", 0.4),
        "max_tokens": p.get("max_tokens", 200),
        "tools": tools,
        "tool_speech": tool_speech,
        "summary_prompt": p.get("summary_prompt", ""),
    }


# ── audio: TTS (edge-tts -> ffmpeg -> wav), STT (faster-whisper), playback ───

def tts_to_wav(text: str, voice: str, rate: str, wav_path: Path) -> None:
    import edge_tts

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH. Install it: https://ffmpeg.org/download.html")
    mp3_path = wav_path.with_suffix(".mp3")
    try:
        asyncio.run(edge_tts.Communicate(text, voice, rate=rate).save(str(mp3_path)))
    except Exception as e:
        raise SystemExit(f"edge-tts failed ({type(e).__name__}: {e}). "
                         "Check your network and that the voice name is valid.")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3_path),
             "-ar", "24000", "-ac", "1", str(wav_path)],
            check=True, timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"ffmpeg conversion failed: {e}")
    finally:
        mp3_path.unlink(missing_ok=True)


def play_wav(path: Path) -> None:
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    elif sys.platform == "darwin":
        subprocess.run(["afplay", str(path)], check=False)
    else:
        for cmd in (["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]):
            if shutil.which(cmd[0]):
                subprocess.run(cmd + [str(path)], check=False)
                return
        print("  [warn] no audio player found (tried aplay, ffplay)", file=sys.stderr)


_stt_model = None


def stt(wav_path: Path) -> str:
    global _stt_model
    if _stt_model is None:
        from faster_whisper import WhisperModel
        print(f"  [stt] loading faster-whisper '{STT_MODEL}' (first run downloads the model)...")
        _stt_model = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
    segments, _info = _stt_model.transcribe(str(wav_path), vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def record_utterance(max_seconds: int = 15, silence_after: float = 1.2) -> Path | None:
    """Energy-VAD mic capture: wait for speech, stop after trailing silence."""
    import numpy as np
    import sounddevice as sd

    block = int(SAMPLE_RATE * 0.1)  # 100 ms chunks
    threshold = 350                  # int16 RMS; raise if a noisy room self-triggers
    frames: list[bytes] = []
    started = False
    silent_blocks = 0
    need_silent = int(silence_after / 0.1)

    print("  [mic] listening... (speak, pause to finish)")
    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=block) as stream:
            for _ in range(int(max_seconds / 0.1)):
                data, _overflow = stream.read(block)
                buf = bytes(data)
                rms = float(np.sqrt(np.mean(np.frombuffer(buf, dtype=np.int16).astype(np.float64) ** 2)))
                if rms > threshold:
                    started = True
                    silent_blocks = 0
                elif started:
                    silent_blocks += 1
                if started:
                    frames.append(buf)
                    if silent_blocks >= need_silent:
                        break
    except sd.PortAudioError as e:
        print(f"  [mic] audio device error: {e}", file=sys.stderr)
        return None
    if not started:
        return None
    path = Path(tempfile.gettempdir()) / "voicezero_utterance.wav"
    try:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(b"".join(frames))
    except OSError as e:
        print(f"  [mic] could not write capture: {e}", file=sys.stderr)
        return None
    return path


# ── the call loop ────────────────────────────────────────────────────────────

class Call:
    """One conversation. Async-tool semantics like the hosted platforms: tools
    return {'status':'queued'} instantly, then a follow-up generation speaks."""

    def __init__(self, persona: dict, variables: dict):
        self.p = persona
        self.variables = variables
        self.history: list[dict] = [{"role": "assistant", "content": persona["first_message"]}]
        self.events: list[dict] = []
        self.ended = False
        self.end_reason = ""

    # Tool handlers. This is where a hosted platform's webhook becomes plain
    # Python: add real side effects here (book the slot, send the message).
    def _handle_tool(self, name: str, args: dict) -> dict:
        self.events.append({"tool": name, "args": args, "t": time.time()})
        if name == "end_call":
            self.ended = True
            self.end_reason = args.get("reason", "")
        return {"status": "queued"}

    def _generate(self) -> "tuple[str, list]":
        _model, resp = chat_with_rotation(
            self.p["model"],
            temperature=self.p["temperature"],
            max_tokens=self.p["max_tokens"],
            tools=self.p["tools"],
            messages=[{"role": "system", "content": self.p["system_prompt"]}] + self.history,
        )
        if not resp.choices:
            return "", []
        m = resp.choices[0].message
        return (m.content or "").strip(), list(m.tool_calls or [])

    def turn(self, user_text: str) -> str:
        """One user turn -> agent speech, running the tool loop."""
        self.history.append({"role": "user", "content": user_text})
        speech_parts: list[str] = []
        fired_tools: list[str] = []
        for _hop in range(3):  # text -> tools -> follow-up speech; hard cap
            try:
                text, tool_calls = self._generate()
            except Exception as e:
                # All (model x key) combinations failed mid-call. End gracefully
                # instead of crashing while the caller is on the line.
                print(f"  [warn] LLM unavailable ({type(e).__name__}: {e}); ending call.",
                      file=sys.stderr)
                self.ended = True
                self.end_reason = "llm_unavailable"
                text = "Sorry, I'm having a technical issue on my end. I'll have someone follow up with you."
                self.history.append({"role": "assistant", "content": text})
                speech_parts.append(text)
                break
            if text:
                speech_parts.append(text)
            if not tool_calls:
                self.history.append({"role": "assistant", "content": text})
                break
            self.history.append({
                "role": "assistant", "content": text,
                "tool_calls": [{"id": t.id, "type": "function", "function": {
                    "name": t.function.name, "arguments": t.function.arguments}} for t in tool_calls],
            })
            for t in tool_calls:
                try:
                    args = json.loads(t.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._handle_tool(t.function.name, args)
                fired_tools.append(t.function.name)
                self.history.append({"role": "tool", "tool_call_id": t.id, "content": json.dumps(result)})
            if self.ended:
                break
        speech = " ".join(p for p in speech_parts if p).strip()
        if not speech and fired_tools:
            # Tool-calls-only turn: speak the tools' fixed lines instead of
            # leaving silence.
            speech = " ".join(self.p["tool_speech"][n] for n in fired_tools
                              if n in self.p["tool_speech"]).strip()
            if speech:
                self.history.append({"role": "assistant", "content": speech})
        return speech

    def _summarize(self) -> dict:
        """Out-of-band post-call summary: the talking model never emits
        structured data (that causes speech-less turns); a separate cheap model
        reads the finished transcript instead. llama-3.1-8b-instant has its own
        Groq daily pool, so this works even when the 70b budget is spent."""
        convo = "\n".join(
            f"{'AGENT' if m['role'] == 'assistant' else 'USER'}: {m['content']}"
            for m in self.history if m["role"] in ("user", "assistant") and m.get("content")
        )
        if not convo:
            return {}
        instruction = self.p["summary_prompt"] or (
            "Summarize this voice call. Return ONLY a JSON object with keys: "
            "outcome (one short phrase), summary (max 2 sentences), "
            "follow_up_needed (true or false).")
        try:
            _model, resp = chat_with_rotation(
                "llama-3.1-8b-instant",
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": instruction},
                          {"role": "user", "content": convo}],
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:
            print(f"  [warn] summary failed ({type(e).__name__})", file=sys.stderr)
            return {}

    def finish(self) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "persona": self.p["name"],
            "variables": self.variables,
            "end_reason": self.end_reason,
            "events": [{k: v for k, v in e.items() if k != "t"} for e in self.events],
            "summary": self._summarize(),
            "transcript": [m for m in self.history if m["role"] in ("user", "assistant") and m.get("content")],
        }
        try:
            with open(CALL_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"  [warn] could not write call log: {e}", file=sys.stderr)
        return record


# ── modes ────────────────────────────────────────────────────────────────────

def run_text(call: Call) -> None:
    print(f"\nAGENT: {call.p['first_message']}")
    print("(type your replies; /quit to end)\n")
    while not call.ended:
        try:
            user = input("YOU:   ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user == "/quit":
            break
        reply = call.turn(user)
        print(f"AGENT: {reply or '(tool action only)'}")
    record = call.finish()
    s = record.get("summary") or {}
    outcome = f" | {s['outcome']}" if s.get("outcome") else ""
    print(f"\n-- call ended: {record['end_reason'] or 'user quit'}{outcome} (log: {CALL_LOG.name})")


def run_voice(call: Call) -> None:
    tmp = Path(tempfile.gettempdir())
    # Warm the STT model BEFORE the call starts, or the first turn stalls.
    silence = tmp / "voicezero_warmup.wav"
    try:
        with wave.open(str(silence), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(b"\x00\x00" * SAMPLE_RATE)  # 1s of silence
        stt(silence)
    except Exception as e:
        print(f"  [warn] STT warmup skipped ({e}); first turn may be slow.", file=sys.stderr)
    finally:
        silence.unlink(missing_ok=True)
    say = tmp / "voicezero_agent_say.wav"
    tts_to_wav(call.p["first_message"], call.p["voice"], call.p["rate"], say)
    print(f"AGENT: {call.p['first_message']}")
    play_wav(say)
    while not call.ended:
        utt = record_utterance()
        if utt is None:
            print("  [mic] no speech detected; ending call.")
            break
        heard = stt(utt)
        if not heard:
            continue
        print(f"YOU:   {heard}")
        reply = call.turn(heard)
        if reply:
            print(f"AGENT: {reply}")
            tts_to_wav(reply, call.p["voice"], call.p["rate"], say)
            play_wav(say)
    record = call.finish()
    s = record.get("summary") or {}
    outcome = f" | {s['outcome']}" if s.get("outcome") else ""
    print(f"\n-- call ended: {record['end_reason'] or 'silence'}{outcome} (log: {CALL_LOG.name})")


def run_loopback(persona: dict) -> None:
    """TTS -> STT round trip. Proves the audio stack with no API key."""
    tmp = Path(tempfile.gettempdir())
    line = "If you can read this back, the audio stack works end to end."
    print(f"[1/2] TTS via edge-tts ({persona['voice']})...")
    t0 = time.time()
    wav = tmp / "voicezero_loopback.wav"
    tts_to_wav(line, persona["voice"], persona["rate"], wav)
    print(f"      ok ({time.time()-t0:.1f}s, {wav.stat().st_size} bytes) at $0.00")
    print(f"[2/2] STT it back via faster-whisper '{STT_MODEL}'...")
    t0 = time.time()
    heard = stt(wav)
    if not heard:
        raise SystemExit("LOOPBACK FAIL: STT returned empty text")
    print(f"      heard: {heard!r} ({time.time()-t0:.1f}s) at $0.00")
    print("\nLOOPBACK PASS. Audio stack verified, no API key used.")


def run_selftest(call: Call) -> None:
    """Headless end to end: proves TTS, STT, LLM and tools with no mic."""
    tmp = Path(tempfile.gettempdir())
    user_line = "Hi, what exactly can you help me with?"
    print(f"[1/4] TTS user line via edge-tts ({call.p['voice']})...")
    t0 = time.time()
    wav = tmp / "voicezero_selftest_user.wav"
    tts_to_wav(user_line, call.p["voice"], call.p["rate"], wav)
    print(f"      ok ({time.time()-t0:.1f}s, {wav.stat().st_size} bytes) at $0.00")

    print(f"[2/4] STT it back via faster-whisper '{STT_MODEL}'...")
    t0 = time.time()
    heard = stt(wav)
    if not heard:
        raise SystemExit("SELFTEST FAIL: STT returned empty text")
    print(f"      heard: {heard!r} ({time.time()-t0:.1f}s) at $0.00")

    print("[3/4] LLM turn via Groq (tools live)...")
    t0 = time.time()
    reply = call.turn(heard)
    if call.end_reason == "llm_unavailable":
        raise SystemExit("SELFTEST FAIL: could not reach Groq on any (model x key) combination. "
                         "Check your key and network.")
    if not (reply or call.events):
        raise SystemExit("SELFTEST FAIL: LLM produced neither speech nor tool calls")
    print(f"      agent: {reply!r}")
    print(f"      tools fired: {[e['tool'] for e in call.events]} ({time.time()-t0:.1f}s)")

    print("[4/4] TTS the reply...")
    out = tmp / "voicezero_selftest_agent.wav"
    tts_to_wav(reply or "Thanks for calling!", call.p["voice"], call.p["rate"], out)
    print(f"      ok ({out})")

    call.finish()
    print("\nSELFTEST PASS. Full voice loop verified at $0.00 per minute.")


def main():
    _load_dotenv()
    ap = argparse.ArgumentParser(description="voicezero: a voice agent at $0.00 per minute.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", action="store_true", help="console conversation, no audio")
    mode.add_argument("--voice", action="store_true", help="mic + speakers conversation")
    mode.add_argument("--selftest", action="store_true", help="headless full-stack check")
    mode.add_argument("--loopback", action="store_true", help="TTS->STT check, no API key needed")
    ap.add_argument("--persona", default=str(DEFAULT_PERSONA), help="path to a persona JSON")
    ap.add_argument("--var", action="append", default=[], metavar="KEY=VALUE",
                    help="fill a {{variable}} in the persona (repeatable)")
    args = ap.parse_args()

    variables = {}
    for pair in args.var:
        k, _, v = pair.partition("=")
        if k:
            variables[k.strip()] = v.strip()

    persona = load_persona(Path(args.persona), variables)

    if args.loopback:
        run_loopback(persona)
        return

    if not _groq_keys():
        raise SystemExit("No Groq key found. Set GROQ_API_KEY (free at console.groq.com/keys), "
                         "or run --loopback to test audio without one.")

    call = Call(persona, variables)
    if args.selftest:
        run_selftest(call)
    elif args.voice:
        run_voice(call)
    else:
        run_text(call)


if __name__ == "__main__":
    main()
