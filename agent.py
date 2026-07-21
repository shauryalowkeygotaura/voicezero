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

import memory  # stdlib-only cross-call memory (SQLite keyed by caller_hash)
import tts_providers  # optional TTS backends + EN/HI language routing (stdlib at import)

HERE = Path(__file__).resolve().parent
DEFAULT_PERSONA = HERE / "personas" / "receptionist.json"
CALL_LOG = HERE / "call_log.jsonl"

# "small" is the sweet spot for Indian English / Hinglish on CPU int8: clearly
# better than "base" without much speed cost. Set WHISPER_MODEL=medium for more
# accuracy if the CPU can handle it (slower); tiny/base only for weak hardware.
STT_MODEL = os.getenv("WHISPER_MODEL", "small")  # tiny/base/small/medium (local backend)
# Per-utterance STT language. A persona sets its own "stt_lang" (e.g. the Hinglish
# caller uses "hi"); this env is only the default when a persona doesn't. Use
# "auto" to let Whisper detect per utterance. NOTE: forcing "en" on a Hindi call
# is the classic reason Hinglish STT comes out garbled — set "hi" or "auto".
WHISPER_LANG = os.getenv("WHISPER_LANG", "en")   # en | hi | auto | any ISO code
# STT backend ladder:
#   auto  (default) best AVAILABLE, each falling to the next on failure:
#         sarvam (if SARVAM_API_KEY) -> groq large-v3 (if a Groq key) -> local.
#         So a LIVE call (which already needs a Groq key) transcribes on Groq's
#         large-v3-turbo automatically — a big jump over local 'small' for Hindi/
#         accents — while --loopback (no key) stays fully on your CPU.
#   local faster-whisper on THIS machine. Free, offline, private. Set this to keep
#         audio fully on-device.
#   groq  Groq cloud Whisper only (production server, nothing local).
#   sarvam Sarvam Saarika only (best Indian/code-switch STT; needs SARVAM_API_KEY).
STT_BACKEND = os.getenv("STT_BACKEND", "auto")  # auto | local | groq | sarvam
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
SAMPLE_RATE = 16000

# Domain vocabulary fed to Whisper as an initial_prompt so it biases toward
# clinic terms and common Indian names instead of guessing generic words.
# Personas can extend this via a "hotwords" list in their JSON.
DEFAULT_HOTWORDS = (
    "appointment, doctor, dentist, clinic, root canal, cleaning, RCT, crowning, "
    "filling, extraction, checkup, Dr, Sharma, Patel, Gupta, Reddy, Khan, "
    "today, tomorrow, morning, evening, afternoon"
)

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
    # Extra STT vocabulary the persona wants Whisper biased toward (clinic name,
    # doctor names, plan names). Accept a list or a comma string; always store a
    # string, plus the persona name itself so the agent hears its own clinic.
    hw = p.get("hotwords", "")
    if isinstance(hw, (list, tuple)):
        hw = ", ".join(str(x) for x in hw)
    hotwords = ", ".join(part for part in (str(hw).strip(), str(p["name"]).strip()) if part)
    return {
        "name": p["name"],
        "system_prompt": fill_vars(p["system_prompt"], variables),
        "first_message": fill_vars(p["first_message"], variables),
        # Indian English voice by default so callers hear a natural local accent.
        # Kept for reference; the actual per-reply routing uses "tts" below.
        "voice": p.get("voice", "en-IN-NeerjaNeural"),
        # Per-language TTS voice profiles. Each agent reply is routed to the
        # profile for its detected language (tts_providers.pick_voice), so the
        # agent speaks English in an English voice and Hindi in a Hindi voice and
        # switches on the fly as the call flows. See _build_tts_map for the schema.
        "tts": _build_tts_map(p),
        # STT language for THIS persona's callers ("en" | "hi" | "auto"). Defaults
        # to the WHISPER_LANG env. A Hinglish persona should set "hi" (or "auto").
        "stt_lang": p.get("stt_lang", WHISPER_LANG),
        # Optional per-persona STT backend override ("auto"|"local"|"groq"|"sarvam").
        # Empty -> use the global STT_BACKEND env. Lets one voice agent stay fully
        # local/private while another uses cloud STT, without touching env vars.
        "stt_backend": p.get("stt_backend", ""),
        "rate": p.get("rate", "+5%"),
        "model": p.get("model", "llama-3.3-70b-versatile"),
        "temperature": p.get("temperature", 0.4),
        "max_tokens": p.get("max_tokens", 200),
        "tools": tools,
        "tool_speech": tool_speech,
        "hotwords": hotwords,
        "summary_prompt": p.get("summary_prompt", ""),
    }


# Free edge-tts fallback voice per language, so on-the-fly routing always has a
# guaranteed voice even when no optional backend (kokoro/sarvam) is installed.
_EDGE_DEFAULTS = {"en": "en-IN-NeerjaNeural", "hi": "hi-IN-SwaraNeural"}


def _build_tts_map(p: dict) -> dict:
    """Normalize a persona's TTS config into {"en": profile, "hi": profile, ...}
    where a profile is {provider, voice, edge_voice, rate}. Two input shapes:

      NEW (enables EN<->HI switching) — a "tts" block, one entry per language:
        "tts": {
          "en": {"provider": "kokoro", "voice": "af_heart", "edge_voice": "en-US-AriaNeural"},
          "hi": {"provider": "sarvam", "voice": "anushka",  "edge_voice": "hi-IN-SwaraNeural"}
        }
      LEGACY (single voice, no switching) — flat "tts_provider"/"tts_voice"/"voice"
        fields; every language routes to that one voice, i.e. old behavior intact.

    provider defaults to "edge" (always-there); if an optional provider is
    unavailable at call time, tts_to_wav falls back to edge_voice for that same
    language. So new personas gain switching and old personas keep working."""
    rate = p.get("rate", "+5%")
    block = p.get("tts")
    m: dict = {}
    if isinstance(block, dict):
        for lang, e in block.items():
            if not isinstance(e, dict):
                continue
            m[lang] = {
                "provider": e.get("provider", "edge"),
                "voice": e.get("voice", ""),
                "edge_voice": e.get("edge_voice", _EDGE_DEFAULTS.get(lang, "en-IN-NeerjaNeural")),
                "rate": e.get("rate", rate),
            }
    if not m:
        # Legacy single-voice persona: one profile used for every language.
        m["en"] = {
            "provider": p.get("tts_provider", "edge"),
            "voice": p.get("tts_voice", ""),
            "edge_voice": p.get("voice", "en-IN-NeerjaNeural"),
            "rate": rate,
        }
    elif "en" not in m:
        # Persona defined only non-English voices: reuse one as the English
        # fallback rather than dropping to an unrelated default.
        m["en"] = next(iter(m.values()))
    return m


# ── audio: TTS (edge-tts -> ffmpeg -> wav), STT (faster-whisper), playback ───

def tts_to_wav(text: str, voice: str, rate: str, wav_path: Path,
               provider: str = "edge", provider_voice: str = "") -> None:
    """Synthesize `text` to a mono WAV at `wav_path`. edge-tts is the default and
    the guaranteed fallback; `provider` can select an OPTIONAL backend (kokoro,
    sarvam — see tts_providers.py). If that backend's package/model/key is missing
    or the call fails, we silently drop back to edge-tts, so a turn is never lost
    to an un-installed extra. `voice` is always a valid edge voice (the fallback);
    `provider_voice` is the provider-specific voice name when a persona sets one."""
    if provider and provider != "edge":
        try:
            if tts_providers.synth(provider, text, provider_voice, rate, wav_path):
                return
        except Exception as e:
            print(f"  [warn] TTS provider {provider!r} unavailable "
                  f"({type(e).__name__}: {e}); falling back to edge-tts.", file=sys.stderr)
    _edge_to_wav(text, voice, rate, wav_path)


def speak(text: str, wav_path: Path, persona: dict) -> dict:
    """Route one agent utterance to the best voice for ITS language and synthesize
    it to wav_path. Detects en/hi, picks the persona's profile for that language,
    then hands off to tts_to_wav (which still edge-falls-back per language). This
    is the on-the-fly EN<->HI switch: consistent voice within an utterance, a
    different voice next turn if the language changed. Returns the chosen profile
    (with a "lang" key) for diagnostics."""
    profile = tts_providers.pick_voice(text, persona["tts"])
    tts_to_wav(text, profile["edge_voice"], profile["rate"], wav_path,
               profile["provider"], profile["voice"])
    return {**profile, "lang": tts_providers.detect_lang(text)}


def _edge_to_wav(text: str, voice: str, rate: str, wav_path: Path) -> None:
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


def _stt_chain(backend: str = "") -> list[str]:
    """Ordered STT backends to try for one utterance. `backend` (a persona's
    stt_backend) overrides the global STT_BACKEND env; empty -> the env. An
    explicit local/groq/sarvam is honored as-is (no surprise fallbacks on a
    server); 'auto' picks the best AVAILABLE and falls through on failure,
    mirroring the LLM model/key rotation: sarvam (if keyed) -> groq -> local."""
    b = (backend or STT_BACKEND or "auto").lower()
    if b in ("local", "groq", "sarvam"):
        return [b]
    chain = []
    if os.getenv("SARVAM_API_KEY", "").strip():
        chain.append("sarvam")
    if _groq_keys():
        chain.append("groq")
    chain.append("local")
    return chain


def stt(wav_path: Path, hotwords: str = "", lang: str | None = None,
        backend: str = "", strict: bool = False) -> str:
    # Domain biasing shared by every backend: seed the recognizer with clinic/name
    # vocabulary so it leans toward the words callers actually say.
    vocab = (DEFAULT_HOTWORDS + (", " + hotwords if hotwords else "")).strip(", ")
    lang = WHISPER_LANG if lang is None else lang
    last: Exception | None = None
    for name in _stt_chain(backend):
        try:
            return _STT_BACKENDS[name](wav_path, vocab, lang)
        except Exception as e:  # unavailable/rate-limited/network: try the next
            last = e
            continue
    # A live call must never crash on a transient STT hiccup: degrade to "no speech
    # heard" so the caller is simply re-prompted. Diagnostics pass strict=True so
    # --selftest/--loopback surface the real error instead.
    if strict:
        raise last or RuntimeError("no STT backend available")
    print(f"  [warn] STT failed ({type(last).__name__ if last else 'NoBackend'}: {last}); "
          "treating as no speech.", file=sys.stderr)
    return ""


def _whisper_lang(lang: str) -> str | None:
    """Map our lang tag to a Whisper `language` arg. 'auto'/'' -> None so Whisper
    detects the language itself instead of being forced into the wrong one."""
    lang = (lang or "").strip().lower()
    return None if lang in ("", "auto") else lang


def _transcribe_local(wav_path: Path, vocab: str, lang: str) -> str:
    """faster-whisper on this machine. Free, offline, no per-call cost, fully
    private. Best for dev/testing; needs your PC on, so not ideal for a live line."""
    global _stt_model
    if _stt_model is None:
        from faster_whisper import WhisperModel
        print(f"  [stt] loading faster-whisper '{STT_MODEL}' (first run downloads the model)...")
        _stt_model = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
    opts = dict(
        beam_size=5,                  # wider search than greedy default -> fewer errors
        temperature=0,                # deterministic; no random fallback decodes
        condition_on_previous_text=False,  # stops one bad turn cascading into the next
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),  # don't clip trailing words
        initial_prompt=vocab,
    )
    wl = _whisper_lang(lang)          # None => auto-detect (don't force a language)
    if wl:
        opts["language"] = wl
    # hotwords is a newer faster-whisper feature; guard it so older installs work.
    try:
        segments, _info = _stt_model.transcribe(str(wav_path), hotwords=vocab, **opts)
    except TypeError:
        segments, _info = _stt_model.transcribe(str(wav_path), **opts)
    return " ".join(s.text.strip() for s in segments).strip()


def _transcribe_groq(wav_path: Path, vocab: str, lang: str) -> str:
    """Cloud STT via Groq's hosted Whisper (large-v3-turbo by default) — clearly
    better on Hindi/Hinglish and accents than local 'small', and nearly free.
    Nothing runs locally, so the agent can live on a tiny always-on server. Reuses
    the same Groq keys and per-key failover as the LLM. Pennies per call."""
    from groq import Groq
    keys = _groq_keys()
    if not keys:
        raise RuntimeError("Groq STT needs a GROQ_API_KEY (free at console.groq.com/keys)")
    audio = wav_path.read_bytes()
    wl = _whisper_lang(lang)
    last = None
    for key in keys:
        try:
            kwargs = dict(
                file=(wav_path.name, audio),
                model=GROQ_STT_MODEL,
                prompt=vocab,                 # same domain biasing as the local path
                temperature=0,
                response_format="text",
            )
            if wl:
                kwargs["language"] = wl       # else let Whisper detect the language
            resp = Groq(api_key=key).audio.transcriptions.create(**kwargs)
            return (resp if isinstance(resp, str) else getattr(resp, "text", "")).strip()
        except Exception as e:  # rate limit, auth, network, API error: try the next key
            last = e
            continue
    raise last or RuntimeError("no usable GROQ key for STT")


def _transcribe_sarvam(wav_path: Path, vocab: str, lang: str) -> str:
    """Sarvam Saarika STT — best-in-class Indian-language and code-switched
    transcription, the STT twin of the sarvam TTS voice. Key-gated (SARVAM_API_KEY)
    and never a default; no package, just a multipart POST over stdlib urllib. Not
    free at scale, so it only runs when explicitly chosen or auto-picked with a key."""
    import json
    import urllib.request
    import uuid

    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("sarvam STT needs SARVAM_API_KEY")
    # Saarika wants a BCP-47 code, or "unknown" to auto-detect (handles code-switch).
    lang_code = {"hi": "hi-IN", "en": "en-IN"}.get(_whisper_lang(lang) or "", "unknown")
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []

    def _field(name: str, value: str) -> None:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8"))

    _field("model", os.getenv("SARVAM_STT_MODEL", "saarika:v2"))
    _field("language_code", lang_code)
    parts.append(
        (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
         f'filename="{wav_path.name}"\r\nContent-Type: audio/wav\r\n\r\n').encode("utf-8")
        + wav_path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        "https://api.sarvam.ai/speech-to-text", data=b"".join(parts),
        headers={"api-subscription-key": key,
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # non-200 raises -> next backend
        payload = json.loads(resp.read().decode("utf-8"))
    return (payload.get("transcript") or "").strip()


_STT_BACKENDS = {
    "local": _transcribe_local,
    "groq": _transcribe_groq,
    "sarvam": _transcribe_sarvam,
}


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

    def __init__(self, persona: dict, variables: dict, caller_hash: str = ""):
        self.p = persona
        self.variables = variables
        # A returning caller (recognized by caller_hash) gets their last context
        # folded into the opening line so the agent greets them by memory instead
        # of starting cold. First-time/unknown caller -> note is "" -> unchanged.
        self.caller_hash = caller_hash if memory._valid_hash(caller_hash) else ""
        first_message = persona["first_message"]
        if self.caller_hash:
            note = memory.greeting_note(self.caller_hash)
            if note:
                first_message = f"{note} {first_message}"
        self.first_message = first_message
        self.history: list[dict] = [{"role": "assistant", "content": first_message}]
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
        summary = self._summarize()
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "persona": self.p["name"],
            "variables": self.variables,
            "end_reason": self.end_reason,
            "events": [{k: v for k, v in e.items() if k != "t"} for e in self.events],
            "summary": summary,
            "transcript": [m for m in self.history if m["role"] in ("user", "assistant") and m.get("content")],
        }
        try:
            with open(CALL_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"  [warn] could not write call log: {e}", file=sys.stderr)
        # Persist this caller's last context so the NEXT call can greet them by
        # memory. Keyed by caller_hash; a no-op (never an exception) when there is
        # no recognized caller, so a missing number never breaks a finished call.
        if self.caller_hash:
            try:
                memory.store(
                    self.caller_hash,
                    summary=(summary.get("summary") or "") if isinstance(summary, dict) else "",
                    outcome=(summary.get("outcome") or "") if isinstance(summary, dict) else "",
                    extra={"persona": self.p["name"], "end_reason": self.end_reason},
                )
            except Exception as e:  # memory is best-effort; never fail the call on it
                print(f"  [warn] could not store caller memory ({type(e).__name__})", file=sys.stderr)
        return record


# ── modes ────────────────────────────────────────────────────────────────────

def run_text(call: Call) -> None:
    print(f"\nAGENT: {call.first_message}")
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
    speak(call.first_message, say, call.p)  # routed to the first message's language
    print(f"AGENT: {call.first_message}")
    play_wav(say)
    while not call.ended:
        utt = record_utterance()
        if utt is None:
            print("  [mic] no speech detected; ending call.")
            break
        heard = stt(utt, call.p["hotwords"], call.p["stt_lang"], call.p["stt_backend"])
        if not heard:
            continue
        print(f"YOU:   {heard}")
        reply = call.turn(heard)
        if reply:
            print(f"AGENT: {reply}")
            speak(reply, say, call.p)  # re-detects language each turn -> EN/HI switch
            play_wav(say)
    record = call.finish()
    s = record.get("summary") or {}
    outcome = f" | {s['outcome']}" if s.get("outcome") else ""
    print(f"\n-- call ended: {record['end_reason'] or 'silence'}{outcome} (log: {CALL_LOG.name})")


def run_loopback(persona: dict) -> None:
    """TTS -> STT round trip. Proves the audio stack with no API key."""
    tmp = Path(tempfile.gettempdir())
    line = "If you can read this back, the audio stack works end to end."
    t0 = time.time()
    wav = tmp / "voicezero_loopback.wav"
    v = speak(line, wav, persona)
    print(f"[1/2] TTS via {v['provider']} ({v['voice'] or v['edge_voice']}) "
          f"ok ({time.time()-t0:.1f}s, {wav.stat().st_size} bytes) at $0.00")
    print(f"[2/2] STT it back via {persona['stt_backend'] or STT_BACKEND} backend...")
    t0 = time.time()
    heard = stt(wav, lang=persona["stt_lang"], backend=persona["stt_backend"], strict=True)
    if not heard:
        raise SystemExit("LOOPBACK FAIL: STT returned empty text")
    print(f"      heard: {heard!r} ({time.time()-t0:.1f}s) at $0.00")
    print("\nLOOPBACK PASS. Audio stack verified, no API key used.")


def run_selftest(call: Call) -> None:
    """Headless end to end: proves TTS, STT, LLM and tools with no mic."""
    tmp = Path(tempfile.gettempdir())
    user_line = "Hi, what exactly can you help me with?"
    t0 = time.time()
    wav = tmp / "voicezero_selftest_user.wav"
    v = speak(user_line, wav, call.p)
    print(f"[1/4] TTS user line via {v['provider']} ({v['voice'] or v['edge_voice']}) "
          f"ok ({time.time()-t0:.1f}s, {wav.stat().st_size} bytes) at $0.00")

    print(f"[2/4] STT it back via {call.p['stt_backend'] or STT_BACKEND} backend...")
    t0 = time.time()
    heard = stt(wav, call.p["hotwords"], call.p["stt_lang"], call.p["stt_backend"], strict=True)
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
    v = speak(reply or "Thanks for calling!", out, call.p)
    print(f"      ok via {v['provider']} [{v['lang']}] ({out})")

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
    ap.add_argument("--caller-number", default="",
                    help="caller phone number; hashed (never stored raw) to recognize "
                         "a returning caller and persist this call's context for next time")
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

    # Hash the number (never store it raw) to recognize a returning caller and to
    # key this call's stored context. Empty/absent number -> "" -> memory inert.
    caller_hash = memory.hash_caller_number(args.caller_number.strip())
    call = Call(persona, variables, caller_hash=caller_hash)
    if args.selftest:
        run_selftest(call)
    elif args.voice:
        run_voice(call)
    else:
        run_text(call)


if __name__ == "__main__":
    main()
