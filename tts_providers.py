"""Optional TTS backends for voicezero — the pluggable half of a two-tier design.

agent.py's edge-tts path is the ALWAYS-THERE default: zero config, zero extra
dependencies, no API key. Everything in this module is an OPTIONAL upgrade you
point a persona at with a "tts_provider" field. The contract mirrors the LLM
model chain in agent.py: try the chosen backend, and if its package/model/key is
missing or the call fails, the caller silently falls back to edge-tts. A turn is
never lost to an un-installed optional backend.

Providers
  kokoro  Fully local, Apache-2.0, CPU-friendly (~0.5 RTF on a normal laptop).
          The best FREE quality bump for English; also speaks Hindi (experimental
          hf_/hm_ voices). Stays $0 forever and runs offline once the ~310 MB
          model is cached. Enable with:  pip install kokoro-onnx
  sarvam  Cloud REST API (pure stdlib urllib, no package). Best-in-class Hindi and
          Indian-language voices — the natural pick for the Hinglish
          outbound_sales persona. NOT free at scale (₹1000 trial credits, then
          paid), so it is never a default and never a hard requirement: it turns
          on only when SARVAM_API_KEY is set, and silently no-ops to edge-tts
          otherwise.
  fish    Cloud REST API (stdlib urllib). The only backend here that can speak in
          a CLONED voice: point a persona's tts_voice at a Fish Audio model id
          and the agent talks in that voice. Metered and NOT free at scale (the
          standing free plan is ~8k credits, roughly 7 minutes of audio a month),
          so like sarvam it is key-gated on FISH_API_KEY and no-ops to edge-tts
          when unset. A mid-run 402 (balance gone) latches Fish off for the rest
          of the process after one warning, so the voice swaps to edge once and
          audibly rather than flapping turn by turn. Still not the right default
          for an unattended calling campaign, for that same reason.
          Clone a voice first with Code/fish-voice/fish_voice.py, whose
          speak_many() checkpoints and resumes a batch across a 402.

Each synth writes a mono 16-bit PCM WAV to wav_path. Whichever sample rate the
backend emits, downstream faster-whisper and native playback both accept it.
"""
from __future__ import annotations

import base64
import os
import re
import time
import wave
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

# Where kokoro model assets are cached (override with VOICEZERO_CACHE). Kept out
# of the repo so the ~310 MB download is never committed and is shared across runs.
CACHE = Path(os.getenv("VOICEZERO_CACHE", str(Path.home() / ".cache" / "voicezero")))

_KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/kokoro-v1.0.onnx"
)
_KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/voices-v1.0.bin"
)


def synth(provider: str, text: str, voice: str, rate: str, wav_path: Path) -> bool:
    """Route to an optional backend. Returns True if wav_path was written, or
    False to tell the caller to fall back to edge-tts (unknown provider, or a
    cloud provider that isn't configured). Any real error is raised so agent.py's
    try/except can log one line before it, too, falls back."""
    fn = _PROVIDERS.get((provider or "").lower())
    if fn is None:
        return False
    return fn(text, voice, rate, wav_path)


# ── helpers ──────────────────────────────────────────────────────────────────

def _rate_to_speed(rate: str) -> float:
    """Map an edge-style rate string ('+5%', '-10%') to a speed multiplier."""
    try:
        pct = int(str(rate).strip().rstrip("%"))
    except (ValueError, TypeError):
        return 1.0
    return max(0.5, min(2.0, 1.0 + pct / 100.0))


def _write_wav_int16(wav_path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)


# ── language routing: switch EN <-> HI voices on the fly, per reply ───────────
#
# The agent should sound English when it speaks English and Hindi when it speaks
# Hindi, switching as the conversation flows. We route WHOLE utterances (not
# mid-sentence): one consistent voice per reply, a different voice next turn if
# the language changed. Mid-sentence engine-swapping would flip timbre/pitch
# inside one sentence and sound broken; whole-utterance routing sounds natural,
# and a Hindi-capable voice speaks the English words embedded in a Hinglish reply
# just fine (the reverse is not true), so "when in doubt, route to Hindi".

# High-frequency ROMANIZED Hindi tokens. Generic langid libraries misclassify
# romanized Hindi because it is Latin script; a tiny function-word lexicon beats
# them here and needs no dependency. We only need "is this reply mostly Hindi?".
# Deliberately EXCLUDES romanizations that collide with common English words
# (is, us, to, me, the, main, sir, madam, na, par) — those caused English replies
# like "this is Maya" / "works for me" to be misrouted to a Hindi voice. True
# Hinglish still scores high on the distinctly-Hindi tokens that remain.
_HINDI_ROMAN = frozenset((
    "hai hain ho hoon hu tha thi ka ki ke ko kaa se mein mujhe muje "
    "aap aapko aapki aapse tum tumhe tumhein hum humein hamein mera meri mere "
    "apna apni nahi nahin haan haa ji kya kyun kyu kaise kaisa kab kahan kaun "
    "kar karo karna karenge karta karti raha rahi rahe gaya gayi gaye diya "
    "liya acha accha achha theek thik bilkul sahi bas abhi phir aur lekin "
    "magar matlab yaar bhai namaste namaskar dhanyavaad shukriya kripya "
    "chahiye chahta chahti sakta sakti sakte hoga hogi paisa rupaye rupay baat "
    "wala wali wale kuch sab yeh ye woh wo jo toh bhi milega chaliye"
).split())


def detect_lang(text: str) -> str:
    """Return 'hi' or 'en' for TTS routing. Any Devanagari char -> Hindi. Else a
    romanized-Hindi function-word density check, biased toward 'hi' on mixed input
    on purpose (see the note above)."""
    if any("ऀ" <= c <= "ॿ" for c in text):
        return "hi"
    toks = re.findall(r"[a-z]+", text.lower())
    if not toks:
        return "en"
    hits = sum(t in _HINDI_ROMAN for t in toks)
    return "hi" if hits / len(toks) >= 0.12 else "en"


def pick_voice(text: str, voice_map: dict) -> dict:
    """Choose the voice profile ({provider, voice, edge_voice, rate}) for the
    detected language of `text`. `voice_map` carries 'en'/'hi' entries built in
    agent.load_persona. Always resolves to a profile ('en' then any) so routing
    never returns None."""
    lang = detect_lang(text)
    return (voice_map.get(lang) or voice_map.get("en")
            or next(iter(voice_map.values())))


# ── kokoro (local, free, Apache-2.0) ─────────────────────────────────────────

_kokoro_engine = None


def _ensure_kokoro_files() -> tuple[Path, Path]:
    """Return (model, voices) paths, downloading them to CACHE on first use.
    Point KOKORO_MODEL_PATH / KOKORO_VOICES_PATH at existing files to skip this."""
    import urllib.request

    model = Path(os.getenv("KOKORO_MODEL_PATH", str(CACHE / "kokoro-v1.0.onnx")))
    voices = Path(os.getenv("KOKORO_VOICES_PATH", str(CACHE / "voices-v1.0.bin")))
    CACHE.mkdir(parents=True, exist_ok=True)
    for path, url in ((model, _KOKORO_MODEL_URL), (voices, _KOKORO_VOICES_URL)):
        if not path.is_file():
            print(f"  [tts] downloading kokoro asset {path.name} (first run only)...")
            urllib.request.urlretrieve(url, str(path))  # raises -> caller falls back
    return model, voices


def _kokoro(text: str, voice: str, rate: str, wav_path: Path) -> bool:
    """kokoro-82M via onnxruntime, entirely on CPU. Voice picks the language:
    hf_*/hm_* speak Hindi, af_*/am_*/bf_*/bm_* speak English."""
    global _kokoro_engine
    import numpy as np

    if _kokoro_engine is None:
        from kokoro_onnx import Kokoro  # optional dep; ImportError -> edge fallback

        model_path, voices_path = _ensure_kokoro_files()
        _kokoro_engine = Kokoro(str(model_path), str(voices_path))

    voice = voice or os.getenv("KOKORO_VOICE", "af_heart")
    lang = os.getenv("KOKORO_LANG") or ("hi" if voice.startswith(("hf_", "hm_")) else "en-us")
    samples, sample_rate = _kokoro_engine.create(
        text, voice=voice, speed=_rate_to_speed(rate), lang=lang
    )
    pcm = (np.clip(np.asarray(samples), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    _write_wav_int16(wav_path, pcm, int(sample_rate))
    return True


# ── sarvam (cloud, premium Hindi, key-gated) ─────────────────────────────────

def _sarvam(text: str, voice: str, rate: str, wav_path: Path) -> bool:
    """Sarvam Bulbul TTS over REST. Best Hindi quality; enabled only when
    SARVAM_API_KEY is set (no key -> return False -> edge fallback). Language,
    model and default speaker are env-overridable so a persona's tts_voice just
    picks the speaker."""
    import json
    import urllib.request

    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        return False  # not configured: let the caller use edge-tts

    body = {
        "text": text[:2500],  # bulbul hard cap
        "target_language_code": os.getenv("SARVAM_LANG", "hi-IN"),
        "model": os.getenv("SARVAM_MODEL", "bulbul:v2"),
        "speech_sample_rate": 24000,
    }
    speaker = voice or os.getenv("SARVAM_SPEAKER", "").strip()
    if speaker:
        body["speaker"] = speaker

    req = urllib.request.Request(
        "https://api.sarvam.ai/text-to-speech",
        data=json.dumps(body).encode("utf-8"),
        headers={"api-subscription-key": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # non-200 raises -> fallback
        payload = json.loads(resp.read().decode("utf-8"))
    audios = payload.get("audios") or []
    if not audios:
        return False
    # Sarvam returns a complete base64-encoded WAV at the requested sample rate.
    wav_path.write_bytes(base64.b64decode(audios[0]))
    return True


# ── fish audio (cloud, cloned voices, key-gated) ─────────────────────────────

def _fish(text: str, voice: str, rate: str, wav_path: Path) -> bool:
    """Fish Audio TTS over REST. The one backend that speaks in a voice cloned
    from a real person: `voice` is a Fish model id (the `_id` returned when you
    clone), falling back to FISH_VOICE_ID so a persona can stay voice-agnostic.

    Enabled only when FISH_API_KEY is set (no key -> return False -> edge
    fallback). Asks for `format: wav` so the caller gets the same PCM WAV every
    other backend here writes, and `latency: low` because this runs inside a
    live call turn, where responsiveness beats the last few percent of quality.

    With no voice id resolved it still works, just in Fish's stock model voice
    rather than a clone.

    402 (out of credit) latches the provider off for the process; 429 buys one
    short Retry-After wait and then gives the turn to edge. See the comment at
    the request below for why those two go opposite ways."""
    import json
    import urllib.request

    import urllib.error

    if _fish_quota_spent():
        # An earlier turn already got a 402. The balance does not refill during a
        # call, so re-asking every turn just adds a round trip to the latency
        # budget before falling back to edge anyway.
        return False

    key = (os.getenv("FISH_API_KEY") or os.getenv("FISH_AUDIO_API_KEY") or "").strip()
    if not key:
        return False  # not configured: let the caller use edge-tts
    if not text.strip():
        return False  # nothing to say: a blank turn would 422, not fall back usefully

    body = {
        # Fish documents no hard text cap, unlike sarvam's 2500. This is a
        # defensive bound only: personas cap max_tokens at ~150, so a reply that
        # reaches it means something upstream ran away, and truncating keeps the
        # call alive instead of 422-ing the turn.
        "text": text[:5000],
        "format": "wav",
        "latency": "low",
        "normalize": True,
        # prosody.speed is a documented multiplier clamped to 0.5-2.0, which is
        # exactly the range _rate_to_speed already returns.
        "prosody": {"speed": _rate_to_speed(rate)},
    }
    reference = (voice or os.getenv("FISH_VOICE_ID", "")).strip()
    if reference:
        body["reference_id"] = reference

    req = urllib.request.Request(
        "https://api.fish.audio/v1/tts",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # s2.1-pro is the current top model; override when the free window
            # closes and you drop to s2.1-pro-free.
            "model": os.getenv("FISH_MODEL", "s2.1-pro"),
        },
    )
    # 402 and 429 are the two metered-API failures worth telling apart, and a
    # live call turn needs the opposite treatment from fish-voice's batch:
    #
    #   402  the balance is gone. There is no batch to checkpoint here (one turn,
    #        one clip, nothing owed), so the equivalent of "stop and resume" is to
    #        LATCH: mark Fish spent for the rest of the process and hand this and
    #        every later turn to edge-tts. Raised once so agent.py logs a single
    #        line; silent from then on. Code/fish-voice/fish_voice.py speak_many()
    #        is the batch path that checkpoints and resumes.
    #
    #   429  transient. Worth exactly one short wait, because the caller is a
    #        human holding a phone: Retry-After is honoured but clamped to
    #        _FISH_RETRY_CEILING, and anything longer is abandoned to edge for
    #        this turn only, with no latch.
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                audio = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 402:
                _FISH_STATE["quota_spent"] = True
                raise RuntimeError(
                    "Fish Audio is out of credit (HTTP 402); disabling it for the "
                    "rest of this process and using edge-tts. Top up at fish.audio "
                    "or unset FISH_API_KEY to silence this."
                ) from e
            if e.code == 429 and attempt == 1:
                wait = _fish_retry_after(e.headers)
                if wait is not None and wait <= _FISH_RETRY_CEILING:
                    time.sleep(wait)
                    continue
                return False  # too long to hold a live turn: fall back, no latch
            raise
    if not audio:
        return False
    # Fish returns a complete WAV container when format=wav.
    wav_path.write_bytes(audio)
    return True


# Longest a live turn will sit on a Fish rate limit before handing the line to
# edge-tts. Beyond a couple of seconds the caller hears dead air, which costs
# more than the cloned voice is worth for one sentence.
_FISH_RETRY_CEILING = 2.0

# Process-local latch. A dict rather than a bare global so the flag can be
# flipped from inside _fish without a `global` statement, and reset in tests.
_FISH_STATE = {"quota_spent": False}


def _fish_quota_spent() -> bool:
    """True once a 402 has been seen this process."""
    return bool(_FISH_STATE["quota_spent"])


def _fish_reset_quota() -> None:
    """Re-enable Fish after a 402. For tests and for a long-lived host that
    knows the balance was topped up."""
    _FISH_STATE["quota_spent"] = False


def _fish_retry_after(headers) -> float | None:
    """Retry-After in seconds, or None when the header is missing.

    Accepts both legal forms (delta-seconds and HTTP-date). An unparseable
    value means "the server said wait but we cannot read how long", which is
    treated as 0.0 (retry now) rather than None (no guidance)."""
    if headers is None:
        return None
    raw = (headers.get("Retry-After") or "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return 0.0
    if when is None:
        return 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


_PROVIDERS = {
    "kokoro": _kokoro,
    "sarvam": _sarvam,
    "fish": _fish,
}
