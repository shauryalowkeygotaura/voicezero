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

Each synth writes a mono 16-bit PCM WAV to wav_path. Whichever sample rate the
backend emits, downstream faster-whisper and native playback both accept it.
"""
from __future__ import annotations

import base64
import os
import re
import wave
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


_PROVIDERS = {
    "kokoro": _kokoro,
    "sarvam": _sarvam,
}
