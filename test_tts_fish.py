"""Metered-failure behaviour of the `fish` TTS provider.

No request ever leaves the machine: urlopen is stubbed and FISH_API_KEY is an
obvious dummy, so the suite spends no Fish credits.

A live call turn cannot checkpoint the way Code/fish-voice/fish_voice.py's
speak_many() does, because there is no batch: one turn is one clip and nothing
is left owed. The equivalent guarantee here is the latch, so these tests pin
down that a 402 disables Fish for the process (one warning, then silence) while
a 429 costs at most one short wait and never latches.

Run:  python -m pytest test_tts_fish.py -v
"""
from __future__ import annotations

import email.message
import io
import json
import urllib.error
import urllib.request

import pytest

import tts_providers


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    hdrs = email.message.Message()
    if retry_after is not None:
        hdrs["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        url="https://api.fish.audio/v1/tts",
        code=code,
        msg="simulated",
        hdrs=hdrs,
        fp=io.BytesIO(json.dumps({"detail": f"simulated {code}"}).encode()),
    )


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class FakeFish:
    """`script[n]` is the HTTPError to raise on the n-th call (0-indexed)."""

    def __init__(self, script=None):
        self.script = script or {}
        self.calls = 0

    def __call__(self, req, timeout=None):
        n = self.calls
        self.calls += 1
        exc = self.script.get(n)
        if exc is not None:
            raise exc
        return _Response(b"RIFF____WAVEfake")


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "dummy-key-never-sent-anywhere")
    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    tts_providers._fish_reset_quota()
    yield
    tts_providers._fish_reset_quota()


@pytest.fixture
def fake(monkeypatch):
    def install(script=None):
        f = FakeFish(script)
        monkeypatch.setattr(urllib.request, "urlopen", f)
        return f

    return install


def _speak(tmp_path, name="turn.wav"):
    return tts_providers.synth("fish", "hello there", "voice-id", "+0%", tmp_path / name)


def test_happy_path_writes_a_wav(tmp_path, fake):
    f = fake()
    assert _speak(tmp_path) is True
    assert (tmp_path / "turn.wav").read_bytes() == b"RIFF____WAVEfake"
    assert f.calls == 1


def test_402_raises_once_then_latches_off_for_the_process(tmp_path, fake):
    f = fake({0: _http_error(402)})

    # First 402: loud, so agent.py logs one "falling back to edge-tts" line.
    with pytest.raises(RuntimeError, match="out of credit"):
        _speak(tmp_path, "turn1.wav")
    assert tts_providers._fish_quota_spent() is True

    # Every later turn: silent False, and no HTTP round trip at all.
    for i in range(3):
        assert _speak(tmp_path, f"turn{i + 2}.wav") is False
    assert f.calls == 1


def test_402_is_not_retried(tmp_path, fake):
    f = fake({0: _http_error(402, "30"), 1: _http_error(402, "30")})
    with pytest.raises(RuntimeError):
        _speak(tmp_path)
    assert f.calls == 1  # a Retry-After on a 402 does not buy a second attempt


def test_429_waits_once_then_succeeds(tmp_path, fake, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(tts_providers.time, "sleep", slept.append)

    f = fake({0: _http_error(429, "1")})
    assert _speak(tmp_path) is True
    assert f.calls == 2
    assert slept == [1.0]
    # A rate limit is transient: it must not disable the provider.
    assert tts_providers._fish_quota_spent() is False


def test_429_with_a_long_retry_after_falls_back_instead_of_holding_the_turn(
    tmp_path, fake, monkeypatch
):
    slept: list[float] = []
    monkeypatch.setattr(tts_providers.time, "sleep", slept.append)

    f = fake({0: _http_error(429, "600")})
    assert _speak(tmp_path) is False  # edge-tts takes this turn
    assert f.calls == 1
    assert slept == []
    assert tts_providers._fish_quota_spent() is False


def test_429_twice_gives_up_without_latching(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(tts_providers.time, "sleep", lambda _s: None)
    f = fake({0: _http_error(429, "1"), 1: _http_error(429, "1")})
    with pytest.raises(urllib.error.HTTPError):
        _speak(tmp_path)
    assert f.calls == 2
    assert tts_providers._fish_quota_spent() is False


def test_other_errors_propagate_and_do_not_latch(tmp_path, fake):
    fake({0: _http_error(401)})
    with pytest.raises(urllib.error.HTTPError):
        _speak(tmp_path)
    assert tts_providers._fish_quota_spent() is False


def test_no_key_means_no_call(tmp_path, fake, monkeypatch):
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    f = fake()
    assert _speak(tmp_path) is False
    assert f.calls == 0


def test_retry_after_forms():
    def hdrs(value):
        m = email.message.Message()
        if value is not None:
            m["Retry-After"] = value
        return m

    assert tts_providers._fish_retry_after(hdrs(None)) is None
    assert tts_providers._fish_retry_after(hdrs("3")) == 3.0
    assert tts_providers._fish_retry_after(hdrs("-1")) == 0.0
    assert tts_providers._fish_retry_after(hdrs("nonsense")) == 0.0
    assert tts_providers._fish_retry_after(hdrs("Wed, 21 Oct 2015 07:28:00 GMT")) == 0.0
    assert tts_providers._fish_retry_after(hdrs("Fri, 01 Jan 2100 00:00:00 GMT")) > 0
