# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-11-06

First tagged release. Recent work:

- Ignore the stray vendored groq_pool.py
- Groq killed the fast-tier models too
- Split 402 from 429 in the Fish TTS provider
- Move off the decommissioned Groq model
- Add Fish Audio as a key-gated TTS backend for cloned voices
- README: emphasize on-the-fly Hindi/English voice switching
- Add on-the-fly EN/HI voice switching and a tiered STT ladder
- Warn once when caller-hash salt is the shared default
- voicezero v0.1: a production-pattern voice agent at $0.00/min
