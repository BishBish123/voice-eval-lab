# Troubleshooting

Real ops gotchas surfaced during development. Each entry: what you'll see, what's actually happening, how to fix it.

## macOS `/tmp` symlink guard

### Symptom: `voice-eval run --out /tmp/report.md` exits with code 2 immediately
**Cause:** On macOS `/tmp` is a symlink to `/private/tmp`. The `compare` command (and all write paths) run a symlink guard on the output parent directory before doing any work, and a symlinked parent is rejected.
**Fix:** Use `/private/tmp/report.md` or a project-relative path such as `--out evals/REPORT.md`. The error message prints the resolved path you can retry with.
**Source:** README Quick start macOS note; loop_history.md Round 1 ("/tmp symlink guard hoisted in `compare`")

### Symptom: Tests using `/tmp` paths fail in CI even on Linux
**Cause:** Test code that hard-codes `/tmp` will hit the symlink guard on macOS. Use `pytest`'s `tmp_path` fixture instead.
**Fix:** Replace `/tmp/...` with `tmp_path / "report.md"` in test code.
**Source:** loop_history.md Round 6 (judge tests "used `/tmp` triggering symlink guard")

## `ResourceWarning: asyncio` in test output

### Symptom: `pytest` output contains `ResourceWarning: Enable tracemalloc...` or unclosed socket warnings
**Cause:** pytest-asyncio + typer.CliRunner GC timing: the CLI runner tears down before the event loop fully closes. Not a real leak.
**Fix:** The `pyproject.toml` `filterwarnings` section adds `ignore::ResourceWarning:asyncio` and sets `asyncio_default_fixture_loop_scope = "session"` to address the root cause. If you see it on a fresh checkout without those settings, `uv sync` to pick up the current config.
**Source:** loop_history.md Round 4 (ResourceWarning suppression) + Round 5 (narrowed to asyncio root cause)

## Whisper model size

### Symptom: `WHISPER_MODEL_NAME=large` set and `make eval` hangs for minutes on first run
**Cause:** The `large` model is ~3 GB and is lazy-loaded on first `transcribe` call. On a cold cache this downloads over the network.
**Fix:** Use `WHISPER_MODEL_NAME=tiny` (~75 MB) for development. `small` (~465 MB) or `medium` (~1.5 GB) for production. The model download is one-time; subsequent runs use the HuggingFace cache.
**Source:** README WhisperSTT model size table

### Symptom: `WHISPER_MODEL_NAME` set but `voice-eval run` still uses `MockSTT`
**Cause:** `openai-whisper` is not installed — the adapter soft-fails to `MockSTT` on `ImportError`.
**Fix:** `uv sync --extra real` to install `openai-whisper` (and `httpx`).
**Source:** README Real adapters section; `[real]` extras group in `pyproject.toml`

## Notes RAG pgvector backend

### Symptom: `voice-eval notes lookup --query "..."` returns no results even after `add`
**Cause:** `NOTES_DSN` is not set, so `InMemoryNotesStore` is used; in-memory store is per-process and cleared on exit.
**Fix:** For persistence across sessions start the container first: `make notes-up`, then `export NOTES_DSN="postgresql://vel:vel@localhost:5433/vel"`.
**Source:** README Notes RAG pgvector backend section

### Symptom: `make notes-up` fails with port conflict
**Cause:** Port 5433 is already bound.
**Fix:** The notes compose file defaults to 5433; override by editing `docker-compose.notes.yml` or `NOTES_DSN` to point at a different port.
**Source:** README pgvector backend activation

## LiveKit / Pipecat without credentials

### Symptom: `voice-eval pipeline serve --room my-room` exits immediately with no error
**Cause:** `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` are not all set. `serve_on_livekit` logs a warning and returns cleanly — this is intentional no-op behaviour.
**Fix:** Export all three env vars before calling `pipeline serve`. Check the log output for the `"LiveKit credentials not set"` warning line.
**Source:** README Pipecat pipeline / Serve on a LiveKit room section

## Real adapters / `[real]` extras

### Symptom: `from voice_eval_lab.adapters import DeepgramSTT` raises `ImportError: httpx`
**Cause:** `httpx` is in the `[real]` optional extras group, not base deps.
**Fix:** `uv sync --extra real`.
**Source:** loop_history.md Round 5 ("`[real]` extras gotcha … httpx wasn't in `dev` extras")

### Symptom: `DeepgramSTT` real-mode logs a warning and falls back to mock even with `DEEPGRAM_API_KEY` set
**Cause:** Deepgram pre-recorded STT requires actual audio bytes. The mock pipeline operates on text-only `Turn` objects with no audio. When `turn.audio_bytes` is absent, `DeepgramSTT` logs a warning and falls back rather than posting text-as-WAV.
**Fix:** Wire audio fixtures: `voice-eval run --audio-fixtures evals/audio/`. The repo ships 54 pre-generated silence WAVs covering all 25 golden conversations.
**Source:** loop_history.md R5 codex (Deepgram audio bytes bug fixed); README DeepgramSTT real-mode note

## LLM judge

### Symptom: `voice-eval run --judge llm` with Anthropic key fails with a stale model error
**Cause:** Older checkout used `claude-3-haiku-20240307` (retired). Fixed in Round 8.
**Fix:** Pull latest — model bumped to `claude-3-5-haiku-20241022`.
**Source:** loop_history.md Round 7 ("`claude-3-haiku-20240307` is 14 months stale") + Round 8 fix

### Symptom: `voice-eval run --judge llm` dumps a full traceback when no key is set
**Cause:** Older CLI only caught `IncompleteRunError`; `OSError`/`ImportError` from `make_judge()` propagated.
**Fix:** Pull latest — the `run` command now catches `(OSError, ImportError)` from judge construction with a friendly message.
**Source:** loop_history.md Round 7 + Round 8 fix
