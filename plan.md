# AI dubbing for Jellyfin (Doblarr) - plan

Status: planning only. Nothing has been installed or changed on the server.
Date: 2026-11-23

## Goal

Jellyfin users can get AI-dubbed audio tracks for movies and series, in
**Polish, Spanish, English, German, Russian**, using voices cloned from the
original actors. Target-language subtitles are always present, so there is no
STT and no translation.

Out of scope for now: how users submit requests (see "Jellyfin integration").

## Doblarr summary (https://github.com/jhd3197/Doblarr)

- Early project: MIT, ~2 stars, "In development". Pin a commit.
- Pipeline: ffmpeg extract -> Demucs `htdemucs_ft` (dialogue / background split)
  -> subtitle cues (pysubs2) -> speaker diarization -> per-speaker voice clone
  -> TTS per line -> ffmpeg `atempo` fit -> `sidechaincompress` ducking -> mux.
- Components: Python FastAPI app + SQLite job queue, web UI on :6363, CLI
  (`doblarr make "<path>" --to es-419`), REST API (`POST /api/jobs`, header
  `X-Api-Key`). Docker image `ghcr.io/jhd3197/doblarr` (may be private; GPU
  image is local-build only).
- Integrations: Radarr, Sonarr, Plex only. No Jellyfin, Jellyseerr or bot.
  Radarr/Sonarr/Plex are optional; files can be dubbed by path.
- `dub.dry_run: true` is the default. Turn it off deliberately.
- Config lives in `config/config.yaml` (from `config.docker.example.yaml`).
  Env vars seen: `ANTHROPIC_API_KEY`, `HF_TOKEN`, `DOBLARR_ALLOWED_HOSTS`,
  `DOBLARR_VOICESTUDIO_API_KEY`, `OLLAMA_ENDPOINT`, `LMSTUDIO_ENDPOINT`.

### Subtitle-only mode (verified in `stages/transcribe.py`)

- `transcribe.source: subtitles`: cues come from `job.subtitle_file` or an
  embedded text track. Track choice order: target language, then source
  language, then any text codec.
- If the track is already in the target language, `script_is_target = True`
  and the subtitle text is used as the translation. Set
  `translate.provider: passthrough`.
- Subtitle timings are NOT re-aligned against the audio (WhisperX alignment is
  refused for target-language tracks). Badly timed subtitles = drifting dub.
- `subtitle_roles` uses subtitle styles to tag voice-over / song / preview cues.
- Still audio-dependent: Demucs (clone reference clips + background) and
  diarization (who speaks which line). With diarization off, every line is
  voiced by one speaker (`SPEAKER_00`). Keep diarization on for films.

### Open questions in Doblarr (files NOT read yet)

- `stages/diarize.py`: is it skipped for subtitle input? Which local ONNX model?
- `stages/separate.py`: is Demucs gated or always run?
- `stages/mux.py`: is the output a full container copy or audio only?
- `stages/translate.py`, `clients/translator.py`: confirm `passthrough`.
- `subtitles.py`: how cues without a speaker get assigned.

## Language coverage (the key finding)

Sources: Qwen docs / DeepWiki results (Qwen3-TTS = 10 languages), voicebox
README (Chatterbox Multilingual = 23 languages). Items marked (unverified) are
from memory and need checking before relying on them.

| Language | Qwen3-TTS (current server) | Chatterbox Multilingual | Notes |
|---|---|---|---|
| English | yes | yes | |
| Spanish | yes | yes | Qwen has Spanish presets (Latin American and Peninsular flavours) |
| German | yes | yes | |
| Russian | yes | yes | |
| **Polish** | **NO** | yes (named in voicebox README) | Qwen3-TTS has no Polish. Cannot be served by the existing qwen-tts container |

Qwen3-TTS's 10 languages: Chinese, English, French, German, Russian, Italian,
Spanish, Portuguese, Japanese, Korean.

Consequences:

1. Four of your five languages (en, es, de, ru) work on the existing Qwen TTS.
2. **Polish needs a second TTS engine.** Candidates, in order of preference:
   - **Chatterbox Multilingual** (23 languages incl. Polish, voice cloning,
     MIT). It is also Doblarr's default voicebox engine, so it is the
     best-trodden path. Quality of Polish: untested.
   - XTTS-v2 (Coqui): Polish and the other four, cloning. Licence is
     CPML (non-commercial). Project is unmaintained (unverified).
   - Fish Speech / F5-TTS multilingual finetunes (unverified).
3. Simplest consistent design: use **Chatterbox for all five languages** via
   voicebox, and keep Qwen as an optional higher-quality engine for en/es/de/ru
   after an A/B test. This also removes the need for the Doblarr fork.
4. To check per language (quality is not proven by "supported"):
   - Cross-lingual cloning: reference clip is in the ORIGINAL language, output
     is in the TARGET language. Qwen3-TTS Base and Chatterbox both advertise
     it; verify identity retention and accent leakage.
   - Whether `atempo` stretching of slow CPU output sounds acceptable.
   - Russian and Polish text normalisation (numbers, abbreviations): TTS
     engines often mispronounce digits. Subtitles contain digits and symbols.

## TTS backend options

### Option 1 - voicebox as-is (recommended starting point)

- Run `jamiepine/voicebox` (MIT, 8 engines, 23 languages) as the speech
  backend. Doblarr already speaks its API (`/profiles`, `/generate`,
  `/history/{id}`, `/audio/{id}`).
- Engines Doblarr knows: `chatterbox`, `chatterbox_turbo` (English only),
  `qwen`, `qwen_custom_voice`, `kokoro`.
- Pro: no Doblarr code change, covers all 5 languages with Chatterbox, can use
  its Qwen engine for the 4 supported languages.
- Con: second TTS stack, bypasses litellm (no virtual keys, budgets, logging).
  Voicebox's "sampling fields work only on the jhd3197 fork" (code comment).
- CPU only on this box (Arc B70 not usable by CUDA/ONNX); GPU box needs CUDA.

### Option 2 - fork Doblarr, OpenAI-compatible `SpeechClient`

- Doblarr never calls `/v1/audio/speech`. Needs a new `SpeechClient` subclass in
  `doblarr/clients/speech.py` plus a branch in `build_speech_client`
  (`BACKENDS` is hard-coded to `("voicebox", "voicestudio")`).
- Required methods: `health`, `transcribe`, `voice_profiles`, `preset_voices`,
  `register_preset`, `clone_voice(name, language, sample, reference_text,
  description)`, `delete_profile`, `generate(...)`, `generation_status`,
  `fetch_audio`, `synthesize_to_file(...)`, `fork`, `canonical_engine`,
  `supports_direction`, `supports_sampling`, `supports_cloning`.
  `llm_generate` may raise `Unsupported`.
- Map `clone_voice` onto the qwen-tts voice registration
  (`/models/voices/manifest.json`), `generate` onto `/v1/audio/speech`
  (via litellm alias, then `tts-chunker`, then qwen-tts). The adapter must
  fake the async `generate -> history -> audio` flow synchronously.
- Pro: uses existing Qwen, litellm keys/budgets/logging.
- Con: maintaining a fork of a young project; **does not solve Polish**.
- Only worth it if Qwen quality is clearly better than Chatterbox for the
  other four languages.

### Decision

Start with Option 1 (voicebox + Chatterbox). Run the quality test below with
Chatterbox and Qwen on the same clip before deciding whether Option 2 is
worth it.

## Doblarr-to-TTS call flow (for reference)

1. Pick reference clip per speaker (`synthesize._resolve_profile`): cues >= 2.0 s,
   ranked by not-overlapped, not-contaminated, 4-15 s, closest to 8 s. Top 5
   tried in order. Extracted with ffmpeg: mono, 16 kHz, `pcm_s16le`,
   `loudnorm=I=-14`, from Demucs vocals when available.
2. Reference transcript = cue `text_src`; if the script language differs from
   the source, or the clip is >15 s, calls the backend's `/transcribe`
   (voicebox `POST /transcribe`). **This is a hidden STT dependency** even in
   subtitle mode: check it behaves when subtitles are in the target language
   but the clip is in the source language.
3. `POST /profiles` then `POST /profiles/{id}/samples` (multipart `file`,
   `reference_text`).
4. `POST /generate` per line (`profile_id`, `text`, `language`, optional
   `seed`, `engine`, `model_size`, `instruct`, sampling fields); poll
   `GET /history/{id}`; `GET /audio/{id}` returns WAV.
5. `voicebox.concurrency` 1-8 parallel sessions.

## Performance and capacity (estimates, measure before committing)

- qwen-tts on this server: CPU, 1.7B Base, Q4_K_M, ~8 s per short sentence
  (from litellm config; the quant was just changed from Q8_0 to Q4_K_M and
  `--cpu-threads=8` added, uncommitted in `qwen-tts/docker-compose.yaml`).
- A feature film has roughly 1,000-1,500 subtitle cues, so a TTS-only pass is
  about 2-3.5 hours per film per language on Qwen CPU. A series episode is
  roughly 400-600 cues. Chatterbox speed on CPU: unknown.
- Add Demucs + diarization on CPU: potentially more than TTS. Unknown.
- 5 languages x full library is not realistic on CPU. Dub on demand, cache the
  result, and queue.
- Decision needed: dedicated GPU box for the dubbing worker, or accept slow
  CPU jobs on bigboy. Note bazzite (Arc B70) cannot use the CUDA/ONNX path.

## Jellyfin integration (UNDECIDED)

### A. Getting dubs into Jellyfin (delivery)

Doblarr writes to `paths.output_dir/<title>/<locale>/`, not next to the source,
and adds a track named by `dub.track_name_template` (default
`"{language_name} AI"`). Original is not modified. Jellyfin only sees changes
after a library rescan (Doblarr auto-refresh is Plex-only).

Options:

1. **Swap into the library:** after a job, replace/rename the dubbed file over
   the original (or mux the track into the original). Simple for users (one
   item, audio track picker). Risk: modifies originals; needs backups or
   `preserve_versions`; file watchers and hash-based tools (Syncthing, Sonarr)
   may react.
2. **Second Jellyfin library** pointed at `output_dir`: no risk to originals,
   but duplicate items, and users must pick the "dubbed" copy.
3. **Jellyfin version grouping:** place dubbed files beside the original with
   Jellyfin's versioning naming (`Movie (2020) - Spanish AI.mkv`). Shows as
   alternate versions in the UI. Needs a post-job copy/move step.
4. **External audio sidecar** (e.g. `Movie.es.mka`): Jellyfin may pick these up
   as external audio tracks (unverified, must test on the installed Jellyfin
   version).

Also: set Jellyfin library path mounts so the dubbing container can read
`/mnt/raid/Movies`, `/mnt/raid/TvShows` (see `jellyfin/docker-compose.yaml`;
Jellyfin runs as `1006:1006`, host network). Output dir needs the same group
ownership, and a post-job Jellyfin library refresh via its API.

### B. Requesting dubs (UI)

Previously floated: custom JS. Options:

1. **Custom JS in Jellyfin web:** inject a "Request dub" button on item pages
   (Jellyfin has a branding / custom JS hook, or a JS injector plugin; there
   is a `linkding-injector` directory in this repo as a precedent for the
   pattern). Button calls a small gateway API.
2. **Jellyfin plugin (C#):** most native, most work.
3. **Standalone request page** behind Authentik (the stack already has
   `auth.osmosis.page`), listing the library via Jellyfin API.
4. **Jellyseerr/Seerr:** supports Jellyfin but requests downloads, not dubs.
   Poor fit.

Whatever the front-end, a **small gateway service** is needed between Jellyfin
users and Doblarr because Doblarr has one shared API key and no user concept:

- Authenticate the caller with their Jellyfin token (`GET /Users/Me`) or
  Authentik, and resolve item id to file path via Jellyfin API.
- Enforce per-user quotas, allowed languages, and an approval or admin gate
  (jobs are hours long on CPU).
- Dedupe: one job per (item, language); show status of existing dubs.
- Call `POST /api/jobs` with the Doblarr API key, poll status, trigger a
  Jellyfin rescan on completion.
- Per-item job state (queued / running / done / failed) exposed back to the
  button.

### Decisions needed

- Delivery mode (A1-A4) - test Jellyfin sidecar audio and version grouping first.
- Front-end (B1-B3).
- Who may request, quotas, and whether requests need approval.
- Which languages are allowed per user.

## Server-side design (when we build it)

Do not build on the current host until decided. Proposed layout if built here:

```
dubbing/                 # new dir under osmosis/
  docker-compose.yaml    # doblarr (+ voicebox) on caddy_proxy
  config/config.yaml     # doblarr config
  data/                  # queue db, output
```

- Network: join `caddy_proxy`; no published ports, expose the UI/API only
  through Caddy with Authentik, like the other services.
- Models/weights on the raid (`/mnt/raid/models/...`), never `/` (root is
  ~87% full per the qwen-tts compose notes; follow `/mnt/raid/models/CONVENTIONS.md`).
- Logging: loki driver like other compose files.
- Secrets in `.env`, not the compose file.
- Sablier wake-on-demand pattern is used for open-webui terminal; consider it
  for the dubbing worker so it only runs during jobs.

## Test plan (first, before any integration work)

1. Pick one short movie scene (about 10 min) with target-language `.srt`.
2. Run Doblarr "tease" job (first 10 minutes) with voicebox:
   a. Chatterbox, for es/de/ru/pl/en
   b. Qwen (via voicebox) for es/de/ru/en
3. Record: wall-clock time per stage (Demucs, diarize, TTS), RAM, subjective
   quality, timing drift, pronunciation of numbers, Polish quality.
4. Listen to the cross-lingual clone quality (original actor voice speaking
   the target language).
5. Test delivery options A1-A4 against the real Jellyfin version.
6. Only then decide: CPU vs GPU host, one engine vs two, fork or not.

## Risks

- Doblarr is very young; API and config may change. Pin a commit/tag.
- GHCR image may be private and GPU image is local-build only: plan to build.
- Polish is not supported by Qwen3-TTS.
- Voice cloning of real actors: legal/ethical limits, keep for private use,
  do not share dubbed files outside the household.
- Hours-long CPU jobs; need queue, cancel, and per-user limits.
- Subtitle timing drift is not corrected.
- Unknown hidden dependency on `/transcribe` for reference text.

## Next steps (ordered)

1. Read the unread Doblarr files listed above (diarize, separate, mux, translate).
2. Decide where the test runs (not on this server, per request).
3. Run the tease test plan with Chatterbox and Qwen.
4. Decide delivery mode and front-end.
5. Design the gateway API.
