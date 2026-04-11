# ATST-SED Service

Minimal GPU-first HTTP service around the official ATST-SED inference stack for non-speech sound event detection and segmentation.

## What It Does

- Exposes `POST /v1/detect` for audio uploads.
- Filters out `Speech` by default so responses only contain non-speech events.
- Downloads the official `atst_as2M.ckpt` and `Stage2_wo_ext.ckpt` artifacts at startup if they are missing.
- Reports readiness through `/healthz` only after the checkpoints are present, the model is loaded, and warmup succeeds.
- Ships with a smoke test for the devcontainer that builds the stack, waits for health, submits `test.opus`, validates the JSON, and shuts everything down cleanly.

## Files

- `compose.yaml`: single-service Docker Compose stack.
- `Dockerfile`: CUDA-enabled API image pinned to the upstream ATST-SED repo.
- `.env` / `.env.example`: runtime configuration and secret inputs.
- `app/`: FastAPI wrapper and model bootstrap logic.
- `.devcontainer/smoke-test.sh`: end-to-end validation script.

## Run

```bash
docker compose up --build
```

Health:

```bash
curl http://localhost:8000/healthz
```

Inference:

```bash
curl -fsS -F file=@test.opus http://localhost:8000/v1/detect
```

## Response Shape

```json
{
  "model": "ATST-SED",
  "audio": {
    "filename": "test.opus",
    "duration_seconds": 3.2,
    "sample_rate_hz": 16000
  },
  "frame_resolution_seconds": 0.064,
  "excluded_labels": ["Speech"],
  "detected_labels": ["Dog"],
  "segment_count": 1,
  "segments": [
    {
      "label": "Dog",
      "start_seconds": 0.32,
      "end_seconds": 1.28
    }
  ]
}
```
