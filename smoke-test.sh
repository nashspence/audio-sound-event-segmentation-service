#!/usr/bin/env bash
set -euo pipefail

set -a
# shellcheck disable=SC1091
source .env
set +a
API_BASE_URL="${API_BASE_URL:-http://host.docker.internal:${SERVICE_PORT}}"
export API_BASE_URL

cleanup() {
  docker compose down -v --remove-orphans
}

trap cleanup EXIT

docker compose down -v --remove-orphans >/dev/null 2>&1 || true
docker compose up --build -d

container_id="$(docker compose ps -q api)"
if [[ -z "${container_id}" ]]; then
  echo "API container did not start." >&2
  exit 1
fi

for _ in $(seq 1 240); do
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
  if [[ "${health}" == "healthy" ]]; then
    break
  fi
  sleep 5
done

health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
if [[ "${health}" != "healthy" ]]; then
  docker compose logs api >&2
  echo "API container did not become healthy." >&2
  exit 1
fi

python3 - <<'PY'
import json
import os
import urllib.request

with urllib.request.urlopen(f"{os.environ['API_BASE_URL']}/healthz") as response:
    payload = json.load(response)

assert payload["ready"] is True
assert payload["device"].startswith("cuda")
assert payload["checkpoint"]["base"]["present"] is True
assert payload["checkpoint"]["stage2"]["present"] is True
PY

curl -fsS \
  -F file=@test.opus \
  "${API_BASE_URL}/v1/detect" \
  > /tmp/atst-sed-detect.json

python3 - <<'PY'
import json

with open("/tmp/atst-sed-detect.json", "r", encoding="utf-8") as handle:
    payload = json.load(handle)

assert payload["model"] == "ATST-SED"
assert payload["audio"]["filename"] == "test.opus"
assert payload["audio"]["duration_seconds"] > 0
assert payload["frame_resolution_seconds"] > 0
assert isinstance(payload["segments"], list)
assert isinstance(payload["detected_labels"], list)
for segment in payload["segments"]:
    assert segment["label"] != "Speech"
    assert segment["start_seconds"] >= 0
    assert segment["end_seconds"] >= segment["start_seconds"]
PY

docker compose down -v --remove-orphans
trap - EXIT
