#!/usr/bin/env bash
# smoke_test_harness.sh — mimic the Track 2 judging harness locally:
# build the linux/amd64 image, run it against a real tasks.json, and
# verify results.json contains all four caption styles.
#
# Usage:
#   FIREWORKS_API_KEY=fw_... VIDEO_URL=https://example.com/clip.mp4 \
#     ./scripts/smoke_test_harness.sh

set -euo pipefail

: "${FIREWORKS_API_KEY:?Set FIREWORKS_API_KEY}"
VIDEO_URL="${VIDEO_URL:?Set VIDEO_URL to a short public MP4 URL}"
IMAGE_TAG="amd-caption-agent:smoke-test"

cd "$(dirname "$0")/.."

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
mkdir -p "$WORKDIR/input" "$WORKDIR/output"

cat > "$WORKDIR/input/tasks.json" <<EOF
[
  { "task_id": "smoke_test_1", "video_url": "${VIDEO_URL}" }
]
EOF

echo "[smoke_test] Building ${IMAGE_TAG} for linux/amd64..."
docker build --platform linux/amd64 -t "${IMAGE_TAG}" .

echo "[smoke_test] Running container against harness-shaped input/output mounts..."
docker run --rm \
    -e FIREWORKS_API_KEY="${FIREWORKS_API_KEY}" \
    -v "${WORKDIR}/input:/input" \
    -v "${WORKDIR}/output:/output" \
    "${IMAGE_TAG}"

RESULTS="${WORKDIR}/output/results.json"
if [ ! -f "${RESULTS}" ]; then
    echo "[smoke_test] FAIL: ${RESULTS} was not written." >&2
    exit 1
fi

python3 - "$RESULTS" <<'PY'
import json
import sys

path = sys.argv[1]
required_styles = {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"}

with open(path) as fh:
    data = json.load(fh)

if not data:
    print("FAIL: results.json is empty.")
    sys.exit(1)

for task_id, styles in data.items():
    missing = required_styles - set(styles)
    if missing:
        print(f"FAIL: task '{task_id}' is missing styles: {sorted(missing)}")
        sys.exit(1)
    for style in required_styles:
        if not isinstance(styles[style], str) or not styles[style].strip():
            print(f"FAIL: task '{task_id}' style '{style}' is empty or not a string.")
            sys.exit(1)

print(f"PASS: {len(data)} task(s), all four styles present and non-empty.")
PY

echo "[smoke_test] Done. Full output:"
cat "${RESULTS}"
