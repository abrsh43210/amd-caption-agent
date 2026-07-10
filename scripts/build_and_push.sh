#!/usr/bin/env bash
# build_and_push.sh — build a public linux/amd64 image for the Track 2 judging harness
# and push it to a registry.
#
# The harness pulls a public linux/amd64 image and runs it against
# /input/tasks.json -> /output/results.json, so the image MUST:
#   1. be built for linux/amd64 (not the arm64 default on Apple Silicon), and
#   2. be pushed somewhere pullable without auth (e.g. Docker Hub, GHCR public repo).
#
# Usage:
#   IMAGE=yourdockerhubuser/amd-caption-agent TAG=latest ./scripts/build_and_push.sh
#
# Prereqs: `docker buildx` (bundled with modern Docker Desktop/Engine) and
# `docker login` already run against your target registry.

set -euo pipefail

IMAGE="${IMAGE:?Set IMAGE, e.g. IMAGE=yourdockerhubuser/amd-caption-agent}"
TAG="${TAG:-latest}"
FULL_TAG="${IMAGE}:${TAG}"

cd "$(dirname "$0")/.."

echo "[build_and_push] Building ${FULL_TAG} for linux/amd64 and pushing..."
docker buildx build \
    --platform linux/amd64 \
    -t "${FULL_TAG}" \
    --push \
    .

echo "[build_and_push] Done. Verify the pushed image is public by pulling it anonymously:"
echo "  docker logout && docker pull ${FULL_TAG}"
