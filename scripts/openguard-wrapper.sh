#!/usr/bin/env bash

set -euo pipefail

IMAGE="${OPENGUARD_DOCKER_IMAGE:-openguard-dev}"
REPO_ROOT="__OPENGUARD_REPO_ROOT__"
WORKSPACE_DIR="${OPENGUARD_MOUNT_DIR:-$PWD}"
PORT="${OPENGUARD_PORT:-23294}"
FORCE_BUILD=0

if [[ "${1:-}" == "--build" ]]; then
  FORCE_BUILD=1
  shift
fi

if [[ "${1:-}" == "--help-wrapper" ]]; then
  cat <<'EOF'
Docker-backed OpenGuard wrapper

Usage:
  openguard [--build] [CLI args...]

Examples:
  openguard
  openguard serve
  openguard launch opencode

Environment:
  OPENGUARD_MOUNT_DIR    Host directory mounted into container as /workspace (default: current PWD)
  OPENGUARD_PORT         Host and container port to expose (default: 23294)
  OPENGUARD_CONFIG       Config path inside /workspace (default: ./guards.yaml)
  OPENGUARD_DOCKER_IMAGE Docker image name (default: openguard-dev)
EOF
  exit 0
fi

if [[ ! -d "$WORKSPACE_DIR" ]]; then
  echo "Mount directory does not exist: $WORKSPACE_DIR" >&2
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1 || [[ "$FORCE_BUILD" -eq 1 ]]; then
  docker build --target dev -t "$IMAGE" "$REPO_ROOT"
fi

if [[ "$#" -eq 0 ]]; then
  set -- serve
fi

exec docker run --rm -it \
  --name "openguard-${USER:-user}-$$" \
  --add-host host.docker.internal:host-gateway \
  -p "${PORT}:${PORT}" \
  --read-only \
  --tmpfs /tmp:rw,nosuid,size=512m \
  --tmpfs /root:rw,nosuid,size=256m \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 256 \
  -e OPENGUARD_HOST="0.0.0.0" \
  -e OPENGUARD_PORT="$PORT" \
  -e OPENGUARD_CONFIG="${OPENGUARD_CONFIG:-./guards.yaml}" \
  -e OPENGUARD_OPENAI_URL_1="${OPENGUARD_OPENAI_URL_1:-http://host.docker.internal:11434/v1}" \
  -e OPENGUARD_OPENAI_KEY_1="${OPENGUARD_OPENAI_KEY_1:-}" \
  -e OPENGUARD_ANTHROPIC_URL_1="${OPENGUARD_ANTHROPIC_URL_1:-}" \
  -e OPENGUARD_ANTHROPIC_KEY_1="${OPENGUARD_ANTHROPIC_KEY_1:-}" \
  -v "${WORKSPACE_DIR}:/workspace" \
  -w /workspace \
  "$IMAGE" openguard "$@"