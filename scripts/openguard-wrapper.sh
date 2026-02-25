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
  OPENGUARD_CONFIG       Config file path (default: /app/presets/agentic.yaml)
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

# 'launch' needs read/write access to host config dirs so integrations
# can discover credentials and persist configuration changes.
LAUNCH_MOUNTS=()
SECURITY_CAPS=(--cap-drop ALL)

if [[ "${1:-}" == "launch" ]]; then
  # launch mounts host config dirs for credential discovery, but does NOT
  # need elevated capabilities — keep --cap-drop ALL.
  for dir in "$HOME/.config" "$HOME/.local/share"; do
    mkdir -p "$dir"
    LAUNCH_MOUNTS+=(-v "$dir:/root${dir#"$HOME"}")
  done
fi

TTY_FLAGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_FLAGS=(-it)
fi

exec docker run --rm ${TTY_FLAGS[@]+"${TTY_FLAGS[@]}"} \
  --name "openguard-${USER:-user}-$$" \
  --add-host host.docker.internal:host-gateway \
  -p "${PORT}:${PORT}" \
  --read-only \
  --tmpfs /tmp:rw,nosuid,exec,size=512m \
  --tmpfs /root:rw,nosuid,exec,size=256m \
  ${SECURITY_CAPS[@]+"${SECURITY_CAPS[@]}"} \
  --security-opt no-new-privileges:true \
  --pids-limit 256 \
  -e OPENGUARD_HOST="0.0.0.0" \
  -e OPENGUARD_PORT="$PORT" \
  -e OPENGUARD_CONFIG="${OPENGUARD_CONFIG:-/app/presets/agentic.yaml}" \
  -e OPENGUARD_OPENAI_URL_1="${OPENGUARD_OPENAI_URL_1:-http://host.docker.internal:11434/v1}" \
  -e OPENGUARD_OPENAI_KEY_1="${OPENGUARD_OPENAI_KEY_1:-}" \
  -e OPENGUARD_ANTHROPIC_URL_1="${OPENGUARD_ANTHROPIC_URL_1:-}" \
  -e OPENGUARD_ANTHROPIC_KEY_1="${OPENGUARD_ANTHROPIC_KEY_1:-}" \
  ${LAUNCH_MOUNTS[@]+"${LAUNCH_MOUNTS[@]}"} \
  -v "${WORKSPACE_DIR}:/workspace" \
  -v "${REPO_ROOT}:/app" \
  -w /workspace \
  "$IMAGE" openguard "$@"