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

echo "[wrapper] IMAGE=$IMAGE REPO_ROOT=$REPO_ROOT WORKSPACE_DIR=$WORKSPACE_DIR PORT=$PORT" >&2
if ! docker image inspect "$IMAGE" >/dev/null 2>&1 || [[ "$FORCE_BUILD" -eq 1 ]]; then
  echo "[wrapper] Building image $IMAGE..." >&2
  docker build --target dev -t "$IMAGE" "$REPO_ROOT"
  echo "[wrapper] Image build complete." >&2
else
  echo "[wrapper] Image $IMAGE already exists, skipping build." >&2
fi

if [[ "$#" -eq 0 ]]; then
  set -- serve
fi

LAUNCH_MOUNTS=()
SECURITY_CAPS=(--cap-drop ALL)
CLAUDE_SEED=""
trap 'rm -rf "${CLAUDE_SEED}"' EXIT

if [[ "${1:-}" == "launch" ]]; then
  echo "[wrapper] Setting up launch mounts for: ${2:-<none>}" >&2
  # Mount host config dirs for integrations that need persistent config
  # (e.g. opencode writes ~/.config/opencode/opencode.json).
  # Skip for claude: it uses a :ro seed mount + tmpfs copy instead (see setup.py).
  if [[ "${2:-}" != "claude" ]]; then
    for dir in "$HOME/.config" "$HOME/.local/share"; do
      mkdir -p "$dir"
      LAUNCH_MOUNTS+=(-v "$dir:/root${dir#"$HOME"}")
      echo "[wrapper] Mounting $dir -> /root${dir#"$HOME"}" >&2
    done
  fi
  if [[ "${2:-}" == "claude" ]]; then
    # The container runs with --cap-drop ALL (no CAP_DAC_OVERRIDE), so container
    # root cannot read mode-600 files owned by the host user (UID 1000).
    # Solution: copy ~/.claude to a host temp dir as the host user (who *can*
    # read them), chmod world-readable, then mount that as the seed.
    CLAUDE_SEED=$(mktemp -d /tmp/.openguard-claude-XXXXXX)
    mkdir -p "$CLAUDE_SEED/claude"
    cp -r "$HOME/.claude/." "$CLAUDE_SEED/claude/" 2>/dev/null || true
    chmod -R a+rX "$CLAUDE_SEED/claude" 2>/dev/null || true
    LAUNCH_MOUNTS+=(-v "$CLAUDE_SEED/claude:/claude-seed:ro")
    echo "[wrapper] Seeded ~/.claude -> $CLAUDE_SEED/claude (world-readable, read-only mount)" >&2
    if [[ -f "$HOME/.claude.json" ]]; then
      cp "$HOME/.claude.json" "$CLAUDE_SEED/claude.json"
      chmod a+r "$CLAUDE_SEED/claude.json"
      LAUNCH_MOUNTS+=(-v "$CLAUDE_SEED/claude.json:/claude-seed.json:ro")
      echo "[wrapper] Seeded ~/.claude.json -> $CLAUDE_SEED/claude.json (world-readable, read-only mount)" >&2
    fi
  fi
fi

TTY_FLAGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_FLAGS=(-it)
fi

CONTAINER_NAME="openguard-${USER:-user}-$$"
echo "[wrapper] TTY_FLAGS=${TTY_FLAGS[*]+"${TTY_FLAGS[*]}"}" >&2
echo "[wrapper] Container name: $CONTAINER_NAME" >&2
echo "[wrapper] Launching: docker run ... $IMAGE openguard $*" >&2

docker run --rm ${TTY_FLAGS[@]+"${TTY_FLAGS[@]}"} \
  --name "$CONTAINER_NAME" \
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
  ${OPENGUARD_DEBUG:+-e OPENGUARD_DEBUG="$OPENGUARD_DEBUG"} \
  ${OPENGUARD_OPENAI_URL_1:+-e OPENGUARD_OPENAI_URL_1="$OPENGUARD_OPENAI_URL_1"} \
  ${OPENGUARD_OPENAI_KEY_1:+-e OPENGUARD_OPENAI_KEY_1="$OPENGUARD_OPENAI_KEY_1"} \
  ${OPENGUARD_ANTHROPIC_URL_1:+-e OPENGUARD_ANTHROPIC_URL_1="$OPENGUARD_ANTHROPIC_URL_1"} \
  ${OPENGUARD_ANTHROPIC_KEY_1:+-e OPENGUARD_ANTHROPIC_KEY_1="$OPENGUARD_ANTHROPIC_KEY_1"} \
  ${LAUNCH_MOUNTS[@]+"${LAUNCH_MOUNTS[@]}"} \
  -v "${WORKSPACE_DIR}:/workspace" \
  -v "${REPO_ROOT}:/app" \
  -w /workspace \
  "$IMAGE" openguard "$@"
EXIT_CODE=$?
echo "[wrapper] docker run exited with code $EXIT_CODE" >&2
exit $EXIT_CODE