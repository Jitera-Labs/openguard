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
SEED_DIR=""
trap 'rm -rf "${SEED_DIR}"' EXIT

if [[ "${1:-}" == "launch" ]]; then
  echo "[wrapper] Setting up launch mounts for: ${2:-<none>}" >&2

  # All integrations use the seed pattern: copy host config to a world-readable
  # temp dir, mount read-only, then copy into the writable /root tmpfs at startup.
  # Direct bind-mounts fail because --cap-drop ALL removes CAP_DAC_OVERRIDE,
  # so container root cannot write to files owned by host UID 1000.

  if [[ "${2:-}" == "claude" ]]; then
    SEED_DIR=$(mktemp -d /tmp/.openguard-claude-XXXXXX)
    mkdir -p "$SEED_DIR/claude"
    cp -r "$HOME/.claude/." "$SEED_DIR/claude/" 2>/dev/null || true
    chmod -R a+rX "$SEED_DIR/claude" 2>/dev/null || true
    LAUNCH_MOUNTS+=(-v "$SEED_DIR/claude:/claude-seed:ro")
    echo "[wrapper] Seeded ~/.claude -> $SEED_DIR/claude (world-readable, read-only mount)" >&2
    if [[ -f "$HOME/.claude.json" ]]; then
      cp "$HOME/.claude.json" "$SEED_DIR/claude.json"
      chmod a+r "$SEED_DIR/claude.json"
      LAUNCH_MOUNTS+=(-v "$SEED_DIR/claude.json:/claude-seed.json:ro")
      echo "[wrapper] Seeded ~/.claude.json -> $SEED_DIR/claude.json (world-readable, read-only mount)" >&2
    fi

  elif [[ "${2:-}" == "opencode" ]]; then
    SEED_DIR=$(mktemp -d /tmp/.openguard-opencode-XXXXXX)

    # Seed ~/.config/opencode
    mkdir -p "$SEED_DIR/config"
    if [[ -d "$HOME/.config/opencode" ]]; then
      cp -r "$HOME/.config/opencode/." "$SEED_DIR/config/" 2>/dev/null || true
    fi
    chmod -R a+rX "$SEED_DIR/config" 2>/dev/null || true
    LAUNCH_MOUNTS+=(-v "$SEED_DIR/config:/opencode-seed/config:ro")
    echo "[wrapper] Seeded ~/.config/opencode -> $SEED_DIR/config (read-only mount)" >&2

    # Seed ~/.local/share/opencode — only auth.json (the rest is large data:
    # snapshots ~73GB, database ~330MB, logs, etc. that we don't need).
    mkdir -p "$SEED_DIR/share"
    if [[ -f "$HOME/.local/share/opencode/auth.json" ]]; then
      cp "$HOME/.local/share/opencode/auth.json" "$SEED_DIR/share/auth.json" 2>/dev/null || true
      chmod a+r "$SEED_DIR/share/auth.json" 2>/dev/null || true
    fi
    LAUNCH_MOUNTS+=(-v "$SEED_DIR/share:/opencode-seed/share:ro")
    echo "[wrapper] Seeded ~/.local/share/opencode/auth.json -> $SEED_DIR/share/ (read-only mount)" >&2

  elif [[ "${2:-}" == "codex" ]]; then
    SEED_DIR=$(mktemp -d /tmp/.openguard-codex-XXXXXX)
    mkdir -p "$SEED_DIR/codex"

    # Seed ~/.codex (auth.json, config.toml, themes/ if present)
    if [[ -d "$HOME/.codex" ]]; then
      for item in auth.json config.toml; do
        if [[ -f "$HOME/.codex/$item" ]]; then
          cp "$HOME/.codex/$item" "$SEED_DIR/codex/$item" 2>/dev/null || true
        fi
      done
      if [[ -d "$HOME/.codex/themes" ]]; then
        cp -r "$HOME/.codex/themes" "$SEED_DIR/codex/themes" 2>/dev/null || true
      fi
    fi
    chmod -R a+rX "$SEED_DIR/codex" 2>/dev/null || true
    LAUNCH_MOUNTS+=(-v "$SEED_DIR/codex:/codex-seed:ro")
    echo "[wrapper] Seeded ~/.codex -> $SEED_DIR/codex (world-readable, read-only mount)" >&2
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