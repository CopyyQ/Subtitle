#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAST_DIR="$ROOT/model/FAST"
FAST_REPO="https://github.com/czczup/FAST.git"
FAST_COMMIT="9cfeda29bfd16c4bc260740a5e39afb2cfdfd23f"
CKPT_NAME="fast_base_ic15_736_finetune_ic17mlt.pth"
CKPT_URL="https://github.com/czczup/FAST/releases/download/release/$CKPT_NAME"
CKPT_SHA256="e9f9f36dd343e04938c1758210f63c78494cf5b38efbf448138f18830bd42992"

mkdir -p "$(dirname "$FAST_DIR")"

if [[ ! -d "$FAST_DIR/.git" ]]; then
  rm -rf "$FAST_DIR"
  git clone "$FAST_REPO" "$FAST_DIR"
fi

git -C "$FAST_DIR" fetch --all --tags
git -C "$FAST_DIR" checkout --detach "$FAST_COMMIT"
git -C "$FAST_DIR" reset --hard "$FAST_COMMIT"
git -C "$FAST_DIR" clean -fd
git -C "$FAST_DIR" apply "$ROOT/patches/fast-minimal-imports.patch"

mkdir -p "$FAST_DIR/checkpoints"
CKPT="$FAST_DIR/checkpoints/$CKPT_NAME"
if [[ ! -f "$CKPT" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$CKPT" "$CKPT_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$CKPT" "$CKPT_URL"
  else
    echo "curl or wget is required" >&2
    exit 1
  fi
fi

echo "$CKPT_SHA256  $CKPT" | sha256sum -c -
echo "FAST ready at $FAST_DIR"
echo "FAST commit: $FAST_COMMIT"
