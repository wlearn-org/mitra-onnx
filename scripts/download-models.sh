#!/usr/bin/env bash
set -euo pipefail

# Download Mitra ONNX models from GitHub Release.
# Usage: bash scripts/download-models.sh [TAG]
#   TAG defaults to the tag in models.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  TAG=$(node -e "console.log(JSON.parse(require('fs').readFileSync('$ROOT/models.json','utf8')).tag)")
fi

BASE_URL="https://github.com/wlearn-org/mitra-onnx/releases/download/$TAG"

for f in mitra-classifier.onnx mitra-regressor.onnx; do
  dest="$ROOT/$f"
  if [[ -f "$dest" ]]; then
    echo "SKIP: $f (already exists)"
    continue
  fi
  echo "Downloading $f from $TAG..."
  curl -fL -o "$dest" "$BASE_URL/$f"
  echo "OK: $f ($(du -h "$dest" | cut -f1))"
done

echo ""
echo "Done. Models are in $ROOT/"
