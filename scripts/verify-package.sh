#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] nix build .#default"
nix build .#default >/dev/null

echo "[2/4] check exit code when Claude credentials are missing"
EMPTY_HOME="$(mktemp -d)"
set +e
OUTPUT=$(env -u ANTHROPIC_OAUTH_API_KEY -u CLAUDE_CONFIG_DIR HOME="$EMPTY_HOME" ./result/bin/llm-quota usage get-claude-usage 2>&1)
STATUS=$?
set -e
rmdir "$EMPTY_HOME"

echo "$OUTPUT"
if [ "$STATUS" -ne 255 ]; then
  echo "expected exit code 255, got $STATUS"
  exit 1
fi

echo "[3/4] check get-codex-usage subcommand is exposed"
HELP_OUTPUT=$(./result/bin/llm-quota usage help 2>&1)
echo "$HELP_OUTPUT" | grep -q "get-codex-usage"

echo "[4/4] package verification passed"
