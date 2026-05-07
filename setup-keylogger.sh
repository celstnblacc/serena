#!/usr/bin/env bash
# serena/setup-keylogger.sh — Register serena MCP server with keylogger-mcp-wrapper
#
# Usage:
#   bash setup-keylogger.sh            # Register with all detected clients
#   bash setup-keylogger.sh --dry-run  # Preview only
#   KEYLOGGER_MCP=0 bash setup-keylogger.sh  # Register without wrapper
#
# The wrapper logs every JSON-RPC message between the client and serena,
# helping diagnose "Not connected" and other MCP transport issues.

set -euo pipefail

KEYLOGGER_MCP="${KEYLOGGER_MCP:-1}"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()    { echo -e "  ${GREEN}[OK]${NC}  $*"; }
info()  { echo -e "  ${YELLOW}[>>]${NC} $*"; }

# ── Determine serena command ──────────────────────────────────────────────────

SERENA_CMD=("uvx" "--from" "git+https://github.com/celstnblacc/serena" "serena" "start-mcp-server" "--context=ide" "--open-web-dashboard" "false" "--project-from-cwd")

# ── Build wrapped command ─────────────────────────────────────────────────────

if [ "$KEYLOGGER_MCP" != "0" ]; then
    OPENCODE_CMD=("keylogger-mcp-wrapper" "--name" "serena" "--" "${SERENA_CMD[@]}")
    CLAUDE_CMD="keylogger-mcp-wrapper"
    CLAUDE_ARGS=("--name" "serena" "--" "${SERENA_CMD[@]}")
    echo "keylogger-mcp-wrapper: ENABLED (KEYLOGGER_MCP=1)"
else
    OPENCODE_CMD=("${SERENA_CMD[@]}")
    CLAUDE_CMD="${SERENA_CMD[0]}"
    CLAUDE_ARGS=("${SERENA_CMD[@]:1}")
    echo "keylogger-mcp-wrapper: DISABLED (KEYLOGGER_MCP=0)"
fi

# ── Register with OpenCode ────────────────────────────────────────────────────

OPENCODE_CFG="$HOME/.config/opencode/opencode.json"
if [ -f "$OPENCODE_CFG" ] || [ "$DRY_RUN" = true ]; then
    if [ "$DRY_RUN" = true ]; then
        info "Dry-run: would register serena in $OPENCODE_CFG"
        echo "    command: ${OPENCODE_CMD[*]}"
    else
        python3 - "$OPENCODE_CFG" "${OPENCODE_CMD[@]}" <<'PYEOF'
import json, sys
cfg_path = sys.argv[1]
wrapper_cmd = list(sys.argv[2:])

with open(cfg_path) as f:
    data = json.load(f)
data.setdefault("mcp", {})
if "serena" in data["mcp"]:
    print("  [skip] serena already registered in", cfg_path)
    sys.exit(0)

data["mcp"]["serena"] = {"command": wrapper_cmd, "enabled": True, "type": "local"}
with open(cfg_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
print(f"  [OK] serena registered in {cfg_path}")
PYEOF
    fi
fi

# ── Register with Claude Code ─────────────────────────────────────────────────

CLAUDE_CFG="$HOME/.claude.json"
if [ -f "$CLAUDE_CFG" ] || [ "$DRY_RUN" = true ]; then
    if [ "$DRY_RUN" = true ]; then
        info "Dry-run: would register serena in $CLAUDE_CFG"
        echo "    command: $CLAUDE_CMD, args: ${CLAUDE_ARGS[*]}"
    else
        python3 - "$CLAUDE_CFG" "$CLAUDE_CMD" "${CLAUDE_ARGS[@]}" <<'PYEOF'
import json, sys
cfg_path = sys.argv[1]
cmd = sys.argv[2]
args = list(sys.argv[3:])

with open(cfg_path) as f:
    data = json.load(f)
data.setdefault("mcpServers", {})
if "serena" in data["mcpServers"]:
    print("  [skip] serena already registered in", cfg_path)
    sys.exit(0)

data["mcpServers"]["serena"] = {"command": cmd, "args": args, "env": {}, "type": "stdio"}
with open(cfg_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
print(f"  [OK] serena registered in {cfg_path}")
PYEOF
    fi
fi

echo ""
ok "Done. Check logs at ~/.keylogger-mcp/proxy/serena/"
info "Restart your MCP client to pick up the changes."
