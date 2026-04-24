#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  run_dashboard.sh  —  Launch the Multi-Strategy Trading Dashboard
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_FILE="$SCRIPT_DIR/trading_system.py"
PORT="${PORT:-8501}"

# ── Colour helpers ─────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${CYAN}[dashboard]${RESET} $*"; }
ok()   { echo -e "${GREEN}[  OK  ]${RESET} $*"; }
warn() { echo -e "${YELLOW}[ WARN ]${RESET} $*"; }
die()  { echo -e "${RED}[ ERR  ]${RESET} $*" >&2; exit 1; }

# ── Sanity checks ──────────────────────────────────────────
[[ -f "$APP_FILE" ]] || die "App file not found: $APP_FILE"

if ! command -v python3 &>/dev/null; then
    die "python3 is not installed or not on PATH."
fi

if ! python3 -c "import streamlit" &>/dev/null; then
    warn "streamlit not found. Attempting to install..."
    pip install streamlit --quiet || die "Failed to install streamlit."
fi

# ── Optional: activate a virtualenv if present ─────────────
if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    log "Activating virtual environment at .venv"
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [[ -f "$SCRIPT_DIR/venv/bin/activate" ]]; then
    log "Activating virtual environment at venv"
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# ── Launch ─────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   Multi-Strategy Trading Dashboard           ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${RESET}"
echo ""
ok "Starting Streamlit on port ${PORT}..."
log "URL  →  http://localhost:${PORT}"
echo ""

cd "$SCRIPT_DIR"

exec streamlit run "$APP_FILE" \
    --server.port "$PORT" \
    --server.headless false \
    --browser.gatherUsageStats false \
    --theme.base dark \
    --theme.primaryColor "#00e5ff" \
    --theme.backgroundColor "#0d0d0d" \
    --theme.secondaryBackgroundColor "#1a1a1a" \
    --theme.textColor "#e0e0e0"
