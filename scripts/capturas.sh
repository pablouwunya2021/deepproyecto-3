#!/usr/bin/env bash
# Toma capturas de la interfaz y del tablero con Chrome headless.
# Uso: scripts/capturas.sh [URL_BASE]   (por defecto http://127.0.0.1:7860)
set -euo pipefail
URL="${1:-http://127.0.0.1:7860}"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$RAIZ/evidencias/capturas"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
mkdir -p "$OUT"
toma() {  # ruta archivo alto
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 \
    --window-size=1280,"$3" --virtual-time-budget=15000 \
    --screenshot="$OUT/$2" "$URL$1" 2>/dev/null
  echo "$OUT/$2"
}
toma "/?n=8&semilla=2026" interfaz.png 900
toma "/monitoreo" monitoreo.png 1400
toma "/docs" api_docs.png 900
