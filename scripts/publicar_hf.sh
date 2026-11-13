#!/usr/bin/env bash
# Publica el servicio en un Hugging Face Space (SDK Docker).
#
# Requisitos (una sola vez):
#   pip install -U huggingface_hub
#   hf auth login                     # token con permiso "write"
#
# Uso:
#   scripts/publicar_hf.sh <usuario_hf>/<nombre_space>
#
# El Space se crea si no existe. Solo se sube lo que necesita la imagen
# (Dockerfile, requirements, app/, lumen/, modelos/), con un README propio que
# lleva los metadatos que pide Hugging Face.
set -euo pipefail
SPACE="${1:?uso: scripts/publicar_hf.sh <usuario>/<space>}"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$RAIZ/build_hf"

rm -rf "$BUILD" && mkdir -p "$BUILD"
cp -r "$RAIZ/Dockerfile" "$RAIZ/requirements.txt" "$RAIZ/app" "$RAIZ/lumen" "$RAIZ/modelos" "$BUILD/"
find "$BUILD" -name "__pycache__" -prune -exec rm -rf {} +
# solo los generadores que sirve el servicio (el resto es para la evaluacion)
python3 - "$BUILD/modelos" <<'PY'
import json, os, sys
d = sys.argv[1]
servidos = json.load(open(os.path.join(d, "tarjeta_modelo.json")))["modelos_servidos"]
for f in os.listdir(d):
    if f.startswith("G_") and f[2:-3] not in servidos:
        os.remove(os.path.join(d, f))
PY

cat > "$BUILD/README.md" <<MD
---
title: Lumen - conceptos de personajes
emoji: ✨
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: DCGAN que genera conceptos de personajes elementales
---

Servicio en producción del Proyecto Final de Deep Learning (UVG 2026).
Interfaz en \`/\`, API en \`/docs\`, monitoreo en \`/monitoreo\`.
Código y documentación técnica en el repositorio de GitHub del proyecto.
MD

python3 - "$SPACE" "$BUILD" <<'PY'
import sys, datetime
from huggingface_hub import HfApi
space, carpeta = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(repo_id=space, repo_type="space", folder_path=carpeta,
                  commit_message=f"Despliegue {datetime.datetime.now():%Y-%m-%d %H:%M}")
PY
echo
echo "Listo. El Space se construye en https://huggingface.co/spaces/$SPACE"
echo "URL del servicio: https://$(echo "$SPACE" | tr '/' '-' | tr '[:upper:]_' '[:lower:]-').hf.space"
