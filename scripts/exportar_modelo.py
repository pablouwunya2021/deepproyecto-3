"""
Exporta los artefactos que usa el servicio a `modelos/`:

  1. G_<nombre>.pt         pesos del generador en fp16 (< 10 MB, sin Git LFS)
  2. banco_features.npy    caracteristicas SqueezeNet de las 5000 imagenes reales (fp16)
  3. tarjeta_modelo.json   tau del filtro de novedad, referencia para monitoreo,
                           metadatos y modelo por defecto

Requiere los checkpoints fp32 en `checkpoints/` (Proyecto 2 + corrida nueva) y el
dataset en `datos/lumen` (`python -m lumen.data`).

Uso:  python scripts/exportar_modelo.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lumen import config as C                                     # noqa: E402
from lumen.data import LumenDataset                                # noqa: E402
from lumen.features import FeatureExtractor                        # noqa: E402
from lumen.inferencia import (BANCO, MODELOS_DIR, TARJETA, cargar_generador,  # noqa: E402
                              describir, z_de_semillas)
from lumen.utils import tensor_to_uint8                             # noqa: E402

MODELOS = {
    "base":                 "Proyecto 2 - DCGAN BCE + BatchNorm en D (linea base en produccion)",
    "hinge":                "Proyecto 2 - experimento A, perdida hinge",
    "spectral":             "Proyecto 2 - experimento B, normalizacion espectral en D",
    "spectral_ema":         "Proyecto Final - normalizacion espectral en D + EMA de G (0.995)",
    "spectral_ema_sin_ema": "Proyecto Final - misma corrida, G sin EMA (control)",
}
SERVIDOS = ["base", "spectral_ema"]
SEMILLA_CALIBRACION = 999        # distinta de la de evaluacion (777)
N_CALIBRACION = 1000


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


@torch.no_grad()
def main():
    os.makedirs(MODELOS_DIR, exist_ok=True)
    torch.set_grad_enabled(False)

    # 1) pesos fp16
    for nombre in list(MODELOS):
        src = os.path.join(C.CKPT_DIR, f"G_{nombre}.pt")
        if not os.path.exists(src):
            print(f"[aviso] falta {src}; se omite {nombre}")
            MODELOS.pop(nombre)
            continue
        estado = torch.load(src, map_location="cpu")
        fp16 = {k: v.half() if v.is_floating_point() else v for k, v in estado.items()}
        torch.save(fp16, os.path.join(MODELOS_DIR, f"G_{nombre}.pt"))
        print(f"[pesos] {nombre}: {os.path.getsize(src)/1e6:.1f} MB -> "
              f"{os.path.getsize(os.path.join(MODELOS_DIR, f'G_{nombre}.pt'))/1e6:.1f} MB")

    # 2) banco de caracteristicas de los datos reales
    fx = FeatureExtractor("cpu", estricto=True)
    ds = LumenDataset(C.IMG_DIR, 64)
    feats, elementos = [], []
    for i in range(0, len(ds), 250):
        lote = torch.stack([ds[j] for j in range(i, min(i + 250, len(ds)))])
        feats.append(fx(lote).numpy())
        elementos += [describir(tensor_to_uint8(x), 0)["elemento"] for x in lote]
    banco = np.concatenate(feats).astype(np.float16)
    np.save(BANCO, banco)
    print(f"[banco] {banco.shape} -> {BANCO}")
    banco_t = torch.from_numpy(banco.astype(np.float32))
    elem_ref = {e: elementos.count(e) / len(elementos) for e in sorted(set(elementos))}

    # 3) calibracion por modelo servido: tau y referencia de monitoreo
    rng = np.random.default_rng(SEMILLA_CALIBRACION)
    semillas = rng.integers(0, 2**31 - 1, size=N_CALIBRACION).tolist()
    info = {}
    for nombre in MODELOS:
        G = cargar_generador(nombre)
        nov, elems = [], []
        for i in range(0, N_CALIBRACION, 250):
            fake = G(z_de_semillas(semillas[i:i + 250]))
            nov.append(torch.cdist(fx(fake), banco_t).min(1).values.numpy())
            elems += [describir(tensor_to_uint8(x), 0)["elemento"] for x in fake]
        nov = np.concatenate(nov)
        info[nombre] = {
            "descripcion": MODELOS[nombre],
            "sha256_16": sha256(os.path.join(MODELOS_DIR, f"G_{nombre}.pt")),
            "tau": float(np.percentile(nov, 10)),
            "novedad_ref_media": float(nov.mean()),
            "novedad_ref_std": float(nov.std()),
            "elementos_ref": {e: elems.count(e) / len(elems) for e in sorted(set(elems))},
        }
        print(f"[calibracion] {nombre}: tau={info[nombre]['tau']:.4f} "
              f"novedad media={nov.mean():.4f}")

    previo = {}
    if os.path.exists(TARJETA):
        with open(TARJETA) as f:
            previo = json.load(f)
    tarjeta = {
        "nombre": "Lumen - generador de conceptos de personajes",
        "proyecto_origen": "Proyecto 2 (DCGAN 64x64), mejorado en el Proyecto Final",
        "exportado": dt.datetime.now().isoformat(timespec="seconds"),
        "arquitectura": "DCGAN: G con 5 ConvTranspose2d + BatchNorm + ReLU, z de 128 dims, salida 3x64x64 tanh",
        "extractor_features": "squeezenet1_1 (ImageNet), 512 dims, L2",
        "datos": {"n": len(ds), "semilla": C.SEED_DATA, "elementos_ref": elem_ref},
        "calibracion": {"semilla": SEMILLA_CALIBRACION, "n": N_CALIBRACION,
                        "tau": "percentil 10 de la novedad de muestras directas"},
        "modelos_servidos": SERVIDOS,
        # el modelo por defecto lo decide scripts/evaluar_hipotesis.py segun H1
        "modelo_por_defecto": previo.get("modelo_por_defecto", "base"),
        "decision_h1": previo.get("decision_h1", "pendiente"),
        "modelos": info,
    }
    with open(TARJETA, "w") as f:
        json.dump(tarjeta, f, indent=2, ensure_ascii=False)
    print(f"[tarjeta] {TARJETA}")


if __name__ == "__main__":
    main()
