"""Dibuja el diagrama del sistema (datos, entrenamiento, servicio, monitoreo).

Uso:  python scripts/diagrama_arquitectura.py  ->  evidencias/arquitectura.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RAIZ, "evidencias", "arquitectura.png")

ETAPAS = {  # x0, color, titulo
    "datos":         (0.2, "#e0f2fe", "1 · Datos"),
    "entrenamiento": (3.7, "#ede9fe", "2 · Entrenamiento"),
    "servicio":      (7.2, "#dcfce7", "3 · Servicio"),
    "monitoreo":     (10.7, "#fef3c7", "4 · Monitoreo"),
}
CAJAS = {
    "gen":   ("datos", 4.6, "lumen/data.py\ngenerador procedural\nSEED_DATA=20260101"),
    "ds":    ("datos", 2.6, "datos/lumen/\n5000 PNG 64x64\n(CC0, reproducible)"),
    "train": ("entrenamiento", 4.6, "lumen/train.py\nDCGAN + SN en D\n+ EMA de G (MPS/GPU)"),
    "exp":   ("entrenamiento", 2.6, "scripts/exportar_modelo.py\npesos fp16 + banco\nSqueezeNet + tau"),
    "eval":  ("entrenamiento", 0.6, "scripts/evaluar_hipotesis.py\nKID, diversidad, fp16\n-> modelo por defecto"),
    "api":   ("servicio", 2.6, "FastAPI + uvicorn\n(Docker, HF Spaces CPU)\n/api/generar, /docs"),
    "ui":    ("servicio", 4.6, "Interfaz web /\nequipo de arte:\ngenera y vota"),
    "ci":    ("servicio", 0.6, "GitHub Actions\npytest + build +\nprueba de humo"),
    "db":    ("monitoreo", 2.6, "SQLite\nsolicitudes y votos\n(sin datos personales)"),
    "dash":  ("monitoreo", 4.6, "/monitoreo\nlatencia, aceptacion,\nderiva de novedad"),
    "prom":  ("monitoreo", 0.6, "/metrics\nPrometheus"),
}
FLECHAS = [("gen", "ds"), ("ds", "train"), ("train", "exp"), ("exp", "eval"),
           ("exp", "api"), ("eval", "api"), ("ui", "api"), ("api", "db"),
           ("db", "dash"), ("api", "prom"), ("ci", "api")]
W, H = 3.1, 1.45


def centro(k):
    etapa, y, _ = CAJAS[k]
    return ETAPAS[etapa][0] + W / 2, y + H / 2


def main():
    fig, ax = plt.subplots(figsize=(14, 6.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.9); ax.axis("off")
    for etapa, (x0, color, titulo) in ETAPAS.items():
        ax.add_patch(FancyBboxPatch((x0 - .1, .35), W + .2, 6.1, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=color, ec="none", alpha=.6))
        ax.text(x0 + W / 2, 6.6, titulo, ha="center", va="center", fontsize=13, fontweight="bold", color="#1e293b")
    for k, (etapa, y, txt) in CAJAS.items():
        x0 = ETAPAS[etapa][0]
        ax.add_patch(FancyBboxPatch((x0, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc="white", ec="#334155", lw=1.2))
        ax.text(x0 + W / 2, y + H / 2, txt, ha="center", va="center", fontsize=9.2, color="#0f172a")
    for a, b in FLECHAS:
        (x1, y1), (x2, y2) = centro(a), centro(b)
        if abs(x1 - x2) < .1:       # misma columna: vertical
            p, q = (x1, y1 - H / 2), (x2, y2 + H / 2)
            if y2 > y1:
                p, q = (x1, y1 + H / 2), (x2, y2 - H / 2)
        else:
            p = (x1 + W / 2 * (1 if x2 > x1 else -1), y1)
            q = (x2 - W / 2 * (1 if x2 > x1 else -1), y2)
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color="#475569", lw=1.3,
                                     connectionstyle="arc3,rad=0.0"))
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(OUT)


if __name__ == "__main__":
    main()
