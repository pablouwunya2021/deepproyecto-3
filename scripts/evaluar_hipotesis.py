"""
Evalua H1, H2 y H3 con el protocolo preregistrado en docs/hipotesis.md y escribe:

  docs/resultados.md                      tabla de resultados y decision
  evidencias/hipotesis/resultados.json    todos los numeros
  evidencias/hipotesis/*.png              figuras para el README y la presentacion
  modelos/tarjeta_modelo.json             modelo_por_defecto segun H1

Uso:  python scripts/evaluar_hipotesis.py
(la parte de latencia de H3 lee evidencias/prueba_carga.json si ya se corrio
 scripts/prueba_carga.py; si no, queda pendiente)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np
import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from lumen import config as C                                         # noqa: E402
from lumen.features import FeatureExtractor                           # noqa: E402
from lumen.inferencia import (BANCO, TARJETA, MotorLumen, cargar_generador,  # noqa: E402
                              describir, z_de_semillas)
from lumen.models import Generator                                    # noqa: E402
from lumen.utils import grid_image, tensor_to_uint8                   # noqa: E402

SEMILLA_EVAL_FINAL = 777
N_GEN = 2000
KID_SUBSETS, KID_TAM = 10, 1000
MODELOS = ["base", "hinge", "spectral", "spectral_ema_sin_ema", "spectral_ema"]
CRITERIO_H1 = 0.85
OUT = os.path.join(RAIZ, "evidencias", "hipotesis")
torch.set_grad_enabled(False)


# --------------------------------------------------------------------------- KID
def _kernel(a, b):
    return (a @ b.T / a.shape[1] + 1.0) ** 3


def kid(reales: np.ndarray, gen: np.ndarray, semilla: int = 0):
    """KID insesgado (Binkowski et al., 2018), media y desviacion sobre subconjuntos."""
    rng = np.random.default_rng(semilla)
    m = min(KID_TAM, len(reales), len(gen))
    vals = []
    for _ in range(KID_SUBSETS):
        x = reales[rng.choice(len(reales), m, replace=False)].astype(np.float64)
        y = gen[rng.choice(len(gen), m, replace=False)].astype(np.float64)
        kxx, kyy, kxy = _kernel(x, x), _kernel(y, y), _kernel(x, y)
        mmd = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
               + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


def feats_de(G, fx, semillas):
    out = []
    for i in range(0, len(semillas), 250):
        fake = G(z_de_semillas(semillas[i:i + 250]))
        out.append(fx(fake).numpy())
    return np.concatenate(out)


def diversidad(f: np.ndarray) -> float:
    d = np.linalg.norm(f[:, None] - f[None], axis=-1)
    n = len(f)
    return float(d.sum() / (n * (n - 1)))


def main():
    os.makedirs(OUT, exist_ok=True)
    fx = FeatureExtractor("cpu", estricto=True)
    reales = np.load(BANCO).astype(np.float32)
    semillas = np.random.default_rng(SEMILLA_EVAL_FINAL).integers(0, 2**31 - 1, N_GEN).tolist()
    res = {"fecha": dt.datetime.now().isoformat(timespec="seconds"),
           "protocolo": {"semilla": SEMILLA_EVAL_FINAL, "n_generadas": N_GEN,
                         "kid_subconjuntos": KID_SUBSETS, "kid_tam": KID_TAM}}

    # ------------------------------------------------------------------ H1
    perm = np.random.default_rng(1).permutation(len(reales))
    piso = kid(reales[perm[:2500]], reales[perm[2500:]])
    kids, gens = {}, {}
    for nombre in MODELOS:
        G = cargar_generador(nombre)
        gens[nombre] = feats_de(G, fx, semillas)
        kids[nombre] = kid(reales, gens[nombre])
        print(f"[H1] KID {nombre:22s} = {kids[nombre][0]*1e3:.3f} ± {kids[nombre][1]*1e3:.3f} (x1e-3)")
    print(f"[H1] KID real vs real (piso) = {piso[0]*1e3:.3f}")
    razon = kids["spectral_ema"][0] / kids["base"][0]
    h1_ok = razon <= CRITERIO_H1
    defecto = "spectral_ema" if h1_ok else "base"
    res["H1"] = {"kid": {k: {"media": v[0], "std": v[1]} for k, v in kids.items()},
                 "kid_piso_real_vs_real": {"media": piso[0], "std": piso[1]},
                 "razon_spectral_ema_vs_base": razon, "criterio": f"<= {CRITERIO_H1}",
                 "cumple": bool(h1_ok), "modelo_por_defecto": defecto}

    with open(TARJETA) as f:
        tarjeta = json.load(f)
    tarjeta["modelo_por_defecto"] = defecto
    tarjeta["decision_h1"] = (f"KID spectral_ema / base = {razon:.3f} "
                              f"({'cumple' if h1_ok else 'no cumple'} <= {CRITERIO_H1})")
    with open(TARJETA, "w") as f:
        json.dump(tarjeta, f, indent=2, ensure_ascii=False)

    # figura KID
    fig, ax = plt.subplots(figsize=(8, 4.2))
    nombres = MODELOS
    ys = [kids[n][0] * 1e3 for n in nombres]
    es = [kids[n][1] * 1e3 for n in nombres]
    col = ["#94a3b8" if n != "spectral_ema" else "#0ea5e9" for n in nombres]
    col[0] = "#475569"
    ax.bar(nombres, ys, yerr=es, color=col, capsize=4)
    ax.axhline(piso[0] * 1e3, ls=":", c="#16a34a", label="piso: real vs real")
    ax.axhline(kids["base"][0] * 1e3 * CRITERIO_H1, ls="--", c="#dc2626",
               label=f"criterio H1: {CRITERIO_H1} x base")
    ax.set_ylabel("KID x 1e-3 (menor es mejor)")
    ax.set_title("H1 - distancia a la distribucion real por generador")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    plt.xticks(rotation=15); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "h1_kid.png"), dpi=130); plt.close(fig)

    # rejillas con las mismas 16 semillas para comparar visualmente
    tiles = []
    for nombre in MODELOS:
        G = cargar_generador(nombre)
        g = grid_image(G(z_de_semillas(semillas[:16])), nrow=8)
        g.save(os.path.join(OUT, f"muestras_{nombre}.png"))
        tiles.append((nombre, g))
    fig, axs = plt.subplots(len(tiles), 1, figsize=(9, 2.6 * len(tiles)))
    for a, (n, g) in zip(axs, tiles):
        a.imshow(g); a.set_title(n, loc="left", fontsize=11); a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "h1_muestras.png"), dpi=110); plt.close(fig)

    # ------------------------------------------------------------------ H2
    motor = MotorLumen(modelos=[defecto], estricto=True)
    motor.por_defecto = defecto
    banco_t = torch.from_numpy(reales)
    div, cob, f_dir, f_fil, nov_dir, nov_fil = {"directo": [], "filtro": []}, {"directo": [], "filtro": []}, [], [], [], []
    for b in range(200):
        for modo, filtro in (("directo", False), ("filtro", True)):
            ps = motor.generar(8, 10_000 + b, defecto, filtro)
            fake = motor.G[defecto](z_de_semillas([p.semilla for p in ps]))
            f = fx(fake).numpy()
            div[modo].append(diversidad(f))
            cob[modo].append(len({p.elemento for p in ps}))
            (f_dir if modo == "directo" else f_fil).append(f)
            (nov_dir if modo == "directo" else nov_fil).extend(p.novedad for p in ps)
    f_dir, f_fil = np.concatenate(f_dir), np.concatenate(f_fil)
    kid_dir, kid_fil = kid(reales, f_dir), kid(reales, f_fil)
    r_div = np.mean(div["filtro"]) / np.mean(div["directo"])
    h2_ok = (r_div >= 1.10 and np.mean(cob["filtro"]) >= np.mean(cob["directo"])
             and kid_fil[0] <= 1.25 * kid_dir[0])
    res["H2"] = {"modelo": defecto, "lotes": 200, "n": 8,
                 "diversidad": {k: float(np.mean(v)) for k, v in div.items()},
                 "razon_diversidad": float(r_div),
                 "cobertura_elementos": {k: float(np.mean(v)) for k, v in cob.items()},
                 "novedad_media": {"directo": float(np.mean(nov_dir)), "filtro": float(np.mean(nov_fil))},
                 "kid": {"directo": kid_dir[0], "filtro": kid_fil[0]},
                 "razon_kid": kid_fil[0] / kid_dir[0], "cumple": bool(h2_ok)}
    print(f"[H2] diversidad x{r_div:.3f} | cobertura {np.mean(cob['directo']):.2f} -> "
          f"{np.mean(cob['filtro']):.2f} | KID x{kid_fil[0]/kid_dir[0]:.3f} -> "
          f"{'CUMPLE' if h2_ok else 'NO CUMPLE'}")

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    axs[0].hist(div["directo"], 25, alpha=.6, label="directo", color="#94a3b8")
    axs[0].hist(div["filtro"], 25, alpha=.6, label="filtro", color="#0ea5e9")
    axs[0].set_title("Diversidad intra-lote (n=8)"); axs[0].legend()
    vals = sorted(set(cob["directo"]) | set(cob["filtro"]))
    w = .38
    axs[1].bar(np.array(vals) - w / 2, [cob["directo"].count(v) for v in vals], w, label="directo", color="#94a3b8")
    axs[1].bar(np.array(vals) + w / 2, [cob["filtro"].count(v) for v in vals], w, label="filtro", color="#0ea5e9")
    axs[1].set_title("Elementos distintos por lote (de 6)"); axs[1].legend()
    axs[2].hist(nov_dir, 30, alpha=.6, label="directo", color="#94a3b8")
    axs[2].hist(nov_fil, 30, alpha=.6, label="filtro", color="#0ea5e9")
    axs[2].axvline(motor.tau(defecto), c="#dc2626", ls="--", label="tau")
    axs[2].set_title("Novedad (dist. al vecino real)"); axs[2].legend()
    fig.suptitle(f"H2 - filtro de novedad y diversidad ({defecto}, 200 lotes)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "h2_filtro.png"), dpi=130); plt.close(fig)

    # ------------------------------------------------------------------ H3
    cfg = C.base_config()
    G32 = Generator(cfg.nz, cfg.ngf, cfg.channels)
    G32.load_state_dict(torch.load(os.path.join(C.CKPT_DIR, f"G_{defecto}.pt"), map_location="cpu"))
    G32.eval()
    G16 = cargar_generador(defecto)
    s500 = semillas[:500]
    a = G32(z_de_semillas(s500)); b = G16(z_de_semillas(s500))
    dif = ((a - b).abs() * 127.5).mean().item()          # escala 0-255
    dif_max = ((a - b).abs() * 127.5).max().item()
    acuerdo = np.mean([describir(tensor_to_uint8(x), 0)["elemento"] == describir(tensor_to_uint8(y), 0)["elemento"]
                       for x, y in zip(a, b)])
    h3 = {"dif_media_pixel_0_255": dif, "dif_max_pixel_0_255": dif_max,
          "acuerdo_elemento": float(acuerdo),
          "tam_mb_fp32": os.path.getsize(os.path.join(C.CKPT_DIR, f"G_{defecto}.pt")) / 1e6,
          "tam_mb_fp16": os.path.getsize(os.path.join(RAIZ, "modelos", f"G_{defecto}.pt")) / 1e6}
    carga = os.path.join(RAIZ, "evidencias", "prueba_carga.json")
    p95 = None
    if os.path.exists(carga):
        with open(carga) as f:
            pc = json.load(f)
        p95 = pc["latencia_ms"]["p95"]
        h3["prueba_carga"] = pc
    h3["latencia_p95_ms"] = p95
    h3["cumple"] = bool(dif < 1 and acuerdo >= .99 and p95 is not None and p95 < 1500) if p95 else None
    res["H3"] = h3
    print(f"[H3] dif media {dif:.4f}/255, max {dif_max:.3f}, acuerdo {acuerdo:.3f}, p95 {p95}")

    with open(os.path.join(OUT, "resultados.json"), "w") as f:
        json.dump(res, f, indent=2)
    escribir_md(res, defecto)


def extra_h3(h3):
    """Mediciones complementarias de latencia (contenedor Docker, --cpus 2)."""
    def leer(nombre):
        ruta = os.path.join(RAIZ, "evidencias", nombre)
        return json.load(open(ruta)) if os.path.exists(ruta) else None
    v1, uno, pc = leer("prueba_carga_v1_sin_candado.json"), leer("prueba_carga_1cliente.json"), h3.get("prueba_carga")
    if not (v1 or uno or pc):
        return ""
    filas = ["", "Mediciones de latencia en el contenedor Docker limitado a 2 CPU (como HF Spaces gratis):", "",
             "| Medición | Clientes | p50 | p95 | Rendimiento | Errores |", "|---|---|---|---|---|---|"]
    for etq, d in (("v1: inferencias en paralelo", v1), ("v2: inferencia serializada (candado)", pc),
                   ("v2: un solo cliente", uno)):
        if d:
            l = d["latencia_ms"]
            filas.append(f"| {etq} | {d['clientes_concurrentes']} | {l['p50']:.0f} ms | {l['p95']:.0f} ms "
                         f"| {d['rps']:.2f} sol/s | {d['errores']}/{d['solicitudes']} |")
    return "\n".join(filas) + "\n"


def escribir_md(r, defecto):
    h1, h2, h3 = r["H1"], r["H2"], r["H3"]
    s = lambda ok: "✅ cumple" if ok else ("⏳ pendiente" if ok is None else "❌ no cumple")
    filas = "\n".join(
        f"| `{k}` | {v['media']*1e3:.3f} ± {v['std']*1e3:.3f} | {v['media']/h1['kid']['base']['media']:.2f} |"
        for k, v in h1["kid"].items())
    md = f"""# Resultados de las hipótesis

> Generado por `scripts/evaluar_hipotesis.py` el {r['fecha']}. Criterios en
> [`hipotesis.md`](hipotesis.md) (registrados antes de entrenar). Números completos en
> `evidencias/hipotesis/resultados.json`.

## H1 — normalización espectral + EMA vs línea base · {s(h1['cumple'])}

| Generador | KID × 10⁻³ (media ± desv.) | relativo a `base` |
|---|---|---|
{filas}
| *piso: real vs real* | {h1['kid_piso_real_vs_real']['media']*1e3:.3f} | — |

Razón `spectral_ema / base` = **{h1['razon_spectral_ema_vs_base']:.3f}** (criterio ≤ {CRITERIO_H1}).
Decisión: el servicio usa **`{defecto}`** por defecto.

![KID](../evidencias/hipotesis/h1_kid.png)
![Muestras](../evidencias/hipotesis/h1_muestras.png)

## H2 — filtro de novedad y diversidad (modelo `{h2['modelo']}`) · {s(h2['cumple'])}

| Métrica | Directo (línea base) | Con filtro | Criterio |
|---|---|---|---|
| Diversidad intra-lote | {h2['diversidad']['directo']:.4f} | {h2['diversidad']['filtro']:.4f} (×{h2['razon_diversidad']:.3f}) | ≥ ×1.10 |
| Elementos distintos por lote | {h2['cobertura_elementos']['directo']:.2f} | {h2['cobertura_elementos']['filtro']:.2f} | ≥ directo |
| KID × 10⁻³ | {h2['kid']['directo']*1e3:.3f} | {h2['kid']['filtro']*1e3:.3f} (×{h2['razon_kid']:.3f}) | ≤ ×1.25 |
| Novedad media | {h2['novedad_media']['directo']:.4f} | {h2['novedad_media']['filtro']:.4f} | — |

![H2](../evidencias/hipotesis/h2_filtro.png)

## H3 — pesos fp16 y latencia en CPU · {s(h3['cumple'])}

| Métrica | Valor | Criterio |
|---|---|---|
| Diferencia media de píxel fp32 vs fp16 | {h3['dif_media_pixel_0_255']:.4f} / 255 | < 1 / 255 |
| Diferencia máxima de píxel | {h3['dif_max_pixel_0_255']:.3f} / 255 | — |
| Mismo elemento asignado (500 muestras) | {100*h3['acuerdo_elemento']:.1f} % | ≥ 99 % |
| Tamaño del archivo de pesos | {h3['tam_mb_fp32']:.1f} MB → {h3['tam_mb_fp16']:.1f} MB | < 10 MB |
| Latencia p95 `POST /api/generar` (n=8, filtro, 4 clientes) | {('%.0f ms' % h3['latencia_p95_ms']) if h3['latencia_p95_ms'] else 'pendiente'} | < 1500 ms |
{extra_h3(h3)}"""
    with open(os.path.join(RAIZ, "docs", "resultados.md"), "w") as f:
        f.write(md)
    print("[ok] docs/resultados.md")


if __name__ == "__main__":
    main()
