"""
Construye presentacion/presentacion.pdf (10 min, ~11 diapositivas) a partir de los
artefactos reales: resultados de hipotesis, prueba de carga, tarjeta del modelo,
diagrama y capturas. Si cambian los resultados, se vuelve a correr y el PDF se
actualiza.

Uso:  python scripts/generar_presentacion.py
"""
from __future__ import annotations

import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(RAIZ, "evidencias")
OUT = os.path.join(RAIZ, "presentacion", "presentacion.pdf")
CFG = os.path.join(RAIZ, "presentacion", "datos_presentacion.json")

OSCURO, TENUE, ACENTO = "#0f172a", "#64748b", "#0284c7"
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})


def leer(ruta, defecto=None):
    try:
        with open(ruta) as f:
            return json.load(f)
    except FileNotFoundError:
        return defecto


def diapo(titulo, sub=None, n=None):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.91, titulo, fontsize=25, fontweight="bold", color=OSCURO)
    if sub:
        fig.text(0.05, 0.865, sub, fontsize=13.5, color=TENUE)
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.845, 0.845], color="#e2e8f0", lw=1.5))
    if n:
        fig.text(0.95, 0.03, str(n), fontsize=11, color=TENUE, ha="right")
    return fig


def vinetas(fig, items, x=0.06, y=0.78, ancho=95, paso=0.062, tam=15.5):
    for it in items:
        sub = it.startswith("  ")
        lineas = textwrap.wrap(it.strip(), ancho - (6 if sub else 0))
        fig.text(x + (0.03 if sub else 0), y, ("–  " if sub else "•  ") + lineas[0],
                 fontsize=tam - (1.5 if sub else 0), color=OSCURO if not sub else "#334155")
        for l in lineas[1:]:
            y -= 0.042
            fig.text(x + (0.03 if sub else 0) + 0.018, y, l, fontsize=tam - (1.5 if sub else 0),
                     color=OSCURO if not sub else "#334155")
        y -= paso
    return y


def imagen(fig, ruta, rect):
    if ruta and os.path.exists(ruta):
        ax = fig.add_axes(rect)
        ax.imshow(mpimg.imread(ruta)); ax.axis("off")
        return True
    ax = fig.add_axes(rect); ax.axis("off")
    ax.text(.5, .5, f"(falta {os.path.relpath(ruta, RAIZ) if ruta else 'imagen'})", ha="center",
            color=TENUE)
    return False


def tabla(fig, filas, rect, anchos=None, tam=12.5, cabecera=True, alto=1.75):
    ax = fig.add_axes(rect); ax.axis("off")
    t = ax.table(cellText=filas[1:] if cabecera else filas, colLabels=filas[0] if cabecera else None,
                 loc="upper center", cellLoc="left", colWidths=anchos)
    t.auto_set_font_size(False); t.set_fontsize(tam); t.scale(1, alto)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if r == 0 and cabecera:
            cell.set_facecolor("#f1f5f9"); cell.set_text_props(fontweight="bold")
    return t


def main():
    d = leer(CFG, {})
    res = leer(os.path.join(EV, "hipotesis", "resultados.json"), {})
    carga = leer(os.path.join(EV, "prueba_carga.json"), {})
    uso = leer(os.path.join(EV, "uso_produccion.json"), {})
    h1, h2, h3 = res.get("H1", {}), res.get("H2", {}), res.get("H3", {})
    url = d.get("url_servicio", "(pendiente: URL del Space)")
    repo = d.get("url_repositorio", "(pendiente: URL del repositorio)")
    integrantes = d.get("integrantes", ["Integrante 1", "Integrante 2", "Integrante 3"])
    ok = lambda b: "CUMPLE" if b else ("PENDIENTE" if b is None else "NO CUMPLE")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with PdfPages(OUT) as pdf:
        # 1 portada
        fig = plt.figure(figsize=(13.33, 7.5)); fig.patch.set_facecolor("#0b0d14")
        fig.text(0.06, 0.70, "Lumen en producción", fontsize=44, fontweight="bold", color="white")
        fig.text(0.06, 0.62, "Un generador de conceptos de personajes (DCGAN) que el equipo de arte ya puede usar",
                 fontsize=18, color="#94a3b8")
        fig.text(0.06, 0.44, "   ·   ".join(integrantes), fontsize=16, color="#e2e8f0")
        fig.text(0.06, 0.38, "Deep Learning y Sistemas Inteligentes · Universidad del Valle · 2026",
                 fontsize=13, color="#94a3b8")
        fig.text(0.06, 0.20, f"Servicio:  {url}", fontsize=14, color="#7dd3fc")
        fig.text(0.06, 0.15, f"Código:    {repo}", fontsize=14, color="#7dd3fc")
        imagen(fig, os.path.join(EV, "hipotesis", "muestras_base.png"), [0.55, 0.06, 0.42, 0.22])
        pdf.savefig(fig); plt.close(fig)

        # 2 problematica
        fig = diapo("Problemática", "Quién lo usa y qué decisión depende del modelo", 2)
        vinetas(fig, [
            "Usuario: el equipo de arte de un estudio indie que diseña los espíritus guardianes del videojuego Lumen.",
            "Problema: cada concepto inicial toma horas de boceto; el equipo necesita muchas ideas variadas para elegir pocas.",
            "Decisión del usuario: qué conceptos pasan a refinamiento ('me sirve') y cuáles se descartan.",
            "Decisión de este comité: mantener o no el modelo en producción. Lo medimos con:",
            "  Tasa de aceptación de conceptos (votos en la interfaz) y novedad respecto a los datos de entrenamiento.",
            "  Latencia y errores del servicio, y alertas de deriva (colapso de modos).",
            "Fuera de alcance: arte final. El modelo produce bocetos 64×64 de partida, no assets terminados.",
        ])
        pdf.savefig(fig); plt.close(fig)

        # 3 planificacion
        fig = diapo("Planificación", "Etapas, fechas y responsables", 3)
        plan = d.get("planificacion", [])
        if plan:
            tabla(fig, [["Etapa", "Fechas", "Responsable", "Entregable"]] + plan,
                  [0.05, 0.08, 0.9, 0.72], anchos=[0.3, 0.14, 0.18, 0.38], tam=13, alto=2.6)
        pdf.savefig(fig); plt.close(fig)

        # 4 supuestos
        fig = diapo("Supuestos", "Sobre los datos, el usuario y el entorno", 4)
        vinetas(fig, [
            "Datos: el dataset procedural (5000 imágenes, CC0, reproducible byte a byte) representa el estilo que quiere el equipo.",
            "Datos: la novedad se mide en el espacio de SqueezeNet (ImageNet); asumimos que ahí 'parecido' ≈ parecido visual.",
            "Usuario: busca variedad e ideas de partida, no fidelidad; acepta resolución 64×64 (se escala ×4 sin suavizado).",
            "Usuario: su voto 'me sirve / descartar' es una señal honesta de utilidad (sin incentivos para votar al azar).",
            "Entorno: CPU compartida gratuita (HF Spaces, 2 vCPU); tráfico bajo (decenas de solicitudes/hora).",
            "Entorno: el disco del Space es efímero → los registros se exportan (/api/registros) antes de reinicios.",
        ], paso=0.075)
        pdf.savefig(fig); plt.close(fig)

        # 5 modelo
        fig = diapo("Modelo planteado", "Proyecto 2 (DCGAN 64×64) + cambios justificados", 5)
        vinetas(fig, [
            "Origen: DCGAN del Proyecto 2. G: 5 ConvTranspose2d + BatchNorm + ReLU, z ∈ R¹²⁸ → 3×64×64 (tanh).",
            "Propuesta 1 · normalización espectral en D: en el P2 llevó D(x), D(G(z)) a ≈0.56 / 0.44 (equilibrio).",
            "Propuesta 2 · EMA de los pesos de G (0.995): servir el promedio suaviza la oscilación adversarial. (H1 decide si se adoptan 1 y 2)",
            "Cambio · filtro en el servicio: 4n candidatos → descarta novedad < τ → elige n por punto más lejano.",
            "Cambio · pesos fp16 (15 → 7.6 MB): caben en HF Spaces sin Git LFS; se ejecutan en fp32.",
            "Reproducible: la corrida nueva repite exactamente las pérdidas del P2 (misma semilla y datos).",
        ], y=0.79, ancho=60, paso=0.083, tam=14)
        imagen(fig, os.path.join(EV, "hipotesis", "h1_muestras.png"), [0.62, 0.05, 0.36, 0.78])
        pdf.savefig(fig); plt.close(fig)

        # 6 H1
        fig = diapo("Validación · H1: ¿el modelo nuevo es mejor?",
                    f"Criterio previo: KID(spectral_ema) ≤ 0.85 × KID(base)  →  {ok(h1.get('cumple'))}", 6)
        imagen(fig, os.path.join(EV, "hipotesis", "h1_kid.png"), [0.04, 0.06, 0.58, 0.76])
        if h1:
            k = h1["kid"]
            vinetas(fig, [
                f"Línea base (base): KID = {k['base']['media']*1e3:.2f}×10⁻³",
                f"spectral_ema: {k['spectral_ema']['media']*1e3:.2f}×10⁻³",
                f"Razón = {h1['razon_spectral_ema_vs_base']:.2f} (criterio ≤ 0.85)",
                "Piso real vs real: ≈ 0 (datos contra datos)",
                f"EMA aislada: sin EMA = {k['spectral_ema_sin_ema']['media']*1e3:.2f}×10⁻³",
                f"Decisión: servir '{h1['modelo_por_defecto']}' por defecto",
            ], x=0.64, y=0.76, ancho=36, paso=0.085, tam=14.5)
        pdf.savefig(fig); plt.close(fig)

        # 7 H2
        fig = diapo("Validación · H2: ¿el filtro entrega lotes más útiles?",
                    f"Criterio previo: diversidad ≥ ×1.10, cobertura ≥ base, KID ≤ ×1.25  →  {ok(h2.get('cumple'))}", 7)
        imagen(fig, os.path.join(EV, "hipotesis", "h2_filtro.png"), [0.03, 0.32, 0.94, 0.5])
        if h2:
            tabla(fig, [["Métrica (200 lotes de 8)", "Directo (línea base)", "Con filtro", "Criterio"],
                        ["Diversidad intra-lote", f"{h2['diversidad']['directo']:.3f}",
                         f"{h2['diversidad']['filtro']:.3f} (×{h2['razon_diversidad']:.2f})", "≥ ×1.10"],
                        ["Elementos distintos / lote", f"{h2['cobertura_elementos']['directo']:.2f}",
                         f"{h2['cobertura_elementos']['filtro']:.2f}", "≥ directo"],
                        ["KID ×10⁻³", f"{h2['kid']['directo']*1e3:.2f}",
                         f"{h2['kid']['filtro']*1e3:.2f} (×{h2['razon_kid']:.2f})", "≤ ×1.25"]],
                  [0.08, 0.04, 0.84, 0.25], anchos=[0.34, 0.2, 0.26, 0.2])
        pdf.savefig(fig); plt.close(fig)

        # 8 H3 + operacion
        fig = diapo("Validación · H3: ¿es operable en CPU?", f"fp16 y latencia  →  {ok(h3.get('cumple'))}", 8)
        if h3:
            lat = carga.get("latencia_ms", {})
            tabla(fig, [["Métrica", "Resultado", "Criterio"],
                        ["Diferencia media de píxel fp32 vs fp16", f"{h3['dif_media_pixel_0_255']:.3f} / 255", "< 1 / 255"],
                        ["Mismo elemento asignado (500 muestras)", f"{100*h3['acuerdo_elemento']:.1f} %", "≥ 99 %"],
                        ["Tamaño de pesos", f"{h3['tam_mb_fp32']:.1f} → {h3['tam_mb_fp16']:.1f} MB", "< 10 MB"],
                        ["Latencia p95 con filtro (n=8, 4 clientes)",
                         f"{lat.get('p95', 0):.0f} ms" if lat else "pendiente", "< 1500 ms"],
                        ["Latencia p50 / rendimiento",
                         f"{lat.get('p50', 0):.0f} ms · {carga.get('rps', 0):.1f} sol/s" if lat else "—", "—"],
                        ["Latencia p95 con 1 solo cliente",
                         f"{leer(os.path.join(EV, 'prueba_carga_1cliente.json'), {}).get('latencia_ms', {}).get('p95', 0):.0f} ms", "—"],
                        ["Errores en la prueba de carga",
                         f"{carga.get('errores', '—')} de {carga.get('solicitudes', '—')}", "0"]],
                  [0.06, 0.1, 0.88, 0.7], anchos=[0.5, 0.3, 0.2], tam=15, alto=3.0)
        pdf.savefig(fig); plt.close(fig)

        # 9 arquitectura
        fig = diapo("Herramientas y arquitectura", "Datos → entrenamiento → servicio → monitoreo", 9)
        imagen(fig, os.path.join(EV, "arquitectura.png"), [0.03, 0.12, 0.94, 0.72])
        fig.text(0.05, 0.06, "PyTorch · FastAPI · Docker · Hugging Face Spaces · GitHub Actions · SQLite · Prometheus",
                 fontsize=14, color=TENUE)
        pdf.savefig(fig); plt.close(fig)

        # 10 evidencias
        fig = diapo("Evidencias del pase a producción", url, 10)
        imagen(fig, os.path.join(EV, "capturas", "interfaz.png"), [0.03, 0.08, 0.46, 0.74])
        imagen(fig, os.path.join(EV, "capturas", "monitoreo.png"), [0.51, 0.08, 0.46, 0.74])
        pdf.savefig(fig); plt.close(fig)

        # 11 uso real + decision
        fig = diapo("Recomendación al comité", "¿Mantener el modelo en producción?", 11)
        items = []
        if uso:
            items += [f"Uso registrado: {uso.get('generaciones', 0)} generaciones, "
                      f"{uso.get('personajes_entregados', 0)} personajes, {uso.get('clientes_distintos', 0)} clientes distintos.",
                      f"Latencia p95 en producción: {uso.get('latencia_ms', {}).get('p95') or 0:.0f} ms · "
                      f"tasa de error {100*uso.get('tasa_error', 0):.1f} %."]
        items += d.get("recomendacion", [
            "Mantener en producción con el modelo elegido por H1 y el filtro activado por defecto.",
            "Condición de retiro: tasa de aceptación < 20 % sostenida, o alerta de deriva de novedad.",
            "Siguiente paso: reentrenar con los conceptos aceptados por el equipo (aprendizaje con retroalimentación).",
        ])
        vinetas(fig, items, paso=0.085)
        pdf.savefig(fig); plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
