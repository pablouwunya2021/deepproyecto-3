"""
Genera presentacion/guion_presentacion.pdf: guion de la presentacion de 10 minutos,
diapositiva por diapositiva, alineado con la rubrica del proyecto final, mas una
guia de preguntas probables del comite.

    python scripts/generar_guion.py

Los numeros salen de evidencias/ (los mismos que usa la presentacion) y los nombres y
la URL de presentacion/datos_presentacion.json.
"""
from __future__ import annotations

import json
import os

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "presentacion", "guion_presentacion.pdf")

# DejaVu (incluida con matplotlib) cubre ≤ ≥ → × ⁻³ τ, que Helvetica no trae
_F = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(_F, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", os.path.join(_F, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", os.path.join(_F, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

OSCURO = colors.HexColor("#0f172a")
ACENTO = colors.HexColor("#0369a1")
TENUE = colors.HexColor("#64748b")
FONDO = colors.HexColor("#f1f5f9")
AVISO = colors.HexColor("#fef3c7")

S = {
    "titulo": ParagraphStyle("titulo", fontName="DV-B", fontSize=22, leading=27, textColor=OSCURO),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=11.5, leading=15, textColor=TENUE),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=15, leading=19, textColor=OSCURO,
                         spaceBefore=6, spaceAfter=4),
    "meta": ParagraphStyle("meta", fontName="DV", fontSize=9.5, leading=13, textColor=ACENTO,
                           spaceAfter=5),
    "etq": ParagraphStyle("etq", fontName="DV-B", fontSize=9.5, leading=13, textColor=TENUE,
                          spaceBefore=5, spaceAfter=2),
    "habla": ParagraphStyle("habla", fontName="DV", fontSize=10.5, leading=15.5, textColor=OSCURO,
                            alignment=TA_LEFT),
    "txt": ParagraphStyle("txt", fontName="DV", fontSize=10, leading=14, textColor=OSCURO),
    "vin": ParagraphStyle("vin", fontName="DV", fontSize=10, leading=14, textColor=OSCURO,
                          leftIndent=12, bulletIndent=2),
    "celda": ParagraphStyle("celda", fontName="DV", fontSize=9, leading=12, textColor=OSCURO),
    "celda_b": ParagraphStyle("celda_b", fontName="DV-B", fontSize=9, leading=12, textColor=OSCURO),
    "preg": ParagraphStyle("preg", fontName="DV-B", fontSize=10, leading=14, textColor=OSCURO,
                           spaceBefore=6),
    "resp": ParagraphStyle("resp", fontName="DV", fontSize=10, leading=14, textColor=OSCURO,
                           leftIndent=10),
}


def leer(ruta, defecto=None):
    try:
        with open(os.path.join(RAIZ, ruta), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return defecto


def caja(parrafos, fondo=FONDO):
    t = Table([[p] for p in parrafos], colWidths=[17 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fondo),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACENTO),
    ]))
    return t


def tabla(filas, anchos):
    datos = [[Paragraph(str(c), S["celda_b"] if i == 0 else S["celda"]) for c in fila]
             for i, fila in enumerate(filas)]
    t = Table(datos, colWidths=[a * cm for a in anchos], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), FONDO),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, OSCURO),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def vinetas(items):
    return [Paragraph(i, S["vin"], bulletText="•") for i in items]


def mmss(seg):
    return f"{seg // 60}:{seg % 60:02d}"


def main():
    d = leer("presentacion/datos_presentacion.json", {})
    res = leer("evidencias/hipotesis/resultados.json", {})
    uso = leer("evidencias/registros/estadisticas_prueba_local.json", {})
    h1, h2, h3 = res["H1"], res["H2"], res["H3"]
    k = h1["kid"]
    pc = h3["prueba_carga"]
    v1 = leer("evidencias/prueba_carga_v1_sin_candado.json", {})
    uno = leer("evidencias/prueba_carga_1cliente.json", {})
    der = uso.get("deriva", {}).get("base", {})

    nombres = [n.split(" (")[0] for n in d.get("integrantes", ["Integrante 1", "Integrante 2",
                                                                 "Integrante 3"])]
    A, B, C = (nombres + ["Integrante 2", "Integrante 3"])[:3]
    url = d.get("url_servicio", "")
    url = url if url.startswith("http") else "[URL del Space]"
    repo = d.get("url_repositorio", "[URL del repositorio]")

    kb, ke, ks = (k["base"]["media"] * 1e3, k["spectral_ema"]["media"] * 1e3,
                  k["spectral_ema_sin_ema"]["media"] * 1e3)
    p95_1 = uno.get("latencia_ms", {}).get("p95", 579) / 1000
    p95_v1 = v1.get("latencia_ms", {}).get("p95", 3014) / 1000
    rps_v1 = v1.get("rps", 1.58)

    # (n, titulo, quien, segundos, criterio, texto hablado, que senalar)
    diapos = [
        (1, "Portada", A, 20, "Presentación (15 pts)",
         f"Buenos días. Somos {A}, {B} y {C}. Les presentamos <b>Lumen</b>: el generador de "
         "personajes del Proyecto 2, convertido en un servicio web que cualquier persona puede "
         f"usar desde el navegador. Hoy venimos a pedirles una decisión: si el modelo se queda "
         "en producción. Nuestra respuesta, y la vamos a sustentar, es <b>que sí, con el modelo "
         "base</b>.",
         ["Decir la URL en voz alta o dejarla visible en la portada."]),
        (2, "Problemática", A, 50, "Problemática, planificación y supuestos (10 pts)",
         "Nuestro usuario es el <b>equipo de arte</b> de un estudio indie que diseña los "
         "espíritus guardianes del videojuego Lumen. Su problema: cada concepto inicial toma "
         "horas de boceto y necesitan muchas ideas variadas para quedarse con unas pocas. Lumen "
         "les entrega un lote de ocho bocetos en menos de un segundo.<br/><br/>"
         "Hay dos decisiones que dependen del modelo. La del artista: qué concepto pasa a "
         "refinamiento; lo marca con «me sirve» o «descartar». Y la de ustedes como comité: "
         "mantenerlo o retirarlo. Para esa segunda decisión medimos cinco cosas: tasa de "
         "aceptación, novedad (que no copie los datos), latencia, errores y deriva. "
         "Fuera de alcance: no es arte final, son bocetos de partida de 64 por 64.",
         ["Las dos decisiones: la del artista y la del comité.",
          "Las cinco métricas: son las mismas que aparecen al final en la recomendación."]),
        (3, "Planificación", A, 40, "Problemática, planificación y supuestos (10 pts)",
         "Lo organizamos en diez etapas. Lo primero fue elegir el modelo y <b>escribir las "
         "hipótesis con su criterio de éxito antes de entrenar</b>; eso está fechado en "
         f"docs/hipotesis.md. Yo porté el código, entrené la mejora y corrí la evaluación. {B} "
         f"hizo el preregistro, las pruebas, Docker y la integración continua. {C} hizo el "
         "servicio, la interfaz, el monitoreo y el despliegue. Las dos últimas etapas, uso "
         "real y documentación, las cerramos los tres hasta el 15 de noviembre.",
         ["Señalar la etapa 3 (preregistro) antes de la 4 (entrenamiento): ese orden es lo que "
          "da validez a las hipótesis."]),
        (4, "Supuestos", A, 50, "Problemática, planificación y supuestos (10 pts)",
         "Tenemos supuestos en tres frentes. <b>Datos</b>: el dataset procedural del Proyecto 2 "
         "representa el estilo que quiere el equipo; lo regeneramos idéntico byte a byte. La "
         "novedad la medimos como la distancia al personaje real más parecido en el espacio de "
         "una red preentrenada, SqueezeNet; suponemos que «cerca» ahí significa «parecido a la "
         "vista». <b>Usuario</b>: busca variedad, no fidelidad; le basta 64 por 64 y vota con "
         "honestidad. <b>Entorno</b>: corre en la CPU gratuita de Hugging Face, con dos núcleos "
         "y poco tráfico; el disco se borra al reiniciar, por eso exportamos los registros. Si "
         "alguno de estos supuestos falla, el monitoreo lo debería mostrar, por ejemplo con una "
         "caída en la aceptación.<br/><br/>"
         f"<i>Transición:</i> «{B} les cuenta qué modelo pusimos y cómo lo validamos.»",
         ["Cerrar conectando supuestos con monitoreo: cada supuesto tiene una señal que lo vigila."]),
        (5, "Modelo planteado", B, 60, "Modelo planteado y validación de hipótesis (20 pts)",
         "Partimos de la <b>DCGAN del Proyecto 2</b>: un generador de cinco capas convolucionales "
         "transpuestas que convierte un vector de ruido de 128 números en una imagen de 64 por "
         "64, con 3.8 millones de parámetros.<br/><br/>"
         "Propusimos dos mejoras, cada una con su justificación. Una: <b>normalización "
         "espectral</b> en el discriminador, que en el Proyecto 2 había llevado al "
         "discriminador al equilibrio. Dos: servir el <b>promedio móvil (EMA) de los pesos</b> "
         "del generador, que suaviza la oscilación propia del entrenamiento adversarial. Si se "
         "adoptaban o no, lo decidía la hipótesis 1.<br/><br/>"
         "Y dos cambios para el servicio: un <b>filtro de novedad</b> que genera cuatro veces más "
         "candidatos, descarta los demasiado parecidos a los datos y se queda con los más "
         "distintos entre sí; y <b>pesos en media precisión</b>, de 15 a 7.6 MB. Además "
         "verificamos que es reproducible: al reentrenar se repiten exactamente las pérdidas "
         "del Proyecto 2.",
         ["Dejar claro que las mejoras eran propuestas y que H1 decidía."]),
        (6, "Validación · H1", B, 50, "Modelo planteado y validación de hipótesis (20 pts)",
         "Hipótesis 1: el modelo con normalización espectral y EMA tiene un KID al menos 15 % "
         "menor que el base. KID mide qué tan distinta es la distribución de imágenes generadas "
         "de la real, usando features de una red neuronal; menor es mejor, y real contra real "
         "da prácticamente cero. El criterio, fijado antes de entrenar: razón menor o igual a "
         f"0.85.<br/><br/>Resultado: base {kb:.2f}, modelo nuevo {ke:.2f} (por 10⁻³). La razón "
         f"es <b>{h1['razon_spectral_ema_vs_base']:.2f}</b>: el modelo nuevo es "
         f"{(h1['razon_spectral_ema_vs_base'] - 1) * 100:.0f} % peor. <b>H1 se refuta.</b> La "
         f"EMA sí ayuda un poco ({ks:.2f} → {ke:.2f}), pero la normalización espectral empeora "
         "la calidad. No movimos el criterio: <b>servimos el modelo base</b>. Haber fijado el "
         "criterio antes nos evitó poner en producción algo peor.",
         ["La desviación estándar del KID es ~0.014×10⁻³: la diferencia no es ruido.",
          "La línea base es el modelo base del Proyecto 2."]),
        (7, "Validación · H2", B, 50, "Modelo planteado y validación de hipótesis (20 pts)",
         f"Hipótesis 2: el filtro entrega lotes más útiles. Comparamos {h2['lotes']} lotes de "
         f"{h2['n']} contra la línea base, que es generar directo, sin filtro. Criterios: "
         "diversidad dentro del lote al menos ×1.10, cobertura de elementos igual o mayor, y que "
         "el KID no empeore más de 25 %.<br/><br/>"
         f"Resultado: la diversidad subió <b>×{h2['razon_diversidad']:.3f}</b>, nos quedamos "
         "cortos del 1.10. Los elementos distintos por lote subieron de "
         f"{h2['cobertura_elementos']['directo']:.1f} a {h2['cobertura_elementos']['filtro']:.1f} "
         f"y la calidad prácticamente no cambió (KID ×{h2['razon_kid']:.3f}). <b>No cumple.</b> "
         "Por eso el filtro queda activado pero es opcional, y cada voto registra si el "
         "personaje venía con filtro o sin él: el uso real lo decide con una comparación A/B.",
         ["Mencionar que la mejora existe pero no llegó al umbral que fijamos."]),
        (8, "Validación · H3", B, 50, "Modelo planteado y validación de hipótesis (20 pts)",
         "Hipótesis 3: se puede operar en CPU. Tiene dos partes. <b>Media precisión</b>: la "
         f"diferencia media de píxel es {h3['dif_media_pixel_0_255']:.2f} sobre 255 y el elemento "
         f"asignado coincide en {h3['acuerdo_elemento'] * 100:.1f} %. Cumple, y los pesos pesan "
         "la mitad.<br/><br/>"
         f"<b>Latencia</b>: con un solo usuario el p95 es {p95_1:.2f} s; con cuatro usuarios "
         f"simultáneos en dos núcleos es {h3['latencia_p95_ms'] / 1000:.1f} s, y el criterio era "
         f"1.5 s. <b>No cumple.</b> La bajamos de {p95_v1:.1f} a "
         f"{h3['latencia_p95_ms'] / 1000:.1f} s haciendo que las inferencias vayan una por una "
         "en lugar de competir por los núcleos, pero no alcanza. Con el tráfico esperado, unas "
         "decenas de solicitudes por hora, casi nunca coinciden dos usuarios; si crece, se "
         "escala con más CPU.<br/><br/>"
         f"<i>Transición:</i> «{C} les muestra cómo está armado y que está funcionando.»",
         [f"Errores en la prueba de carga: {pc['errores']} de {pc['solicitudes']}."]),
        (9, "Herramientas y arquitectura", C, 60, "Herramientas y arquitectura (10 pts)",
         "El sistema tiene cuatro bloques, de izquierda a derecha. <b>Datos</b>: el generador "
         "procedural del Proyecto 2, reproducible con semilla. <b>Entrenamiento</b>: PyTorch; los "
         "checkpoints se exportan a media precisión junto con la tarjeta del modelo y el banco "
         "de features que usamos para medir novedad. <b>Servicio</b>: FastAPI dentro de un "
         "contenedor Docker en Hugging Face Spaces; expone la interfaz web, la API documentada "
         "en /docs, y permite regenerar cualquier personaje exacto a partir de su semilla. "
         "<b>Monitoreo</b>: cada solicitud y cada voto quedan en SQLite, hay métricas para "
         "Prometheus y un tablero en /monitoreo con latencia, aceptación y dos alertas de "
         "deriva: si la novedad se aleja más de tres errores estándar de la referencia, o si "
         "cambia la mezcla de elementos. En cada push, GitHub Actions corre las 16 pruebas y "
         "construye la imagen.",
         ["Recorrer el diagrama con el dedo o el puntero, columna por columna.",
          "Explicar la deriva en un generador: no hay datos de entrada, se vigila la salida."]),
        (10, "Evidencias del pase a producción + demo", C, 110, "Evidencias del pase a producción (20 pts)",
         f"El servicio está en producción en {url}. Se los muestro.<br/><br/>"
         "<i>(Demo, ver pasos abajo.)</i><br/><br/>"
         "Además de lo que acaban de ver, en el repositorio están las evidencias: capturas, "
         f"registros de uso exportados, la prueba de carga de {pc['solicitudes']} solicitudes con "
         f"{pc['errores']} errores, el monitoreo sin alertas de deriva (z = "
         f"{der.get('z_novedad_sin_filtro', 0):.2f}, TV = {der.get('distancia_tv_elementos', 0):.2f}) "
         "y la integración continua en verde. Los votos de la prueba de carga son simulados y "
         "así lo dice el README; los votos reales son los de [N] personas que lo usaron "
         "[completar con los datos reales antes de presentar].",
         ["Si falla internet o el Space no arranca: plan B con las capturas de la diapositiva.",
          "No decir números de uso real que no estén exportados en evidencias/registros/."]),
        (11, "Recomendación al comité", C, 30, "Presentación (15 pts)",
         "Nuestra recomendación: <b>mantener Lumen en producción con el modelo base</b>. Es "
         "estable, sin errores y sin deriva. No adoptamos la mejora porque no pasó su criterio. "
         "El filtro sigue en prueba A/B. Lo retiraríamos si la aceptación cae por debajo de 20 % "
         "de forma sostenida o salta una alerta de deriva. El siguiente paso es reentrenar con "
         "los conceptos que el equipo de arte aceptó. Gracias; quedamos atentos a sus preguntas.",
         ["Terminar a tiempo: la recomendación es lo que el comité tiene que llevarse."]),
    ]

    doc = SimpleDocTemplate(SALIDA, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.6 * cm,
                            title="Guion de presentación · Lumen en producción",
                            author=", ".join(nombres))

    def pie(canvas, doc_):
        canvas.saveState()
        canvas.setFont("DV", 8)
        canvas.setFillColor(TENUE)
        canvas.drawString(2 * cm, 1 * cm, "Lumen en producción · guion de presentación")
        canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, str(doc_.page))
        canvas.restoreState()

    h = [Paragraph("Guion de presentación", S["titulo"]),
         Paragraph(f"Lumen en producción · {A} · {B} · {C}", S["sub"]),
         Paragraph(f"Servicio: {url} · Repositorio: {repo}", S["sub"]),
         Spacer(1, 10)]

    h.append(Paragraph("Formato y mensaje central", S["h1"]))
    h += vinetas([
        "10 minutos de presentación y 5 de preguntas. Presentan los tres y <b>cualquiera puede "
        "recibir cualquier pregunta</b>.",
        "La audiencia es un comité técnico y de negocio que decide si el modelo se mantiene en "
        "producción. Cada diapositiva debe ayudar a tomar esa decisión.",
        "El guion acompaña a <b>presentacion/presentacion.pdf</b> (11 diapositivas). No hay que "
        "leerlo: sirve para ensayar qué decir y cuánto tiempo dura cada parte.",
    ])
    h.append(Spacer(1, 4))
    h.append(caja([Paragraph(
        "<b>Mensaje en una frase:</b> Lumen ya está en producción y recomendamos mantenerlo con "
        "el modelo base: es estable, barato y medible. La mejora que probamos no superó a la "
        "línea base con el criterio que fijamos antes de entrenar, y por eso no la adoptamos.",
        S["habla"])]))

    # reparto y tiempos
    h.append(Paragraph("Reparto y tiempos", S["h1"]))
    filas = [["#", "Diapositiva", "Presenta", "Duración", "Inicia en"]]
    t = 0
    for n, tit, quien, seg, *_ in diapos:
        filas.append([n, tit, quien, mmss(seg), mmss(t)])
        t += seg
    filas.append(["", "<b>Total</b>", "", f"<b>{mmss(t)}</b>", f"margen {mmss(600 - t)}"])
    h.append(tabla(filas, [1, 7.5, 3.5, 2.3, 2.7]))
    tot = {}
    for _, _, quien, seg, *_ in diapos:
        tot[quien] = tot.get(quien, 0) + seg
    h.append(Spacer(1, 4))
    h.append(Paragraph(" · ".join(f"{q}: {mmss(s)}" for q, s in tot.items()), S["meta"]))

    h.append(Paragraph("Cómo cubre la rúbrica", S["h1"]))
    h.append(tabla([
        ["Criterio", "Pts", "Dónde se cubre", "Responsable"],
        ["Problemática, planificación y supuestos", "10", "Diapositivas 2, 3 y 4 · README §1-3", A],
        ["Modelo planteado y validación de hipótesis", "20",
         "Diapositivas 5 a 8 · README §4-5 · docs/hipotesis.md y docs/resultados.md", B],
        ["Herramientas y arquitectura", "10", "Diapositiva 9 · README §6", C],
        ["Repositorio: organizado, reproducible y documentado", "10",
         "Se menciona en 3, 5 y 9 (reproducibilidad, pruebas, CI); el README tiene las "
         "instrucciones de ejecución", "Todos"],
        ["Evidencias del pase a producción", "20", "Diapositiva 10 + demo en vivo · README §7", C],
        ["Presentación: claridad en 10 minutos", "15", "Mensaje central, tiempos y transiciones", "Todos"],
        ["Respuesta a preguntas", "15", "Sección «Preguntas probables» de este guion", "Todos"],
    ], [5.2, 1, 7.3, 3.5]))

    # diapositivas
    h.append(PageBreak())
    h.append(Paragraph("Guion por diapositiva", S["titulo"]))
    h.append(Paragraph("El texto en el recuadro es una propuesta para decir en voz alta; "
                       "adáptenlo a su forma de hablar.", S["sub"]))
    h.append(Spacer(1, 8))
    t = 0
    for n, tit, quien, seg, crit, habla, senalar in diapos:
        bloque = [
            Paragraph(f"{n}. {tit}", S["h1"]),
            Paragraph(f"<b>{quien}</b> · {mmss(seg)} (de {mmss(t)} a {mmss(t + seg)}) · "
                      f"Rúbrica: {crit}", S["meta"]),
            caja([Paragraph(habla, S["habla"])]),
        ]
        if senalar:
            bloque.append(Paragraph("Qué señalar", S["etq"]))
            bloque += vinetas(senalar)
        if n == 10:
            bloque.append(Paragraph("Demo en vivo (unos 60 segundos)", S["etq"]))
            bloque += vinetas([
                f"Abrir {url}/?n=8&amp;semilla=2026: genera un lote de 8 personajes con nombre, "
                "elemento, rol y novedad. Decir cuánto tardó.",
                "Votar «Me sirve» en uno y «Descartar» en otro.",
                f"Cambiar a la pestaña {url}/monitoreo: aparecen la solicitud y los votos, la "
                "latencia y las alertas de deriva en verde.",
                f"Si sobra tiempo: {url}/docs para mostrar que también es una API.",
            ])
        bloque.append(Spacer(1, 6))
        h.append(KeepTogether(bloque))
        t += seg

    # preguntas
    h.append(PageBreak())
    h.append(Paragraph("Preguntas probables del comité", S["titulo"]))
    h.append(Paragraph("Respuestas cortas: primero la respuesta directa, después el dato. Cualquier "
                       "integrante puede recibir cualquier pregunta, así que los tres deben "
                       "conocerlas todas.", S["sub"]))
    preguntas = {
        "Problema y negocio": [
            ("¿Por qué una GAN y no un modelo de difusión como Stable Diffusion?",
             "El proyecto pide llevar a producción un modelo de los proyectos 1 a 3. Además la "
             "DCGAN pesa 7.6 MB y responde en menos de un segundo en CPU gratuita; un modelo de "
             "difusión necesitaría GPU. Para bocetos de partida de 64×64 alcanza."),
            ("¿Cómo saben que le sirve al usuario?",
             "Con la tasa de aceptación de los votos «me sirve / descartar». Los votos de la "
             "prueba de carga son simulados; los reales son los registrados en el Space "
             "[completar con el número antes de presentar]. El criterio de retiro es una "
             "aceptación menor a 20 % sostenida."),
            ("¿Por qué 20 % como umbral de retiro?",
             "Si de un lote de 8 al menos uno o dos sirven, el artista ahorra horas de boceto. Por "
             "debajo, pierde más tiempo revisando que el que ahorra. Es un supuesto de negocio que "
             "el uso real debe confirmar."),
            ("¿Qué pasa con los derechos de las imágenes?",
             "El dataset es procedural y CC0, generado por código del Proyecto 2; no se entrenó con "
             "arte de terceros."),
        ],
        "Modelo e hipótesis": [
            ("¿Qué es el KID y por qué no usaron FID?",
             "Es la distancia entre las distribuciones de features de imágenes reales y generadas, "
             "calculada con un kernel. Con 2000 muestras no tiene sesgo, a diferencia del FID, y da "
             "una desviación estándar al promediar 10 subconjuntos. Usamos SqueezeNet porque corre "
             "en CPU y es la misma red del Proyecto 2."),
            ("¿Por qué empeoró la normalización espectral?",
             "Nuestra explicación: limita demasiado al discriminador y le da gradientes más débiles "
             "al generador en 60 épocas. Que el discriminador esté en equilibrio no implica mejores "
             "imágenes. No probamos otros hiperparámetros a propósito: hubiera sido ajustar después "
             "de ver el resultado."),
            ("¿La diferencia de KID es significativa?",
             f"Sí. La desviación entre subconjuntos es de ~0.014×10⁻³ y la diferencia entre modelos "
             f"es de {ke - kb:.2f}×10⁻³, unas 60 veces mayor."),
            ("Si las tres hipótesis no se cumplen, ¿el proyecto fracasó?",
             "No. Los criterios se fijaron antes de entrenar (docs/hipotesis.md, con fecha en git) y "
             "decidimos con ellos: no servimos el modelo peor, dejamos el filtro como opcional y "
             "documentamos el límite de concurrencia. El servicio funciona y la parte de fp16 sí "
             "cumple."),
            ("¿Cómo evitan que el modelo copie los datos?",
             "Medimos la novedad como la distancia al vecino real más cercano. El filtro descarta "
             "los candidatos por debajo de τ (el percentil 10 de la calibración) y el tablero vigila "
             "la novedad media servida."),
            ("¿Qué es la EMA y por qué la probaron?",
             "Es un promedio móvil exponencial de los pesos del generador (decaimiento 0.995). El "
             "entrenamiento adversarial oscila, y el promedio suele dar muestras más estables. Aislada "
             f"mejoró un poco ({ks:.2f} → {ke:.2f}), pero no compensó la normalización espectral."),
            ("¿Por qué guardan en fp16 pero ejecutan en fp32?",
             "fp16 reduce el archivo a la mitad (cabe sin Git LFS), pero la CPU no acelera fp16; por "
             "eso los pesos se convierten al cargar. La diferencia en las imágenes es despreciable."),
        ],
        "Producción y operación": [
            ("¿Qué pasa si el Space se reinicia?",
             "El disco es efímero y se pierde la base SQLite. Por eso exportamos /api/registros a "
             "evidencias/. Para operar a largo plazo montaríamos almacenamiento persistente o una "
             "base externa."),
            ("¿Qué significa deriva en un generador, si no recibe datos de entrada?",
             "Se vigila la salida. Si la novedad media cambia mucho (posible colapso de modos o pesos "
             "dañados) o cambia la mezcla de elementos, salta la alerta. La referencia se calibró con "
             "1000 muestras."),
            ("¿Por qué la alerta de novedad solo usa solicitudes sin filtro?",
             "Porque el filtro sube la novedad a propósito; si las mezcláramos, daría falsas alertas."),
            ("¿Cómo escalarían si crece el uso?",
             "Más vCPU o réplicas del contenedor detrás de un balanceador. Dentro de cada réplica las "
             "inferencias van una por una: con 2 núcleos eso subió el rendimiento de "
             f"{rps_v1:.2f} a {pc['rps']:.2f} solicitudes por segundo."),
            ("¿El mismo personaje sale igual siempre?",
             "Sí. Cada semilla tiene su propio generador aleatorio y BatchNorm está en modo "
             "evaluación, así que un personaje no depende del lote; /api/personaje/{semilla}.png lo "
             "regenera exacto."),
            ("¿Qué cubren las pruebas?",
             "Son 16. De la API: validación, determinismo por semilla, votos, estadísticas y "
             "registros. Del modelo: carga, forma de salida, selección sin repetidos y fp16. La "
             "integración continua además construye la imagen Docker y la prueba."),
            ("¿Guardan datos personales?",
             "No. El cliente se registra como un hash truncado de IP + user-agent, solo para contar "
             "usuarios distintos."),
            ("¿Cómo usaron la IA?",
             "Está declarado en el README: un asistente (Claude) para programar y redactar. El "
             "problema, las hipótesis, los criterios, la interpretación y la decisión fueron "
             "nuestros, y cada uno puede explicar el código."),
        ],
    }
    for grupo, qs in preguntas.items():
        h.append(Paragraph(grupo, S["h1"]))
        for q, r in qs:
            h.append(KeepTogether([Paragraph(q, S["preg"]), Paragraph(r, S["resp"])]))

    # lista final
    h.append(Spacer(1, 8))
    h.append(KeepTogether([Paragraph("Antes de presentar", S["h1"]), caja([Paragraph(x, S["habla"]) for x in [
        "☐ Abrir el Space 5 minutos antes: si estuvo inactivo, puede estar dormido y tardar en "
        "arrancar.",
        "☐ Dejar abiertas dos pestañas: /?n=8&amp;semilla=2026 y /monitoreo.",
        "☐ Completar en este guion y en la diapositiva 10 el número real de usuarios y votos "
        "(exportar /api/registros a evidencias/registros/).",
        "☐ Ensayar con cronómetro: el guion dura 9:30 y deja 30 s de margen.",
        "☐ Tener las capturas como plan B si falla la conexión.",
        "☐ Los tres leen el README completo y estas preguntas.",
    ]], fondo=AVISO)]))

    doc.build(h, onFirstPage=pie, onLaterPages=pie)
    print(SALIDA)


if __name__ == "__main__":
    main()
