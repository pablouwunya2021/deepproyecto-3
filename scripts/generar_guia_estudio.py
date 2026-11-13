"""
Genera presentacion/guia_estudio.pdf: guia para que cada integrante pueda explicar
cualquier parte del proyecto (conceptos, codigo, resultados y operacion).

    python scripts/generar_guia_estudio.py

Reutiliza los estilos de generar_guion.py y toma los numeros de evidencias/ y de la
tarjeta del modelo, asi que se mantiene al dia si se reevalua.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (KeepTogether, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer)

from generar_guion import (AVISO, RAIZ, S, TENUE, caja, leer, tabla,  # noqa: E402
                           vinetas)

SALIDA = os.path.join(RAIZ, "presentacion", "guia_estudio.pdf")

# un titulo nunca queda solo al pie de una pagina
for _e in ("h1", "etq"):
    S[_e].keepWithNext = True


def P(texto, estilo="txt"):
    return Paragraph(texto, S[estilo])


def formula(texto):
    return caja([P(texto, "habla")])


def main():
    d = leer("presentacion/datos_presentacion.json", {})
    res = leer("evidencias/hipotesis/resultados.json", {})
    tarjeta = leer("modelos/tarjeta_modelo.json", {})
    h1, h2, h3 = res["H1"], res["H2"], res["H3"]
    k = h1["kid"]
    m = tarjeta["modelos"]
    uno = leer("evidencias/prueba_carga_1cliente.json", {})
    v1 = leer("evidencias/prueba_carga_v1_sin_candado.json", {})
    pc = h3["prueba_carga"]
    nombres = [n.split(" (")[0] for n in d.get("integrantes", [])]

    doc = SimpleDocTemplate(SALIDA, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.7 * cm, bottomMargin=1.6 * cm,
                            title="Guía de estudio · Lumen en producción",
                            author=", ".join(nombres))

    def pie(canvas, doc_):
        canvas.saveState()
        canvas.setFont("DV", 8)
        canvas.setFillColor(TENUE)
        canvas.drawString(2 * cm, 1 * cm, "Lumen en producción · guía de estudio")
        canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, str(doc_.page))
        canvas.restoreState()

    h = []
    h += [P("Guía de estudio", "titulo"),
          P("Lo necesario para explicar cualquier parte del proyecto frente al comité", "sub"),
          Spacer(1, 8)]

    # ------------------------------------------------------------------ 0
    h.append(P("Cómo usar esta guía", "h1"))
    h += vinetas([
        "La rúbrica dice que <b>cada integrante debe poder explicar cualquier parte del código y "
        "de las decisiones</b>, y cualquiera puede recibir cualquier pregunta. No basta con "
        "dominar la parte propia.",
        "Orden sugerido (unas 2 horas): secciones 1 y 2 (el proyecto y las GAN), 3 y 4 (mejoras y "
        "métricas), 5 (hipótesis), 6 a 8 (servicio, monitoreo y pruebas). Al final, la "
        "autoevaluación de la sección 10, sin mirar las respuestas.",
        "Complementos: <b>README.md</b> (documento técnico), <b>docs/hipotesis.md</b> "
        "(preregistro), <b>docs/resultados.md</b> y <b>presentacion/guion_presentacion.pdf</b> "
        "(guion y preguntas del comité).",
    ])
    h.append(P("Mapa del repositorio", "etq"))
    h.append(tabla([
        ["Carpeta / archivo", "Qué contiene", "Pregunta que responde"],
        ["lumen/data.py", "Generador procedural del dataset (5000 imágenes, 6 elementos)",
         "¿De dónde salen los datos?"],
        ["lumen/models.py · losses.py", "Generador, discriminador y pérdidas (BCE, hinge, WGAN-GP)",
         "¿Qué arquitectura es?"],
        ["lumen/train.py", "Bucle de entrenamiento, EMA, checkpoints", "¿Cómo se entrenó?"],
        ["lumen/features.py", "Extractor SqueezeNet (512 features)", "¿Cómo se mide calidad y novedad?"],
        ["lumen/inferencia.py", "MotorLumen: carga, genera, filtro de novedad", "¿Qué hace el servicio con el modelo?"],
        ["app/main.py · monitoreo.py", "API FastAPI, registro SQLite, estadísticas y deriva",
         "¿Cómo se sirve y se vigila?"],
        ["app/static/", "Interfaz web y tablero /monitoreo", "¿Qué ve el usuario?"],
        ["scripts/", "Exportar pesos, evaluar hipótesis, prueba de carga, publicar, PDF",
         "¿Cómo se reproduce todo?"],
        ["modelos/", "Pesos fp16, banco de features, tarjeta del modelo", "¿Qué se despliega?"],
        ["tests/ · .github/workflows/", "16 pruebas y la CI", "¿Cómo sabemos que funciona?"],
        ["evidencias/", "Capturas, registros, prueba de carga, figuras de hipótesis",
         "¿Dónde está la prueba?"],
    ], [4.4, 7.0, 5.6]))

    # ------------------------------------------------------------------ 1
    h.append(P("1. El proyecto en una página", "h1"))
    h.append(tabla([
        ["Pregunta", "Respuesta corta"],
        ["¿Qué pide el curso?", "Llevar a producción un modelo de los proyectos 1-3, de modo que "
         "alguien ajeno al equipo lo use, y sustentar cada decisión con las 7 secciones del README."],
        ["¿Qué modelo?", "La DCGAN 64×64 del Proyecto 2 («Lumen», espíritus guardianes "
         "elementales). Corre en CPU y ya estaba entrenada."],
        ["¿Quién la usa?", "El equipo de arte de un estudio indie, para obtener bocetos de partida."],
        ["¿Qué decisión depende del modelo?", "Del artista: qué concepto pasa a refinamiento "
         "(voto «me sirve» / «descartar»). Del comité: mantener o retirar el modelo."],
        ["¿Qué mejoramos?", "Probamos normalización espectral + EMA (H1), un filtro de novedad (H2) "
         "y pesos fp16 (H3)."],
        ["¿Qué resultó?", "H1 refutada (peor), H2 casi (×1.08 de ×1.10), H3 fp16 cumple y la "
         "latencia con concurrencia no. Servimos el modelo <b>base</b>."],
        ["¿Dónde corre?", "FastAPI en Docker en Hugging Face Spaces (CPU gratuita, 2 vCPU)."],
        ["¿Cómo se vigila?", "SQLite + tablero /monitoreo: latencia, errores, aceptación, alertas de "
         "deriva; métricas Prometheus en /metrics."],
        ["¿Recomendación?", "Mantener con el modelo base. Retirar si la aceptación queda por debajo "
         "de 20 % sostenida o salta una alerta de deriva."],
    ], [4.5, 12.5]))

    # ------------------------------------------------------------------ 2
    h.append(P("2. Fundamentos: GAN y DCGAN", "h1"))
    h.append(P("Una <b>GAN</b> entrena dos redes que compiten. El <b>generador G</b> convierte ruido "
               "z en una imagen; el <b>discriminador D</b> recibe una imagen y da un puntaje de "
               "«real». D aprende a distinguir reales de generadas y G aprende a engañar a D. "
               "Cuando funciona, la distribución de G(z) se acerca a la de los datos."))
    h.append(Spacer(1, 4))
    h.append(formula(
        "<b>Pérdida de D (BCE):</b> −[log σ(D(x)) + log(1 − σ(D(G(z))))]<br/>"
        "<b>Pérdida de G no saturante:</b> −log σ(D(G(z)))  (maximiza que D crea que es real; da "
        "gradientes fuertes al inicio, cuando D gana fácil)<br/>"
        "<b>Hinge (experimento del P2):</b> D: E[max(0, 1 − D(x))] + E[max(0, 1 + D(G(z)))]; "
        "G: −E[D(G(z))]"))
    h.append(P("Por qué es inestable", "etq"))
    h += vinetas([
        "<b>Oscilación:</b> G y D se persiguen; las pérdidas no bajan de forma monótona y no indican "
        "calidad por sí solas.",
        "<b>Colapso de modos:</b> G aprende a producir pocas variantes que engañan a D. Se nota "
        "como poca diversidad (lo vigilamos con la novedad y la mezcla de elementos).",
        "<b>D demasiado fuerte:</b> si D gana siempre, los gradientes hacia G se vuelven "
        "pequeños. Por eso se busca equilibrio (D(x) y D(G(z)) cerca de 0.5).",
    ])
    h.append(P("Reglas DCGAN que seguimos", "etq"))
    h += vinetas([
        "Convoluciones con stride en vez de pooling: transpuestas en G (suben resolución), "
        "normales con stride 2 en D (bajan resolución).",
        "BatchNorm en G y en D (salvo la primera capa de D); ReLU en G, LeakyReLU(0.2) en D; "
        "<b>tanh</b> a la salida de G (imágenes en [−1, 1]).",
        "Pesos iniciales N(0, 0.02); Adam con lr = 2×10⁻⁴, β₁ = 0.5, β₂ = 0.999; lote de 128; "
        "60 épocas.",
    ])
    h.append(P("Formas del generador (ngf = 64)", "etq"))
    h.append(tabla([
        ["Capa", "Operación", "Salida"],
        ["entrada", "z de 128 números", "128×1×1"],
        ["1", "ConvTranspose 4×4, stride 1 + BN + ReLU", "512×4×4"],
        ["2", "ConvTranspose 4×4, stride 2 + BN + ReLU", "256×8×8"],
        ["3", "ConvTranspose 4×4, stride 2 + BN + ReLU", "128×16×16"],
        ["4", "ConvTranspose 4×4, stride 2 + BN + ReLU", "64×32×32"],
        ["5", "ConvTranspose 4×4, stride 2 + tanh", "3×64×64"],
    ], [2, 10, 5]))
    h.append(P("El discriminador hace el camino inverso: 3×64×64 → 64×32×32 → 128×16×16 → "
               "256×8×8 → 512×4×4 → 1 logit. G tiene 3.8 millones de parámetros.", "txt"))
    h.append(P("Los datos", "etq"))
    h += vinetas([
        "5000 imágenes 64×64 generadas por código (lumen/data.py): siluetas simétricas "
        "bioluminiscentes sobre fondo oscuro. Origen propio, licencia CC0.",
        "6 elementos por tono de color: brasa (rojo), marea (azul), musgo (verde), céfiro (verde "
        "azulado), vacío (púrpura), solar (dorado); 4 arquetipos de silueta.",
        "Semilla de datos 20260101: el dataset se regenera <b>idéntico byte a byte</b>, por eso no "
        "se sube al repositorio.",
        "El elemento de un personaje generado se asigna por el <b>tono dominante</b> de sus "
        "píxeles brillantes (HSV), y con él se escoge el nombre y el rol.",
    ])

    # ------------------------------------------------------------------ 3
    h.append(P("3. Las mejoras que probamos", "h1"))
    h.append(P("Normalización espectral (SN) en D", "etq"))
    h.append(P("Divide cada matriz de pesos de D por su mayor valor singular σ(W), que se estima "
               "con una iteración de potencia por paso. Así cada capa cambia su salida, como "
               "máximo, en la misma proporción que su entrada (D queda aproximadamente "
               "1-Lipschitz). Eso evita que D se vuelva brusco y da gradientes más estables a G. "
               "Al usar SN quitamos la BatchNorm de D, como en SNGAN."))
    h.append(Spacer(1, 3))
    h.append(formula("W_SN = W / σ(W)  ·  σ(W) = mayor valor singular de W"))
    h.append(P("<b>Por qué la probamos:</b> en el Proyecto 2 llevó a D(x) ≈ 0.56 y D(G(z)) ≈ 0.44, "
               "o sea, más equilibrio. <b>Lección:</b> equilibrio de D no es lo mismo que calidad "
               "de las imágenes."))
    h.append(P("EMA de los pesos del generador", "etq"))
    h.append(P("Se mantiene una copia de G cuyos pesos son un promedio móvil exponencial de los del "
               "G que se entrena. No se entrena; se actualiza después de cada paso de G y es la "
               "que se sirve. Las estadísticas de BatchNorm se copian tal cual."))
    h.append(Spacer(1, 3))
    h.append(formula("θ_ema ← 0.995 · θ_ema + 0.005 · θ   (promedia aprox. los últimos 1/(1−0.995) "
                     "= 200 pasos)"))
    h.append(P("<b>Por qué:</b> el entrenamiento adversarial oscila; el promedio suaviza esa "
               "oscilación. El control <b>spectral_ema_sin_ema</b> es la misma corrida sin "
               "promediar, para aislar el efecto de la EMA."))
    h.append(P("Cambios para producción", "etq"))
    h += vinetas([
        "<b>Filtro de novedad</b> (sección 4) aplicado al servir, sin reentrenar.",
        "<b>Pesos fp16:</b> el archivo baja de 15.2 a 7.6 MB y cabe en Hugging Face sin Git LFS. "
        "Se convierten a fp32 al cargar porque la CPU no acelera fp16.",
        "<b>Una inferencia a la vez</b> (threading.Lock): con 2 vCPU, varias en paralelo compiten "
        "por los núcleos y el rendimiento total baja.",
    ])
    h.append(P("Reproducibilidad (dato fuerte para defender)", "etq"))
    h.append(P("Con la misma semilla de entrenamiento (42) y los mismos datos, la corrida nueva "
               "repitió <b>exactamente</b> las pérdidas de las 60 épocas del Proyecto 2, y los pesos "
               "de G sin EMA son idénticos bit a bit a los del experimento spectral del P2. Por eso "
               "spectral y spectral_ema_sin_ema tienen el mismo KID."))

    # ------------------------------------------------------------------ 4
    h.append(P("4. Cómo medimos: features, KID, novedad y diversidad", "h1"))
    h.append(P("Todas las métricas se calculan en el <b>espacio de features de SqueezeNet 1.1</b> "
               "preentrenada en ImageNet: la imagen se escala a 224×224, se normaliza como "
               "ImageNet y se toma un vector de 512 números, normalizado a largo 1. Elegimos "
               "SqueezeNet porque es pequeña y corre en CPU (Inception, la de FID, es mucho más "
               "pesada)."))
    h.append(P("KID (Kernel Inception Distance)", "etq"))
    h.append(formula(
        "KID = MMD² entre features reales X y generadas Y, con kernel polinomial "
        "k(a, b) = (aᵀb / 512 + 1)³<br/>"
        "MMD² = media k(x, x′) + media k(y, y′) − 2 · media k(x, y)   (sin la diagonal: estimador "
        "insesgado)<br/>"
        "Se promedia sobre 10 subconjuntos de 1000 → media ± desviación. Menor es mejor; real vs "
        "real ≈ 0."))
    h.append(P("<b>KID vs FID:</b> FID asume distribuciones gaussianas y tiene sesgo con muestras "
               "pequeñas; KID es insesgado y da una medida de incertidumbre, útil con 2000 "
               "muestras.", "txt"))
    h.append(P("Novedad, τ y diversidad", "etq"))
    h += vinetas([
        "<b>Novedad</b> de un personaje = distancia L2 a su vecino más cercano entre los 5000 de "
        "entrenamiento (el «banco de features»). Baja = parecido a algo que ya existe.",
        f"<b>τ</b> = percentil 10 de la novedad en 1000 muestras de calibración (semilla 999). Para "
        f"base, τ = {m['base']['tau']:.3f}; media de referencia {m['base']['novedad_ref_media']:.3f} "
        f"± {m['base']['novedad_ref_std']:.3f}.",
        "<b>Diversidad intra-lote</b> = distancia media entre todos los pares de personajes de un "
        "lote. <b>Cobertura</b> = cuántos de los 6 elementos distintos aparecen en un lote.",
    ])
    h.append(P("Filtro de novedad (lumen/inferencia.py · seleccionar)", "etq"))
    h += vinetas([
        "1) Genera 4n candidatos con la semilla del lote. 2) Descarta los que tienen novedad < τ.",
        "3) Toma el más novedoso y, repetidamente, añade el candidato <b>más lejano</b> de los ya "
        "elegidos (farthest-point sampling) hasta tener n. Nunca repite un candidato.",
        "4) Si no hay suficientes por encima de τ, completa con los más novedosos restantes.",
    ])

    # ------------------------------------------------------------------ 5
    h.append(P("5. Validación de hipótesis", "h1"))
    h.append(P("Lo que la rúbrica pide es: hipótesis, métrica, <b>criterio de éxito definido antes "
               "de entrenar</b> y resultado frente a una <b>línea base</b>. El preregistro está en "
               "docs/hipotesis.md con fecha del 13 de noviembre de 2026, anterior al "
               "entrenamiento. Los criterios no se cambiaron después de ver los resultados: eso es "
               "lo que da validez al resultado, aunque sea negativo."))
    h.append(Spacer(1, 4))
    p95_1 = uno.get("latencia_ms", {}).get("p95", 0)
    p95_v1 = v1.get("latencia_ms", {}).get("p95", 0)
    h.append(tabla([
        ["", "H1 · ¿modelo nuevo mejor?", "H2 · ¿filtro útil?", "H3 · ¿operable en CPU?"],
        ["Línea base", "modelo base (P2)", "generar directo, sin filtro", "pesos fp32; criterio fijo"],
        ["Métrica", "KID (2000 muestras, semilla 777)", "diversidad, cobertura y KID en 200 lotes de 8",
         "diferencia de píxel, acuerdo de elemento, p95 de latencia"],
        ["Criterio previo", "KID(spectral_ema) ≤ 0.85 × KID(base)",
         "diversidad ≥ ×1.10, cobertura ≥ base, KID ≤ ×1.25",
         "dif. < 1/255, acuerdo ≥ 99 %, p95 < 1.5 s con 4 clientes"],
        ["Resultado",
         f"base {k['base']['media'] * 1e3:.2f} vs nuevo {k['spectral_ema']['media'] * 1e3:.2f} "
         f"(×10⁻³) → razón {h1['razon_spectral_ema_vs_base']:.2f}",
         f"diversidad ×{h2['razon_diversidad']:.3f}; cobertura "
         f"{h2['cobertura_elementos']['directo']:.2f} → {h2['cobertura_elementos']['filtro']:.2f}; "
         f"KID ×{h2['razon_kid']:.3f}",
         f"dif. {h3['dif_media_pixel_0_255']:.3f}/255; acuerdo {h3['acuerdo_elemento'] * 100:.1f} %; "
         f"p95 {h3['latencia_p95_ms'] / 1000:.2f} s"],
        ["Veredicto", "<b>No cumple</b> (65 % peor)", "<b>No cumple</b> (le faltó poco a la diversidad)",
         "<b>No cumple</b>: fp16 sí, latencia no"],
        ["Decisión", "servir base", "filtro activo pero opcional + A/B de votos",
         "documentar el límite; escalar con más CPU si crece el uso"],
    ], [2.6, 4.6, 4.9, 4.9]))
    h.append(P("Cómo leer los resultados", "etq"))
    h += vinetas([
        f"<b>H1:</b> la desviación del KID es ~{k['base']['std'] * 1e3:.3f}×10⁻³ y la diferencia "
        f"entre modelos es {(k['spectral_ema']['media'] - k['base']['media']) * 1e3:.2f}×10⁻³, "
        "así que no es ruido. Pista que lo explica: los modelos con SN tienen novedad media "
        f"{m['spectral_ema']['novedad_ref_media']:.2f} contra {m['base']['novedad_ref_media']:.2f} "
        "de base; sus imágenes se alejan más de los datos, coherente con un KID peor.",
        f"<b>EMA aislada:</b> {k['spectral_ema_sin_ema']['media'] * 1e3:.2f} → "
        f"{k['spectral_ema']['media'] * 1e3:.2f}: ayuda un poco, pero no compensa la SN. "
        f"El mejor KID fue hinge ({k['hinge']['media'] * 1e3:.2f}), "
        "pero no era la hipótesis; cambiar el modelo por eso sería elegir después de ver los "
        "datos.",
        "<b>H2:</b> el filtro sí mejora todo un poco y casi no cuesta calidad; no llegó al umbral "
        "que nos pusimos. Lo decide el uso real con la tasa de aceptación con y sin filtro.",
        f"<b>H3:</b> 1 cliente p95 {p95_1 / 1000:.2f} s; 4 clientes p95 "
        f"{p95_v1 / 1000:.1f} s sin candado y {h3['latencia_p95_ms'] / 1000:.1f} s con candado "
        f"({v1.get('rps', 0):.2f} → {pc['rps']:.2f} solicitudes/s); {pc['errores']} errores en "
        f"{pc['solicitudes']} solicitudes. Con 4 a la vez, cada uno espera su turno: es cola, no "
        "fallo.",
    ])

    # ------------------------------------------------------------------ 6
    h.append(P("6. El servicio", "h1"))
    h.append(tabla([
        ["Ruta", "Qué hace"],
        ["GET /", "Interfaz web: generar, ver nombre/elemento/rol/novedad, votar, descargar PNG"],
        ["POST /api/generar", "n (1-16), semilla, filtro, modelo, tam → personajes en PNG base64"],
        ["GET /api/personaje/{semilla}.png", "Regenera exacto un personaje a partir de su semilla"],
        ["POST /api/retroalimentacion", "Voto me_sirve / descartar (con modelo y si venía con filtro)"],
        ["GET /api/estadisticas · /monitoreo", "Uso, latencia p50/p95, errores, aceptación, deriva"],
        ["GET /api/registros", "Últimas solicitudes y votos (para exportar evidencias)"],
        ["GET /salud · /metrics · /docs", "Chequeo de vida · Prometheus · documentación OpenAPI"],
    ], [5.5, 11.5]))
    h += vinetas([
        "<b>FastAPI + Pydantic:</b> valida la entrada (n entre 1 y 16, voto solo con dos valores); "
        "si algo es inválido responde 422 sin tocar el modelo.",
        "<b>Determinismo:</b> cada semilla tiene su propio torch.Generator y BatchNorm está en modo "
        "evaluación, así que el personaje no depende del lote en que salió.",
        "<b>Carga una sola vez:</b> los generadores, SqueezeNet y el banco de features se cargan al "
        "arrancar (lifespan); cada solicitud solo hace inferencia.",
        "<b>Docker:</b> python:3.12-slim + torch CPU; SqueezeNet se descarga al construir la imagen "
        "(no al arrancar); corre como usuario 1000 en el puerto 7860, que exige Hugging Face. Usa "
        "unos 330 MB de RAM.",
        "<b>Hugging Face Spaces:</b> CPU gratuita (2 vCPU); el Space se duerme si no se usa y "
        "su disco se borra al reiniciar, así que los registros se exportan.",
    ])

    # ------------------------------------------------------------------ 7
    h.append(P("7. Monitoreo y deriva", "h1"))
    h.append(P("En un generador <b>no hay datos de entrada</b> que puedan cambiar, así que la deriva "
               "se vigila en la <b>salida</b>: que lo servido siga pareciéndose a la referencia "
               "calibrada (1000 muestras). Un cambio puede indicar colapso de modos, pesos dañados "
               "o un despliegue con el modelo equivocado."))
    h.append(Spacer(1, 4))
    h.append(formula(
        "<b>Alerta de novedad:</b> z = (media servida − μ_ref) / (σ_ref / √n). Alerta si |z| > 3. "
        "Solo usa solicitudes <i>sin filtro</i>, porque el filtro sube la novedad a propósito.<br/>"
        "<b>Alerta de mezcla de elementos:</b> TV = ½ Σ |p_servido(e) − p_ref(e)|, la distancia de "
        "variación total entre proporciones de elementos. Alerta si TV > 0.25 con al menos 50 "
        "personajes.<br/>Ambas se calculan sobre las últimas 500 solicitudes por modelo."))
    h += vinetas([
        "<b>Métrica de negocio:</b> tasa de aceptación = me_sirve / (me_sirve + descartar), por "
        "modelo y con/sin filtro (A/B).",
        "<b>Métricas operativas:</b> latencia p50/p95/máx, tasa de error, solicitudes por hora, "
        "clientes distintos.",
        "<b>SQLite:</b> tablas solicitudes y retroalimentacion. <b>Prometheus:</b> contadores de "
        "solicitudes, personajes y votos, e histograma de latencia en /metrics.",
        "<b>Privacidad:</b> no se guarda la IP; el cliente es un hash truncado de IP + "
        "user-agent, solo para contar usuarios distintos.",
        "En la prueba de carga: z = 0.62 y TV = 0.11, sin alertas. Los votos de esa prueba son "
        "<b>simulados</b> (así lo dice el README); los reales salen del Space.",
    ])

    # ------------------------------------------------------------------ 8
    h.append(P("8. Pruebas, CI y reproducibilidad", "h1"))
    h += vinetas([
        "<b>tests/test_modelo.py (5):</b> los pesos fp16 cargan y dan valores en [−1, 1]; un "
        "personaje no depende del lote; el banco de features tiene la forma correcta; la "
        "selección respeta τ, no repite y completa si faltan candidatos.",
        "<b>tests/test_api.py (11):</b> salud, tarjeta, n personajes, determinismo por semilla, "
        "filtro ≥ τ, el PNG por semilla reproduce el lote, todos los modelos servidos, validación "
        "de entrada (422), votos y estadísticas, registros y /metrics, páginas HTML.",
        "<b>GitHub Actions</b> (cada push): job «pruebas» corre pytest; job «imagen» construye "
        "Docker, levanta el contenedor y prueba /salud y /api/generar.",
    ])
    h.append(tabla([
        ["Semilla", "Uso"],
        ["20260101", "generar el dataset"],
        ["42", "inicializar pesos y ordenar lotes en el entrenamiento"],
        ["7", "ruido fijo de las rejillas de evolución"],
        ["777", "las 2000 muestras de la evaluación de H1 (KID)"],
        ["999", "las 1000 muestras de calibración (τ y referencias de deriva)"],
        ["10000 + b", "los 200 lotes de la evaluación de H2"],
    ], [3, 14]))
    h.append(P("Comandos esenciales", "etq"))
    h.append(caja([P(x, "habla") for x in [
        "pytest -v  ·  python -m uvicorn app.main:app --port 7860",
        "docker build -t lumen .  ·  docker run -p 7860:7860 lumen",
        "python -m lumen.data  ·  python -m lumen.train --name spectral_ema --spectral-norm --ema-decay 0.995",
        "python scripts/exportar_modelo.py  ·  python scripts/evaluar_hipotesis.py",
        "python scripts/prueba_carga.py --url http://127.0.0.1:7860 --solicitudes 200 --clientes 4",
        "scripts/publicar_hf.sh USUARIO/lumen",
    ]]))

    # ------------------------------------------------------------------ 9
    h.append(P("9. Glosario rápido", "h1"))
    h.append(tabla([
        ["Término", "Definición en una línea"],
        ["Espacio latente (z)", "Vector de 128 números aleatorios N(0,1) que G convierte en imagen."],
        ["Logit", "Salida de D antes de la sigmoide; puntaje sin acotar de «real»."],
        ["BatchNorm (eval)", "Normaliza con estadísticas fijas aprendidas, no las del lote actual."],
        ["Lipschitz", "Límite a cuánto puede cambiar la salida cuando cambia la entrada."],
        ["Colapso de modos", "G produce pocas variantes que engañan a D; baja la diversidad."],
        ["MMD", "Distancia entre distribuciones usando un kernel; KID es MMD² en features."],
        ["Percentil 10 (τ)", "Valor bajo el cual queda el 10 % de las novedades de calibración."],
        ["Farthest-point", "Elegir cada vez el punto más lejano de los ya elegidos: maximiza variedad."],
        ["p50 / p95", "Latencia que no supera el 50 % / 95 % de las solicitudes."],
        ["rps", "Solicitudes por segundo atendidas (rendimiento)."],
        ["Deriva", "Lo servido se aleja de la referencia con que se validó el modelo."],
        ["Variación total (TV)", "Mitad de la suma de diferencias absolutas entre dos distribuciones."],
        ["Prueba A/B", "Comparar dos variantes (con/sin filtro) con la misma métrica real."],
        ["Preregistro", "Escribir hipótesis y criterios antes de ver resultados."],
        ["Tarjeta del modelo", "modelos/tarjeta_modelo.json: modelos, hash, τ, referencias, decisión."],
        ["fp16 / fp32", "Números de 16 / 32 bits; fp16 ocupa la mitad."],
        ["Contenedor", "Imagen Docker con el código, dependencias y pesos; corre igual en todas partes."],
        ["CI", "Integración continua: pruebas automáticas en cada push."],
    ], [4.2, 12.8]))

    # ------------------------------------------------------------------ 10
    h.append(P("10. Autoevaluación", "h1"))
    h.append(P("Respóndanlas en voz alta, sin mirar. Las respuestas están debajo de cada pregunta.",
               "sub"))
    preguntas = [
        ("¿Por qué la pérdida de G es «no saturante»?",
         "Porque usa −log D(G(z)) en vez de log(1 − D(G(z))); al inicio D rechaza todo y la "
         "segunda versión casi no da gradiente."),
        ("¿Qué controla la semilla en una solicitud con filtro?",
         "Los 4n candidatos: misma semilla y mismo modelo dan exactamente el mismo lote."),
        ("¿Por qué no confiar en las pérdidas de la GAN para decidir el modelo?",
         "Porque oscilan y miden el juego entre G y D, no la calidad; por eso usamos KID."),
        ("Si KID real vs real ≈ 0, ¿qué significa 1.41×10⁻³ del base?",
         "Que hay una diferencia pequeña pero medible entre lo generado y lo real; sirve para comparar "
         "modelos con el mismo protocolo, no como nota absoluta."),
        ("¿Por qué spectral y spectral_ema_sin_ema tienen el mismo KID?",
         "Son el mismo modelo: el entrenamiento es determinista y los pesos son idénticos bit a bit."),
        ("¿Qué pasaría con el filtro si τ fuera muy alto?",
         "Pocos candidatos lo pasarían; el filtro completa con los más novedosos y las imágenes se "
         "alejarían de los datos (podría empeorar la calidad)."),
        ("¿Por qué la alerta de novedad ignora solicitudes con filtro?",
         "El filtro sube la novedad a propósito; daría falsas alertas."),
        ("¿Por qué el candado mejora el rendimiento con 2 vCPU?",
         "Varias inferencias a la vez compiten por los mismos núcleos (sobresuscripción de hilos); "
         "en fila cada una usa todos los núcleos."),
        ("¿Qué se pierde si se reinicia el Space y cómo se mitiga?",
         "La base SQLite; se exporta /api/registros a evidencias/; a largo plazo, almacenamiento "
         "persistente."),
        ("¿Cuál es la línea base de cada hipótesis?",
         "H1: el modelo base del P2. H2: generar sin filtro. H3: fp32 y el criterio de latencia."),
        ("¿Por qué no cambiar al modelo hinge si tuvo el mejor KID?",
         "No era la hipótesis preregistrada; elegirlo ahora sería decidir después de ver los datos. "
         "Queda como propuesta para una siguiente iteración con su propia hipótesis."),
        ("¿Qué demuestra que está en producción?",
         "Una URL pública que usan personas ajenas al equipo, con sus solicitudes y votos "
         "registrados, más capturas, prueba de carga y CI en verde."),
    ]
    for q, r in preguntas:
        h.append(KeepTogether([P(q, "preg"), P(r, "resp")]))
    h.append(Spacer(1, 8))
    h.append(caja([P("<b>Si no saben una respuesta:</b> digan qué saben, dónde está en el repositorio "
                     "y qué harían para comprobarlo. Inventar un número es peor que decir «no lo "
                     "medimos».", "habla")], fondo=AVISO))

    doc.build(h, onFirstPage=pie, onLaterPages=pie)
    print(SALIDA)


if __name__ == "__main__":
    main()
