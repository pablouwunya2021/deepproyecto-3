# Hipótesis del Proyecto Final (preregistro)

> **Registrado el 2026-11-13, antes de entrenar el modelo nuevo (`spectral_ema`) y
> antes de calcular cualquiera de las métricas de abajo.** No se cambian los
> criterios después de ver los resultados. Los resultados se escriben en
> [`resultados.md`](resultados.md), que genera `scripts/evaluar_hipotesis.py`.

Línea base de producción: **`G_base`**, el generador del Proyecto 2 con el que se
armó la galería original (DCGAN, BCE no saturante, BatchNorm en D, 60 épocas).

Protocolo común de medición (fijo para todas las hipótesis):

- Extractor de características: SqueezeNet 1.1 preentrenada en ImageNet (la misma
  del Proyecto 2), vector de 512 dimensiones normalizado L2.
- Reales: las 5000 imágenes del dataset Lumen (`SEED_DATA = 20260101`).
- Generadas: 2000 muestras por modelo con la semilla de evaluación `SEED_EVAL_FINAL = 777`.
- **KID** (Kernel Inception Distance, kernel polinomial de grado 3), promedio de
  10 subconjuntos de 1000. Menor es mejor. Referencia inferior: KID entre dos
  mitades disjuntas de los datos reales.

---

## H1 — Modelo: normalización espectral + EMA del generador

**Hipótesis.** Entrenar con normalización espectral en D (que en el Proyecto 2
mantuvo D(x) ≈ D(G(z)) ≈ 0.5) y servir un promedio exponencial (EMA, decay 0.995)
de los pesos de G produce muestras más cercanas a la distribución real que `G_base`,
porque la EMA suaviza la oscilación de G entre pasos del juego adversarial.

- **Métrica:** KID (protocolo común).
- **Criterio de éxito (definido antes de entrenar):** KID(`spectral_ema`) ≤ 0.85 × KID(`G_base`),
  es decir, una reducción de al menos 15 %.
- **Control para aislar la EMA:** se compara también contra `G_spectral` (Proyecto 2)
  y contra el G sin EMA de la misma corrida (`spectral_ema_sin_ema`). Mismos datos,
  semilla (`SEED_TRAIN = 42`), épocas (60) y batch (128).
- **Decisión:** si se cumple, `spectral_ema` pasa a ser el modelo por defecto del
  servicio; si no, se queda `G_base`.

## H2 — Servicio: filtro de novedad y diversidad

**Hipótesis.** Un filtro en el servicio (muestrear 4 × n candidatos, descartar los
que estén a menos de τ de su vecino de entrenamiento y elegir n por punto más lejano)
entrega lotes más diversos que el muestreo directo sin alejarse de la distribución.

- **Métricas:** (a) diversidad intra-lote = distancia media por pares de las
  características dentro de un lote de 8; (b) cobertura = número de elementos
  distintos (de 6) en el lote; (c) KID de las muestras filtradas.
- **Línea base:** muestreo directo de n = 8 con el mismo modelo, 200 lotes.
- **Criterio de éxito:** diversidad ≥ 1.10 × la línea base **y** cobertura media
  ≥ línea base **y** KID filtrado ≤ 1.25 × KID directo.
- τ se fija como el percentil 10 de la distancia al vecino de entrenamiento de las
  muestras directas del modelo servido (se calcula una vez al exportar el modelo y
  queda en `modelos/tarjeta_modelo.json`).

## H3 — Operación: pesos en fp16 y latencia en CPU

**Hipótesis.** Guardar los pesos en fp16 (para que cada archivo pese < 10 MB y entre
en Hugging Face Spaces sin Git LFS) y ejecutar en fp32 no cambia las salidas de forma
apreciable, y el servicio responde rápido en CPU.

- **Criterio de éxito:** diferencia absoluta media de píxel entre fp32 y fp16
  < 1/255, mismo elemento asignado en ≥ 99 % de 500 muestras, y latencia
  p95 de `POST /api/generar` (n = 8, con filtro) < 1.5 s en la prueba de carga
  local con 4 clientes concurrentes.

---

Un resultado negativo también se reporta como tal; el criterio no se ajusta.
