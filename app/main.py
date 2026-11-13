"""
Servicio web de Lumen: API REST + interfaz para el equipo de arte.

    uvicorn app.main:app --host 0.0.0.0 --port 7860

Rutas principales
    GET  /                    interfaz web (generar, revisar y votar conceptos)
    GET  /monitoreo           tablero de monitoreo
    GET  /docs                documentacion interactiva de la API (OpenAPI)
    POST /api/generar         genera n personajes (con o sin filtro de novedad)
    GET  /api/personaje/{s}.png  regenera un personaje exacto a partir de su semilla
    POST /api/retroalimentacion  voto "me_sirve" / "descartar"
    GET  /api/modelo          tarjeta del modelo
    GET  /api/estadisticas    metricas de uso, latencia, aceptacion y deriva
    GET  /api/registros       ultimas solicitudes registradas
    GET  /salud               chequeo de vida
    GET  /metrics             metricas en formato Prometheus
"""
from __future__ import annotations

import base64
import io
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from app.monitoreo import Registro, hash_cliente
from lumen.inferencia import SEMILLA_MAX, MotorLumen

VERSION = "1.0.0"
STATIC = os.path.join(os.path.dirname(__file__), "static")
RUTA_DB = os.environ.get("LUMEN_DB", os.path.join(os.path.dirname(__file__), "..",
                                                  "registros", "lumen.sqlite"))
torch.set_num_threads(int(os.environ.get("LUMEN_HILOS", "2")))
torch.set_grad_enabled(False)

log = logging.getLogger("lumen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

M_SOLICITUDES = Counter("lumen_solicitudes_total", "Solicitudes por ruta y estado",
                        ["ruta", "estado"])
M_LATENCIA = Histogram("lumen_generacion_segundos", "Latencia de /api/generar", ["modelo"],
                       buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5))
M_PERSONAJES = Counter("lumen_personajes_total", "Personajes entregados", ["modelo", "elemento"])
M_VOTOS = Counter("lumen_votos_total", "Retroalimentacion del equipo de arte", ["modelo", "voto"])

estado: dict = {}
# Una inferencia a la vez: con 2 vCPU, varias en paralelo compiten por los mismos
# nucleos (sobresuscripcion de hilos de torch) y bajan el rendimiento total.
INFERENCIA = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    estado["motor"] = MotorLumen()
    estado["registro"] = Registro(RUTA_DB)
    estado["inicio"] = time.time()
    log.info("modelos cargados en %.1fs: %s (por defecto: %s) | registros en %s",
             time.time() - t0, list(estado["motor"].G), estado["motor"].por_defecto, RUTA_DB)
    yield


app = FastAPI(
    title="Lumen - generador de conceptos de personajes",
    description="DCGAN 64x64 del Proyecto 2 en produccion. Genera espiritus guardianes "
                "elementales para el equipo de arte de un videojuego.",
    version=VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------- esquemas
class SolicitudGenerar(BaseModel):
    n: int = Field(8, ge=1, le=16, description="Cantidad de personajes (1-16)")
    semilla: Optional[int] = Field(None, ge=0, le=SEMILLA_MAX,
                                   description="Semilla del lote; si se omite es aleatoria")
    filtro: bool = Field(True, description="Filtro de novedad y diversidad (H2)")
    modelo: Optional[str] = Field(None, description="Nombre del generador; por defecto el de la tarjeta")
    tam: int = Field(256, ge=64, le=512, description="Lado en pixeles del PNG devuelto")


class PersonajeOut(BaseModel):
    semilla: int
    nombre: str
    elemento: str
    rol: str
    novedad: float
    url: str
    png_base64: str


class RespuestaGenerar(BaseModel):
    modelo: str
    semilla: int
    filtro: bool
    latencia_ms: float
    personajes: list[PersonajeOut]


class Voto(BaseModel):
    semilla: int = Field(..., ge=0, le=SEMILLA_MAX)
    modelo: str
    elemento: Optional[str] = None
    voto: Literal["me_sirve", "descartar"]
    filtro: Optional[bool] = Field(None, description="Si el personaje vino de un lote con filtro")


# ---------------------------------------------------------------------- utilidades
def _png(img: np.ndarray, tam: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).resize((tam, tam), Image.NEAREST).save(buf, format="PNG")
    return buf.getvalue()


def _cliente(req: Request) -> str:
    ip = req.headers.get("x-forwarded-for", req.client.host if req.client else None)
    return hash_cliente(ip, req.headers.get("user-agent"))


def _modelo_valido(nombre: Optional[str]) -> str:
    motor: MotorLumen = estado["motor"]
    nombre = nombre or motor.por_defecto
    if nombre not in motor.G:
        raise HTTPException(422, f"modelo desconocido '{nombre}'. Opciones: {list(motor.G)}")
    return nombre


@app.middleware("http")
async def contar(request: Request, call_next):
    resp = await call_next(request)
    ruta = request.scope.get("route")
    M_SOLICITUDES.labels(getattr(ruta, "path", "otra"), str(resp.status_code)).inc()
    return resp


# ---------------------------------------------------------------------- rutas
@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/monitoreo", include_in_schema=False)
def tablero():
    return FileResponse(os.path.join(STATIC, "monitoreo.html"))


@app.get("/salud")
def salud():
    motor: MotorLumen = estado["motor"]
    return {"estado": "ok", "version": VERSION, "modelos": list(motor.G),
            "modelo_por_defecto": motor.por_defecto,
            "activo_hace_s": round(time.time() - estado["inicio"], 1)}


@app.get("/api/modelo")
def tarjeta():
    return estado["motor"].tarjeta


@app.post("/api/generar", response_model=RespuestaGenerar)
def generar(sol: SolicitudGenerar, request: Request):
    motor: MotorLumen = estado["motor"]
    reg: Registro = estado["registro"]
    modelo = _modelo_valido(sol.modelo)
    semilla = sol.semilla if sol.semilla is not None else secrets.randbelow(SEMILLA_MAX)
    t0 = time.perf_counter()
    try:
        with INFERENCIA:
            ps = motor.generar(sol.n, semilla, modelo, sol.filtro)
    except Exception:
        log.exception("fallo al generar")
        reg.solicitud("generar", modelo, sol.n, sol.filtro, semilla, None, 500,
                      cliente=_cliente(request))
        raise HTTPException(500, "error interno al generar")
    personajes = [PersonajeOut(
        semilla=p.semilla, nombre=p.nombre, elemento=p.elemento, rol=p.rol,
        novedad=round(p.novedad, 4), url=f"/api/personaje/{p.semilla}.png?modelo={modelo}",
        png_base64=base64.b64encode(_png(p.imagen, sol.tam)).decode()) for p in ps]
    lat = (time.perf_counter() - t0) * 1000
    M_LATENCIA.labels(modelo).observe(lat / 1000)
    for p in ps:
        M_PERSONAJES.labels(modelo, p.elemento).inc()
    reg.solicitud("generar", modelo, sol.n, sol.filtro, semilla, lat, 200,
                  float(np.mean([p.novedad for p in ps])), [p.elemento for p in ps],
                  _cliente(request))
    log.info("generar modelo=%s n=%d filtro=%s semilla=%d %.0fms", modelo, sol.n,
             sol.filtro, semilla, lat)
    return RespuestaGenerar(modelo=modelo, semilla=semilla, filtro=sol.filtro,
                            latencia_ms=round(lat, 1), personajes=personajes)


@app.get("/api/personaje/{semilla}.png", response_class=Response,
         responses={200: {"content": {"image/png": {}}}})
def personaje_png(semilla: int, request: Request, modelo: Optional[str] = None,
                  tam: int = Query(256, ge=64, le=512)):
    if not 0 <= semilla <= SEMILLA_MAX:
        raise HTTPException(422, "semilla fuera de rango")
    modelo = _modelo_valido(modelo)
    with INFERENCIA:
        p = estado["motor"].personaje(semilla, modelo)
    estado["registro"].solicitud("personaje", modelo, 1, None, semilla, None, 200,
                                 p.novedad, [p.elemento], _cliente(request))
    return Response(_png(p.imagen, tam), media_type="image/png",
                    headers={"X-Lumen-Nombre": p.nombre, "X-Lumen-Elemento": p.elemento,
                             "Cache-Control": "public, max-age=86400"})


@app.post("/api/retroalimentacion")
def retroalimentacion(v: Voto, request: Request):
    _modelo_valido(v.modelo)
    estado["registro"].voto(v.semilla, v.modelo, v.elemento, v.voto, v.filtro, _cliente(request))
    M_VOTOS.labels(v.modelo, v.voto).inc()
    return {"ok": True}


@app.get("/api/estadisticas")
def estadisticas():
    return estado["registro"].estadisticas(estado["motor"].tarjeta)


@app.get("/api/registros")
def registros(limite: int = Query(100, ge=1, le=1000)):
    reg: Registro = estado["registro"]
    return {"solicitudes": reg.recientes(limite), "retroalimentacion": reg.votos_recientes(limite)}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(ValueError)
async def valor_invalido(_, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})
