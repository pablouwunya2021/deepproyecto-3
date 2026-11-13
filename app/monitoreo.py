"""
Registro de uso y monitoreo del servicio.

- Cada solicitud de generacion queda en SQLite (latencia, modelo, filtro, novedad
  media, elementos entregados). No se guardan IP ni datos personales: el cliente se
  identifica con un hash truncado de IP + user-agent para contar usuarios distintos.
- La retroalimentacion ("me sirve" / "descartar") de cada personaje es la metrica
  de negocio: tasa de aceptacion de conceptos.
- `estadisticas()` compara lo servido con la referencia de la tarjeta del modelo
  para detectar deriva (caida de novedad = posible colapso de modos, o cambio en la
  mezcla de elementos).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from collections import Counter

ESQUEMA = """
CREATE TABLE IF NOT EXISTS solicitudes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    endpoint TEXT NOT NULL,
    modelo TEXT,
    n INTEGER,
    filtro INTEGER,
    semilla INTEGER,
    latencia_ms REAL,
    estado INTEGER,
    novedad_media REAL,
    elementos TEXT,
    cliente TEXT
);
CREATE TABLE IF NOT EXISTS retroalimentacion (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    semilla INTEGER NOT NULL,
    modelo TEXT NOT NULL,
    elemento TEXT,
    voto TEXT NOT NULL,
    filtro INTEGER,
    cliente TEXT
);
"""

UMBRAL_DERIVA_NOVEDAD = 3.0   # alerta si la media servida se aleja > 3 errores estandar
UMBRAL_DERIVA_TV = 0.25       # alerta si la distancia de variacion total de elementos > 0.25
VENTANA = 500                 # ultimas solicitudes consideradas para deriva


def hash_cliente(ip: str | None, ua: str | None) -> str:
    return hashlib.sha256(f"{ip}|{ua}".encode()).hexdigest()[:12]


def _percentil(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


class Registro:
    def __init__(self, ruta: str):
        os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
        self.ruta = ruta
        self._lock = threading.Lock()
        self._con = sqlite3.connect(ruta, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(ESQUEMA)
        self._con.commit()

    def _exec(self, sql, params=()):
        with self._lock:
            cur = self._con.execute(sql, params)
            self._con.commit()
            return cur

    def _query(self, sql, params=()):
        with self._lock:
            return [dict(r) for r in self._con.execute(sql, params).fetchall()]

    # ------------------------------------------------------------------ escritura
    def solicitud(self, endpoint, modelo=None, n=None, filtro=None, semilla=None,
                  latencia_ms=None, estado=200, novedad_media=None, elementos=None,
                  cliente=None):
        self._exec(
            "INSERT INTO solicitudes (ts, endpoint, modelo, n, filtro, semilla, latencia_ms,"
            " estado, novedad_media, elementos, cliente) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), endpoint, modelo, n, None if filtro is None else int(filtro),
             semilla, latencia_ms, estado, novedad_media,
             json.dumps(elementos) if elementos is not None else None, cliente))

    def voto(self, semilla, modelo, elemento, voto, filtro=None, cliente=None):
        self._exec(
            "INSERT INTO retroalimentacion (ts, semilla, modelo, elemento, voto, filtro, cliente)"
            " VALUES (?,?,?,?,?,?,?)", (time.time(), semilla, modelo, elemento, voto,
                                       None if filtro is None else int(filtro), cliente))

    # ------------------------------------------------------------------ lectura
    def recientes(self, limite=100):
        filas = self._query("SELECT * FROM solicitudes ORDER BY id DESC LIMIT ?", (limite,))
        for f in filas:
            f["elementos"] = json.loads(f["elementos"]) if f["elementos"] else None
        return filas

    def votos_recientes(self, limite=100):
        return self._query("SELECT * FROM retroalimentacion ORDER BY id DESC LIMIT ?",
                           (limite,))

    def estadisticas(self, tarjeta: dict) -> dict:
        gen = self._query("SELECT * FROM solicitudes WHERE endpoint='generar' ORDER BY id")
        ok = [g for g in gen if g["estado"] == 200]
        lat = [g["latencia_ms"] for g in ok if g["latencia_ms"] is not None]
        votos = self._query("SELECT modelo, voto, COUNT(*) c FROM retroalimentacion "
                            "GROUP BY modelo, voto")
        votos_filtro = self._query("SELECT filtro, voto, COUNT(*) c FROM retroalimentacion "
                                   "WHERE filtro IS NOT NULL GROUP BY filtro, voto")
        total = self._query("SELECT COUNT(*) c, COUNT(DISTINCT cliente) u FROM solicitudes")[0]

        aceptacion = {}
        for v in votos:
            a = aceptacion.setdefault(v["modelo"], {"me_sirve": 0, "descartar": 0})
            a[v["voto"]] = v["c"]
        # comparacion A/B del filtro (H2 no cumplio su criterio: se decide con uso real)
        ab = {}
        for v in votos_filtro:
            a = ab.setdefault("con_filtro" if v["filtro"] else "sin_filtro",
                              {"me_sirve": 0, "descartar": 0})
            a[v["voto"]] = v["c"]
        for a in list(aceptacion.values()) + list(ab.values()):
            t = a["me_sirve"] + a["descartar"]
            a["tasa_aceptacion"] = a["me_sirve"] / t if t else None

        # deriva por modelo sobre la ventana reciente
        deriva = {}
        for modelo, ref in tarjeta["modelos"].items():
            filas = [g for g in ok if g["modelo"] == modelo][-VENTANA:]
            if not filas:
                continue
            nov = [g["novedad_media"] for g in filas if g["novedad_media"] is not None]
            sin_filtro = [g["novedad_media"] for g in filas
                          if not g["filtro"] and g["novedad_media"] is not None]
            cnt = Counter(e for g in filas for e in (g["elementos"] and json.loads(g["elementos"]) or []))
            tot = sum(cnt.values()) or 1
            dist = {e: cnt.get(e, 0) / tot for e in ref["elementos_ref"]}
            tv = 0.5 * sum(abs(dist.get(e, 0) - ref["elementos_ref"].get(e, 0))
                           for e in set(dist) | set(ref["elementos_ref"]))
            # la novedad solo se compara con la referencia en solicitudes sin filtro
            # (el filtro la sube a proposito)
            z = None
            if len(sin_filtro) >= 5:
                media = sum(sin_filtro) / len(sin_filtro)
                ee = ref["novedad_ref_std"] / math.sqrt(len(sin_filtro))
                z = (media - ref["novedad_ref_media"]) / ee if ee else None
            alertas = []
            if z is not None and abs(z) > UMBRAL_DERIVA_NOVEDAD:
                alertas.append(f"novedad fuera de referencia (z={z:.1f})")
            if tot >= 50 and tv > UMBRAL_DERIVA_TV:
                alertas.append(f"mezcla de elementos distinta a la referencia (TV={tv:.2f})")
            deriva[modelo] = {
                "solicitudes_ventana": len(filas),
                "novedad_media_servida": sum(nov) / len(nov) if nov else None,
                "novedad_ref_media": ref["novedad_ref_media"],
                "z_novedad_sin_filtro": z,
                "elementos_servidos": dist,
                "elementos_ref": ref["elementos_ref"],
                "distancia_tv_elementos": tv,
                "alertas": alertas,
            }

        # serie por hora para el tablero
        por_hora = Counter(time.strftime("%Y-%m-%d %H:00", time.localtime(g["ts"])) for g in gen)
        return {
            "solicitudes_totales": total["c"],
            "clientes_distintos": total["u"],
            "generaciones": len(gen),
            "errores": len(gen) - len(ok),
            "tasa_error": (len(gen) - len(ok)) / len(gen) if gen else 0.0,
            "personajes_entregados": sum(g["n"] or 0 for g in ok),
            "latencia_ms": {"p50": _percentil(lat, 50), "p95": _percentil(lat, 95),
                            "max": max(lat) if lat else None},
            "uso_por_modelo": dict(Counter(g["modelo"] for g in ok)),
            "uso_filtro": sum(1 for g in ok if g["filtro"]) / len(ok) if ok else None,
            "retroalimentacion": aceptacion,
            "aceptacion_por_filtro": ab,
            "deriva": deriva,
            "solicitudes_por_hora": dict(sorted(por_hora.items())),
        }
