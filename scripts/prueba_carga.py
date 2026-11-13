"""
Prueba de carga contra el servicio (local o desplegado).

    python scripts/prueba_carga.py --url http://127.0.0.1:7860 --solicitudes 200 --clientes 4

Mezcla realista: 80 % con filtro, n=8; 20 % sin filtro; y vota sobre ~30 % de los
personajes con una regla simulada (para poblar el tablero de monitoreo).
Escribe evidencias/prueba_carga.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def una(cliente: httpx.Client, url: str, i: int, votar: bool):
    rnd = random.Random(i)
    filtro = rnd.random() < 0.8
    t0 = time.perf_counter()
    r = cliente.post(f"{url}/api/generar", json={"n": 8, "filtro": filtro})
    lat = (time.perf_counter() - t0) * 1000
    ok = r.status_code == 200
    if ok and votar:
        d = r.json()
        for p in d["personajes"]:
            if rnd.random() < 0.3:
                # regla simulada del "equipo de arte": prefiere conceptos mas novedosos
                voto = "me_sirve" if p["novedad"] > 0.62 or rnd.random() < 0.3 else "descartar"
                cliente.post(f"{url}/api/retroalimentacion",
                             json={"semilla": p["semilla"], "modelo": d["modelo"],
                                   "elemento": p["elemento"], "voto": voto,
                                   "filtro": filtro})
    return {"ok": ok, "estado": r.status_code, "lat": lat, "filtro": filtro,
            "servidor_ms": r.json().get("latencia_ms") if ok else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--solicitudes", type=int, default=200)
    ap.add_argument("--clientes", type=int, default=4)
    ap.add_argument("--sin-votos", action="store_true")
    ap.add_argument("--salida", default=os.path.join(RAIZ, "evidencias", "prueba_carga.json"))
    a = ap.parse_args()

    with httpx.Client(timeout=60, headers={"user-agent": "lumen-prueba-carga"}) as c:
        c.get(f"{a.url}/salud").raise_for_status()
        t0 = time.time()
        with ThreadPoolExecutor(a.clientes) as ex:
            res = list(ex.map(lambda i: una(c, a.url, i, not a.sin_votos), range(a.solicitudes)))
        dur = time.time() - t0

    lat = sorted(r["lat"] for r in res if r["ok"])
    lat_f = sorted(r["lat"] for r in res if r["ok"] and r["filtro"])
    q = lambda xs, p: xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None
    out = {
        "url": a.url, "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "solicitudes": a.solicitudes, "clientes_concurrentes": a.clientes,
        "duracion_s": round(dur, 2), "rps": round(a.solicitudes / dur, 2),
        "errores": sum(not r["ok"] for r in res),
        "latencia_ms": {"p50": q(lat, 50), "p95": q(lat_f, 95), "p99": q(lat, 99),
                        "media": statistics.mean(lat) if lat else None, "max": lat[-1] if lat else None,
                        "nota": "p95 calculado sobre solicitudes con filtro (criterio H3)"},
        "latencia_servidor_ms_media": statistics.mean(r["servidor_ms"] for r in res if r["ok"]),
    }
    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    with open(a.salida, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
