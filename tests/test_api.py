"""Pruebas de integracion del servicio (se ejecutan en CI en cada push)."""
import base64
import io

import numpy as np
from PIL import Image


def _img(b64):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


def test_salud(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200
    d = r.json()
    assert d["estado"] == "ok" and d["modelo_por_defecto"] in d["modelos"]


def test_tarjeta_modelo(cliente):
    t = cliente.get("/api/modelo").json()
    assert set(t["modelos_servidos"]) <= set(t["modelos"])
    for m in t["modelos_servidos"]:
        assert t["modelos"][m]["tau"] > 0


def test_generar_devuelve_n_personajes(cliente):
    r = cliente.post("/api/generar", json={"n": 5, "semilla": 123})
    assert r.status_code == 200
    d = r.json()
    assert len(d["personajes"]) == 5
    p = d["personajes"][0]
    assert p["elemento"] in {"brasa", "marea", "musgo", "cefiro", "vacio", "solar"}
    assert _img(p["png_base64"]).shape == (256, 256, 3)


def test_generar_es_determinista(cliente):
    a = cliente.post("/api/generar", json={"n": 4, "semilla": 42}).json()
    b = cliente.post("/api/generar", json={"n": 4, "semilla": 42}).json()
    assert [p["semilla"] for p in a["personajes"]] == [p["semilla"] for p in b["personajes"]]
    assert a["personajes"][0]["png_base64"] == b["personajes"][0]["png_base64"]


def test_filtro_respeta_tau(cliente):
    t = cliente.get("/api/modelo").json()
    d = cliente.post("/api/generar", json={"n": 8, "semilla": 7, "filtro": True}).json()
    tau = t["modelos"][d["modelo"]]["tau"]
    assert all(p["novedad"] >= tau - 1e-4 for p in d["personajes"])
    assert len({p["semilla"] for p in d["personajes"]}) == 8   # sin repetidos


def test_personaje_png_reproduce_el_lote(cliente):
    d = cliente.post("/api/generar", json={"n": 1, "semilla": 99, "filtro": False}).json()
    p = d["personajes"][0]
    r = cliente.get(p["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.headers["x-lumen-nombre"] == p["nombre"]
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(r.content))), _img(p["png_base64"]))


def test_todos_los_modelos_servidos(cliente):
    for m in cliente.get("/salud").json()["modelos"]:
        r = cliente.post("/api/generar", json={"n": 2, "semilla": 1, "modelo": m})
        assert r.status_code == 200 and r.json()["modelo"] == m


def test_validacion_de_entrada(cliente):
    assert cliente.post("/api/generar", json={"n": 0}).status_code == 422
    assert cliente.post("/api/generar", json={"n": 17}).status_code == 422
    assert cliente.post("/api/generar", json={"semilla": -1}).status_code == 422
    assert cliente.post("/api/generar", json={"modelo": "no_existe"}).status_code == 422
    assert cliente.post("/api/retroalimentacion",
                        json={"semilla": 1, "modelo": "base", "voto": "talvez"}).status_code == 422


def test_retroalimentacion_y_estadisticas(cliente):
    m = cliente.get("/salud").json()["modelo_por_defecto"]
    antes = cliente.get("/api/estadisticas").json()
    r = cliente.post("/api/retroalimentacion",
                     json={"semilla": 5, "modelo": m, "elemento": "marea", "voto": "me_sirve",
                           "filtro": True})
    assert r.status_code == 200
    s = cliente.get("/api/estadisticas").json()
    assert s["generaciones"] >= 1
    assert s["latencia_ms"]["p95"] is not None
    assert s["retroalimentacion"][m]["me_sirve"] >= antes.get("retroalimentacion", {}).get(m, {}).get("me_sirve", 0) + 1
    assert m in s["deriva"]
    assert s["aceptacion_por_filtro"]["con_filtro"]["me_sirve"] >= 1


def test_registros_y_metricas(cliente):
    reg = cliente.get("/api/registros?limite=5").json()
    assert reg["solicitudes"] and "latencia_ms" in reg["solicitudes"][0]
    assert "ip" not in reg["solicitudes"][0]          # no se guardan datos personales
    met = cliente.get("/metrics").text
    assert "lumen_generacion_segundos_bucket" in met and "lumen_personajes_total" in met


def test_paginas(cliente):
    assert "Lumen" in cliente.get("/").text
    assert "Monitoreo" in cliente.get("/monitoreo").text
    assert cliente.get("/docs").status_code == 200
