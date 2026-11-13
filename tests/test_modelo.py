"""Pruebas unitarias del modelo exportado y de la logica de inferencia."""
import numpy as np
import torch

from lumen.inferencia import BANCO, cargar_generador, leer_tarjeta, seleccionar, z_de_semillas


def test_pesos_fp16_cargan_y_generan_rango_valido():
    for m in leer_tarjeta()["modelos_servidos"]:
        G = cargar_generador(m)
        with torch.no_grad():
            x = G(z_de_semillas([0, 1, 2]))
        assert x.shape == (3, 3, 64, 64)
        assert x.min() >= -1 and x.max() <= 1
        assert torch.isfinite(x).all()


def test_independencia_del_lote():
    """BatchNorm en eval: un personaje no depende de con quien se genere."""
    G = cargar_generador(leer_tarjeta()["modelo_por_defecto"])
    with torch.no_grad():
        solo = G(z_de_semillas([77]))
        lote = G(z_de_semillas([5, 77, 9]))[1:2]
    torch.testing.assert_close(solo, lote, atol=1e-5, rtol=0)


def test_banco_de_caracteristicas():
    b = np.load(BANCO)
    assert b.shape == (5000, 512)
    np.testing.assert_allclose(np.linalg.norm(b.astype(np.float32), axis=1), 1, atol=1e-2)


def test_seleccionar_respeta_tau_y_no_repite():
    g = torch.Generator().manual_seed(0)
    feats = torch.nn.functional.normalize(torch.randn(32, 16, generator=g), dim=1)
    nov = torch.linspace(0, 1, 32)
    idx = seleccionar(feats, nov, 8, tau=0.5)
    assert len(idx) == 8 == len(set(idx))
    assert all(nov[i] >= 0.5 for i in idx)


def test_seleccionar_completa_si_faltan_candidatos():
    feats = torch.eye(10)
    nov = torch.zeros(10)
    assert len(set(seleccionar(feats, nov, 6, tau=1.0))) == 6
