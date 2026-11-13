"""
Inferencia del generador Lumen, compartida por el servicio, la exportacion y la
evaluacion. Todo es determinista: cada personaje se identifica por una semilla
entera y la misma semilla siempre produce la misma imagen (BatchNorm en modo eval,
asi que el resultado no depende del resto del lote).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from lumen import config as C
from lumen.features import FeatureExtractor
from lumen.gallery import NOMBRES, ROLES, _elemento_de
from lumen.models import Generator
from lumen.utils import tensor_to_uint8

MODELOS_DIR = os.path.join(C.ROOT, "modelos")
TARJETA = os.path.join(MODELOS_DIR, "tarjeta_modelo.json")
BANCO = os.path.join(MODELOS_DIR, "banco_features.npy")
SEMILLA_MAX = 2**31 - 1


def ruta_pesos(nombre: str) -> str:
    return os.path.join(MODELOS_DIR, f"G_{nombre}.pt")


def cargar_generador(nombre: str, device="cpu") -> Generator:
    """Carga pesos exportados (fp16) y ejecuta en fp32."""
    cfg = C.base_config()
    G = Generator(cfg.nz, cfg.ngf, cfg.channels)
    estado = torch.load(ruta_pesos(nombre), map_location="cpu")
    G.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in estado.items()})
    return G.to(device).eval()


def z_de_semillas(semillas, nz: int = 128) -> torch.Tensor:
    """Un vector z por semilla: N(0, I) con torch.Generator propio."""
    zs = [torch.randn(nz, generator=torch.Generator().manual_seed(int(s))) for s in semillas]
    return torch.stack(zs).view(len(zs), nz, 1, 1)


def describir(img_uint8: np.ndarray, semilla: int) -> dict:
    elem = _elemento_de(img_uint8)
    pool = NOMBRES[elem]
    return {"elemento": elem, "nombre": pool[semilla % len(pool)], "rol": ROLES[elem]}


def leer_tarjeta() -> dict:
    with open(TARJETA) as f:
        return json.load(f)


@dataclass
class Personaje:
    semilla: int
    imagen: np.ndarray        # (64, 64, 3) uint8
    elemento: str
    nombre: str
    rol: str
    novedad: float            # distancia L2 al vecino de entrenamiento mas cercano


class MotorLumen:
    """Carga los generadores, el extractor y el banco de caracteristicas una vez."""

    def __init__(self, modelos=None, device="cpu", estricto=True):
        self.device = torch.device(device)
        self.tarjeta = leer_tarjeta()
        nombres = modelos or list(self.tarjeta["modelos_servidos"])
        self.G = {n: cargar_generador(n, self.device) for n in nombres}
        self.fx = FeatureExtractor(self.device, estricto=estricto)
        self.banco = torch.from_numpy(np.load(BANCO).astype(np.float32)).to(self.device)
        self.por_defecto = self.tarjeta["modelo_por_defecto"]

    def tau(self, modelo: str) -> float:
        return float(self.tarjeta["modelos"][modelo]["tau"])

    @torch.no_grad()
    def _generar(self, modelo: str, semillas):
        fake = self.G[modelo](z_de_semillas(semillas).to(self.device))
        feats = self.fx(fake)
        nov = torch.cdist(feats, self.banco).min(dim=1).values
        return fake, feats, nov

    def generar(self, n: int, semilla: int, modelo: str | None = None,
                filtro: bool = True) -> list[Personaje]:
        modelo = modelo or self.por_defecto
        rng = np.random.default_rng(semilla)
        k = 4 * n if filtro else n
        semillas = rng.integers(0, SEMILLA_MAX, size=k).tolist()
        fake, feats, nov = self._generar(modelo, semillas)
        idx = list(range(n))
        if filtro:
            idx = seleccionar(feats, nov, n, self.tau(modelo))
        out = []
        for i in idx:
            img = tensor_to_uint8(fake[i])
            d = describir(img, semillas[i])
            out.append(Personaje(semillas[i], img, d["elemento"], d["nombre"], d["rol"],
                                 float(nov[i])))
        return out

    def personaje(self, semilla: int, modelo: str | None = None) -> Personaje:
        modelo = modelo or self.por_defecto
        fake, _, nov = self._generar(modelo, [semilla])
        img = tensor_to_uint8(fake[0])
        d = describir(img, semilla)
        return Personaje(semilla, img, d["elemento"], d["nombre"], d["rol"], float(nov[0]))


def seleccionar(feats: torch.Tensor, nov: torch.Tensor, n: int, tau: float) -> list[int]:
    """Filtro de produccion (H2): descarta candidatos con novedad < tau y elige n
    por punto-mas-lejano empezando por el mas novedoso. Si no alcanzan, completa
    con los mas novedosos restantes."""
    nov_np = nov.cpu().numpy()
    cand = [i for i in np.argsort(-nov_np) if nov_np[i] >= tau]
    if len(cand) < n:
        cand = list(np.argsort(-nov_np)[:max(n, len(cand))])
    F = feats[cand]
    elegidos = [0]
    d = torch.cdist(F, F[0:1]).squeeze(1)
    while len(elegidos) < n:
        d[elegidos] = -1.0          # nunca repetir un candidato
        j = int(torch.argmax(d))
        elegidos.append(j)
        d = torch.minimum(d, torch.cdist(F, F[j:j + 1]).squeeze(1))
    return [int(cand[j]) for j in elegidos]
