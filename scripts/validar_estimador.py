"""Validación del estimador del trío con varias semillas del generador.

Para cada semilla regenera el mundo sintético, corre la prueba del trío y compara cada estimación
con la verdad (mismo estimador sobre los litros esperados sin ruido). Reporta sesgo medio,
dispersión y cobertura del intervalo de confianza.

    python scripts/validar_estimador.py --semillas 8
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from share_diag import comun, datos, tactica  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=100)
    a = ap.parse_args()
    cfg = comun.config()
    original = cfg["semilla"], cfg["experimento"]["bootstrap"]
    cfg["experimento"]["bootstrap"] = a.bootstrap
    filas = []
    try:
        for s in range(1, a.semillas + 1):
            cfg["semilla"] = 1000 + s
            d = datos.generar()
            exp = tactica.experimento(tactica.Panel(d["venta_clientes"], d["clientes"]))
            filas.append(exp.assign(semilla=cfg["semilla"]))
            print(f"semilla {cfg['semilla']}: {len(exp)} eventos")
    finally:
        cfg["semilla"], cfg["experimento"]["bootstrap"] = original
    r = pd.concat(filas)
    r["error"] = r["efecto_pp"] - r["verdad_pp"]
    r["cubre"] = (r["ic_inf"] <= r["verdad_pp"]) & (r["verdad_pp"] <= r["ic_sup"])
    r["error_ingenuo"] = r["diferencia_ingenua_pp"] - r["verdad_pp"]
    res = r.groupby(["tipo", "clase"]).agg(
        eventos=("error", "size"), sesgo_pp=("error", "mean"), dispersion_pp=("error", "std"),
        cobertura_ic=("cubre", "mean"), sesgo_ingenuo_pp=("error_ingenuo", "mean")).round(2)
    destino = comun.ruta("tablas", "validacion_estimador.csv")
    res.to_csv(destino)
    print(res.to_string())
    print("→", destino)


if __name__ == "__main__":
    main()
