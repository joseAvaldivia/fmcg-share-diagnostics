"""Rutas, configuración y utilidades compartidas."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data"
SALIDAS = RAIZ / "salidas"


@lru_cache(maxsize=1)
def config() -> dict:
    return yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))


def meses() -> pd.PeriodIndex:
    c = config()["meses"]
    return pd.period_range(c["inicio"], c["fin"], freq="M")


def ruta(*partes: str) -> Path:
    p = SALIDAS.joinpath(*partes)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def guardar(df: pd.DataFrame, nombre: str, carpeta: Path = DATOS) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    p = carpeta / f"{nombre}.parquet"
    df.to_parquet(p, index=False)
    return p


def leer(nombre: str, carpeta: Path = DATOS) -> pd.DataFrame:
    return pd.read_parquet(carpeta / f"{nombre}.parquet")


def guardar_json(obj, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json), encoding="utf-8")


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, pd.Period):
        return str(o)
    raise TypeError(type(o))


def mes_idx(mes: str | pd.Period) -> int:
    """Posición del mes dentro del horizonte (0 = primer mes)."""
    return int((pd.Period(mes, "M") - meses()[0]).n)
