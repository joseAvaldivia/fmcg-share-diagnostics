"""Orquestador: corre solo los pasos desactualizados y deja todo regenerable.

    python -m share_diag.pipeline            # pasos pendientes
    python -m share_diag.pipeline --todo     # todo
    python -m share_diag.pipeline --solo tactica
    python -m share_diag.pipeline --desde precios
    python -m share_diag.pipeline --lista    # estado de cada paso

Un paso está desactualizado si cambió su script, config.yaml o alguna de sus entradas (tamaño y
fecha), o si falta alguna salida. Las salidas de un paso son entradas del siguiente, así que un
cambio se propaga solo hacia adelante. Las firmas se guardan en salidas/manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .comun import DATOS, RAIZ, SALIDAS

T = SALIDAS / "tablas"
TABLAS_DATOS = ["panel", "macro", "lista_precios", "hexagonos", "clientes", "venta_clientes", "ejecucion"]


@dataclass(frozen=True)
class Paso:
    nombre: str
    modulo: str
    entradas: tuple[Path, ...]
    salidas: tuple[Path, ...]


def _d(*n): return tuple(DATOS / f"{x}.parquet" for x in n)  # noqa: E704


PASOS = [
    Paso("datos", "share_diag.datos", (), _d(*TABLAS_DATOS)),
    Paso("share", "share_diag.share", _d("panel"), (T / "m1_share_mensual.csv", T / "m1_share_periodos.csv")),
    Paso("precios", "share_diag.precios", _d("panel", "macro", "lista_precios", "venta_clientes"),
         (T / "m9_episodios.csv", T / "m5_elasticidad.csv", T / "alerta_alzas.csv", T / "trio_escalera.csv",
          SALIDAS / "precio.json")),
    Paso("tactica", "share_diag.tactica", _d("venta_clientes", "clientes"),
         (T / "trio_experimento.csv", SALIDAS / "trio.json")),
    Paso("semaforo", "share_diag.semaforo", _d("venta_clientes", "clientes", "ejecucion", "hexagonos"),
         (T / "semaforo_hex.csv", T / "semaforo_resumen.csv", SALIDAS / "mapas" / "mapa_semaforo.html")),
    Paso("reporte", "share_diag.reporte",
         (T / "m1_share_mensual.csv", T / "m9_episodios.csv", T / "trio_experimento.csv", T / "semaforo_hex.csv"),
         (SALIDAS / "ppt" / "diagnostico_share.pptx", SALIDAS / "graficos" / "trio.png")),
]


def firma(p: Paso) -> str:
    h = hashlib.sha256()
    src = importlib.util.find_spec(p.modulo).origin
    h.update(Path(src).read_bytes())
    h.update((RAIZ / "config.yaml").read_bytes())
    for e in p.entradas:
        st = e.stat() if e.exists() else None
        h.update(f"{e.name}:{st.st_size if st else -1}:{st.st_mtime_ns if st else -1}".encode())
    return h.hexdigest()[:16]


def _manifest() -> dict:
    f = SALIDAS / "manifest.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def pendiente(p: Paso, man: dict) -> bool:
    return man.get(p.nombre, {}).get("firma") != firma(p) or not all(s.exists() for s in p.salidas)


def correr(seleccion: list[Paso], forzar: bool = False) -> list[str]:
    SALIDAS.mkdir(exist_ok=True)
    man, corridos = _manifest(), []
    for p in seleccion:
        if not forzar and not pendiente(p, man):
            print(f"  = {p.nombre:9s} al día")
            continue
        t = time.time()
        print(f"  ▶ {p.nombre}")
        importlib.import_module(p.modulo).main()
        man[p.nombre] = {"firma": firma(p), "segundos": round(time.time() - t, 1),
                         "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}
        (SALIDAS / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
        corridos.append(p.nombre)
    return corridos


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--todo", action="store_true")
    ap.add_argument("--solo")
    ap.add_argument("--desde")
    ap.add_argument("--lista", action="store_true")
    a = ap.parse_args(argv)
    nombres = [p.nombre for p in PASOS]
    if a.lista:
        man = _manifest()
        for p in PASOS:
            print(f"  {p.nombre:9s} {'pendiente' if pendiente(p, man) else 'al día'}")
        return
    if a.solo:
        correr([PASOS[nombres.index(a.solo)]], forzar=True)
    elif a.desde:
        correr(PASOS[nombres.index(a.desde):], forzar=True)
    else:
        correr(PASOS, forzar=a.todo)


if __name__ == "__main__":
    main()
