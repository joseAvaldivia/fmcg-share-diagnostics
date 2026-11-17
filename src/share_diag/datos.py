"""Mundo sintético: panel de auditoría minorista, macro, lista de precios, venta interna por
cliente, auditoría de ejecución y población por hexágono.

Todo sale de config.yaml y de una semilla. Los mecanismos que los análisis deben descubrir están
escritos de forma explícita (sección `verdad` del config):
- share con utilidad logit en precio relativo y distribución;
- rondas de ajuste de precio que el rival sigue o no;
- clientes con el "trío" que pasan volumen al retornable cuando sube solo el no retornable (OW);
- hexágonos donde el rival domina y hexágonos más sensibles al precio;
- disrupciones de abastecimiento y un feriado que golpea solo al OW (placebo).
Las columnas `verdad_*` nunca se usan en los análisis: solo en las pruebas.
"""
from __future__ import annotations

import math

import h3
import numpy as np
import pandas as pd

from .comun import config, guardar, mes_idx, meses

ESTACIONALIDAD = 0.12


def _estacion(per: pd.PeriodIndex) -> np.ndarray:
    return 1 + ESTACIONALIDAD * np.cos(2 * np.pi * (per.month.to_numpy() - 1) / 12)


def indices_lista(cfg: dict) -> pd.DataFrame:
    """Índice de precio de lista (base 1) de la Compañía A y de cada rival, por mes."""
    m = meses()
    T = len(m)
    idx = {k: np.ones(T) for k in ["ow", "retornable"] + cfg["fabricantes"][1:]}
    for r in cfg["rondas"]:
        t0 = mes_idx(r["mes"])
        for lista in ("ow", "retornable"):
            idx[lista][t0:] *= 1 + r[lista]
        for rival in cfg["fabricantes"][1:3]:
            seg = r.get(rival)
            if seg:
                rezago, alza = seg
                idx[rival][min(t0 + rezago, T):] *= 1 + alza
    out = pd.DataFrame(idx)
    out.insert(0, "mes", m.astype(str))
    return out


def macro(cfg: dict) -> pd.DataFrame:
    m = meses()
    c = cfg["macro"]
    salto = mes_idx(c["tipo_cambio_salto"])
    t = np.arange(len(m))
    infl = np.where(t < salto, c["inflacion_mensual_antes"], c["inflacion_mensual_despues"])
    ipc = 100 * np.cumprod(1 + infl)
    tc = np.where(t < salto, 100.0, 100 + (c["tipo_cambio_final"] - 100) * (t - salto + 1) / (len(m) - salto))
    return pd.DataFrame({"mes": m.astype(str), "ipc": ipc.round(2), "tipo_cambio": tc.round(2),
                         "costo": (0.5 * ipc + 0.5 * tc).round(2)})


def panel(cfg: dict, rng: np.random.Generator, lista: pd.DataFrame, mac: pd.DataFrame) -> pd.DataFrame:
    """Panel mercado × clase × fabricante × mes (litros, valor, precio por litro, distribución)."""
    v = cfg["verdad"]
    m = meses()
    est = _estacion(m)
    T = len(m)
    foco, control = cfg["mercados"]["foco"], cfg["mercados"]["control"]
    alfa = {  # preferencia de base por mercado y fabricante
        foco: {"Compañía A": 0.0, "Rival B": 0.25, "Rival C": -0.8, "Otros": -0.9},
        control: {"Compañía A": 0.35, "Rival B": -0.6, "Rival C": -0.5, "Otros": -0.9},
    }
    ajuste_clase = {"Multiserve retornable": {"Compañía A": 0.6, "Rival B": -0.3}}
    precio_rel = {"Compañía A": 1.0, "Rival B": 0.82, "Rival C": 1.0, "Otros": 0.75}
    dist_base = {foco: {"Compañía A": 0.88, "Rival B": 0.75, "Rival C": 0.60, "Otros": 0.50},
                 control: {"Compañía A": 0.90, "Rival B": 0.45, "Rival C": 0.60, "Otros": 0.50}}
    escala = {foco: 1.0e6, control: 0.8e6}
    ipc = mac["ipc"].to_numpy() / 100
    filas = []
    for mercado in (foco, control):
        disr = np.isin(m.astype(str), cfg["disrupciones"].get(mercado, []))
        for clase, c in cfg["clases"].items():
            litros_cat = (escala[mercado] * c["peso"] * est * (1 + 0.002 * np.arange(T))
                          * np.where(disr, 0.85, 1.0) * rng.lognormal(0, 0.02, T))
            precios, util, dist = {}, {}, {}
            for f in cfg["fabricantes"]:
                if f == "Compañía A":
                    ind = lista[c["lista"]].to_numpy()
                elif f in lista:
                    ind = lista[f].to_numpy()
                else:
                    ind = ipc ** 0.5
                precios[f] = c["precio_litro"] * precio_rel[f] * ind
                if f != "Compañía A":
                    # Promociones y descuentos propios del rival en cada mercado: AR(1) sobre el precio
                    ar = np.zeros(T)
                    choques = rng.normal(0, 0.02, T)
                    for k in range(1, T):
                        ar[k] = 0.6 * ar[k - 1] + choques[k]
                    precios[f] = precios[f] * np.exp(ar)
                d = np.full(T, dist_base[mercado][f])
                if f == "Compañía A":
                    d = d - 0.18 * disr
                dist[f] = np.clip(d + rng.normal(0, 0.01, T), 0.05, 0.99)
                a = alfa[mercado][f] + ajuste_clase.get(clase, {}).get(f, 0.0)
                util[f] = (a + v["beta_precio"] * np.log(precios[f] / c["precio_litro"])
                           + v["beta_distribucion"] * dist[f] + rng.normal(0, 0.03, T))
            U = np.vstack([util[f] for f in cfg["fabricantes"]])
            sh = np.exp(U - U.max(axis=0)) / np.exp(U - U.max(axis=0)).sum(axis=0)
            for i, f in enumerate(cfg["fabricantes"]):
                litros = litros_cat * sh[i]
                filas.append(pd.DataFrame({
                    "mercado": mercado, "clase": clase, "fabricante": f, "mes": m.astype(str),
                    "litros": litros.round(0), "valor": (litros * precios[f]).round(0),
                    "precio_litro": precios[f].round(3), "distribucion": dist[f].round(3),
                    "disrupcion": disr,
                }))
    return pd.concat(filas, ignore_index=True)


def ciudad(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Hexágonos H3 (res 7) de la ciudad foco: distrito, población y rasgos ocultos."""
    c = cfg["clientes"]
    centro = h3.latlng_to_cell(*c["centro"], 7)
    lat0, lng0 = c["centro"]
    filas = []
    for celda in h3.grid_disk(centro, c["radio_anillos"]):
        lat, lng = h3.cell_to_latlng(celda)
        ang = math.atan2(lat - lat0, (lng - lng0) * math.cos(math.radians(lat0)))
        dist = h3.grid_distance(centro, celda)
        filas.append({"h3": celda, "lat": lat, "lng": lng, "anillo": dist,
                      "distrito": f"Distrito {int(((ang + math.pi) / (2 * math.pi)) * c['distritos']) % c['distritos'] + 1}"})
    hx = pd.DataFrame(filas)
    hx["poblacion"] = (22000 * np.exp(-hx["anillo"] / 3.0) * rng.lognormal(0, 0.3, len(hx))).round(0)
    # Rasgos ocultos: dónde domina el rival y dónde la gente es más sensible al precio
    hx["verdad_dominio_rival"] = (hx["distrito"].isin(["Distrito 2"]) & (hx["anillo"] >= 3)) | (rng.random(len(hx)) < 0.06)
    hx["verdad_sensible_precio"] = (~hx["verdad_dominio_rival"]) & (
        (hx["distrito"].isin(["Distrito 5"]) & (hx["anillo"] >= 2)) | (rng.random(len(hx)) < 0.08))
    return hx


def clientes_y_venta(cfg: dict, rng: np.random.Generator, lista: pd.DataFrame, hx: pd.DataFrame):
    v, c, trio = cfg["verdad"], cfg["clientes"], cfg["trio"]
    m = meses()
    T, N = len(m), c["n"]
    skus = list(cfg["skus_cliente"])
    S = len(skus)

    hex_i = rng.choice(len(hx), size=N, p=(hx["poblacion"] / hx["poblacion"].sum()).to_numpy())
    rojo = hx["verdad_dominio_rival"].to_numpy()[hex_i]
    sensible = hx["verdad_sensible_precio"].to_numpy()[hex_i]
    r = np.clip(rng.beta(2, 2, N) - 0.15 * rojo, 0.01, 0.99)       # afinidad por retornable
    base = rng.lognormal(np.log(120), 0.6, N) * np.where(rojo, 0.6, 1.0)  # litros/mes

    # Probabilidad de manejar cada SKU y peso dentro del cliente
    p_carry = np.column_stack([np.full(N, 0.8), 0.6 * (1 - 0.5 * r), 0.3 + 0.5 * r, 0.2 + 0.6 * r,
                               0.35 + 0.2 * r, 0.15 + 0.5 * r])
    peso = np.column_stack([np.full(N, 1.0), np.full(N, 0.9), 0.8 + r, 0.7 + r, np.full(N, 0.4), 0.2 + 0.5 * r])
    maneja = rng.random((N, S)) < p_carry
    maneja[maneja.sum(axis=1) == 0, 0] = True
    n_trio = maneja[:, [skus.index(s) for s in trio["skus"]]].sum(axis=1)
    theta = np.where(n_trio >= trio["minimo_habitual"], v["traspaso_trio"], v["traspaso_resto"])

    I_ow = lista["ow"].to_numpy()
    I_ret = lista["retornable"].to_numpy()
    brecha = np.log(I_ow / I_ret)
    e_ow = np.where(sensible, -2.8, v["elasticidad_ow_cliente"])
    es_ow = np.array([cfg["skus_cliente"][s]["lista"] == "ow" for s in skus])

    t = np.arange(T)
    tendencia = np.exp(np.outer((r - 0.5) * 0.004, t))                       # N×T
    estac = _estacion(m)
    disr = np.isin(m.astype(str), cfg["disrupciones"].get(cfg["mercados"]["foco"], []))
    placebo = (m.astype(str) == cfg["mes_placebo"])

    esperado = np.empty((N, T, S))
    for s in range(S):
        if es_ow[s]:
            mult = I_ow[None, :] ** e_ow[:, None] * np.where(placebo, 0.85, 1.0)[None, :]
        else:
            mult = (I_ret[None, :] ** v["elasticidad_retornable_cliente"]
                    * (1 + theta[:, None] * v["escala_traspaso"] * brecha[None, :]))
        esperado[:, :, s] = (base[:, None] * peso[:, s, None] * maneja[:, s, None] * estac[None, :]
                             * tendencia * mult * np.where(disr, 0.7, 1.0)[None, :])
    esperado /= peso.sum(axis=1)[:, None, None] / 2.5   # normaliza el total por cliente

    compra = rng.random((N, T, S)) < 0.85
    esporadico = (~maneja[:, None, :]) & (rng.random((N, T, S)) < 0.04)
    ruido = rng.lognormal(-0.35 ** 2 / 2, 0.35, (N, T, S))
    obs = np.where(compra & maneja[:, None, :], esperado / 0.85 * ruido, 0.0)
    obs += np.where(esporadico, base[:, None, None] * 0.05 * ruido, 0.0)

    ids = np.array([f"C{i:05d}" for i in range(N)])
    venta = pd.DataFrame({
        "cliente": np.repeat(ids, T * S),
        "mes": np.tile(np.repeat(m.astype(str), S), N),
        "sku": np.tile(skus, N * T),
        "litros": obs.reshape(-1).round(1),
        "verdad_litros_esperados": esperado.reshape(-1).round(2),
    })
    venta = venta[(venta["litros"] > 0) | (venta["verdad_litros_esperados"] > 0)].reset_index(drop=True)

    cli = pd.DataFrame({
        "cliente": ids, "h3": hx["h3"].to_numpy()[hex_i], "distrito": hx["distrito"].to_numpy()[hex_i],
        "lat": hx["lat"].to_numpy()[hex_i] + rng.normal(0, 0.004, N),
        "lng": hx["lng"].to_numpy()[hex_i] + rng.normal(0, 0.004, N),
        "verdad_afinidad_retornable": r.round(3), "verdad_skus_trio": n_trio,
    })

    # Auditoría de ejecución: 35 % de los clientes visitados
    vis = rng.random(N) < 0.35
    z = 0.3 - 1.5 * rojo + 0.8 * (r - 0.5) + rng.normal(0, 0.5, N)
    ejec = pd.DataFrame({
        "cliente": ids[vis],
        "share_frentes_propio": (1 / (1 + np.exp(-z)))[vis].round(3),
        "presencia_rival_b": (rng.random(N) < 0.45 + 0.4 * rojo)[vis],
        "presencia_rival_c": (rng.random(N) < 0.35)[vis],
        "tiene_enfriador": (rng.random(N) < 0.5 - 0.2 * rojo)[vis],
    })
    return cli, venta, ejec


def generar() -> dict[str, pd.DataFrame]:
    cfg = config()
    rng = np.random.default_rng(cfg["semilla"])
    lista = indices_lista(cfg)
    mac = macro(cfg)
    pan = panel(cfg, rng, lista, mac)
    hx = ciudad(cfg, rng)
    cli, venta, ejec = clientes_y_venta(cfg, rng, lista, hx)
    return {"lista_precios": lista, "macro": mac, "panel": pan, "hexagonos": hx,
            "clientes": cli, "venta_clientes": venta, "ejecucion": ejec}


def main() -> None:
    for nombre, df in generar().items():
        guardar(df, nombre)
        print(f"{nombre:16s} {len(df):>9,d} filas")


if __name__ == "__main__":
    main()
