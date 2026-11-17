"""Táctica de portafolio "trío": tres SKU propios (retornable 2,5 L, sabores 2,5 L y vidrio 1,5 L)
contra el SKU ancla del rival (multiserve 3 L).

Mecanismo que se quiere probar: cuando sube el no retornable (OW) y el retornable NO sube, el
consumidor de un cliente con el trío se queda en nuestro retornable en vez de irse al rival.

Prueba (experimento natural con las rondas de ajuste de precio):
1. Estado del cliente medido 9 a 4 meses antes de la ronda (sin cruzarse con la ventana base):
   SKU del trío que compra de forma habitual (≥ 1/3 de sus meses con compra).
2. Resultado: crecimiento del volumen multiserve, meses 0 a +2 vs. −3 a −1.
3. Con trío (≥ 2 SKU) vs. sin trío (0–1 SKU) DENTRO de celdas distrito × decil de volumen ×
   tercil del peso del retornable: los clientes con trío viven más del retornable y cualquier
   golpe al OW los separa aunque el trío no haga nada.
4. Intervalo de confianza por bootstrap agrupado por hexágono H3.
5. Placebo en meses sin cambio de precio. Cada ronda se clasifica con la lista de precios:
   "retornable protegido" (solo subió el OW) o "retornable subió".
6. Verdad: el mismo estimador aplicado a los litros esperados sin ruido del generador.
7. Piloto: efecto mínimo detectable con asignación aleatoria de clientes a los que les falta un SKU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .comun import config, guardar_json, leer, mes_idx, meses, ruta
from .precios import matriz


class Panel:
    """Cubos cliente × mes × SKU (observado y esperado) y atributos por cliente."""

    def __init__(self, venta: pd.DataFrame, clientes: pd.DataFrame):
        cfg = config()
        self.obs, self.clientes, self.meses, self.skus = matriz(venta, "litros")
        self.esp, *_ = matriz(venta, "verdad_litros_esperados")
        cl = clientes.set_index("cliente").loc[self.clientes]
        self.hex = cl["h3"].to_numpy()
        self.distrito = cl["distrito"].to_numpy()
        self.trio = [self.skus.index(s) for s in cfg["trio"]["skus"]]
        self.ret = [i for i, s in enumerate(self.skus) if cfg["skus_cliente"][s]["lista"] == "retornable"]
        self.ow = [i for i, s in enumerate(self.skus) if cfg["skus_cliente"][s]["lista"] == "ow"]

    def skus_habituales(self, t0: int, t1: int) -> np.ndarray:
        """Cantidad de SKU del trío comprados en ≥ umbral de los meses con compra (t0..t1)."""
        umbral = config()["trio"]["umbral_habitual"]
        v = self.obs[:, t0:t1 + 1, :]
        meses_compra = (v.sum(axis=2) > 0).sum(axis=1)
        frec = (v[:, :, self.trio] > 0).sum(axis=1) / np.maximum(meses_compra, 1)[:, None]
        return (frec >= umbral).sum(axis=1)


def _celdas(pnl: Panel, t: int) -> np.ndarray:
    """Distrito × decil de volumen × tercil del peso del retornable, medidos en la ventana de estado.

    El decil NO se mide en la ventana "pre": condicionar en el denominador del crecimiento produce
    regresión a la media, más fuerte en los clientes con menos SKU (más ruidosos)."""
    e0, e1 = config()["experimento"]["meses_estado"]
    v = pnl.obs[:, t + e0:t + e1 + 1, :]
    vol = v.sum(axis=(1, 2))
    ret = v[:, :, pnl.ret].sum(axis=(1, 2)) / np.maximum(vol, 1e-9)
    decil = pd.qcut(pd.Series(vol).rank(method="first"), 10, labels=False).to_numpy()
    tercil = pd.qcut(pd.Series(ret).rank(method="first"), 3, labels=False).to_numpy()
    return pd.Series(pnl.distrito).astype(str).to_numpy() + "|" + decil.astype(str) + "|" + tercil.astype(str)


def _dif_celdas(g: np.ndarray, trat: np.ndarray, celda: np.ndarray, w_idx: np.ndarray | None = None) -> float:
    """Diferencia de medias tratados − control dentro de celda, ponderada por n_T·n_C/(n_T+n_C)."""
    d = pd.DataFrame({"g": g, "t": trat, "c": celda})
    if w_idx is not None:
        d = d.iloc[w_idx]
    a = d.groupby(["c", "t"])["g"].agg(["mean", "size"]).unstack("t")
    a = a.dropna()
    if a.empty:
        return np.nan
    nT, nC = a[("size", True)], a[("size", False)]
    w = nT * nC / (nT + nC)
    return float(((a[("mean", True)] - a[("mean", False)]) * w).sum() / w.sum())


def evento(pnl: Panel, t: int, rng: np.random.Generator, B: int) -> dict:
    cfg = config()
    ex = cfg["experimento"]
    e0, e1 = ex["meses_estado"]
    p0, p1 = ex["meses_pre"]
    q0, q1 = ex["meses_post"]
    k = pnl.skus_habituales(t + e0, t + e1)
    trat = k >= cfg["trio"]["minimo_habitual"]
    ctrl = k <= cfg["trio"]["minimo_habitual"] - 1

    def crec(cubo):
        # Crecimiento logarítmico: el ruido multiplicativo aparece igual en pre y post y se cancela,
        # aunque un grupo sea más ruidoso que el otro (con post/pre − 1 no pasa: desigualdad de Jensen).
        pre = cubo[:, t + p0:t + p1 + 1, :].sum(axis=(1, 2))
        post = cubo[:, t + q0:t + q1 + 1, :].sum(axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.clip(np.log(post / pre), -2.3, 1.4)
        return pre, post, g

    pre, post, g = crec(pnl.obs)
    ok = (pre > 0) & (post > 0) & (trat | ctrl)
    celda = _celdas(pnl, t)
    est = _dif_celdas(g[ok], trat[ok], celda[ok]) * 100

    # Verdad: mismo estimador sobre litros esperados (sin ruido), mismos grupos y celdas
    _, _, g_v = crec(pnl.esp)
    verdad = _dif_celdas(g_v[ok], trat[ok], celda[ok]) * 100

    # Bootstrap agrupado por hexágono
    hexes = pnl.hex[ok]
    unicos = np.unique(hexes)
    pos = {h: np.flatnonzero(hexes == h) for h in unicos}
    boot = []
    for _ in range(B):
        idx = np.concatenate([pos[h] for h in rng.choice(unicos, len(unicos), replace=True)])
        boot.append(_dif_celdas(g[ok], trat[ok], celda[ok], idx) * 100)
    a = (1 - ex["confianza"]) / 2
    lo, hi = np.nanquantile(boot, [a, 1 - a])
    return {"efecto_pp": est, "ic_inf": lo, "ic_sup": hi, "verdad_pp": verdad,
            "n_con_trio": int((ok & trat).sum()), "n_sin_trio": int((ok & ctrl).sum()),
            "diferencia_ingenua_pp": float((np.nanmean(g[ok & trat]) - np.nanmean(g[ok & ctrl])) * 100)}


def experimento(pnl: Panel) -> pd.DataFrame:
    cfg = config()
    ex = cfg["experimento"]
    rng = np.random.default_rng(cfg["semilla"])
    T = len(pnl.meses)
    filas = []
    rondas = {mes_idx(r["mes"]): r for r in cfg["rondas"]}
    for t, r in rondas.items():
        if t + ex["meses_estado"][0] < 0 or t + ex["meses_post"][1] >= T:
            continue
        res = evento(pnl, t, rng, ex["bootstrap"])
        filas.append({"tipo": "ronda", "mes": pnl.meses[t],
                      "clase": "retornable protegido" if r["retornable"] == 0 else "retornable subió", **res})
    # Placebos: meses sin ninguna ronda dentro de la ventana pre/post (t−3 .. t+2)
    for t in range(-ex["meses_estado"][0], T - ex["meses_post"][1]):
        if any(t + ex["meses_pre"][0] <= tr <= t + ex["meses_post"][1] for tr in rondas):
            continue
        res = evento(pnl, t, rng, max(ex["bootstrap"] // 3, 50))
        filas.append({"tipo": "placebo", "mes": pnl.meses[t],
                      "clase": "feriado (solo OW)" if pnl.meses[t] == cfg["mes_placebo"] else "sin cambio", **res})
    return pd.DataFrame(filas).round(3)


def potencia_piloto(pnl: Panel, n_sim: int = 400, potencia: float = 0.8) -> dict:
    """Efecto mínimo detectable: se reparten al azar los clientes a los que les falta un SKU del trío
    entre tratamiento y control, en ventanas históricas sin intervención."""
    cfg = config()
    ex = cfg["experimento"]
    rng = np.random.default_rng(cfg["semilla"] + 1)
    T = len(pnl.meses)
    k = pnl.skus_habituales(T - 9, T - 4)
    elegibles = np.flatnonzero(k == cfg["trio"]["minimo_habitual"] - 1)
    brazo = int(min(ex["piloto_por_brazo"], len(elegibles) // 2))
    difs = []
    tot = pnl.obs.sum(axis=2)
    for _ in range(n_sim):
        t = rng.integers(3, T - 3)
        sel = rng.choice(elegibles, 2 * brazo, replace=False)
        pre, post = tot[sel, t - 3:t].sum(axis=1), tot[sel, t:t + 3].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where((pre > 0) & (post > 0), np.clip(np.log(post / pre), -2.3, 1.4), np.nan)
        difs.append(np.nanmean(g[:brazo]) - np.nanmean(g[brazo:]))
    ee = float(np.std(difs))
    z = stats.norm.ppf(1 - (1 - ex["confianza"]) / 2) + stats.norm.ppf(potencia)
    return {"clientes_elegibles": int(len(elegibles)), "clientes_por_brazo": brazo,
            "error_estandar_pp": round(ee * 100, 2), "efecto_minimo_detectable_pp": round(z * ee * 100, 2),
            "confianza": ex["confianza"], "potencia": potencia}


def main() -> None:
    pnl = Panel(leer("venta_clientes"), leer("clientes"))
    exp = experimento(pnl)
    exp.to_csv(ruta("tablas", "trio_experimento.csv"), index=False)
    pot = potencia_piloto(pnl)
    rondas = exp[exp["tipo"] == "ronda"]
    resumen = {
        "rondas": rondas[["mes", "clase", "efecto_pp", "ic_inf", "ic_sup", "verdad_pp",
                          "diferencia_ingenua_pp"]].to_dict(orient="records"),
        "placebo_medio_pp": float(exp.loc[exp["tipo"] == "placebo", "efecto_pp"].mean()),
        "piloto": pot,
    }
    guardar_json(resumen, ruta("trio.json"))
    print(exp[["tipo", "mes", "clase", "efecto_pp", "ic_inf", "ic_sup", "verdad_pp", "diferencia_ingenua_pp",
               "n_con_trio", "n_sin_trio"]].to_string(index=False))
    print(pot)


if __name__ == "__main__":
    main()
