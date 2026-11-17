"""M3 · M5 · M9 — Precio: episodios de alza frente a la macro y la competencia, elasticidad del
share al precio relativo, alerta de alzas con la venta interna y escalera del trío vs. el ancla.

Pregunta de negocio: ¿cuánto share se perdió por subir precio y cuánto de eso era inevitable
porque subieron los costos? Hipótesis a probar (no a asumir): subir cuesta share cuando abre
brecha con el rival; si el alza responde al costo y el mercado la sigue, la pérdida es chica.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .comun import config, guardar_json, leer, mes_idx, meses, ruta


def _share_propio(panel: pd.DataFrame, mercado: str) -> pd.Series:
    p = panel[panel["mercado"] == mercado]
    tot = p.groupby("mes")["litros"].sum()
    return p[p["fabricante"] == config()["propio"]].groupby("mes")["litros"].sum() / tot


def _laspeyres(panel: pd.DataFrame, mercado: str, fabricantes: list[str], t_base: int, t: int) -> float:
    """Índice de precio por litro con pesos fijos (litros de la ventana base por clase)."""
    m = meses().astype(str)
    p = panel[(panel["mercado"] == mercado) & panel["fabricante"].isin(fabricantes)]

    def ventana(t_fin: int) -> pd.DataFrame:
        # Promedio de 2 meses para no confundir una promoción puntual del rival con un alza
        sel = m[max(t_fin - 1, 0):min(t_fin, len(m) - 1) + 1]
        return p[p["mes"].isin(sel)].groupby(["clase", "fabricante"]).agg(
            precio_litro=("precio_litro", "mean"), litros=("litros", "sum"))

    base, act = ventana(t_base), ventana(t)
    w = base["litros"]
    return float((act["precio_litro"] * w).sum() / (base["precio_litro"] * w).sum() - 1)


def episodios(panel: pd.DataFrame, mac: pd.DataFrame) -> pd.DataFrame:
    cfg = config()
    foco, control = cfg["mercados"]["foco"], cfg["mercados"]["control"]
    propio, rivales = cfg["propio"], [f for f in cfg["fabricantes"] if f != cfg["propio"] and f != "Otros"]
    s_foco, s_ctrl = _share_propio(panel, foco), _share_propio(panel, control)
    costo = mac.set_index("mes")["costo"]
    m = meses().astype(str)
    filas, previo = [], 0
    for r in cfg["rondas"]:
        t0 = mes_idx(r["mes"])
        pre, post = m[max(t0 - 3, 0):t0], m[t0:t0 + 3]
        alza = _laspeyres(panel, foco, [propio], t0 - 1, t0 + 1)
        riv1 = _laspeyres(panel, foco, rivales, t0 - 1, t0 + 1)
        riv3 = _laspeyres(panel, foco, rivales, t0 - 1, t0 + 4)
        ancla3 = _laspeyres(panel, foco, [cfg["rival_ancla"]], t0 - 1, t0 + 4)
        seguimiento = riv3 / alza if alza > 0 else np.nan
        d_foco = (s_foco[post].mean() - s_foco[pre].mean()) * 100
        d_ctrl = (s_ctrl[post].mean() - s_ctrl[pre].mean()) * 100
        presion = costo.iloc[t0 - 1] / costo.iloc[previo] - 1
        filas.append({
            "ronda": r["mes"], "alza_ow_lista": r["ow"], "alza_retornable_lista": r["retornable"],
            "alza_propia_medida": alza, "alza_rivales_t1": riv1, "alza_rivales_t3": riv3,
            "alza_ancla_t3": ancla3, "seguimiento": seguimiento,
            "veredicto_seguimiento": "seguida" if seguimiento >= 0.6 else ("parcial" if seguimiento >= 0.25 else "no seguida"),
            "brecha_abierta_pp": (alza - riv3) * 100,
            "dshare_pp_foco": d_foco, "dshare_pp_control": d_ctrl,
            "presion_costo_desde_ronda_previa": presion,
            "lectura": ("costo + seguida" if presion >= 0.03 and seguimiento >= 0.6 else
                        "costo, sin seguimiento" if presion >= 0.03 else "sin presión de costo"),
        })
        previo = t0
    return pd.DataFrame(filas).round(4)


def pendiente_brecha(ep: pd.DataFrame) -> dict:
    """Cuánto share se pierde por cada punto de brecha que abre una ronda (MCO simple entre rondas)."""
    x, y = ep["brecha_abierta_pp"].to_numpy(), ep["dshare_pp_foco"].to_numpy()
    b, a = np.polyfit(x, y, 1)
    return {"pp_share_por_pp_brecha": round(float(b), 3), "intercepto": round(float(a), 3),
            "r2": round(float(np.corrcoef(x, y)[0, 1] ** 2), 3), "n_rondas": int(len(x))}


def elasticidad(panel: pd.DataFrame, mac: pd.DataFrame) -> pd.DataFrame:
    """ln(share/(1−share)) propio ~ ln(P propio/P rivales) + ln(P propio real) + disrupción,
    con efectos fijos de mercado×clase y de mes, errores agrupados por mercado×clase."""
    propio = config()["propio"]
    g = panel.assign(es_propio=panel["fabricante"] == propio).groupby(
        ["mercado", "clase", "mes", "es_propio"]).agg(litros=("litros", "sum"), valor=("valor", "sum"),
                                                     disrupcion=("disrupcion", "max")).reset_index()
    a = g[g["es_propio"]].set_index(["mercado", "clase", "mes"])
    b = g[~g["es_propio"]].set_index(["mercado", "clase", "mes"])
    d = pd.DataFrame({"l_a": a["litros"], "l_b": b["litros"], "p_a": a["valor"] / a["litros"],
                      "p_b": b["valor"] / b["litros"], "disrupcion": a["disrupcion"].astype(int)}).reset_index()
    d = d.merge(mac[["mes", "ipc"]], on="mes")
    s = d["l_a"] / (d["l_a"] + d["l_b"])
    d["ln_odds"] = np.log(s / (1 - s))
    d["ln_precio_relativo"] = np.log(d["p_a"] / d["p_b"])
    d["ln_precio_real"] = np.log(d["p_a"] / d["ipc"] * 100)
    d["entidad"] = d["mercado"] + " · " + d["clase"]
    mod = smf.ols("ln_odds ~ ln_precio_relativo + ln_precio_real + disrupcion + C(entidad) + C(mes)", data=d)
    res = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["entidad"])[0]})
    vars_ = ["ln_precio_relativo", "ln_precio_real", "disrupcion"]
    ci = res.conf_int(0.10).loc[vars_]
    return pd.DataFrame({"coef": res.params[vars_], "ee": res.bse[vars_], "p": res.pvalues[vars_],
                         "ic90_inf": ci[0], "ic90_sup": ci[1], "n": int(res.nobs)}).round(4)


def matriz(venta: pd.DataFrame, valor: str = "litros") -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Cubo cliente × mes × SKU (0 si no hubo compra)."""
    cl = sorted(venta["cliente"].unique())
    m = list(meses().astype(str))
    sk = list(config()["skus_cliente"])
    ic = pd.Index(cl).get_indexer(venta["cliente"])
    im = pd.Index(m).get_indexer(venta["mes"])
    isk = pd.Index(sk).get_indexer(venta["sku"])
    cubo = np.zeros((len(cl), len(m), len(sk)))
    np.add.at(cubo, (ic, im, isk), venta[valor].to_numpy())
    return cubo, cl, m, sk


def alerta_alzas(venta: pd.DataFrame) -> pd.DataFrame:
    """Con la venta interna: SKU que subieron vs. SKU que no, en cada ronda con grupo de control.
    Δ = ln(T/C) post − pre; si hay año anterior, se descuenta el mismo Δ un año antes."""
    cfg = config()
    cubo, _, m, sk = matriz(venta)
    es_ow = np.array([cfg["skus_cliente"][s]["lista"] == "ow" for s in sk])
    tot = cubo.sum(axis=0)  # mes × SKU
    filas = []
    for r in cfg["rondas"]:
        sube = es_ow * (r["ow"] > 0) | (~es_ow) * (r["retornable"] > 0)
        if sube.all() or not sube.any():
            continue
        t0 = mes_idx(r["mes"])

        def delta(t):
            lr = np.log(tot[:, sube].sum(axis=1) / tot[:, ~sube].sum(axis=1))
            return lr[t:t + 3].mean() - lr[t - 3:t].mean()

        d = delta(t0)
        d_prev = delta(t0 - 12) if t0 - 12 >= 3 else np.nan
        filas.append({"ronda": r["mes"], "skus_que_subieron": ", ".join(np.array(sk)[sube]),
                      "delta_simple": d, "delta_anio_previo": d_prev,
                      "efecto": d - d_prev if np.isfinite(d_prev) else d,
                      "ajuste_estacional": bool(np.isfinite(d_prev))})
    return pd.DataFrame(filas).round(4)


def escalera_trio(lista: pd.DataFrame) -> pd.DataFrame:
    """Precio sugerido vigente de los SKU del trío vs. el SKU ancla del rival (desembolso y litro)."""
    cfg = config()
    t = cfg["trio"]
    ult = lista.iloc[-1]
    ancla_p = t["ancla_precio"] * ult[cfg["rival_ancla"]]
    filas = [{"sku": t["ancla"], "rol": "ancla (rival)", "precio": ancla_p, "litros": 3.0}]
    for s in t["skus"]:
        c = cfg["skus_cliente"][s]
        filas.append({"sku": s, "rol": "trío", "precio": c["precio"] * ult[c["lista"]], "litros": c["litros"]})
    d = pd.DataFrame(filas)
    d["precio_litro"] = d["precio"] / d["litros"]
    d["indice_desembolso"] = d["precio"] / ancla_p * 100
    d["indice_litro"] = d["precio_litro"] / (ancla_p / 3.0) * 100
    return d.round(2)


def main() -> None:
    panel, mac, lista = leer("panel"), leer("macro"), leer("lista_precios")
    ep = episodios(panel, mac)
    ep.to_csv(ruta("tablas", "m9_episodios.csv"), index=False)
    el = elasticidad(panel, mac)
    el.to_csv(ruta("tablas", "m5_elasticidad.csv"))
    al = alerta_alzas(leer("venta_clientes"))
    al.to_csv(ruta("tablas", "alerta_alzas.csv"), index=False)
    esc = escalera_trio(lista)
    esc.to_csv(ruta("tablas", "trio_escalera.csv"), index=False)
    pend = pendiente_brecha(ep)
    guardar_json({"elasticidad": el.reset_index(names="variable").to_dict(orient="records"),
                  "brecha": pend}, ruta("precio.json"))
    print(ep[["ronda", "alza_propia_medida", "seguimiento", "veredicto_seguimiento", "brecha_abierta_pp",
              "dshare_pp_foco", "dshare_pp_control", "presion_costo_desde_ronda_previa", "lectura"]].to_string(index=False))
    print(pend)
    print(el.to_string())
    print(al.to_string(index=False))


if __name__ == "__main__":
    main()
