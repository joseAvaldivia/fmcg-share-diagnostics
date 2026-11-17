"""M1 · Share de mercado: evolución mensual, comparación de periodos y quién ganó lo perdido."""
from __future__ import annotations

import pandas as pd

from .comun import config, guardar, leer, ruta


def share_mensual(panel: pd.DataFrame, por: list[str] | None = None) -> pd.DataFrame:
    por = por or ["mercado", "clase"]
    g = panel.groupby(por + ["mes", "fabricante"], as_index=False)["litros"].sum()
    g["share"] = g["litros"] / g.groupby(por + ["mes"])["litros"].transform("sum")
    return g


def comparar_periodos(panel: pd.DataFrame, meses_ventana: int = 12) -> pd.DataFrame:
    """Share de los primeros vs. los últimos `meses_ventana` meses (mismos meses del año)."""
    todos = sorted(panel["mes"].unique())
    ini, fin = set(todos[:meses_ventana]), set(todos[-meses_ventana:])
    filas = []
    for nombre, filtro in (("mercado total", None), ("por clase", "clase")):
        por = ["mercado"] + ([filtro] if filtro else [])
        for periodo, sel in (("inicio", ini), ("final", fin)):
            g = panel[panel["mes"].isin(sel)].groupby(por + ["fabricante"])["litros"].sum()
            sh = g / g.groupby(level=por).transform("sum")
            filas.append(sh.rename(periodo).reset_index().assign(nivel=nombre))
    a = pd.concat(filas[0::2]).merge(pd.concat(filas[1::2]), on=["nivel", "mercado", "clase", "fabricante"], how="outer") \
        if "clase" in pd.concat(filas).columns else None
    a["delta_pp"] = (a["final"] - a["inicio"]) * 100
    return a[["nivel", "mercado", "clase", "fabricante", "inicio", "final", "delta_pp"]].round(4)


def main() -> None:
    panel = leer("panel")
    mensual = share_mensual(panel, ["mercado"])
    guardar(mensual, "share_mensual")
    mensual.to_csv(ruta("tablas", "m1_share_mensual.csv"), index=False)
    comp = comparar_periodos(panel)
    comp.to_csv(ruta("tablas", "m1_share_periodos.csv"), index=False)
    foco = config()["mercados"]["foco"]
    tot = comp[(comp["nivel"] == "mercado total") & (comp["mercado"] == foco)]
    print(tot[["fabricante", "inicio", "final", "delta_pp"]].to_string(index=False))


if __name__ == "__main__":
    main()
