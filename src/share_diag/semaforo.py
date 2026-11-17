"""Mapa semáforo por hexágono H3 (resolución 7): dónde estamos bien, dónde perdimos con las alzas
y dónde la competencia siempre dominó.

Métricas por hexágono:
- tendencia: volumen de los últimos 12 meses vs. los 12 anteriores, comparada con la mediana del mercado;
- efecto de las alzas: en las rondas donde subió solo el OW, ln(OW/retornable) post − pre;
- ejecución (clientes visitados): share de frentes propio y presencia del rival ancla;
- penetración: litros por habitante al año.

Clases:
- rojo (domina la competencia): share de frentes bajo el p25 o rival sobre el p75, y penetración bajo el p33;
- amarillo (perdido con las alzas): el volumen cae y el efecto de las alzas es igual o peor que el umbral;
- verde (fortaleza): volumen igual o mejor que la mediana del mercado;
- naranja (cae por otras razones): cae, pero no se concentra en lo que subió;
- gris: pocos clientes.
"""
from __future__ import annotations

import json

import h3
import numpy as np
import pandas as pd

from .comun import config, leer, mes_idx, ruta
from .precios import matriz

COLORES = {"rojo": "#c0392b", "amarillo": "#f1c40f", "verde": "#27ae60", "naranja": "#e67e22", "gris": "#bdc3c7"}


def metricas_hex(venta: pd.DataFrame, clientes: pd.DataFrame, ejec: pd.DataFrame, hx: pd.DataFrame) -> pd.DataFrame:
    cfg = config()
    cubo, ids, m, sk = matriz(venta)
    es_ow = np.array([cfg["skus_cliente"][s]["lista"] == "ow" for s in sk])
    T = len(m)
    cl = clientes.set_index("cliente").loc[ids].reset_index()
    tot = cubo.sum(axis=2)
    cl["vol_ult12"] = tot[:, T - 12:].sum(axis=1)
    cl["vol_ant12"] = tot[:, T - 24:T - 12].sum(axis=1)
    ow, ret = cubo[:, :, es_ow].sum(axis=2), cubo[:, :, ~es_ow].sum(axis=2)
    for k, r in enumerate([r for r in cfg["rondas"] if r["retornable"] == 0]):
        t0 = mes_idx(r["mes"])
        cl[f"ow_pre_{k}"], cl[f"ow_post_{k}"] = ow[:, t0 - 3:t0].sum(1), ow[:, t0:t0 + 3].sum(1)
        cl[f"ret_pre_{k}"], cl[f"ret_post_{k}"] = ret[:, t0 - 3:t0].sum(1), ret[:, t0:t0 + 3].sum(1)
    cl = cl.merge(ejec, on="cliente", how="left")

    g = cl.groupby("h3")
    out = pd.DataFrame({
        "clientes": g.size(), "vol_ult12": g["vol_ult12"].sum(), "vol_ant12": g["vol_ant12"].sum(),
        "share_frentes": g["share_frentes_propio"].mean(), "presencia_rival": g["presencia_rival_b"].mean(),
        "visitados": g["share_frentes_propio"].count(),
    })
    out["tendencia"] = out["vol_ult12"] / out["vol_ant12"] - 1
    efectos = []
    k = 0
    while f"ow_pre_{k}" in cl:
        s = g[[f"ow_pre_{k}", f"ow_post_{k}", f"ret_pre_{k}", f"ret_post_{k}"]].sum()
        efectos.append(np.log(s[f"ow_post_{k}"] / s[f"ret_post_{k}"]) - np.log(s[f"ow_pre_{k}"] / s[f"ret_pre_{k}"]))
        k += 1
    out["efecto_alzas"] = pd.concat(efectos, axis=1).mean(axis=1)
    out = out.join(hx.set_index("h3")[["distrito", "poblacion", "lat", "lng"]])
    out["penetracion_l_hab"] = out["vol_ult12"] / out["poblacion"]
    return out.reset_index()


def clasificar(df: pd.DataFrame) -> pd.DataFrame:
    cfg = config()["semaforo"]
    d = df.copy()
    base = d[d["clientes"] >= cfg["min_clientes"]]
    p25_frentes = base["share_frentes"].quantile(0.25)
    p75_rival = base["presencia_rival"].quantile(0.75)
    p33_pen = base["penetracion_l_hab"].quantile(0.33)
    efecto_ref = base["efecto_alzas"].median()
    rojo = ((d["share_frentes"] < p25_frentes) | (d["presencia_rival"] > p75_rival)) & (d["penetracion_l_hab"] < p33_pen)
    # "Cae" relativo al mercado: si todo el mercado baja por las alzas acumuladas, un hexágono que
    # baja igual que la mediana no es una alerta.
    cae = d["tendencia"] < base["tendencia"].median() - 0.02
    # El efecto de las alzas se mide relativo a la mediana: lo común a todos (estacionalidad) se descuenta
    alzas = (d["efecto_alzas"] - efecto_ref) <= cfg["umbral_efecto_alza"]
    d["clase"] = np.select(
        [d["clientes"] < cfg["min_clientes"], rojo, cae & alzas, ~cae],
        ["gris", "rojo", "amarillo", "verde"], default="naranja")
    d["umbral_frentes_p25"], d["umbral_rival_p75"], d["umbral_penetracion_p33"] = p25_frentes, p75_rival, p33_pen
    return d


def resumen(d: pd.DataFrame) -> pd.DataFrame:
    r = d.groupby("clase").agg(hexagonos=("h3", "size"), clientes=("clientes", "sum"), volumen=("vol_ult12", "sum"),
                               poblacion=("poblacion", "sum"), tendencia=("tendencia", "median"))
    r["peso_volumen"] = r["volumen"] / r["volumen"].sum()
    return r.reindex([c for c in COLORES if c in r.index]).round(3)


PLANTILLA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mapa semáforo por hexágono</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body{margin:0;height:100%;font-family:system-ui,sans-serif}#m{position:absolute;inset:0;background:#eef1f4}
.p{position:absolute;z-index:1000;background:#fff;border-radius:8px;padding:10px 12px;box-shadow:0 2px 10px rgba(0,0,0,.15);font-size:12.5px}
.c{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-2px}</style></head>
<body><div id="m"></div><div class="p" style="top:10px;left:50px;max-width:340px"><b>Mapa semáforo · Metro Norte</b><br>
Datos sintéticos. Clic en las clases para prenderlas o apagarlas.</div><div class="p" id="ley" style="bottom:20px;left:10px"></div>
<script>
const D=__DATOS__, COL=__COLORES__, NOM={rojo:"Domina la competencia",amarillo:"Perdido con las alzas",verde:"Fortaleza",naranja:"Cae por otras razones",gris:"Pocos clientes"};
const m=L.map("m",{attributionControl:false}); m.fitBounds(L.latLngBounds(D.map(d=>[d.lat,d.lng])),{padding:[20,20]});
const capas={}; Object.keys(COL).forEach(c=>capas[c]=L.layerGroup().addTo(m));
const f=x=>x==null||isNaN(x)?"–":x.toLocaleString("es",{maximumFractionDigits:2});
D.forEach(d=>L.polygon(d.borde,{color:"#fff",weight:.8,fillColor:COL[d.clase],fillOpacity:.8}).bindTooltip(
 `<b>${NOM[d.clase]}</b> · ${d.distrito}<br>Clientes: ${d.clientes}<br>Tendencia: ${f(d.tendencia*100)} %<br>`+
 `Efecto alzas: ${f(d.efecto_alzas)}<br>Share de frentes: ${f(d.share_frentes*100)} %<br>Penetración: ${f(d.penetracion_l_hab)} L/hab`).addTo(capas[d.clase]));
document.getElementById("ley").innerHTML=Object.keys(COL).map(c=>`<div style="cursor:pointer" onclick="m.hasLayer(capas['${c}'])?m.removeLayer(capas['${c}']):m.addLayer(capas['${c}'])"><span class="c" style="background:${COL[c]}"></span>${NOM[c]} (${D.filter(d=>d.clase===c).length})</div>`).join("");
</script></body></html>"""


def mapa_html(d: pd.DataFrame) -> str:
    datos = []
    for r in d.itertuples(index=False):
        datos.append({"h3": r.h3, "clase": r.clase, "distrito": r.distrito, "clientes": int(r.clientes),
                      "tendencia": None if pd.isna(r.tendencia) else round(float(r.tendencia), 4),
                      "efecto_alzas": None if pd.isna(r.efecto_alzas) else round(float(r.efecto_alzas), 4),
                      "share_frentes": None if pd.isna(r.share_frentes) else round(float(r.share_frentes), 3),
                      "penetracion_l_hab": None if pd.isna(r.penetracion_l_hab) else round(float(r.penetracion_l_hab), 3),
                      "lat": round(r.lat, 5), "lng": round(r.lng, 5),
                      "borde": [[round(a, 5), round(b, 5)] for a, b in h3.cell_to_boundary(r.h3)]})
    return PLANTILLA.replace("__DATOS__", json.dumps(datos, ensure_ascii=False)).replace("__COLORES__", json.dumps(COLORES))


def main() -> None:
    hx = leer("hexagonos")
    d = clasificar(metricas_hex(leer("venta_clientes"), leer("clientes"), leer("ejecucion"), hx))
    d.to_csv(ruta("tablas", "semaforo_hex.csv"), index=False)
    res = resumen(d)
    res.to_csv(ruta("tablas", "semaforo_resumen.csv"))
    ruta("mapas", "mapa_semaforo.html").write_text(mapa_html(d), encoding="utf-8")
    print(res.to_string())


if __name__ == "__main__":
    main()
