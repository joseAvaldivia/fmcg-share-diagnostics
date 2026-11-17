"""Deck ejecutivo en PowerPoint con gráficos NATIVOS (editables), más figuras PNG para el README.

Estilo: un mensaje por lámina, pocos puntos, la cifra sobre cada punto, sin cuadrícula.
"""
from __future__ import annotations

import json

import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.chart.data import CategoryChartData  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from .comun import config, leer, ruta  # noqa: E402
from .semaforo import COLORES  # noqa: E402

AZUL, GRIS, ROJO, VERDE = "1F5AA6", "9AA5B1", "C0392B", "27AE60"
TAB = ruta("tablas")


def _rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


def _cargar() -> dict:
    return {
        "share": pd.read_csv(TAB / "m1_share_mensual.csv"),
        "periodos": pd.read_csv(TAB / "m1_share_periodos.csv"),
        "episodios": pd.read_csv(TAB / "m9_episodios.csv"),
        "elasticidad": pd.read_csv(TAB / "m5_elasticidad.csv", index_col=0),
        "escalera": pd.read_csv(TAB / "trio_escalera.csv"),
        "experimento": pd.read_csv(TAB / "trio_experimento.csv"),
        "semaforo": pd.read_csv(TAB / "semaforo_resumen.csv", index_col=0),
        "semaforo_hex": pd.read_csv(TAB / "semaforo_hex.csv"),
        "trio": json.loads(ruta("trio.json").read_text(encoding="utf-8")),
        "precio": json.loads(ruta("precio.json").read_text(encoding="utf-8")),
    }


# ----------------------------------------------------------------- PowerPoint
def _n(x: float, d: int = 1) -> str:
    """Número con coma decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def _m(x: int) -> str:
    """Entero con punto de miles."""
    return f"{int(x):,}".replace(",", ".")


def _titulo(prs, texto: str, sub: str | None = None):
    s = prs.slides.add_slide(prs.slide_layouts[5])
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.85)
    t.text = texto
    t.text_frame.paragraphs[0].font.size = Pt(26)
    t.text_frame.paragraphs[0].font.bold = True
    t.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    if sub:
        tb = s.shapes.add_textbox(Inches(0.6), Inches(1.25), Inches(12), Inches(0.5))
        tb.text_frame.text = sub
        tb.text_frame.paragraphs[0].font.size = Pt(14)
        tb.text_frame.paragraphs[0].font.color.rgb = _rgb("555555")
    return s


def _limpiar(chart, formato="0.0", leyenda=True):
    chart.has_legend = leyenda
    if leyenda:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(11)
    va = chart.value_axis
    va.has_major_gridlines = False
    va.visible = False
    chart.category_axis.tick_labels.font.size = Pt(11)
    chart.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW   # debajo aunque haya negativos
    chart.category_axis.has_major_gridlines = False
    pl = chart.plots[0]
    pl.has_data_labels = True
    pl.data_labels.number_format = formato
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.font.size = Pt(10)


def _grafico(slide, tipo, datos, x=0.6, y=1.8, w=12.1, h=4.9):
    return slide.shapes.add_chart(tipo, Inches(x), Inches(y), Inches(w), Inches(h), datos).chart


def _bullets(slide, lineas: list[str], x=0.6, y=1.8, w=12.1, h=5.0, tam=16):
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, l in enumerate(lineas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "• " + l
        p.font.size = Pt(tam)
        p.space_after = Pt(8)


def _tabla(slide, df: pd.DataFrame, x, y, w, h, tam=11):
    t = slide.shapes.add_table(len(df) + 1, len(df.columns), Inches(x), Inches(y), Inches(w), Inches(h)).table
    for j, c in enumerate(df.columns):
        t.cell(0, j).text = str(c)
    for i, fila in enumerate(df.itertuples(index=False), start=1):
        for j, v in enumerate(fila):
            if isinstance(v, str):
                t.cell(i, j).text = v
            elif float(v).is_integer() and abs(v) >= 1:
                t.cell(i, j).text = _m(v)
            else:
                t.cell(i, j).text = _n(v, 2)
    for celda in (t.cell(i, j) for i in range(len(df) + 1) for j in range(len(df.columns))):
        for p in celda.text_frame.paragraphs:
            p.font.size = Pt(tam)


def deck(d: dict, figs: dict) -> None:
    cfg = config()
    foco, propio = cfg["mercados"]["foco"], cfg["propio"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    per = d["periodos"]
    tot = per[(per["nivel"] == "mercado total") & (per["mercado"] == foco)].set_index("fabricante")
    ep = d["episodios"]
    no_seg = ep[ep["veredicto_seguimiento"] == "no seguida"]["dshare_pp_foco"].mean()
    seg = ep[ep["veredicto_seguimiento"] == "seguida"]["dshare_pp_foco"].mean()
    el = d["elasticidad"]
    exp = pd.DataFrame(d["trio"]["rondas"])
    prot = exp[exp["clase"] == "retornable protegido"]
    sem = d["semaforo"]
    pil = d["trio"]["piloto"]

    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = f"Diagnóstico de share · {foco}"
    s.placeholders[1].text = "Precio, competencia y tácticas de portafolio\nCaso con datos sintéticos"

    s = _titulo(prs, "Resumen ejecutivo")
    _bullets(s, [
        f"{propio} perdió {_n(abs(tot.loc[propio, 'delta_pp']), 1)} pp de share en {foco} entre el primer y el último año; "
        f"lo ganaron Rival B (+{_n(tot.loc['Rival B', 'delta_pp'], 1)} pp) y marcas chicas (+{_n(tot.loc['Otros', 'delta_pp'], 1)} pp).",
        f"Las rondas de precio que el rival NO siguió costaron {_n(abs(no_seg), 1)} pp en promedio; las seguidas, {_n(abs(seg), 1)} pp: "
        "la pérdida viene de la brecha, no del alza.",
        f"Elasticidad del share al precio relativo: {_n(el.loc['ln_precio_relativo', 'coef'], 2)} "
        f"(IC 90 %: {_n(el.loc['ln_precio_relativo', 'ic90_inf'], 2)} a {_n(el.loc['ln_precio_relativo', 'ic90_sup'], 2)}); "
        f"el nivel real del precio no es significativo.",
        f"Los clientes con el trío crecen {_n(prot['efecto_pp'].mean(), 1)} pp más cuando sube solo el OW "
        "(el retornable queda como refugio); cuando sube también el retornable la ventaja se achica.",
        f"Piloto propuesto: {_m(pil['clientes_por_brazo'])} + {_m(pil['clientes_por_brazo'])} clientes; "
        f"detecta efectos desde {_n(pil['efecto_minimo_detectable_pp'], 1)} pp.",
        f"Mapa semáforo: {sem.loc['rojo', 'peso_volumen']:.0%} del volumen está en zonas donde domina la competencia "
        f"y {sem.loc['amarillo', 'peso_volumen']:.0%} en zonas perdidas con las alzas.",
    ])

    # Share mensual
    sh = d["share"]
    sh = sh[sh["mercado"] == foco].pivot(index="mes", columns="fabricante", values="share")
    trim = sh.groupby(pd.PeriodIndex(sh.index, freq="M").asfreq("Q")).mean()
    cd = CategoryChartData()
    cd.categories = [str(q).replace("Q", " T") for q in trim.index]
    for f in cfg["fabricantes"]:
        cd.add_series(f, (trim[f] * 100).round(1).tolist())
    s = _titulo(prs, f"{propio} cede share a Rival B y a marcas chicas", f"Share en volumen, {foco}, promedio trimestral (%)")
    ch = _grafico(s, XL_CHART_TYPE.LINE_MARKERS, cd)
    _limpiar(ch)
    ch.plots[0].data_labels.position = XL_LABEL_POSITION.ABOVE
    for serie, color in zip(ch.series, [AZUL, ROJO, GRIS, "C8CDD2"]):
        serie.format.line.color.rgb = _rgb(color)
        serie.smooth = False
        serie.marker.format.fill.solid()
        serie.marker.format.fill.fore_color.rgb = _rgb(color)
        serie.marker.format.line.color.rgb = _rgb(color)

    # Episodios
    cd = CategoryChartData()
    cd.categories = [f"{r.ronda}\n{r.veredicto_seguimiento}" for r in ep.itertuples()]
    cd.add_series("Δ share propio (pp)", ep["dshare_pp_foco"].round(1).tolist())
    cd.add_series("Brecha abierta vs. rivales (pp)", ep["brecha_abierta_pp"].round(1).tolist())
    s = _titulo(prs, "Se pierde share cuando la ronda abre brecha con el rival",
                "Cambio de share 3 meses después vs. 3 meses antes de cada ronda de ajuste de precio")
    ch = _grafico(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cd, h=4.2)
    _limpiar(ch)
    for serie, color in zip(ch.series, [AZUL, GRIS]):
        serie.format.fill.solid()
        serie.format.fill.fore_color.rgb = _rgb(color)
        serie.invert_if_negative = False
    pb = d["precio"]["brecha"]
    _bullets(s, [f"Cada punto de brecha costó {_n(abs(pb['pp_share_por_pp_brecha']), 2)} pp de share (R² = {_n(pb['r2'], 2)}, "
                 f"{pb['n_rondas']} rondas). Las rondas con presión de costo y seguidas costaron poco."], y=6.1, h=0.8, tam=13)

    # Elasticidad
    s = _titulo(prs, "El share responde al precio RELATIVO, no al nivel",
                "ln(share/(1−share)) con efectos fijos de mercado×clase y de mes; errores agrupados")
    t = el.reset_index().rename(columns={"index": "variable"})
    t = t.rename(columns={t.columns[0]: "variable"})[["variable", "coef", "ee", "p", "ic90_inf", "ic90_sup"]]
    _tabla(s, t, 0.6, 1.9, 12.1, 1.6, tam=13)
    _bullets(s, ["Un 10 % de precio por encima del rival baja las chances de compra (odds) cerca de "
                 f"{_n(abs(el.loc['ln_precio_relativo', 'coef']) * 10, 0)} %.",
                 "Las disrupciones de abastecimiento pesan aparte: bajan la distribución y el share en el mes."],
             y=4.0, h=2, tam=15)

    # Escalera del trío
    esc = d["escalera"]
    cd = CategoryChartData()
    cd.categories = esc["sku"].tolist()
    cd.add_series("Índice de desembolso (ancla = 100)", esc["indice_desembolso"].round(0).tolist())
    cd.add_series("Índice de precio por litro (ancla = 100)", esc["indice_litro"].round(0).tolist())
    s = _titulo(prs, f"El trío rodea al ancla del rival: {cfg['trio']['ancla']}",
                "Precio sugerido vigente; arriba en desembolso o abajo en precio por litro")
    ch = _grafico(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cd)
    _limpiar(ch, "0")
    for serie, color in zip(ch.series, [AZUL, GRIS]):
        serie.format.fill.solid()
        serie.format.fill.fore_color.rgb = _rgb(color)

    # Prueba del trío
    e = d["experimento"]
    cd = CategoryChartData()
    cd.categories = [f"{r.mes}\n{r.clase}" for r in e.itertuples()]
    cd.add_series("Estimado (pp)", e["efecto_pp"].round(1).tolist())
    cd.add_series("Verdad del generador (pp)", e["verdad_pp"].round(1).tolist())
    s = _titulo(prs, "Prueba del trío: el retornable es el refugio cuando sube solo el OW",
                "Diferencia de crecimiento multiserve, con trío vs. sin trío, dentro de celdas comparables")
    ch = _grafico(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cd, h=3.6)
    _limpiar(ch)
    for serie, color in zip(ch.series, [AZUL, "C8CDD2"]):
        serie.format.fill.solid()
        serie.format.fill.fore_color.rgb = _rgb(color)
        serie.invert_if_negative = False
    t = e[["tipo", "mes", "clase", "efecto_pp", "ic_inf", "ic_sup", "diferencia_ingenua_pp"]]
    _tabla(s, t, 0.6, 5.5, 12.1, 1.6, tam=10)

    # Piloto
    s = _titulo(prs, "Siguiente paso: un piloto aleatorio para pasar de asociación a causa")
    _bullets(s, [
        f"Universo: clientes a los que les falta un SKU del trío ({_m(pil['clientes_elegibles'])} elegibles).",
        f"Diseño: {pil['clientes_por_brazo']} clientes con activación del SKU faltante y {pil['clientes_por_brazo']} de control, al azar.",
        f"Error estándar estimado con la historia: {_n(pil['error_estandar_pp'], 1)} pp → efecto mínimo detectable "
        f"{_n(pil['efecto_minimo_detectable_pp'], 1)} pp (confianza {pil['confianza']:.0%}, potencia {pil['potencia']:.0%}).",
        "Lectura: si el efecto real se parece al de las rondas con retornable protegido, el piloto lo detecta.",
    ])

    # Semáforo
    s = _titulo(prs, "Dónde actuar: mapa semáforo por hexágono", "Volumen de los últimos 12 meses por clase")
    s.shapes.add_picture(str(figs["semaforo"]), Inches(0.5), Inches(1.7), height=Inches(5.6))
    t = sem.reset_index()[["clase", "hexagonos", "clientes", "peso_volumen", "tendencia"]]
    _tabla(s, t, 7.4, 2.0, 5.5, 2.4, tam=12)

    # Anexo
    s = _titulo(prs, "Anexo · método y límites")
    _bullets(s, [
        "Datos 100 % sintéticos con mecanismos conocidos; los análisis no usan las columnas de verdad.",
        "Episodios: índice de Laspeyres de precio por litro (pesos fijos), seguimiento del rival a 3–4 meses, "
        "share 3 meses antes vs. 3 después.",
        "Trío: estado del cliente 9–4 meses antes; celdas distrito × decil de volumen × tercil del peso del retornable; "
        "crecimiento logarítmico; bootstrap agrupado por hexágono; placebos en meses sin rondas.",
        "Límites: asociación, no causalidad (de ahí el piloto); venta interna = sell-in; sin calendario de promociones.",
    ], tam=14)

    destino = ruta("ppt", "diagnostico_share.pptx")
    prs.save(destino)
    print("Deck:", destino)


# ----------------------------------------------------------------- PNG
def figuras(d: dict) -> dict:
    cfg = config()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    out = {}
    foco = cfg["mercados"]["foco"]

    sh = d["share"][d["share"]["mercado"] == foco].pivot(index="mes", columns="fabricante", values="share") * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for f, c in zip(cfg["fabricantes"], ["#" + AZUL, "#" + ROJO, "#" + GRIS, "#c8cdd2"]):
        ax.plot(range(len(sh)), sh[f], color=c, lw=2 if f == cfg["propio"] else 1.3, label=f)
        ax.annotate(f"{sh[f].iloc[-1]:.0f}", (len(sh) - 1, sh[f].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", color=c, fontsize=8)
    for r in cfg["rondas"]:
        ax.axvline(list(sh.index).index(r["mes"]), color="#ddd", lw=0.8, zorder=0)
    ax.set_xticks(range(0, len(sh), 6), sh.index[::6])
    ax.set_ylabel("share en volumen (%)")
    ax.set_title(f"Share mensual en {foco} · líneas grises = rondas de ajuste de precio", loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    out["share"] = ruta("graficos", "share.png")
    fig.savefig(out["share"], dpi=150)

    ep = d["episodios"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    col = ["#" + ROJO if v == "no seguida" else "#" + AZUL for v in ep["veredicto_seguimiento"]]
    ax.scatter(ep["brecha_abierta_pp"], ep["dshare_pp_foco"], c=col, s=50)
    for r in ep.itertuples():
        ax.annotate(f"{r.ronda} · {r.veredicto_seguimiento}", (r.brecha_abierta_pp, r.dshare_pp_foco),
                    xytext=(5, 4), textcoords="offset points", fontsize=7.5)
    b = d["precio"]["brecha"]
    xs = np.linspace(ep["brecha_abierta_pp"].min(), ep["brecha_abierta_pp"].max(), 10)
    ax.plot(xs, b["intercepto"] + b["pp_share_por_pp_brecha"] * xs, color="#999", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("brecha de precio abierta vs. rivales (pp)")
    ax.set_ylabel("Δ share propio (pp)")
    ax.set_title("La pérdida sigue a la brecha, no al alza", loc="left")
    fig.tight_layout()
    out["episodios"] = ruta("graficos", "episodios.png")
    fig.savefig(out["episodios"], dpi=150)

    e = d["experimento"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(e))
    colores = ["#" + AZUL if c == "retornable protegido" else ("#" + GRIS if t == "ronda" else "#c8cdd2")
               for c, t in zip(e["clase"], e["tipo"])]
    ax.bar(x, e["efecto_pp"], color=colores, yerr=[e["efecto_pp"] - e["ic_inf"], e["ic_sup"] - e["efecto_pp"]],
           capsize=3, ecolor="#555")
    ax.scatter(x, e["verdad_pp"], marker="D", color="#" + ROJO, zorder=3, label="verdad del generador")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x, [f"{m}\n{c}" for m, c in zip(e["mes"], e["clase"])], fontsize=7.5)
    ax.set_ylabel("con trío − sin trío (pp, log)")
    ax.set_title("Prueba del trío · barras = estimado con IC 90 %, rombo = verdad", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    out["trio"] = ruta("graficos", "trio.png")
    fig.savefig(out["trio"], dpi=150)

    hx = d["semaforo_hex"]
    polys = [[(lng, lat) for lat, lng in h3.cell_to_boundary(c)] for c in hx["h3"]]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(PolyCollection(polys, facecolors=[COLORES[c] for c in hx["clase"]], edgecolors="white", linewidths=0.6))
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.radians(hx["lat"].mean())))
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    from matplotlib.patches import Patch
    nombres = {"rojo": "domina la competencia", "amarillo": "perdido con las alzas", "verde": "fortaleza",
               "naranja": "cae por otras razones", "gris": "pocos clientes"}
    ax.legend(handles=[Patch(color=COLORES[k], label=v) for k, v in nombres.items()], loc="lower left",
              frameon=False, fontsize=8)
    ax.set_title("Mapa semáforo (H3 res. 7)", loc="left")
    fig.tight_layout()
    out["semaforo"] = ruta("graficos", "semaforo.png")
    fig.savefig(out["semaforo"], dpi=150)
    plt.close("all")
    return out


def main() -> None:
    d = _cargar()
    figs = figuras(d)
    deck(d, figs)


if __name__ == "__main__":
    main()
