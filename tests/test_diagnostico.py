import h3
import numpy as np
import pandas as pd
import pytest

from share_diag import comun, datos, precios, semaforo, tactica
from share_diag.pipeline import PASOS, Paso, pendiente


@pytest.fixture(scope="module")
def mundo():
    return datos.generar()


@pytest.fixture(scope="module")
def panel_trio(mundo):
    return tactica.Panel(mundo["venta_clientes"], mundo["clientes"])


def test_reproducible(mundo):
    otro = datos.generar()
    pd.testing.assert_frame_equal(mundo["panel"], otro["panel"])
    pd.testing.assert_frame_equal(mundo["clientes"], otro["clientes"])


def test_ciudad_ficticia_en_mar_abierto(mundo):
    hx = mundo["hexagonos"]
    assert hx["lat"].between(-28, -25).all() and hx["lng"].between(-23, -20).all()
    assert all(h3.get_resolution(c) == 7 for c in hx["h3"])


def test_share_cae_en_el_foco(mundo):
    from share_diag.share import comparar_periodos
    c = comparar_periodos(mundo["panel"])
    foco, propio = comun.config()["mercados"]["foco"], comun.config()["propio"]
    fila = c[(c["nivel"] == "mercado total") & (c["mercado"] == foco) & (c["fabricante"] == propio)]
    assert fila["delta_pp"].iloc[0] < -2


def test_elasticidad_recupera_la_verdad(mundo):
    el = precios.elasticidad(mundo["panel"], mundo["macro"])
    beta = comun.config()["verdad"]["beta_precio"]
    assert abs(el.loc["ln_precio_relativo", "coef"] - beta) < 0.35
    # Lo que importa es la brecha: el nivel real del precio no es significativo
    assert el.loc["ln_precio_real", "ic90_inf"] < 0 < el.loc["ln_precio_real", "ic90_sup"] or \
        abs(el.loc["ln_precio_real", "coef"]) < 0.3


def test_episodios_detectan_quien_siguio(mundo):
    ep = precios.episodios(mundo["panel"], mundo["macro"]).set_index("ronda")
    for r in comun.config()["rondas"]:
        esperado = "no seguida" if r["Rival B"] is None else "seguida"
        assert ep.loc[r["mes"], "veredicto_seguimiento"] == esperado
    assert precios.pendiente_brecha(ep.reset_index())["pp_share_por_pp_brecha"] < 0


def test_prueba_del_trio(panel_trio):
    cfg = comun.config()
    b0 = cfg["experimento"]["bootstrap"]
    cfg["experimento"]["bootstrap"] = 60
    try:
        exp = tactica.experimento(panel_trio)
    finally:
        cfg["experimento"]["bootstrap"] = b0
    rondas = exp[exp["tipo"] == "ronda"]
    prot = rondas[rondas["clase"] == "retornable protegido"]
    sube = rondas[rondas["clase"] == "retornable subió"]
    assert (prot["ic_inf"] > 0).all()                                # efecto claro con el retornable protegido
    assert prot["efecto_pp"].mean() > sube["efecto_pp"].mean()       # y más chico cuando sube el retornable
    assert (rondas["efecto_pp"] - rondas["verdad_pp"]).abs().max() < 4
    assert (exp.loc[exp["tipo"] == "placebo", "verdad_pp"].abs() < 1).all()


def test_potencia_piloto(panel_trio):
    p = tactica.potencia_piloto(panel_trio, n_sim=80)
    assert 0 < p["efecto_minimo_detectable_pp"] < 15
    assert 2 * p["clientes_por_brazo"] <= p["clientes_elegibles"]


def test_semaforo_encuentra_zonas_rojas(mundo):
    d = semaforo.clasificar(semaforo.metricas_hex(mundo["venta_clientes"], mundo["clientes"],
                                                  mundo["ejecucion"], mundo["hexagonos"]))
    d = d.merge(mundo["hexagonos"][["h3", "verdad_dominio_rival"]], on="h3")
    rojos = d[d["clase"] == "rojo"]
    assert len(rojos) > 5
    assert rojos["verdad_dominio_rival"].mean() >= 0.8
    assert set(d["clase"]) <= set(semaforo.COLORES)


def test_pipeline_detecta_salidas_faltantes(tmp_path):
    p = Paso("prueba", "share_diag.share", (), (tmp_path / "no_existe.csv",))
    assert pendiente(p, {"prueba": {"firma": "x"}})
    assert [p.nombre for p in PASOS][0] == "datos"


def test_ninguna_salida_expone_la_verdad(mundo):
    # Las columnas verdad_* existen en los datos, pero ningún análisis las devuelve
    ep = precios.episodios(mundo["panel"], mundo["macro"])
    assert not [c for c in ep.columns if c.startswith("verdad")]
    assert np.isfinite(ep["dshare_pp_foco"]).all()
