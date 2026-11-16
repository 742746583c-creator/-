import datetime as dt

import numpy as np
import pytest

from options_analyzer import analysis as A
from options_analyzer import strategy as S
from options_analyzer.data import DemoSource
from options_analyzer.pricing import bs_price, greeks, implied_vol


def test_bs_known_value():
    # 教科书例子：S=100, K=100, T=1, r=5%, sigma=20% -> call≈10.4506, put≈5.5735
    assert bs_price(100, 100, 1, 0.05, 0.2, "call") == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(100, 100, 1, 0.05, 0.2, "put") == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity():
    S0, K, T, r, q, sig = 120, 110, 0.4, 0.03, 0.01, 0.35
    c = bs_price(S0, K, T, r, sig, "call", q)
    p = bs_price(S0, K, T, r, sig, "put", q)
    assert c - p == pytest.approx(S0 * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-9)


def test_greeks_match_finite_difference():
    S0, K, T, r, sig = 100, 105, 0.25, 0.04, 0.3
    g = greeks(S0, K, T, r, sig, "call")
    h = 1e-3
    d = (bs_price(S0 + h, K, T, r, sig) - bs_price(S0 - h, K, T, r, sig)) / (2 * h)
    v = (bs_price(S0, K, T, r, sig + 0.01) - bs_price(S0, K, T, r, sig - 0.01)) / 2
    th = bs_price(S0, K, T - 1 / 365, r, sig) - bs_price(S0, K, T, r, sig)
    assert float(g["delta"]) == pytest.approx(d, abs=1e-5)
    assert float(g["vega"]) == pytest.approx(v, abs=1e-4)
    assert float(g["theta"]) == pytest.approx(th, abs=2e-3)


def test_implied_vol_roundtrip_and_bad_price():
    for typ in ("call", "put"):
        price = bs_price(100, 90, 0.5, 0.04, 0.42, typ)
        assert float(implied_vol(price, 100, 90, 0.5, 0.04, typ)) == pytest.approx(0.42, abs=1e-5)
    assert np.isnan(implied_vol(5.0, 100, 90, 0.5, 0.04, "call"))  # 低于内在价值


@pytest.fixture
def demo():
    chain = DemoSource(spot=100, asof=dt.date(2026, 1, 2)).get_chain("DEMO")
    return chain, A.enrich(chain)


def test_enrich_recovers_demo_iv(demo):
    chain, df = demo
    src = DemoSource(spot=100, asof=dt.date(2026, 1, 2))
    atm = df[(df["strike"] == 100) & (df["dte"] == 30)]
    expected = src.iv_surface(100, 30)
    assert np.allclose(atm["iv"], expected, atol=0.01)  # 买卖价四舍五入带来少量误差


def test_max_pain_and_pcr(demo):
    chain, df = demo
    exp = chain.expiries[0]
    mp, table = A.max_pain(df, exp)
    assert mp == table.loc[table["total_payout"].idxmin(), "price"]
    pcr = A.put_call_ratio(df, exp)
    assert pcr["oi_pcr"] > 0


def test_screen_filters(demo):
    _, df = demo
    res = A.screen(df, option_type="put", min_delta=0.2, max_delta=0.3, max_dte=45, min_oi=0)
    assert len(res) > 0
    assert (res["type"] == "put").all()
    assert res["delta"].abs().between(0.2, 0.3).all()
    assert (res["dte"] <= 45).all()


def test_iron_condor_numbers():
    legs = [S.Leg("put", 1, 85, 0.26), S.Leg("put", -1, 90, 0.61),
            S.Leg("call", -1, 110, 0.45), S.Leg("call", 1, 115, 0.10)]
    res = S.analyze(legs, 100, 30, vol=0.3)
    credit = (0.61 + 0.45 - 0.26 - 0.10) * 100
    assert res.net_premium == pytest.approx(-credit)
    assert res.max_profit == pytest.approx(credit)
    assert res.max_loss == pytest.approx(-(500 - credit))
    assert res.breakevens == pytest.approx([90 - 0.70, 110 + 0.70], abs=0.02)
    assert 0 < res.pop < 1


def test_unlimited_risk_detection():
    naked_call = S.analyze([S.Leg("call", -1, 110, 2.0)], 100, 30, vol=0.3)
    assert naked_call.max_loss == float("-inf")
    long_call = S.analyze([S.Leg("call", 1, 110, 2.0)], 100, 30, vol=0.3)
    assert long_call.max_profit == float("inf") and long_call.max_loss == pytest.approx(-200)


def test_parse_legs_and_presets():
    assert S.parse_legs("+1C105,-2P95,+1S") == [("call", 1, 105.0), ("put", -2, 95.0), ("stock", 1, None)]
    assert S.preset_legs("bull-put", [95, 90]) == "-1P95,+1P90"
    with pytest.raises(ValueError):
        S.parse_legs("buy call")
