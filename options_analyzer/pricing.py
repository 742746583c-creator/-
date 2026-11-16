"""Black-Scholes-Merton 定价、希腊值与隐含波动率（支持 numpy 向量化）。

约定：
- T 以“年”为单位；r、q、sigma 为年化小数（4% 写 0.04）。
- theta 为“每天”的变化；vega、rho 为波动率/利率每变动 1 个百分点的变化。
- 美股个股期权是美式期权，这里用欧式公式近似，对不派息股票的看涨期权完全准确，
  对深度实值看跌期权会略微低估。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

MIN_T = 1e-6
MIN_SIGMA = 1e-6


def _prep(S, K, T, r, sigma, q):
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    T = np.maximum(T, MIN_T)
    sigma = np.maximum(sigma, MIN_SIGMA)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S, K, T, r, sigma, q, sqrtT, d1, d2


def _is_call(option_type):
    t = np.asarray(option_type)
    if t.dtype == bool:
        return t
    return np.char.lower(t.astype(str)) == "call"


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    S, K, T, r, sigma, q, _, d1, d2 = _prep(S, K, T, r, sigma, q)
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return np.where(_is_call(option_type), call, put)


def greeks(S, K, T, r, sigma, option_type="call", q=0.0) -> dict:
    """返回 delta / gamma / theta(每天) / vega(每1%) / rho(每1%) / prob_itm（风险中性到期实值概率）。"""
    S, K, T, r, sigma, q, sqrtT, d1, d2 = _prep(S, K, T, r, sigma, q)
    is_call = _is_call(option_type)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    delta = np.where(is_call, disc_q * norm.cdf(d1), -disc_q * norm.cdf(-d1))
    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT / 100.0
    decay = -S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
    theta_call = decay - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    theta_put = decay + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
    theta = np.where(is_call, theta_call, theta_put) / 365.0
    rho = np.where(is_call, K * T * disc_r * norm.cdf(d2), -K * T * disc_r * norm.cdf(-d2)) / 100.0
    prob_itm = np.where(is_call, norm.cdf(d2), norm.cdf(-d2))
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho, "prob_itm": prob_itm}


def implied_vol_scalar(price, S, K, T, r, option_type="call", q=0.0, lo=1e-4, hi=5.0) -> float:
    """单个期权的隐含波动率；价格不合理（低于内在价值或高于上限）时返回 NaN。"""
    if not all(np.isfinite([price, S, K, T])) or price <= 0 or S <= 0 or K <= 0:
        return float("nan")
    T = max(T, MIN_T)
    is_call = str(option_type).lower() == "call"
    disc_S, disc_K = S * np.exp(-q * T), K * np.exp(-r * T)
    lower = max(disc_S - disc_K, 0.0) if is_call else max(disc_K - disc_S, 0.0)
    upper = disc_S if is_call else disc_K
    if price <= lower + 1e-10 or price >= upper:
        return float("nan")

    def f(sig):
        return float(bs_price(S, K, T, r, sig, option_type, q)) - price

    try:
        if f(lo) > 0 or f(hi) < 0:
            return float("nan")
        return brentq(f, lo, hi, xtol=1e-7, maxiter=200)
    except (ValueError, RuntimeError):
        return float("nan")


def implied_vol(price, S, K, T, r, option_type="call", q=0.0) -> np.ndarray:
    """向量化的隐含波动率（逐个求解）。"""
    b = np.broadcast_arrays(*(np.asarray(x, dtype=object) for x in (price, S, K, T, r, option_type, q)))
    out = [implied_vol_scalar(float(p), float(s), float(k), float(t), float(rr), str(ot), float(qq))
           for p, s, k, t, rr, ot, qq in zip(*(a.ravel() for a in b))]
    return np.asarray(out, dtype=float).reshape(b[0].shape)
