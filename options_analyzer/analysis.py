"""期权链分析：补全希腊值、预期波动、最大痛点、PCR、波动率偏斜与期限结构、合约筛选。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import OptionChain
from .pricing import greeks, implied_vol


@dataclass
class MarketParams:
    r: float = 0.04   # 无风险利率（可参考 3 个月美债收益率）
    q: float = 0.0    # 股息率


def enrich(chain: OptionChain, params: MarketParams | None = None, recompute_iv: bool = False) -> pd.DataFrame:
    """在期权链上添加 mid、dte、iv、希腊值、到期实值概率、价差比例等列。"""
    p = params or MarketParams()
    df = chain.df.copy()
    S = chain.spot
    has_quote = (df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])
    df["mid"] = np.where(has_quote, (df["bid"] + df["ask"]) / 2, df["last"])
    df["spread_pct"] = np.where(has_quote & (df["mid"] > 0), (df["ask"] - df["bid"]) / df["mid"], np.nan)
    df["dte"] = [(e - chain.asof).days for e in df["expiry"]]
    df["T"] = np.maximum(df["dte"], 0.5) / 365.0  # 当天到期按半天算

    need = df["iv"].isna() | (df["iv"] <= 0.001) if not recompute_iv else pd.Series(True, index=df.index)
    if need.any():
        sub = df[need]
        df.loc[need, "iv"] = implied_vol(sub["mid"].values, S, sub["strike"].values, sub["T"].values,
                                         p.r, sub["type"].values, p.q)

    g = greeks(S, df["strike"].values, df["T"].values, p.r, df["iv"].fillna(0).values, df["type"].values, p.q)
    for k, v in g.items():
        df[k] = np.where(df["iv"].notna(), v, np.nan)

    is_call = df["type"] == "call"
    df["intrinsic"] = np.where(is_call, np.maximum(S - df["strike"], 0), np.maximum(df["strike"] - S, 0))
    df["extrinsic"] = df["mid"] - df["intrinsic"]
    df["moneyness"] = df["strike"] / S - 1
    df["breakeven"] = np.where(is_call, df["strike"] + df["mid"], df["strike"] - df["mid"])
    # 卖方年化收益：卖 put 以占用现金(行权价)计；卖 call 以正股市值计
    base = np.where(is_call, S, df["strike"])
    df["sell_yield_ann"] = df["mid"] / base * 365 / np.maximum(df["dte"], 1)
    return df


def nearest_expiry(df: pd.DataFrame, target_dte: int):
    exp = df.groupby("expiry")["dte"].first()
    return exp.index[np.argmin(np.abs(exp.values - target_dte))]


def atm_iv(df: pd.DataFrame, spot: float, expiry) -> float:
    """平值 IV：在离现价最近的两个行权价间插值，看涨/看跌取平均。"""
    sub = df[(df["expiry"] == expiry) & df["iv"].notna()]
    vals = []
    for typ in ("call", "put"):
        s = sub[sub["type"] == typ].sort_values("strike")
        if len(s) >= 2:
            vals.append(float(np.interp(spot, s["strike"], s["iv"])))
        elif len(s) == 1:
            vals.append(float(s["iv"].iloc[0]))
    return float(np.mean(vals)) if vals else float("nan")


def term_structure(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    rows = []
    for exp, g in df.groupby("expiry"):
        dte = int(g["dte"].iloc[0])
        iv = atm_iv(df, spot, exp)
        rows.append({"expiry": exp, "dte": dte, "atm_iv": iv})
    return pd.DataFrame(rows)


def expected_move(df: pd.DataFrame, spot: float, expiry) -> dict:
    """到期前的预期波动幅度：(1) 平值跨式价格；(2) IV 推算的 1 个标准差。"""
    sub = df[df["expiry"] == expiry]
    dte = int(sub["dte"].iloc[0])
    k = sub.loc[(sub["strike"] - spot).abs().idxmin(), "strike"]
    c = sub[(sub["type"] == "call") & (sub["strike"] == k)]["mid"]
    p = sub[(sub["type"] == "put") & (sub["strike"] == k)]["mid"]
    straddle = float(c.iloc[0] + p.iloc[0]) if len(c) and len(p) else float("nan")
    iv = atm_iv(df, spot, expiry)
    sigma_move = spot * iv * np.sqrt(max(dte, 0.5) / 365.0)
    return {"expiry": expiry, "dte": dte, "atm_strike": float(k), "straddle": straddle,
            "straddle_pct": straddle / spot, "atm_iv": iv, "one_sigma": sigma_move,
            "range_low": spot - sigma_move, "range_high": spot + sigma_move}


def max_pain(df: pd.DataFrame, expiry) -> tuple[float, pd.DataFrame]:
    """最大痛点：到期时让期权买方总价值（=卖方总赔付）最小的价格。"""
    sub = df[df["expiry"] == expiry]
    calls, puts = sub[sub["type"] == "call"], sub[sub["type"] == "put"]
    strikes = np.sort(sub["strike"].unique())
    pay = []
    for P in strikes:
        c = (np.maximum(P - calls["strike"], 0) * calls["open_interest"]).sum()
        pt = (np.maximum(puts["strike"] - P, 0) * puts["open_interest"]).sum()
        pay.append(100 * (c + pt))
    table = pd.DataFrame({"price": strikes, "total_payout": pay})
    return float(strikes[int(np.argmin(pay))]), table


def put_call_ratio(df: pd.DataFrame, expiry=None) -> dict:
    sub = df if expiry is None else df[df["expiry"] == expiry]
    agg = sub.groupby("type")[["volume", "open_interest"]].sum()
    c, p = agg.reindex(["call", "put"]).fillna(0).values
    return {"volume_pcr": p[0] / c[0] if c[0] else float("nan"),
            "oi_pcr": p[1] / c[1] if c[1] else float("nan"),
            "call_oi": int(c[1]), "put_oi": int(p[1])}


def skew(df: pd.DataFrame, spot: float, expiry, delta: float = 0.25) -> dict:
    """25 Delta 偏斜：看跌 IV − 看涨 IV。正值说明市场更愿意花钱买下跌保护。"""
    sub = df[(df["expiry"] == expiry) & df["iv"].notna() & df["delta"].notna()]
    out = {"expiry": expiry, "atm_iv": atm_iv(df, spot, expiry)}
    for typ, target in (("put", -delta), ("call", delta)):
        s = sub[sub["type"] == typ].sort_values("delta")
        out[f"{typ}_iv"] = float(np.interp(target, s["delta"], s["iv"])) if len(s) >= 2 else float("nan")
    out["skew"] = out["put_iv"] - out["call_iv"]
    return out


def oi_walls(df: pd.DataFrame, expiry, top: int = 3) -> dict:
    """未平仓量最大的行权价（常被视为支撑/阻力参考）。"""
    sub = df[df["expiry"] == expiry]
    res = {}
    for typ in ("call", "put"):
        s = sub[sub["type"] == typ].nlargest(top, "open_interest")
        res[typ] = list(zip(s["strike"].astype(float), s["open_interest"].astype(int)))
    return res


def screen(df: pd.DataFrame, option_type: str | None = None, min_dte: int = 0, max_dte: int = 10_000,
           min_delta: float | None = None, max_delta: float | None = None, min_oi: int = 0,
           min_volume: int = 0, max_spread_pct: float | None = None, sort_by: str = "sell_yield_ann",
           ascending: bool = False, limit: int = 20) -> pd.DataFrame:
    """按条件筛选合约。delta 用绝对值比较（看跌期权的 delta 0.3 即 -0.3）。"""
    m = (df["dte"] >= min_dte) & (df["dte"] <= max_dte)
    m &= (df["open_interest"] >= min_oi) & (df["volume"] >= min_volume) & df["iv"].notna()
    if option_type:
        m &= df["type"] == option_type
    ad = df["delta"].abs()
    if min_delta is not None:
        m &= ad >= min_delta
    if max_delta is not None:
        m &= ad <= max_delta
    if max_spread_pct is not None:
        m &= df["spread_pct"] <= max_spread_pct
    return df[m].sort_values(sort_by, ascending=ascending).head(limit)
