"""期权组合策略：到期盈亏、当前盈亏（按 BS 模型）、盈亏平衡点、最大盈亏、盈利概率、组合希腊值。

腿（Leg）的写法（用于命令行 --legs）：
    +1C105   买 1 张 行权价 105 的看涨
    -2P95    卖 2 张 行权价 95 的看跌
    +1S      买 100 股正股（S 表示 1 手=100 股）
多条腿用逗号分隔，例如牛市看涨价差："+1C100,-1C110"
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .pricing import bs_price, greeks

MULT = 100
LEG_RE = re.compile(r"^\s*([+-]?)(\d*)\s*([CPS])\s*(\d+(?:\.\d+)?)?\s*$", re.I)


@dataclass
class Leg:
    kind: str            # 'call' / 'put' / 'stock'
    qty: int             # 正数=买入，负数=卖出（张数；正股为手，1 手=100 股）
    strike: float = 0.0
    premium: float = 0.0  # 每股价格（期权为权利金，正股为买入价）
    iv: float = float("nan")

    def payoff_at_expiry(self, prices: np.ndarray) -> np.ndarray:
        if self.kind == "stock":
            val = prices - self.premium
        elif self.kind == "call":
            val = np.maximum(prices - self.strike, 0) - self.premium
        else:
            val = np.maximum(self.strike - prices, 0) - self.premium
        return val * self.qty * MULT

    def label(self) -> str:
        side = "买" if self.qty > 0 else "卖"
        if self.kind == "stock":
            return f"{side} {abs(self.qty) * MULT} 股 @ {self.premium:.2f}"
        name = "看涨" if self.kind == "call" else "看跌"
        return f"{side} {abs(self.qty)} 张 {self.strike:g} {name} @ {self.premium:.2f} (IV {self.iv:.1%})"


def parse_legs(text: str) -> list[tuple[str, int, float | None]]:
    out = []
    for part in text.split(","):
        m = LEG_RE.match(part)
        if not m:
            raise ValueError(f"看不懂这条腿：{part!r}（示例：+1C105 / -1P95 / +1S）")
        sign, n, k, strike = m.groups()
        qty = int(n or 1) * (-1 if sign == "-" else 1)
        kind = {"C": "call", "P": "put", "S": "stock"}[k.upper()]
        if kind != "stock" and strike is None:
            raise ValueError(f"期权腿需要行权价：{part!r}")
        out.append((kind, qty, float(strike) if strike else None))
    return out


PRESETS = {
    # 名称: (说明, 需要的行权价个数, 生成腿的函数)
    "long-call":     ("买入看涨", 1, lambda k: f"+1C{k[0]}"),
    "long-put":      ("买入看跌", 1, lambda k: f"+1P{k[0]}"),
    "covered-call":  ("备兑看涨（持股 + 卖 call）", 1, lambda k: f"+1S,-1C{k[0]}"),
    "csp":           ("现金担保卖 put", 1, lambda k: f"-1P{k[0]}"),
    "bull-call":     ("牛市看涨价差（低买高卖）", 2, lambda k: f"+1C{k[0]},-1C{k[1]}"),
    "bear-put":      ("熊市看跌价差（高买低卖）", 2, lambda k: f"+1P{k[1]},-1P{k[0]}"),
    "bull-put":      ("牛市看跌价差（收权利金）", 2, lambda k: f"-1P{k[1]},+1P{k[0]}"),
    "bear-call":     ("熊市看涨价差（收权利金）", 2, lambda k: f"-1C{k[0]},+1C{k[1]}"),
    "straddle":      ("买入跨式", 1, lambda k: f"+1C{k[0]},+1P{k[0]}"),
    "strangle":      ("买入宽跨式", 2, lambda k: f"+1P{k[0]},+1C{k[1]}"),
    "iron-condor":   ("铁鹰（4 个行权价由低到高）", 4,
                      lambda k: f"+1P{k[0]},-1P{k[1]},-1C{k[2]},+1C{k[3]}"),
    "butterfly":     ("看涨蝶式（3 个行权价）", 3, lambda k: f"+1C{k[0]},-2C{k[1]},+1C{k[2]}"),
}


def preset_legs(name: str, strikes: list[float]) -> str:
    if name not in PRESETS:
        raise ValueError(f"未知策略 {name}，可选：{', '.join(PRESETS)}")
    desc, n, fn = PRESETS[name]
    if len(strikes) != n:
        raise ValueError(f"{name}（{desc}）需要 {n} 个行权价，你给了 {len(strikes)} 个")
    strikes = sorted(strikes)
    return fn([f"{s:g}" for s in strikes])


def legs_from_chain(text: str, enriched: pd.DataFrame, spot: float, expiry) -> list[Leg]:
    """用期权链里的中间价和 IV 填充每条腿；行权价会对齐到最近的可交易行权价。"""
    sub = enriched[enriched["expiry"] == expiry]
    legs = []
    for kind, qty, strike in parse_legs(text):
        if kind == "stock":
            legs.append(Leg("stock", qty, 0.0, spot))
            continue
        s = sub[sub["type"] == kind]
        if s.empty:
            raise ValueError(f"{expiry} 没有 {kind} 合约")
        row = s.loc[(s["strike"] - strike).abs().idxmin()]
        legs.append(Leg(kind, qty, float(row["strike"]), float(row["mid"]), float(row["iv"])))
    return legs


@dataclass
class StrategyResult:
    legs: list[Leg]
    net_premium: float          # 正数=净支出，负数=净收入（整组美元）
    breakevens: list[float]
    max_profit: float           # inf 表示理论无限
    max_loss: float             # -inf 表示理论无限
    pop: float                  # 到期盈利概率（风险中性、对数正态）
    greeks: dict
    table: pd.DataFrame         # 不同股价下的到期盈亏与当前盈亏


def analyze(legs: list[Leg], spot: float, dte: int, r: float = 0.04, q: float = 0.0,
            vol: float | None = None, width: float = 0.35, n: int = 2001) -> StrategyResult:
    T = max(dte, 0.5) / 365.0
    opt_ivs = [l.iv for l in legs if l.kind != "stock" and np.isfinite(l.iv)]
    sigma = vol if vol else (float(np.mean(opt_ivs)) if opt_ivs else 0.3)
    strikes = [l.strike for l in legs if l.kind != "stock"]
    lo = min([spot * (1 - width)] + [k * 0.8 for k in strikes])
    hi = max([spot * (1 + width)] + [k * 1.2 for k in strikes])
    prices = np.linspace(max(lo, 0.01), hi, n)
    pnl = sum(l.payoff_at_expiry(prices) for l in legs)

    # 盈亏平衡点：盈亏曲线穿过 0 的位置（线性插值）
    sign = np.sign(pnl)
    idx = np.where(np.diff(sign) != 0)[0]
    bes = []
    for i in idx:
        x0, x1, y0, y1 = prices[i], prices[i + 1], pnl[i], pnl[i + 1]
        bes.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x0))
    bes = sorted({round(b, 2) for b in bes})

    # 最大盈亏：看两端的斜率判断是否无限
    slope_hi = sum(l.qty * (1 if l.kind in ("call", "stock") else 0) for l in legs)
    max_p, max_l = float(pnl.max()), float(pnl.min())
    if slope_hi > 0:
        max_p = float("inf")
    if slope_hi < 0:
        max_l = float("-inf")
    # 价格跌到 0 时盈亏有限，直接计算
    at_zero = float(sum(l.payoff_at_expiry(np.array([0.0]))[0] for l in legs))
    max_p, max_l = max(max_p, at_zero), min(max_l, at_zero)

    # 到期盈利概率：对数正态分布下 pnl>0 的概率质量
    mu = np.log(spot) + (r - q - 0.5 * sigma**2) * T
    sd = sigma * np.sqrt(T)
    edges = np.concatenate([[0.0], (prices[1:] + prices[:-1]) / 2, [np.inf]])
    with np.errstate(divide="ignore"):
        cdf = norm.cdf((np.log(edges) - mu) / sd)
    pop = float(np.sum(np.diff(cdf)[pnl > 0]))

    # 组合希腊值（整组美元口径）
    tot = {k: 0.0 for k in ("delta", "gamma", "theta", "vega")}
    for l in legs:
        if l.kind == "stock":
            tot["delta"] += l.qty * MULT
            continue
        g = greeks(spot, l.strike, T, r, l.iv if np.isfinite(l.iv) else sigma, l.kind, q)
        for k in tot:
            tot[k] += float(g[k]) * l.qty * MULT

    # 盈亏表：在几个代表性股价上展示到期盈亏和“现在”的理论盈亏
    marks = sorted({round(spot * (1 + x), 2) for x in (-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2)}
                   | {round(k, 2) for k in strikes} | set(bes))
    marks = np.array(marks)
    exp_pnl = sum(l.payoff_at_expiry(marks) for l in legs)
    now_pnl = np.zeros_like(marks)
    for l in legs:
        if l.kind == "stock":
            now_pnl += (marks - l.premium) * l.qty * MULT
        else:
            iv = l.iv if np.isfinite(l.iv) else sigma
            now_pnl += (bs_price(marks, l.strike, T, r, iv, l.kind, q) - l.premium) * l.qty * MULT
    table = pd.DataFrame({"股价": marks, "较现价": marks / spot - 1, "到期盈亏": exp_pnl, "当前理论盈亏": now_pnl})

    net = float(sum(l.premium * l.qty * MULT for l in legs))
    return StrategyResult(legs, net, bes, max_p, max_l, pop, tot, table)


def ascii_payoff(legs: list[Leg], spot: float, width: int = 60, height: int = 15, span: float = 0.3) -> str:
    """在终端里画一个简单的到期盈亏图。"""
    strikes = [l.strike for l in legs if l.kind != "stock"]
    lo = min([spot * (1 - span)] + [k * 0.9 for k in strikes])
    hi = max([spot * (1 + span)] + [k * 1.1 for k in strikes])
    xs = np.linspace(lo, hi, width)
    ys = sum(l.payoff_at_expiry(xs) for l in legs)
    ymin, ymax = min(ys.min(), 0), max(ys.max(), 0)
    if ymax == ymin:
        ymax = ymin + 1
    rows = []
    zero_row = int(round((ymax - 0) / (ymax - ymin) * (height - 1)))
    spot_col = int(round((spot - lo) / (hi - lo) * (width - 1)))
    grid = [[" "] * width for _ in range(height)]
    for c in range(width):
        grid[zero_row][c] = "-"
    for rr in range(height):
        grid[rr][spot_col] = ":"
    for c, y in enumerate(ys):
        rr = int(round((ymax - y) / (ymax - ymin) * (height - 1)))
        grid[rr][c] = "+" if y > 0 else "x" if y < 0 else "o"
    for rr in range(height):
        yv = ymax - rr * (ymax - ymin) / (height - 1)
        rows.append(f"{yv:>10,.0f} |" + "".join(grid[rr]))
    rows.append(" " * 11 + "+" + "-" * width)
    rows.append(" " * 12 + f"{lo:<10.2f}" + f"现价 {spot:.2f}".center(width - 20) + f"{hi:>10.2f}")
    rows.append(" " * 12 + "（+ 盈利  x 亏损  : 当前股价）")
    return "\n".join(rows)
