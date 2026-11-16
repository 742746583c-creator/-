"""期权链数据源。

统一的期权链格式（pandas.DataFrame，每行一个合约）：
    symbol, type('call'/'put'), strike, expiry(datetime.date), bid, ask, last,
    volume, open_interest, iv(可为空，小数形式)

数据源：
- YahooSource：通过 yfinance 获取 Yahoo 财经的期权链（免费、有延迟，需要能访问 Yahoo）。
- CSVSource：读取你从券商（IBKR、富途、老虎等）导出的 CSV。
- DemoSource：生成一份带波动率微笑的模拟期权链，用于离线学习和测试。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pricing import bs_price

COLUMNS = ["symbol", "type", "strike", "expiry", "bid", "ask", "last", "volume", "open_interest", "iv"]


@dataclass
class OptionChain:
    symbol: str
    spot: float
    df: pd.DataFrame
    asof: dt.date
    source: str = ""

    @property
    def expiries(self) -> list[dt.date]:
        return sorted(self.df["expiry"].unique())


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS].copy()
    df["type"] = df["type"].astype(str).str.lower().map(
        lambda x: "call" if x.startswith("c") else "put")
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    for c in ["strike", "bid", "ask", "last", "volume", "open_interest", "iv"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[["volume", "open_interest"]] = df[["volume", "open_interest"]].fillna(0)
    return df.sort_values(["expiry", "type", "strike"]).reset_index(drop=True)


class YahooSource:
    name = "yahoo"

    def __init__(self, max_expiries: int | None = 8):
        self.max_expiries = max_expiries

    def get_chain(self, symbol: str, expiries: list[str] | None = None) -> OptionChain:
        try:
            return self._get_chain(symbol, expiries)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"从 Yahoo 获取 {symbol} 失败（{type(e).__name__}: {e}）。"
                               "请检查网络能否访问 finance.yahoo.com，或改用 --source csv / demo") from e

    def _get_chain(self, symbol: str, expiries: list[str] | None = None) -> OptionChain:
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("需要先安装 yfinance：pip install yfinance") from e

        tk = yf.Ticker(symbol)
        spot = None
        try:
            spot = float(tk.fast_info["last_price"])
        except Exception:
            hist = tk.history(period="5d")
            if not hist.empty:
                spot = float(hist["Close"].iloc[-1])
        if not spot:
            raise RuntimeError(f"拿不到 {symbol} 的股价，请检查代码或网络")

        all_exp = list(tk.options or [])
        if not all_exp:
            raise RuntimeError(f"{symbol} 没有期权数据")
        wanted = expiries or all_exp[: self.max_expiries]
        frames = []
        for exp in wanted:
            if exp not in all_exp:
                raise RuntimeError(f"到期日 {exp} 不存在，可选：{', '.join(all_exp[:12])} ...")
            oc = tk.option_chain(exp)
            for typ, part in (("call", oc.calls), ("put", oc.puts)):
                frames.append(pd.DataFrame({
                    "symbol": part["contractSymbol"], "type": typ, "strike": part["strike"],
                    "expiry": exp, "bid": part["bid"], "ask": part["ask"], "last": part["lastPrice"],
                    "volume": part["volume"], "open_interest": part["openInterest"],
                    "iv": part["impliedVolatility"],
                }))
        df = _normalize(pd.concat(frames, ignore_index=True))
        return OptionChain(symbol.upper(), spot, df, dt.date.today(), self.name)


class CSVSource:
    """读取 CSV。列名不区分大小写，支持常见别名；必须提供股价 spot。"""
    name = "csv"
    ALIASES = {
        "type": ["type", "right", "put/call", "cp", "option_type"],
        "strike": ["strike", "strike price"],
        "expiry": ["expiry", "expiration", "exp", "expiration date", "lasttradedate_or_contractmonth"],
        "bid": ["bid"], "ask": ["ask"], "last": ["last", "lastprice", "last price", "mark"],
        "volume": ["volume", "vol"],
        "open_interest": ["open_interest", "openinterest", "open interest", "oi"],
        "iv": ["iv", "impliedvolatility", "implied volatility", "implied vol"],
        "symbol": ["symbol", "contractsymbol", "contract"],
    }

    def __init__(self, path: str, spot: float, iv_in_percent: bool | None = None):
        self.path, self.spot, self.iv_in_percent = path, spot, iv_in_percent

    def get_chain(self, symbol: str, expiries: list[str] | None = None) -> OptionChain:
        raw = pd.read_csv(self.path)
        lower = {c.lower().strip(): c for c in raw.columns}
        df = pd.DataFrame()
        for std, names in self.ALIASES.items():
            for n in names:
                if n in lower:
                    df[std] = raw[lower[n]]
                    break
        missing = {"type", "strike", "expiry"} - set(df.columns)
        if missing:
            raise RuntimeError(f"CSV 缺少必要列：{missing}；现有列：{list(raw.columns)}")
        if "symbol" not in df:
            df["symbol"] = symbol.upper()
        df = _normalize(df)
        ivs = df["iv"].dropna()
        if self.iv_in_percent or (self.iv_in_percent is None and len(ivs) and ivs.median() > 3):
            df["iv"] = df["iv"] / 100.0
        if expiries:
            keep = {pd.to_datetime(e).date() for e in expiries}
            df = df[df["expiry"].isin(keep)].reset_index(drop=True)
        return OptionChain(symbol.upper(), float(self.spot), df, dt.date.today(), self.name)


class DemoSource:
    """模拟期权链：带左偏波动率微笑和期限结构，结果可复现。"""
    name = "demo"

    def __init__(self, spot: float = 100.0, base_iv: float = 0.30, r: float = 0.04,
                 asof: dt.date | None = None, seed: int = 7):
        self.spot, self.base_iv, self.r = spot, base_iv, r
        self.asof = asof or dt.date.today()
        self.seed = seed

    def iv_surface(self, strike, dte):
        T = np.maximum(dte, 1) / 365.0
        m = np.log(np.asarray(strike, float) / self.spot) / np.sqrt(T)
        term = self.base_iv * (0.9 + 0.1 * np.exp(-T * 4))  # 近月略高
        return np.clip(term - 0.10 * m + 0.08 * m**2, 0.05, 2.0)

    def get_chain(self, symbol: str = "DEMO", expiries: list[str] | None = None) -> OptionChain:
        rng = np.random.default_rng(self.seed)
        exps = ([pd.to_datetime(e).date() for e in expiries] if expiries else
                [self.asof + dt.timedelta(days=d) for d in (7, 14, 30, 45, 60, 90)])
        step = 1.0 if self.spot < 50 else 2.5 if self.spot < 200 else 5.0
        strikes = np.arange(np.floor(self.spot * 0.7 / step) * step, self.spot * 1.3 + step, step)
        rows = []
        for exp in exps:
            dte = (exp - self.asof).days
            T = max(dte, 1) / 365.0
            iv = self.iv_surface(strikes, dte)
            for typ in ("call", "put"):
                mid = bs_price(self.spot, strikes, T, self.r, iv, typ)
                spread = np.maximum(0.01, mid * 0.04) + 0.02
                bid = np.maximum(np.round(mid - spread / 2, 2), 0.0)
                ask = np.round(mid + spread / 2, 2)
                dist = np.abs(np.log(strikes / self.spot)) / (0.3 * np.sqrt(T) + 1e-9)
                oi = (5000 * np.exp(-dist**2) * rng.uniform(0.5, 1.5, len(strikes))
                      * (1.3 if typ == "put" else 1.0)).astype(int)
                vol = (oi * rng.uniform(0.05, 0.4, len(strikes))).astype(int)
                for k, b, a, m_, o, v in zip(strikes, bid, ask, mid, oi, vol):
                    rows.append({"symbol": f"{symbol}{exp:%y%m%d}{typ[0].upper()}{int(k*1000):08d}",
                                 "type": typ, "strike": k, "expiry": exp, "bid": b, "ask": a,
                                 "last": round(float(m_), 2), "volume": v, "open_interest": o,
                                 "iv": np.nan})
        df = _normalize(pd.DataFrame(rows))
        return OptionChain(symbol.upper(), self.spot, df, self.asof, self.name)
