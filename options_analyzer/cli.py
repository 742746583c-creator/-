"""命令行入口：python -m options_analyzer <命令> <股票代码> [选项]

命令：
  summary   一页总览：期限结构、预期波动、最大痛点、PCR、偏斜、持仓墙
  chain     查看某个到期日的期权链（含希腊值）
  screen    按 Delta / 到期天数 / 流动性筛选合约
  strategy  分析组合策略的盈亏（预设策略或自定义腿）
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import analysis as A
from . import strategy as S
from .data import CSVSource, DemoSource, YahooSource

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def _pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.1%}"


def _money(x):
    if x == float("inf"):
        return "无限"
    if x == float("-inf"):
        return "无限"
    return f"-${-x:,.2f}" if x < 0 else f"${x:,.2f}"


def load(args):
    if args.source == "demo":
        src = DemoSource(spot=args.spot or 100.0)
    elif args.source == "csv":
        if not args.csv or not args.spot:
            sys.exit("使用 CSV 数据源需要同时提供 --csv 文件路径 和 --spot 股价")
        src = CSVSource(args.csv, args.spot)
    else:
        src = YahooSource(max_expiries=args.max_expiries)
    expiries = [args.expiry] if getattr(args, "expiry", None) and args.source != "demo" else None
    chain = src.get_chain(args.symbol, expiries)
    params = A.MarketParams(r=args.rate, q=args.div)
    df = A.enrich(chain, params, recompute_iv=args.recompute_iv)
    return chain, df, params


def pick_expiry(df, args):
    if getattr(args, "expiry", None):
        target = pd.to_datetime(args.expiry).date()
        if target in set(df["expiry"]):
            return target
        print(f"（没有 {target} 这个到期日，改用最接近的）")
        return A.nearest_expiry(df, (target - df["expiry"].min()).days + int(df["dte"].min()))
    return A.nearest_expiry(df, args.dte)


def header(chain, df):
    print(f"\n=== {chain.symbol}  现价 ${chain.spot:,.2f}  数据源: {chain.source}  日期: {chain.asof} ===")
    if chain.source == "demo":
        print("（注意：这是模拟数据，仅用于学习和测试）")


def cmd_summary(args):
    chain, df, _ = load(args)
    header(chain, df)
    S0 = chain.spot

    print("\n【IV 期限结构 + 预期波动】")
    rows = []
    for exp in chain.expiries:
        em = A.expected_move(df, S0, exp)
        sk = A.skew(df, S0, exp)
        pcr = A.put_call_ratio(df, exp)
        mp, _ = A.max_pain(df, exp)
        rows.append({"到期日": exp, "天数": em["dte"], "平值IV": _pct(em["atm_iv"]),
                     "跨式价格": f"{em['straddle']:.2f}", "预期波动": f"±{em['straddle_pct']:.1%}",
                     "1σ区间": f"{em['range_low']:.2f} ~ {em['range_high']:.2f}",
                     "25Δ偏斜": _pct(sk["skew"]), "PCR(持仓)": f"{pcr['oi_pcr']:.2f}",
                     "最大痛点": f"{mp:g}"})
    print(pd.DataFrame(rows).to_string(index=False))

    exp = pick_expiry(df, args)
    walls = A.oi_walls(df, exp)
    total = A.put_call_ratio(df)
    print(f"\n【{exp} 持仓墙（未平仓量最多的行权价）】")
    print("  看涨（上方阻力参考）: " + ", ".join(f"{k:g}({oi:,})" for k, oi in walls["call"]))
    print("  看跌（下方支撑参考）: " + ", ".join(f"{k:g}({oi:,})" for k, oi in walls["put"]))
    print(f"\n【全部到期日合计】成交量 PCR {total['volume_pcr']:.2f}，持仓 PCR {total['oi_pcr']:.2f} "
          f"（看涨持仓 {total['call_oi']:,} / 看跌持仓 {total['put_oi']:,}）")
    print("\n解读提示：预期波动 = 平值跨式价格 ≈ 市场认为到期前的平均涨跌幅；"
          "25Δ偏斜为正表示下跌保护更贵；PCR > 1 表示看跌期权更多。")


CHAIN_COLS = ["strike", "bid", "ask", "mid", "iv", "delta", "gamma", "theta", "vega",
              "prob_itm", "volume", "open_interest"]


def cmd_chain(args):
    chain, df, _ = load(args)
    header(chain, df)
    exp = pick_expiry(df, args)
    sub = df[df["expiry"] == exp]
    sub = sub[(sub["moneyness"].abs() <= args.range)]
    print(f"\n到期日 {exp}（{int(sub['dte'].iloc[0])} 天）  平值 IV {_pct(A.atm_iv(df, chain.spot, exp))}")
    fmt = {"iv": "{:.1%}".format, "prob_itm": "{:.0%}".format, "delta": "{:+.3f}".format,
           "gamma": "{:.4f}".format, "theta": "{:+.3f}".format, "vega": "{:.3f}".format}
    for typ, name in (("call", "看涨 CALL"), ("put", "看跌 PUT")):
        if args.type and args.type != typ:
            continue
        t = sub[sub["type"] == typ][CHAIN_COLS]
        print(f"\n--- {name} ---")
        print(t.to_string(index=False, formatters=fmt))
    print("\n列说明：theta=每天时间价值损耗；vega=IV 每变 1% 的价格变化；prob_itm=到期实值概率（风险中性）")


def cmd_screen(args):
    chain, df, _ = load(args)
    header(chain, df)
    res = A.screen(df, option_type=args.type, min_dte=args.min_dte, max_dte=args.max_dte,
                   min_delta=args.min_delta, max_delta=args.max_delta, min_oi=args.min_oi,
                   min_volume=args.min_volume, max_spread_pct=args.max_spread,
                   sort_by=args.sort, ascending=args.asc, limit=args.limit)
    if res.empty:
        print("\n没有符合条件的合约，试着放宽条件。")
        return
    cols = ["expiry", "dte", "type", "strike", "mid", "iv", "delta", "theta", "prob_itm",
            "breakeven", "sell_yield_ann", "spread_pct", "open_interest", "volume"]
    fmt = {"iv": "{:.1%}".format, "prob_itm": "{:.0%}".format, "delta": "{:+.2f}".format,
           "theta": "{:+.3f}".format, "sell_yield_ann": "{:.1%}".format, "spread_pct": "{:.1%}".format,
           "breakeven": "{:.2f}".format}
    print(f"\n筛选结果（按 {args.sort} {'升序' if args.asc else '降序'}，共 {len(res)} 条）")
    print(res[cols].to_string(index=False, formatters=fmt))
    print("\nsell_yield_ann = 卖出该合约的年化收益（卖 put 按行权价占用资金，卖 call 按正股市值）")


def cmd_strategy(args):
    chain, df, params = load(args)
    header(chain, df)
    exp = pick_expiry(df, args)
    dte = int(df[df["expiry"] == exp]["dte"].iloc[0])
    if args.legs:
        text = args.legs
    elif args.preset:
        strikes = [float(x) for x in args.strikes.split(",")] if args.strikes else []
        text = S.preset_legs(args.preset, strikes)
    else:
        sys.exit("请用 --preset（预设策略）或 --legs（自定义腿）指定策略")
    legs = S.legs_from_chain(text, df, chain.spot, exp)
    res = S.analyze(legs, chain.spot, dte, params.r, params.q)

    title = S.PRESETS[args.preset][0] if args.preset and not args.legs else "自定义组合"
    print(f"\n【{title}】到期日 {exp}（{dte} 天）")
    for l in legs:
        print("  " + l.label())
    kind = "净支出" if res.net_premium > 0 else "净收入"
    print(f"\n  {kind}:       {_money(abs(res.net_premium))}")
    print(f"  最大盈利:     {_money(res.max_profit)}")
    print(f"  最大亏损:     {_money(res.max_loss)}")
    print(f"  盈亏平衡点:   {', '.join(f'{b:.2f}' for b in res.breakevens) or '无'}")
    print(f"  到期盈利概率: {res.pop:.1%}（基于 IV 的风险中性估计，仅供参考）")
    g = res.greeks
    print(f"  组合希腊值:   Delta {g['delta']:+.1f} 股 | Gamma {g['gamma']:+.2f} | "
          f"Theta {g['theta']:+.2f} $/天 | Vega {g['vega']:+.2f} $/1%IV")
    print("\n" + S.ascii_payoff(legs, chain.spot))
    t = res.table.copy()
    t[["到期盈亏", "当前理论盈亏"]] = t[["到期盈亏", "当前理论盈亏"]].round(0) + 0.0  # 去掉 -0
    print("\n" + t.to_string(index=False, formatters={"较现价": "{:+.1%}".format,
                                                      "到期盈亏": "{:,.0f}".format,
                                                      "当前理论盈亏": "{:,.0f}".format}))


def build_parser():
    p = argparse.ArgumentParser(prog="options_analyzer", description="美股期权分析工具",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("symbol", help="股票代码，如 AAPL、TSLA、SPY；演示模式随便填")
    common.add_argument("--source", choices=["yahoo", "csv", "demo"], default="yahoo", help="数据源（默认 yahoo）")
    common.add_argument("--csv", help="CSV 文件路径（--source csv 时使用）")
    common.add_argument("--spot", type=float, help="股价（csv 必填；demo 可选，默认 100）")
    common.add_argument("--rate", type=float, default=0.04, help="无风险利率，默认 0.04")
    common.add_argument("--div", type=float, default=0.0, help="股息率，默认 0")
    common.add_argument("--max-expiries", type=int, default=8, help="从 Yahoo 最多取几个到期日")
    common.add_argument("--recompute-iv", action="store_true", help="忽略数据源的 IV，用中间价重新计算")
    common.add_argument("--expiry", help="指定到期日 YYYY-MM-DD")
    common.add_argument("--dte", type=int, default=30, help="没指定到期日时，选最接近这个天数的（默认 30）")

    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary", parents=[common], help="一页总览").set_defaults(func=cmd_summary)

    c = sub.add_parser("chain", parents=[common], help="查看期权链")
    c.add_argument("--type", choices=["call", "put"])
    c.add_argument("--range", type=float, default=0.15, help="只显示现价 ±多少的行权价，默认 0.15")
    c.set_defaults(func=cmd_chain)

    s = sub.add_parser("screen", parents=[common], help="筛选合约")
    s.add_argument("--type", choices=["call", "put"])
    s.add_argument("--min-dte", type=int, default=0)
    s.add_argument("--max-dte", type=int, default=60)
    s.add_argument("--min-delta", type=float)
    s.add_argument("--max-delta", type=float)
    s.add_argument("--min-oi", type=int, default=100, help="最小未平仓量，默认 100")
    s.add_argument("--min-volume", type=int, default=0)
    s.add_argument("--max-spread", type=float, default=0.15, help="最大买卖价差比例，默认 0.15")
    s.add_argument("--sort", default="sell_yield_ann",
                   help="排序列：sell_yield_ann / iv / delta / theta / open_interest / volume ...")
    s.add_argument("--asc", action="store_true", help="升序")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_screen)

    st = sub.add_parser("strategy", parents=[common], help="组合策略盈亏分析")
    st.add_argument("--preset", choices=list(S.PRESETS), help="预设策略")
    st.add_argument("--strikes", help="预设策略的行权价，逗号分隔，如 95,100,110,115")
    st.add_argument("--legs", help='自定义腿，如 "+1C100,-1C110"（详见 strategy.py）')
    st.set_defaults(func=cmd_strategy)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (RuntimeError, ValueError) as e:
        sys.exit(f"出错了：{e}")


if __name__ == "__main__":
    main()
