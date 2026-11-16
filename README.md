# 美股期权分析工具 📈

一个用 Python 写的命令行工具，用来**分析**美股期权（不会下单）：

- **期权链 + 希腊值**：Delta、Gamma、Theta、Vega、到期实值概率
- **隐含波动率（IV）**：数据源没给 IV 时，会用买卖中间价自己算
- **一页总览**：IV 期限结构、预期波动幅度、最大痛点、Put/Call 比率、25Δ 偏斜、持仓墙
- **合约筛选**：按 Delta、到期天数、未平仓量、买卖价差筛选，可以按卖方年化收益排序
- **策略分析**：备兑、卖 put、垂直价差、跨式、铁鹰、蝶式，也可以自定义组合。输出最大盈亏、盈亏平衡点、盈利概率、组合希腊值，还会在终端画出盈亏图

## 安装

```bash
git clone <你的仓库地址>
cd <仓库目录>
python -m venv .venv && source .venv/bin/activate   # Windows：.venv\Scripts\activate
pip install -r requirements.txt
```

## 快速上手

先用模拟数据试一试（不需要联网）：

```bash
python -m options_analyzer summary DEMO --source demo
```

看真实数据（Yahoo 财经，免费，有约 15 分钟延迟）：

```bash
python -m options_analyzer summary AAPL                 # 一页总览
python -m options_analyzer chain AAPL --dte 30          # 最接近 30 天的期权链
python -m options_analyzer chain AAPL --expiry 2026-12-18 --type put --range 0.1
```

### 筛选：比如找适合卖 put 的合约

```bash
# 7~45 天到期，Delta 在 0.15~0.30 之间，未平仓量至少 500，按年化收益从高到低排
python -m options_analyzer screen TSLA --type put --min-dte 7 --max-dte 45 \
    --min-delta 0.15 --max-delta 0.30 --min-oi 500
```

### 策略分析

```bash
# 预设策略（行权价会自动对齐到最近的可交易行权价）
python -m options_analyzer strategy SPY --preset iron-condor --strikes 560,570,610,620 --dte 30
python -m options_analyzer strategy AAPL --preset csp --strikes 220
python -m options_analyzer strategy NVDA --preset bull-call --strikes 180,200 --expiry 2027-01-15

# 自定义组合：+ 是买，- 是卖，C 是看涨，P 是看跌，S 是 100 股正股
python -m options_analyzer strategy AAPL --legs "+1S,-1C240"         # 备兑看涨
python -m options_analyzer strategy AAPL --legs "-1P220,+1P210,-1C250,+1C260"
```

可用的预设策略：`long-call` `long-put` `covered-call` `csp` `bull-call` `bear-put`
`bull-put` `bear-call` `straddle` `strangle` `iron-condor` `butterfly`

### 用券商导出的 CSV

从 IBKR、富途、老虎等平台导出期权链 CSV。至少要有**类型（Call/Put）、行权价、到期日**三列，最好再有 bid、ask、未平仓量。

```bash
python -m options_analyzer summary AAPL --source csv --csv my_chain.csv --spot 231.5
```

程序能认出常见的列名，比如 `Right` `Strike` `Expiration` `Open Interest` `Implied Volatility`。

## 常用参数

| 参数 | 说明 |
|---|---|
| `--expiry YYYY-MM-DD` | 指定到期日 |
| `--dte 30` | 没指定到期日时，选最接近这个天数的 |
| `--rate 0.04` | 无风险利率，可参考 3 个月美债收益率 |
| `--div 0.005` | 股息率 |
| `--recompute-iv` | 不用数据源给的 IV，自己从中间价重新算 |

## 指标怎么看

- **预期波动**：平值跨式（同时买平值 call 和 put）的价格，大约就是市场预期到期前的平均涨跌幅度。财报前通常会明显变大。
- **IV 期限结构**：一般远月 IV 比近月高。如果近月更高，说明市场预期近期有大事，比如财报。
- **25Δ 偏斜**：看跌 IV 减看涨 IV。正值越大，说明市场越愿意花钱买下跌保护。
- **PCR**：看跌和看涨的比值。很高时可能代表恐慌，也可能只是机构在对冲，要结合其他指标一起看。
- **最大痛点 / 持仓墙**：只是参考，并不能可靠地预测到期价格。
- **prob_itm / 盈利概率**：按 IV 算出来的“风险中性”概率，不是真实概率，但适合用来比较不同合约。

## 项目结构

```
options_analyzer/
  pricing.py    Black-Scholes 定价、希腊值、隐含波动率
  data.py       数据源：Yahoo / CSV / 演示数据
  analysis.py   期权链分析：预期波动、最大痛点、PCR、偏斜、筛选
  strategy.py   组合策略盈亏分析和终端盈亏图
  cli.py        命令行入口
tests/          单元测试（运行：python -m pytest）
```

## 局限和提醒

- 用的是欧式 BS 模型。美股个股期权是美式的，深度实值 put 的价格会有一点偏差。
- Yahoo 的数据是延迟的，偶尔会有缺失或异常的报价（比如 bid=0）。
- 本工具只做分析和学习，**不构成投资建议**。期权有杠杆，卖方可能面临很大亏损。
