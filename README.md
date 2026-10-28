# QuantLab

A browser-based backtester for Python trading strategies on daily price data from Nasdaq.com.
Pick an instrument, write or pick a strategy, and get a full set of statistics,
charts and research tools. Everything runs in the browser through Pyodide; there is no server
and no API calls.

## Run it

```bash
cd quantlab
python -m http.server 8000
# open http://localhost:8000
```

Opening `index.html` straight from disk won't work, because browsers block it from reading `engine.py`.
To put it online, push this folder to GitHub and turn on GitHub Pages.

## Add data

On nasdaq.com open a ticker's **Historical Quotes** page (e.g. `nasdaq.com/market-activity/etf/spy/historical`),
choose **10Y**, download, and drop the CSV on the **Data** tab. To bundle files with the site, copy them
into `data/` and add them to `data/manifest.json`:

```json
{"instruments": [{"ticker": "SPY", "name": "SPDR S&P 500 ETF", "file": "SPY.csv"}]}
```

## Write a strategy

```python
PARAMS = {"fast": 50, "slow": 200}

def on_bar(ctx):
    if isnan(ctx.sma(PARAMS["slow"])):
        return 0.0
    return 1.0 if ctx.sma(PARAMS["fast"]) > ctx.sma(PARAMS["slow"]) else 0.0
```

`on_bar` runs at each day's close and returns the share of equity to hold (1.0 = all in,
0 = cash, -1 = short, None = no change). The trade fills at the next day's open.
Full reference: `docs.html` (also linked from the app).

## Portfolios, dividends and carry

A strategy that sets `UNIVERSE = ["SPY", "TLT", "GLD"]` runs as a portfolio: `on_bar(ctx)` returns
`{ticker: weight}` and `ctx["SPY"]` has every indicator. Dividends are paid on ex-dates (from an
Adj Close column or a dividend file dropped on the Data tab), idle cash earns interest, leverage and
shorts pay financing and borrow fees, and margin calls cut positions. Every instrument gets a
data-quality check against the NYSE calendar when it loads.

## History, exports and audit

Every run is saved in the browser with its code hash, parameters, settings, data fingerprints and
engine version (History tab: compare, restore, re-run to check it reproduces). Export CSVs, the run as
JSON, or a printable tear sheet; back up or move the whole workspace as one file (Data tab).

## Risk and capacity

The Research Tools tab has a **Risk Report** (multi-factor regression on loaded tickers, spreads such as
`IWM-SPY` or uploaded Fama-French files, with Newey-West t-statistics and return/variance attribution;
rolling beta; each holding's share of the risk; ten historical stress windows replayed on the final
holdings; historical, normal and Cornish-Fisher VaR with a Kupiec back-test) and a **Capacity** curve
that re-runs the strategy at growing account sizes with square-root impact and participation limits.
An optional **Options Overlay** tool adds covered calls, protective puts or a collar to a strategy's long
position and shows a delta-hedged option's volatility P&L. Option prices are modelled (Black-Scholes on
trailing realized volatility or a loaded volatility index), not market quotes.
**Start/End Surfaces** score every window: S(s, e), the strategy backtested from s to e, and B(s, e),
the benchmark over the same window, for a chosen statistic (return, CAGR, Sharpe, drawdown …), as
heatmaps or 3D surfaces. **Strategy Ranking** backtests every built-in strategy on your market and settings,
ranks them by a chosen statistic, and deflates the winner's Sharpe ratio for the number tried. **Every Instrument** runs the
strategy in the editor on every loaded instrument against each one's buy & hold.

## Options

Strategies can trade listed options from end-of-day option chains you import (CBOE DataShop, OptionsDX,
ORATS, OptionMetrics or any date/expiration/strike/call-put/bid/ask file; QuantLab ships no option data).
`ctx.options.find("put", dte=30, delta=0.25)` picks a contract, `buy`/`sell` fill at the next day's ask or
bid plus commission, positions are marked at the mid and cash-settle at intrinsic value on expiry.

## Stock dashboard

The **Charts** tab shows one instrument with moving averages, Bollinger, Keltner and Donchian channels
(line or candles) above panes for volume, RSI, MACD, stochastics, ADX, ATR, OBV, Williams %R, CCI, rate
of change and volatility, with a shared crosshair, the latest readings, and a table of classic patterns
(golden crosses, MACD and RSI signals, band breaks, 52-week highs …) with the returns that followed them.

## Paper trading

Every run shows the orders the strategy would send for the next session. The **Paper** tab freezes a
strategy and start date and runs it forward as new data arrives, logging each update and flagging any
change to past results. Routing orders to a broker is a separate, optional server you run yourself:
`tools/bridge/broker_bridge.py` (dry run by default; Alpaca paper or live with flags).

## Sharing and review

Runs can be shared as a link (code, settings, results and comments in the URL fragment; the receiver
re-runs and checks it reproduces), carry comment threads and an auto-generated model card, and be
approved or rejected into a SHA-256 hash-chained audit log that `engine.verify_audit_log` can check.

## Modules, editor and tests

Shared code lives in **modules**: Python files in the browser's module library that any strategy
imports (`from my_signals import trend`). The editor has tabs with unsaved markers, pyflakes problems
(plus checks for `ctx` attributes that don't exist), completions for `ctx.`, `qt.`, `data.` and your
modules, go-to-definition (F12), and find & replace. Functions named `test_...` are unit tests, run
from the full editor with helpers such as `run_strategy(make_data([...]))`; the Console there runs
Python next to your data and latest result. Optional **GitHub Sync** (Data tab, `github-sync.js`)
pushes and pulls strategies and modules as `.py` files using your own token; nothing else leaves the
browser.

Sweeps and walk-forward grids run on several Python workers at once, unchanged reruns are shown from
a cache, and `signals()` output is reused when only settings change.

## Machine learning

Strategies can `import numpy`, `pandas`, `scipy` and `sklearn`; they load on first use
(NumPy is bundled in `pyodide/`, the rest come from the Pyodide CDN). `walk_forward(X, y, fit_predict, ...)`
trains and retrains a model on past data only. See the *Machine Learning* templates.

For neural networks, the built-in `nn` module (`lib/nn.py`, plain NumPy) does the boilerplate:

```python
import nn
LAYERS = [nn.Dense(32, "relu"), nn.Dropout(0.1), nn.Dense(16, "relu"), nn.Dense(1, "sigmoid")]

def signals(data):
    return nn.signals(data, LAYERS)   # features, labels, walk-forward training, early stopping, positions, report
```

It has Dense, Dropout and LSTM layers, Adam, early stopping and ensembles. To change how any of it
works, choose **Modules → Edit a Copy** next to `nn.py`: your copy replaces the built-in one.
PyTorch can't run in a browser, but works when you run the engine from Python as below.

## Use the engine from Python

```python
import engine
ds = engine.load_csv("SPY.csv", "SPY")
res = engine.backtest(engine.Strategy.from_code(open("strategy.py").read()), ds)
stats, bench = engine.summarize(res)

# portfolios
sets = [engine.load_csv(f"{t}.csv", t) for t in ("SPY", "TLT", "GLD")]
res = engine.backtest_portfolio(engine.Strategy.from_code(code), sets, engine.Config(cash_rate=0.04))
print(engine.attribution(res)["rows"])
```

## Files

| Path | What it is |
|---|---|
| `engine.py` | Indicators, backtest loop, statistics, research tools, JSON API (pure Python, no dependencies) |
| `templates.py` | 36 built-in strategies, including 5 portfolio, 3 options and 6 machine-learning ones |
| `lib/` | Built-in modules strategies can import: `nn.py` (neural networks) |
| `index.html` | The web app |
| `runtime.js`, `worker.js` | Start Pyodide in a background Web Worker (plus helper workers for sweeps) |
| `github-sync.js` | Optional GitHub Sync, loaded only when used |
| `docs.html` | Documentation (rebuild with `python tools/build_docs.py`) |
| `pyodide/`, `vendor/` | Bundled Pyodide 0.26.4 runtime, Chart.js 4.4.1 and pyflakes 3.2.0 (MIT), so it works offline |
| `data/` | Bundled Nasdaq CSVs for 23 instruments (10 years, daily); `tests/fixtures/` holds two synthetic series for tests |
| `tests/test_engine.py` | Unit tests: `python tests/test_engine.py` |
| `tools/bridge/` | Optional broker bridge (separate server); tests in `tests/test_bridge.py` |
