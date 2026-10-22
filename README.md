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

## Use the engine from Python

```python
import engine
ds = engine.load_csv("SPY.csv", "SPY")
res = engine.backtest(engine.Strategy.from_code(open("strategy.py").read()), ds)
stats, bench = engine.summarize(res)
```

## Files

| Path | What it is |
|---|---|
| `engine.py` | Indicators, backtest loop, statistics, research tools, JSON API (pure Python, no dependencies) |
| `templates.py` | 20 built-in strategies |
| `index.html` | The web app |
| `runtime.js`, `worker.js` | Start Pyodide in a background Web Worker |
| `docs.html` | Documentation (rebuild with `python tools/build_docs.py`) |
| `pyodide/`, `vendor/` | Bundled Pyodide 0.26.4 runtime and Chart.js 4.4.1, so it works offline |
| `data/` | Bundled Nasdaq CSVs for 23 instruments (10 years, daily); `tests/fixtures/` holds two synthetic series for tests |
| `tests/test_engine.py` | Unit tests: `python tests/test_engine.py` |
