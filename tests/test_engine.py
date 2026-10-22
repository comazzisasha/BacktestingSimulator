"""Unit tests for engine.py.

Run from the project folder with either:
    python -m pytest tests
    python tests/test_engine.py        (no pytest needed)
"""
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import engine  # noqa: E402

DEMO = engine.load_csv(os.path.join(HERE, "fixtures", "DEMO-STOCK.csv"), "DEMO-STOCK")
BOND = engine.load_csv(os.path.join(HERE, "fixtures", "DEMO-BOND.csv"), "DEMO-BOND")


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def run(code, ds=DEMO, **cfg):
    cfg.setdefault("slippage_bps", 0.0)
    return engine.backtest(engine.Strategy.from_code(code), ds, engine.Config(**cfg))


def load_templates():
    text = open(os.path.join(ROOT, "templates.py"), encoding="utf-8").read()
    parts = re.split(r"(?m)^#%% id: ", text)[1:]
    out = {}
    for part in parts:
        tid = part.split("\n", 1)[0].strip()
        code = "\n".join(l for l in part.split("\n")[1:] if not l.startswith("#%%"))
        out[tid] = code
    return out


# ---------------------------------------------------------------- data -----

def test_csv_parsing_nasdaq_format():
    assert DEMO.dates == sorted(DEMO.dates), "dates must be oldest first"
    assert DEMO.n > 2500
    assert all(c > 0 for c in DEMO.close)
    assert all(h >= max(o, c) - 1e-9 and l <= min(o, c) + 1e-9
               for o, h, l, c in zip(DEMO.open, DEMO.high, DEMO.low, DEMO.close))


def test_csv_parsing_handles_dollars_and_na():
    text = ("Date,Close/Last,Volume,Open,High,Low\n"
            "01/03/2024,$10.50,N/A,$10.00,$11.00,$9.90\n"
            "01/02/2024,$10.00,1000,$9.80,$10.10,$9.70\n")
    ds = engine.parse_csv_text(text, "X")
    assert ds.dates == ["2024-01-02", "2024-01-03"]
    assert ds.close == [10.0, 10.5]
    assert ds.volume == [1000.0, 0.0]


def test_series_view_blocks_future():
    v = engine.SeriesView([1, 2, 3, 4, 5], 3)
    assert v[-1] == 3 and v[0] == 1 and len(v) == 3
    assert v[-2:] == [2, 3]
    assert v[0:10] == [1, 2, 3]
    try:
        v[3]
        raise AssertionError("future bar should not be visible")
    except IndexError:
        pass


# ---------------------------------------------------------- indicators -----

def test_sma_and_ema():
    x = [float(i) for i in range(1, 11)]
    s = engine.sma(x, 3)
    assert all(math.isnan(v) for v in s[:2])
    assert s[2] == 2.0 and s[-1] == 9.0
    e = engine.ema(x, 3)
    assert e[2] == 2.0  # seeded with the SMA
    assert approx(e[3], 0.5 * 4 + 0.5 * 2.0)


def test_rsi_bounds_and_extremes():
    up = [float(i) for i in range(30)]
    assert engine.rsi(up, 14)[-1] == 100.0
    down = [float(30 - i) for i in range(30)]
    assert engine.rsi(down, 14)[-1] == 0.0
    r = engine.rsi(DEMO.close, 14)
    assert all(0 <= v <= 100 for v in r if not math.isnan(v))


def test_indicators_are_causal():
    """The value at bar k must not change when later bars are removed."""
    k = 1500
    c, h, l, vol = DEMO.close, DEMO.high, DEMO.low, DEMO.volume
    checks = {
        "sma": (lambda C, H, L, V: engine.sma(C, 50)),
        "ema": (lambda C, H, L, V: engine.ema(C, 20)),
        "rsi": (lambda C, H, L, V: engine.rsi(C, 14)),
        "macd": (lambda C, H, L, V: engine.macd(C)[1]),
        "bb": (lambda C, H, L, V: engine.bbands(C)[1]),
        "atr": (lambda C, H, L, V: engine.atr(H, L, C)),
        "adx": (lambda C, H, L, V: engine.adx(H, L, C)[0]),
        "stoch": (lambda C, H, L, V: engine.stochastic(H, L, C)[1]),
        "cci": (lambda C, H, L, V: engine.cci(H, L, C)),
        "vol": (lambda C, H, L, V: engine.realized_vol(C)),
        "z": (lambda C, H, L, V: engine.zscore(C, 20)),
        "obv": (lambda C, H, L, V: engine.obv(C, V)),
        "max": (lambda C, H, L, V: engine.rolling_max(H, 55)),
    }
    for name, fn in checks.items():
        full = fn(c, h, l, vol)[k]
        part = fn(c[:k + 1], h[:k + 1], l[:k + 1], vol[:k + 1])[k]
        assert approx(full, part), name


def test_indicators_match_pandas_if_available():
    try:
        import pandas as pd
    except ImportError:
        return
    s = pd.Series(DEMO.close)
    ref = s.rolling(20).mean().tolist()
    got = engine.sma(DEMO.close, 20)
    assert all(approx(a, b, 1e-9) for a, b in zip(got[19:], ref[19:]))
    ref_sd = s.rolling(20).std(ddof=0).tolist()
    got_sd = engine.rolling_std(DEMO.close, 20)
    assert all(abs(a - b) < 1e-6 for a, b in zip(got_sd[19:], ref_sd[19:]))
    mx = s.rolling(30).max().tolist()
    assert engine.rolling_max(DEMO.close, 30)[29:] == mx[29:]


# ------------------------------------------------------------ backtest -----

def test_buy_and_hold_matches_price_ratio():
    res = run("def on_bar(ctx):\n    return 1.0\n", fractional=True)
    # decision on bar 0, fill at bar 1 open, held to the last close
    expected = DEMO.close[-1] / DEMO.open[1]
    assert approx(res.equity[-1] / res.equity[0], expected, 1e-9)
    assert len(res.fills) == 1
    assert res.trades[-1]["open"] is True


def test_no_lookahead_fill_is_next_bar():
    res = run("def on_bar(ctx):\n    return 1.0 if ctx.bar == 10 else None\n")
    assert res.fills[0]["i"] == 11
    assert res.fills[0]["price"] == DEMO.open[11]
    res2 = run("def on_bar(ctx):\n    return 1.0 if ctx.bar == 10 else None\n", execution="close")
    assert res2.fills[0]["price"] == DEMO.close[11]


def test_original_weekday_bug_cannot_happen():
    """The notebook version created free shares. Every fill must now cost real money."""
    code = load_templates()["day_of_week"]
    res = run(code, cash=1000.0, fractional=False)
    for f in res.fills:
        assert f["price"] > 1.0, "fills must use the market price"
    assert res.equity[-1] < 1000.0 * 5, "a weekday rule cannot plausibly 5x"
    assert min(res.position) >= 0


def test_costs_reduce_returns():
    code = load_templates()["sma_cross"]
    free = run(code)
    costly = run(code, commission_bps=10.0, slippage_bps=10.0, commission_fixed=1.0)
    assert costly.equity[-1] < free.equity[-1]
    assert costly.commission > 0 and costly.slippage > 0


def test_cash_never_negative_without_leverage():
    code = load_templates()["vol_target"]
    res = run(code, commission_fixed=5.0, slippage_bps=5.0)
    # cash can only dip below zero by rounding
    acct_cash_min = min(res.equity[i] - res.position[i] * res.close[i] for i in range(len(res.equity)))
    assert acct_cash_min > -1e-6


def test_short_selling_flag():
    code = "def on_bar(ctx):\n    return -1.0\n"
    long_only = run(code)
    assert all(p == 0 for p in long_only.position)
    shorted = run(code, allow_short=True)
    assert min(shorted.position) < 0


def test_share_orders_are_clipped_to_cash():
    res = run("def on_bar(ctx):\n    ctx.order(1000)\n", cash=1000.0, fractional=False)
    assert res.clipped > 0
    assert res.fills[0]["qty"] * res.fills[0]["price"] <= 1000.0 + 1e-6


def test_trade_bookkeeping():
    res = run(load_templates()["rsi_reversion"])
    closed = [t for t in res.trades if not t["open"]]
    assert closed, "expected at least one closed trade"
    total = sum(t["pnl"] for t in res.trades)
    assert approx(total, res.equity[-1] - res.equity[0], 1e-6)


def test_signals_mode_matches_on_bar():
    t = load_templates()
    a = run(t["sma_cross"])
    b = run(t["sma_cross_vector"])
    assert approx(a.equity[-1], b.equity[-1], 1e-9)


def test_lookahead_check_flags_cheating():
    cheat = ("def signals(data):\n"
             "    c = data.close\n"
             "    return [1.0 if i + 1 < len(c) and c[i + 1] > c[i] else 0.0 for i in range(len(c))]\n")
    strat = engine.Strategy.from_code(cheat)
    full = strat.signals(engine.DataView(DEMO))
    chk = engine.lookahead_check(strat, DEMO, full)
    assert chk["violations"], "peeking at tomorrow's close must be detected"
    honest = engine.Strategy.from_code(load_templates()["sma_cross_vector"])
    full = honest.signals(engine.DataView(DEMO))
    assert not engine.lookahead_check(honest, DEMO, full)["violations"]


def test_all_templates_run():
    for tid, code in load_templates().items():
        res = run(code, allow_short=(tid == "zscore_ls"))
        m, b = engine.summarize(res)
        assert m["final_equity"] > 0, tid
        assert not math.isnan(m["cagr"]), tid


def test_user_errors_point_at_line():
    code = "PARAMS = {}\n\ndef on_bar(ctx):\n    x = ctx.close[-1]\n    return x / 0\n"
    try:
        run(code)
        raise AssertionError("expected a StrategyError")
    except engine.StrategyError as e:
        assert e.info["line"] == 5
        assert "ZeroDivisionError" in e.info["message"]
    try:
        engine.Strategy.from_code("def on_bar(ctx)\n    return 1\n")
        raise AssertionError("expected a syntax error")
    except engine.StrategyError as e:
        assert e.info["line"] == 1


# ------------------------------------------------------------- metrics -----

def test_metrics_against_direct_formulas():
    res = run("def on_bar(ctx):\n    return 1.0\n")
    m, b = engine.summarize(res)
    r = engine.daily_returns(res.equity)
    mu, sd = sum(r) / len(r), engine.stdev(r)
    assert approx(m["sharpe"], mu / sd * math.sqrt(252), 1e-9)
    assert approx(m["volatility"], sd * math.sqrt(252), 1e-9)
    peak, mdd = 0, 0
    for v in res.equity:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    assert approx(m["max_drawdown"], mdd)
    # vs itself (buy and hold vs buy and hold), beta ~ 1
    assert 0.95 < m["beta"] < 1.05
    assert m["correlation"] > 0.99


def test_benchmark_alignment_and_beta():
    strat = engine.Strategy.from_code("def on_bar(ctx):\n    return 1.0\n")
    res = engine.backtest(strat, DEMO, engine.Config(), bench=BOND)
    assert len(res.bench_equity) == len(res.equity)
    m, _ = engine.summarize(res)
    assert m["beta"] < 0.5


def test_monte_carlo_shapes():
    res = run("def on_bar(ctx):\n    return 1.0\n")
    mc = engine.monte_carlo(engine.daily_returns(res.equity), n_paths=50, block=10, seed=1)
    assert len(mc["fan"]["50"]) == len(mc["checkpoints"])
    assert mc["final"]["5"] <= mc["final"]["50"] <= mc["final"]["95"]
    assert 0 <= mc["prob_loss"] <= 1


def test_split_metrics_cover_whole_range():
    res = run(load_templates()["sma_cross"])
    sm = engine.split_metrics(res, len(res.dates) // 2)
    assert sm["in_sample"]["end"] == sm["out_of_sample"]["start"]
    total = (1 + sm["in_sample"]["metrics"]["total_return"]) * (1 + sm["out_of_sample"]["metrics"]["total_return"]) - 1
    m, _ = engine.summarize(res)
    assert approx(total, m["total_return"], 1e-9)


def test_bundled_data_parses():
    import json
    man = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
    for m in man["instruments"]:
        ds = engine.load_csv(os.path.join(ROOT, "data", m["file"]), m["ticker"])
        assert ds.n > 1000, m["ticker"]
        assert all(l <= min(o, c) + 1e-9 and h >= max(o, c) - 1e-9 for o, h, l, c in zip(ds.open, ds.high, ds.low, ds.close)), m["ticker"]


def test_json_api_roundtrip():
    import json
    engine.DATA.clear()
    for ds in (DEMO, BOND):
        payload = {"ticker": ds.ticker, "data": {"dates": ds.dates, "open": ds.open, "high": ds.high,
                                                  "low": ds.low, "close": ds.close, "volume": ds.volume}}
        out = json.loads(engine.handle("load_data", json.dumps(payload)))
        assert out["ok"], out
    out = json.loads(engine.handle("run", json.dumps({
        "ticker": "DEMO-STOCK", "benchmark": "DEMO-BOND", "code": load_templates()["sma_cross"],
        "config": {"cash": 10000}})))
    assert out["ok"], out
    r = out["result"]
    assert len(r["equity"]) == len(r["dates"]) == len(r["bench_equity"])
    bad = json.loads(engine.handle("run", json.dumps({"ticker": "DEMO-STOCK", "code": "def on_bar(ctx):\n  return 'x'\n"})))
    assert not bad["ok"] and "on_bar must return" in bad["error"]["message"]
    an = json.loads(engine.handle("analytics", json.dumps({"tickers": ["DEMO-STOCK", "DEMO-BOND"]})))
    assert an["ok"] and len(an["result"]["matrix"]) == 2


if __name__ == "__main__":
    failed = 0
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print("ok   ", name)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL ", name, "-", type(exc).__name__, exc)
    print("\n%d passed, %d failed" % (len(tests) - failed, failed))
    sys.exit(1 if failed else 0)
