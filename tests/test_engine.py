"""Unit tests for engine.py.

Run from the project folder with either:
    python -m pytest tests
    python tests/test_engine.py        (no pytest needed)
"""
import json
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


def missing_imports(code):
    """Top-level packages a template imports that aren't installed (e.g. numpy)."""
    import importlib.util
    mods = re.findall(r"(?m)^(?:import|from) (\w+)", code)
    return [m for m in mods if importlib.util.find_spec(m) is None]


def test_walk_forward_never_trains_on_unknown_labels():
    n, h = 300, 5
    X = [[float(i)] for i in range(n)]          # feature = the bar's own index
    y = [float(i % 3) if i + h < n else engine.NAN for i in range(n)]
    seen = []

    def fit_predict(X_train, y_train, X_new):
        newest_label = max(r[0] for r in X_train)
        first_new = min(r[0] for r in X_new)
        assert newest_label + h <= first_new, "trained on a label from the future"
        seen.append((len(X_train), first_new))
        return [first_new] * len(X_new)

    out = engine.walk_forward(X, y, fit_predict, min_train=50, retrain_every=20, horizon=h, window=100)
    # labels 0..49 are all known at bar 54 (49 + horizon), so the first model starts there
    assert all(math.isnan(v) for v in out[:54]) and out[54] == 54
    assert out[73] == 54 and out[74] == 74, "predictions come from the latest model"
    assert all(size <= 100 for size, _ in seen), "window caps the training set"
    X[60] = [engine.NAN]
    assert math.isnan(engine.walk_forward(X, y, fit_predict, min_train=50, retrain_every=20, horizon=h)[60])


def test_ml_templates_are_causal():
    t = load_templates()
    for tid in ("ml_logistic", "ml_mlp", "ml_forest"):
        if missing_imports(t[tid]):
            print("skip  %s (needs %s)" % (tid, ", ".join(missing_imports(t[tid]))))
            continue
        strat = engine.Strategy.from_code(t[tid])
        res = engine.backtest(strat, DEMO, engine.Config())
        assert 0 < engine.summarize(res)[0]["exposure"] < 1, tid
        chk = engine.lookahead_check(strat, DEMO, res.signals, samples=3)
        assert chk["checked"] == 3 and not chk["violations"], tid


def test_all_templates_run():
    for tid, code in load_templates().items():
        if tid.startswith("pf_"):
            continue                    # portfolio templates: test_portfolio_templates_run_and_are_causal
        if missing_imports(code):
            print("skip  %s (needs %s)" % (tid, ", ".join(missing_imports(code))))
            continue
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


def test_metric_tiers_and_coverage():
    keys = [m[0] for m in engine.METRICS]
    assert len(keys) == len(set(keys)), "metric keys must be unique"
    assert all(m[4] in ("core", "std", "extra") for m in engine.METRICS)
    res = engine.backtest(engine.Strategy.from_code(load_templates()["sma_cross"]), DEMO,
                          engine.Config(slippage_bps=0.0), bench=BOND)
    m, _ = engine.summarize(res)
    missing = [k for k in keys if k not in m]
    assert not missing, "every metric should be computed for a strategy with trades: %s" % missing


def test_new_metrics_against_direct_formulas():
    res = run(load_templates()["rsi_reversion"])
    m, _ = engine.summarize(res)
    r = engine.daily_returns(res.equity)
    n, mu, sd = len(r), sum(r) / len(r), engine.stdev(r)
    assert approx(m["t_stat"], mu / sd * math.sqrt(n))
    assert 0 <= m["p_value"] <= 1 and 0 <= m["psr"] <= 1 and 0 <= m["jb_p"] <= 1
    assert m["sharpe_lo"] < m["sharpe"] < m["sharpe_hi"]
    assert approx(m["var_99"], -engine.percentile(r, 1))
    assert m["cvar_99"] >= m["var_99"] >= m["var_95"] - 1e-12
    assert approx(m["kelly"], mu / sd ** 2)
    assert approx(m["recovery_factor"], m["total_return"] / abs(m["max_drawdown"]))
    dd = engine.drawdown(res.equity)
    assert approx(m["pain_index"], -sum(dd) / len(dd))
    assert 0 <= m["stability"] <= 1
    assert approx(m["payoff_ratio"], m["avg_win"] / abs(m["avg_loss"]))
    closed = [t["ret"] for t in res.trades if not t["open"]]
    assert approx(m["sqn"], math.sqrt(len(closed)) * (sum(closed) / len(closed)) / engine.stdev(closed))
    assert approx(m["trades_per_year"], m["n_trades"] / m["years"])
    # a strategy identical to its benchmark beats it on no day and has no alpha to test
    bh, _ = engine.summarize(run("def on_bar(ctx):\n    return 1.0\n"))
    assert bh["beat_days"] < 0.6


def test_return_tests_detect_signal_and_noise():
    rng = random.Random(3)
    noise = [rng.gauss(0, 0.01) for _ in range(2000)]
    drift = [x + 0.002 for x in noise]
    tn, td = engine.return_tests(noise), engine.return_tests(drift)
    assert td["p_value"] < 0.001 < tn["p_value"]
    assert td["psr"] > 0.999
    assert tn["jb_p"] > 0.01, "gaussian returns should pass the normality test"
    assert tn["autocorr_p"] > 0.01
    trending = [0.01 if i % 2 else -0.01 for i in range(500)]
    assert engine.return_tests(trending)["autocorr"] < -0.9


def test_subperiods_and_costs():
    code = load_templates()["sma_cross"]
    res = run(code)
    sp = engine.subperiod_metrics(res, 4)
    assert sp["n"] == 4 and len(sp["periods"]) == 4
    growth = 1.0
    for p in sp["periods"]:
        growth *= 1 + p["metrics"]["total_return"]
    assert approx(growth - 1, engine.summarize(res)[0]["total_return"], 1e-9)
    cs = engine.cost_sensitivity(lambda: engine.Strategy.from_code(code), DEMO, engine.Config(), [20, 0, 5])
    cagrs = [row["cagr"] for row in cs["rows"]]
    assert [row["bps"] for row in cs["rows"]] == [0, 5, 20]
    assert cagrs == sorted(cagrs, reverse=True), "higher costs can't raise returns"
    assert cs["rows"][0]["costs"] == 0


def test_permutation_test():
    # The weight set at bar i fills at bar i + 1's close and earns bar i + 2's return.
    cheat = ("def signals(data):\n"
             "    c = data.close\n"
             "    return [1.0 if i + 2 < len(c) and c[i + 2] > c[i + 1] else 0.0 for i in range(len(c))]\n")
    res = run(cheat, execution="close")
    pt = engine.permutation_test(res, 200, seed=1)
    assert pt["p_value"] < 0.01, "perfect foresight must beat random timing"
    bh = engine.permutation_test(run("def on_bar(ctx):\n    return 1.0\n"), 50)
    assert bh["constant"]


def synthetic(ticker, closes, start="2020-01-02", calendar="NYSE", **cols):
    """A DataSet on real exchange days with the given closes (open = previous close)."""
    days = engine.session_days(start, "2035-12-31", calendar)[:len(closes)]
    opens = cols.pop("opens", [closes[0]] + closes[:-1])
    return engine.DataSet(ticker, days, closes, opens, None, None, cols.pop("volume", [1e6] * len(closes)), **cols)


def bundled(ticker):
    return engine.load_csv(os.path.join(ROOT, "data", ticker + ".csv"), ticker)


# ------------------------------------------------- portfolio engine -----

def test_single_asset_results_unchanged_by_portfolio_engine():
    golden = json.load(open(os.path.join(HERE, "fixtures", "golden_single_asset.json")))
    configs = {"default": {}, "costs": {"commission_bps": 5, "commission_fixed": 1.0, "slippage_bps": 10},
               "short_lev2": {"allow_short": True, "max_leverage": 2.0}, "whole_shares": {"fractional": False, "cash": 3000},
               "close_fill": {"execution": "close", "rebalance_band": 0.0}}
    t = load_templates()
    checked = 0
    for key, want in golden.items():
        if key.startswith("_"):
            continue
        tid, cn = key.split("|")
        cfg = dict(configs[cn], maintenance_margin=0.0)   # margin calls didn't exist before
        res = engine.backtest(engine.Strategy.from_code(t[tid]), DEMO, engine.Config(**cfg), bench=BOND)
        assert abs(res.equity[-1] - want["final"]) < 1e-5, (key, res.equity[-1], want["final"])
        assert (len(res.fills), len(res.trades), res.clipped) == (want["fills"], want["trades"], want["clipped"]), key
        assert abs(sum(tr["pnl"] for tr in res.trades) - want["trade_pnl"]) < 1e-5, key
        checked += 1
    assert checked == 105


def test_one_instrument_portfolio_equals_single_backtest():
    single = load_templates()["sma_cross"]
    port = ("def on_bar(ctx):\n    a = ctx['DEMO-STOCK']\n    slow = a.sma(200)\n"
            "    if isnan(slow):\n        return {}\n    return {'DEMO-STOCK': 1.0 if a.sma(50) > slow else 0.0}\n")
    a = run(single)
    b = engine.backtest_portfolio(engine.Strategy.from_code(port), [DEMO], engine.Config(slippage_bps=0.0), bench=DEMO)
    assert a.equity == b.equity and len(a.fills) == len(b.fills) and len(a.trades) == len(b.trades)
    # weights given as a list in ticker order work too
    code = "UNIVERSE = ['DEMO-STOCK']\ndef on_bar(ctx):\n    return [1.0]\n"
    res = engine.backtest_portfolio(engine.Strategy.from_code(code), [DEMO], engine.Config(slippage_bps=0.0), bench=DEMO)
    bh = run("def on_bar(ctx):\n    return 1.0\n")
    assert approx(res.equity[-1], bh.equity[-1], 1e-12)


def test_daily_rebalanced_portfolio_matches_numpy_reference():
    try:
        import numpy as np
    except ImportError:
        print("skip  numpy reference (needs numpy)")
        return
    sets = [bundled("SPY"), bundled("TLT"), bundled("GLD")]     # same calendar, so every day rebalances
    code = "def signals(data):\n    k = len(data.tickers)\n    return [{t: 1 / k for t in data.tickers} for _ in data.dates]\n"
    cfg = engine.Config(slippage_bps=0.0, rebalance_band=0.0, execution="close", maintenance_margin=0.0)
    res = engine.backtest_portfolio(engine.Strategy.from_code(code), sets, cfg)
    closes = np.array([[engine.Panel(sets).mark(a, t) for a in range(3)] for t in range(len(res.dates))])
    r = closes[1:] / closes[:-1] - 1
    # weights decided at close t fill at close t+1, so the equal-weight return starts on day 2
    ref = np.concatenate([[1.0, 1.0], np.cumprod(1 + r[1:].mean(axis=1))]) * cfg.cash
    assert np.max(np.abs(np.array(res.equity) - ref) / ref) < 1e-10


def test_unified_calendar_listing_and_missing_days():
    a = synthetic("AAA", [100.0 + i for i in range(60)])
    late = synthetic("NEW", [50.0] * 40, start=a.dates[20])            # starts trading 20 bars later
    gappy_days = [d for i, d in enumerate(a.dates) if i not in (30, 31)]
    gappy = engine.DataSet("GAP", gappy_days, [10.0 + 0.1 * i for i in range(len(gappy_days))])
    code = "def on_bar(ctx):\n    return {'AAA': 0.3, 'NEW': 0.3, 'GAP': 0.3}\n"
    res = engine.backtest_portfolio(engine.Strategy.from_code(code), [a, late, gappy], engine.Config(slippage_bps=0.0))
    assert res.dates == a.dates, "the calendar is the union of trading days"
    first_new = min(f["i"] for f in res.fills if f["ticker"] == "NEW")
    assert res.dates[first_new] == late.dates[0], "no trade before an instrument lists"
    assert all(res.weights["NEW"][t] == 0 for t in range(20))
    assert not any(f["ticker"] == "GAP" and f["i"] in (30, 31) for f in res.fills), "no trades on missing days"
    assert abs(res.weights["GAP"][31] - res.weights["GAP"][29]) < 0.02, "a missing day is valued at the last price"
    view = engine.PanelView(res.panel)
    assert math.isnan(view["NEW"].close[0]) and view["GAP"].close[30] == view["GAP"].close[29]
    assert view.listed["GAP"][30] is False and view.listed["GAP"][29] is True


def test_portfolio_weights_are_scaled_to_the_leverage_limit():
    code = "def on_bar(ctx):\n    return {'DEMO-STOCK': 1.0, 'DEMO-BOND': 1.0}\n"
    res = engine.backtest_portfolio(engine.Strategy.from_code(code), [DEMO, BOND], engine.Config(slippage_bps=0.0))
    assert max(res.gross) < 1.06 and any("scaled down" in n for n in res.notes)
    assert min(res.equity[1:]) > 0 and min(e - sum(res.weights[t][i] for t in res.tickers) * e
                                          for i, e in enumerate(res.equity)) > -1e-6, "no borrowing at 1x"
    lev = engine.backtest_portfolio(engine.Strategy.from_code(code), [DEMO, BOND],
                                    engine.Config(slippage_bps=0.0, max_leverage=2.0))
    assert 1.8 < max(lev.gross) < 2.2


def test_attribution_adds_up_with_carry():
    div = [(d, 0.5) for d in DEMO.dates[60::63]]
    stock = engine.DataSet("DIVCO", DEMO.dates, DEMO.close, DEMO.open, DEMO.high, DEMO.low, DEMO.volume, dividends=div)
    code = "def on_bar(ctx):\n    return {'DIVCO': 0.8, 'DEMO-BOND': -0.4}\n"
    cfg = engine.Config(allow_short=True, max_leverage=1.5, cash_rate=0.03, borrow_fee=0.02, commission_bps=2)
    res = engine.backtest_portfolio(engine.Strategy.from_code(code), [stock, BOND], cfg)
    att = engine.attribution(res)
    assert abs(att["check"]) < 1e-6, att["check"]
    assert res.carry["dividends_received"] > 0 and res.carry["borrow_paid"] > 0 and res.carry["interest_earned"] > 0
    row = {r["ticker"]: r for r in att["rows"]}
    assert row["DIVCO"]["dividends"] > 0 and row["DEMO-BOND"]["costs"] > 0
    m, _ = engine.summarize(res)
    assert approx(m["avg_net"], 0.4, 0.05) and approx(m["avg_gross"], 1.2, 0.05) and m["avg_short"] > 0.35


# ----------------------------------------------------- execution -----

def bars(rows, volume=1e9, start="2021-03-01"):
    """DataSet from (open, high, low, close) rows on consecutive NYSE days."""
    days = engine.session_days(start, "2030-12-31")[:len(rows)]
    o, h, l, c = zip(*rows)
    return engine.DataSet("BAR", days, list(c), list(o), list(h), list(l), [volume] * len(rows))


def run_bars(code, rows, **cfg):
    cfg.setdefault("slippage_bps", 0.0)
    cfg.setdefault("maintenance_margin", 0.0)
    return engine.backtest(engine.Strategy.from_code(code), bars(rows, cfg.pop("volume", 1e9)), engine.Config(**cfg))


FLAT = (100, 100, 100, 100)


def test_limit_orders_fill_against_the_bar_range():
    entry = "def on_bar(ctx):\n    if ctx.bar == 0:\n        ctx.order(10, limit=%s)\n"
    res = run_bars(entry % 99.5, [FLAT, (101, 102, 99, 100), FLAT])
    assert [(f["price"], f["kind"]) for f in res.fills] == [(99.5, "limit")]
    gap = run_bars(entry % 99.5, [FLAT, (98, 99, 97, 98), FLAT])
    assert gap.fills[0]["price"] == 98, "a gap through the limit fills at the open"
    touch = run_bars(entry % 99, [FLAT, (101, 102, 99, 100), FLAT])
    assert touch.fills and touch.fills[0]["price"] == 99
    strict = run_bars(entry % 99, [FLAT, (101, 102, 99, 100), FLAT], fill_model="pessimistic")
    assert not strict.fills, "pessimistic: touching the limit isn't enough"
    missed = run_bars(entry % 95, [FLAT, (101, 102, 99, 100), (100, 100, 90, 92)])
    assert not missed.fills and missed.book_stats["expired"] == 1, "a day order lives for one bar"
    gtc = run_bars((entry % 95).replace(")\n", ", tif='gtc')\n"), [FLAT, (101, 102, 99, 100), (96, 97, 94, 95)])
    assert gtc.fills[0]["price"] == 95 and gtc.fills[0]["i"] == 2
    costs = run_bars(entry % 99.5, [FLAT, (101, 102, 99, 100), FLAT], slippage_bps=10, spread_bps=20, commission_bps=5)
    f = costs.fills[0]
    assert f["price"] == 99.5 and approx(f["commission"], 10 * 99.5 * 5e-4), "limits pay commission only"


def test_stop_and_stop_limit_orders():
    exit_ = ("def on_bar(ctx):\n    if ctx.bar == 0:\n        return 1.0\n"
             "    if ctx.bar == 1:\n        ctx.order_target_shares(0, stop=95, tif='gtc')\n")
    res = run_bars(exit_, [FLAT, FLAT, FLAT, (90, 91, 88, 89), FLAT], slippage_bps=10)
    sell = res.fills[-1]
    assert sell["kind"] == "stop" and approx(sell["price"], 90 * (1 - 0.001)), "a stop gapped through fills at the open, as a market order"
    sl = "def on_bar(ctx):\n    if ctx.bar == 0:\n        ctx.order(10, stop=105, limit=106)\n"
    hit = run_bars(sl, [FLAT, (104, 108, 103, 107), FLAT])
    assert hit.fills[0]["price"] == 105 and hit.fills[0]["kind"] == "stop_limit"
    gapped = run_bars(sl, [FLAT, (107, 108, 103, 104), FLAT])
    assert gapped.fills[0]["price"] == 106, "triggered above the limit, filled when the price came back to it"
    waits = run_bars(sl.replace(")\n", ", tif='gtc')\n"), [FLAT, (107, 108, 103, 104), (105, 106, 104, 105)], fill_model="pessimistic")
    assert waits.fills[0]["i"] == 2 and waits.fills[0]["price"] == 105, "pessimistic: the triggered order waits as a limit"


def test_bracket_orders_and_pessimistic_ordering():
    code = ("def on_bar(ctx):\n    if ctx.bar == 0:\n        return 1.0\n"
            "    if ctx.bar == 1:\n"
            "        ctx.order_target_shares(0, limit=110, tif='gtc', oco='exit')\n"
            "        ctx.order_target_shares(0, stop=95, tif='gtc', oco='exit')\n")
    rows = [FLAT, FLAT, FLAT, (105, 111, 94, 100), FLAT]          # both levels inside one bar
    std = run_bars(code, rows)
    assert std.fills[-1]["price"] == 110, "open nearer the high: the target is reached first"
    pes = run_bars(code, rows, fill_model="pessimistic")
    assert pes.fills[-1]["price"] == 95, "pessimistic: a long hits its stop first"
    for r in (std, pes):
        assert r.position[-1] == 0 and r.book_stats["cancelled"] == 1, "oco cancels the other exit"
    already_flat = run_bars("def on_bar(ctx):\n    if ctx.bar == 0:\n        ctx.order_target_shares(0, stop=95)\n",
                            [FLAT, (90, 91, 88, 89), FLAT])
    assert not already_flat.fills, "a target exit with nothing to exit does nothing"


def test_participation_limit_spreads_fills_over_days():
    code = "def on_bar(ctx):\n    if ctx.bar == 0:\n        ctx.order(450)\n"
    res = run_bars(code, [FLAT] * 8, volume=1000.0, max_participation=0.1, cash=100000.0)
    assert [f["qty"] for f in res.fills] == [100.0, 100.0, 100.0, 100.0, 50.0]
    assert res.position[-1] == 450 and res.partial_fills == 4
    m, _ = engine.summarize(res)
    assert m["partial_fills"] == 4
    w = run_bars("def on_bar(ctx):\n    return 1.0\n", [FLAT] * 8, volume=10.0, max_participation=0.5)
    assert w.position[1] == 5.0 and w.position[2] == 10.0 and all(abs(f["qty"]) <= 5.0 + 1e-9 for f in w.fills)


def test_square_root_impact_and_spread():
    import random as rnd
    rng = rnd.Random(5)
    px, rows = 100.0, []
    for _ in range(60):
        nxt = px * (1 + rng.gauss(0, 0.01))
        rows.append((px, max(px, nxt), min(px, nxt), nxt))
        px = nxt

    def cost(q, coef=1.0):
        r = run_bars("def on_bar(ctx):\n    if ctx.bar == 30:\n        ctx.order(%d)\n" % q, rows,
                     volume=1e5, impact_coef=coef, cash=1e7)
        return r, r.impact_paid
    r1, c1 = cost(1000)
    _, c4 = cost(4000)
    _, c1x2 = cost(1000, 2.0)
    assert c1 > 0 and approx(c4 / c1, 8.0, 1e-6), "impact cost grows as size^1.5 (price impact as sqrt)"
    assert approx(c1x2 / c1, 2.0, 1e-9)
    f = r1.fills[0]
    sig = engine._trailing_vol(r1.panel.sets[0].close, 20)[31]
    assert approx(f["price"], rows[31][0] * (1 + sig * math.sqrt(1000 / 1e5)), 1e-12)
    sp = run_bars("def on_bar(ctx):\n    if ctx.bar == 0:\n        ctx.order(10)\n", [FLAT, FLAT, FLAT], spread_bps=10)
    assert approx(sp.spread_paid, 10 * 100 * 0.0005) and approx(sp.fills[0]["price"], 100.05)


def test_implementation_shortfall_and_market_on_close():
    rows = [FLAT, (102, 103, 101, 104), FLAT]
    res = run_bars("def on_bar(ctx):\n    if ctx.bar == 0:\n        ctx.order(10)\n", rows, slippage_bps=10)
    m, _ = engine.summarize(res)
    assert res.fills[0]["arrival"] == 100
    assert approx(m["is_bps"], (102 * 1.001 / 100 - 1) * 1e4, 1e-9), "delay to the open plus slippage"
    assert approx(m["is_cost"], 10 * (102 * 1.001 - 100), 1e-9)
    moc = run_bars("def on_bar(ctx):\n    if ctx.bar == 0:\n        ctx.order(10, at='close')\n", rows)
    assert moc.fills[0]["price"] == 104 and moc.fills[0]["i"] == 1, "at='close' fills at the next close"
    try:
        run_bars("def on_bar(ctx):\n    ctx.order(1, limt=5)\n", rows)
        raise AssertionError("misspelled order option must raise")
    except engine.StrategyError as e:
        assert "unknown order option limt" in e.info["message"]


# ------------------------------------------------- overfitting -----

def test_norm_ppf_inverts_norm_cdf():
    for q in (1e-10, 0.001, 0.02, 0.3, 0.5, 0.8, 0.975, 0.999999):
        assert abs(engine.norm_cdf(engine.norm_ppf(q)) - q) < 1e-12 * max(1, 1 / q), q
    assert abs(engine.norm_ppf(0.975) - 1.959963984540054) < 1e-12


def test_expected_max_sharpe_matches_simulation():
    rng = random.Random(11)
    for trials in (10, 50, 200):
        sims = [max(rng.gauss(0, 0.2) for _ in range(trials)) for _ in range(3000)]
        want = sum(sims) / len(sims)
        got = engine.expected_max_sharpe(trials, 0.04)
        assert abs(got / want - 1) < 0.05, (trials, got, want)
    assert engine.expected_max_sharpe(1, 0.04) == 0.0


def test_deflated_sharpe_penalizes_many_trials():
    rng = random.Random(2)
    good = [rng.gauss(0.0006, 0.01) for _ in range(2500)]
    one = engine.deflated_sharpe(good, 1)
    assert approx(one["dsr"], engine.return_tests(good)["psr"], 1e-12), "with one trial DSR is PSR"
    many = engine.deflated_sharpe(good, 100)
    assert many["dsr"] < one["dsr"] and many["sr_hurdle"] > 0
    noise = [[rng.gauss(0, 0.01) for _ in range(1000)] for _ in range(50)]
    srs = [sum(r) / len(r) / engine.stdev(r) * math.sqrt(252) for r in noise]
    best = noise[srs.index(max(srs))]
    var = engine.stdev(srs) ** 2
    assert engine.return_tests(best)["psr"] > 0.9, "the best of 50 noise strategies looks significant..."
    assert engine.deflated_sharpe(best, 50, var)["dsr"] < 0.7, "...until it is deflated"
    res = run("def on_bar(ctx):\n    return 1.0\n")
    m, _ = engine.summarize(res, trials={"n": 30, "var": 0.1})
    assert m["trials"] == 30 and 0 <= m["dsr"] <= 1 and m["sr_hurdle"] > 0


def test_pbo_cscv():
    rng = random.Random(4)
    noise = [engine.block_stats([rng.gauss(0, 0.01) for _ in range(1600)], 10) for _ in range(20)]
    out = engine.pbo_cscv(noise)
    assert out["splits"] == 252 and 0.25 < out["pbo"] < 0.75, out["pbo"]
    skill = [engine.block_stats([rng.gauss(0.0015 if c == 0 else 0, 0.01) for _ in range(1600)], 10) for c in range(20)]
    out = engine.pbo_cscv(skill, labels=["c%d" % c for c in range(20)])
    assert out["pbo"] < 0.1 and out["most_chosen"][0]["label"] == "c0"
    try:
        engine.pbo_cscv([engine.block_stats([0.01] * 100, 5)] * 2)
        raise AssertionError("odd block count must raise")
    except ValueError:
        pass


def test_walk_forward_optimization_stitches_out_of_sample():
    T = 600
    a = [0.002 if t < 300 else -0.002 for t in range(T)]        # good, then bad
    b = [-0.002 if t < 300 else 0.002 for t in range(T)]        # bad, then good
    c = [0.0001 * ((-1) ** t) for t in range(T)]
    dates = engine.session_days("2020-01-02", "2030-01-01")[:T + 1]
    out = engine.walk_forward_optimization(dates, [a, b, c], ["A", "B", "C"], train=100, test=50, metric="total_return")
    chosen = [f["chosen"] for f in out["folds"]]
    assert chosen[0] == "A" and chosen[-1] == "B", chosen
    stitched = []
    for f in out["folds"]:
        i, j = dates.index(f["test_start"]), dates.index(f["test_end"])
        stitched += [a, b, c][f["config"]][i:j]
    eq = 10000.0
    for x in stitched:
        eq *= 1 + x
    assert approx(out["equity"][-1], eq, 1e-12) and len(out["equity"]) == T - 100 + 1
    anchored = engine.walk_forward_optimization(dates, [a, b, c], ["A", "B", "C"], 100, 50, anchored=True, metric="total_return")
    assert "B" not in [f["chosen"] for f in anchored["folds"]], "anchored windows remember the whole past, so adapt slowly"


def test_purged_kfold_and_cpcv():
    n, h, e = 100, 5, 3
    folds = list(engine.purged_kfold(n, 5, horizon=h, embargo=e))
    assert sorted(j for _, test in folds for j in test) == list(range(n))
    for train, test in folds:
        a, b = test[0], test[-1] + 1
        assert not any(a - h <= j < b + e for j in train), "purged before and embargoed after the test fold"
    splits = engine.cpcv_splits(n, 6, 2, horizon=h, embargo=e)
    assert len(splits) == 15
    counts = [sum(1 for _, _, g in splits if k in g) for k in range(6)]
    assert counts == [5] * 6
    for train, test, _ in splits:
        assert not set(train) & set(test)
    X = [[float(i)] for i in range(200)]
    y = [float(i % 2) for i in range(200)]
    seen = []

    def fp(xt, yt, xn):
        newest = max(r[0] for r in xt if r[0] < xn[0][0])
        seen.append(xn[0][0] - newest)
        return [0.0] * len(xn)
    engine.walk_forward(X, y, fp, min_train=50, retrain_every=25, horizon=2, embargo=4)
    assert min(seen) >= 6, "embargo widens the gap between training and prediction"
    cv = engine.cv_score(X, y, lambda xt, yt, xn: [0.5] * len(xn), lambda t, p: -sum((a - b) ** 2 for a, b in zip(t, p)) / len(t), k=4)
    assert approx(cv["mean"], -0.25) and len(cv["folds"]) == 4


# ---------------------------------------------- dividends & carry -----

def test_dividends_are_paid_and_total_return_benchmark():
    closes = [100.0] * 260
    ds = synthetic("FLAT", closes, dividends=[("2020-03-20", 1.0), ("2020-06-19", 1.0), ("2020-09-18", 1.0)])
    assert ds.dividend_source == "events" and sum(ds.dividends) == 3.0
    long_ = engine.backtest(engine.Strategy.from_code("def on_bar(ctx):\n    return 1.0\n"), ds, engine.Config(slippage_bps=0))
    sh = 100.0                                       # $10,000 at $100; the dividend cash is reinvested later
    assert approx(long_.equity[-1], 10000 + 3.0 * sh, 1e-9), "a holder receives each dividend"
    assert approx(long_.bench_equity[-1], 10000 * (1.01 ** 3), 1e-9), "the benchmark reinvests dividends"
    m, _ = engine.summarize(long_)
    assert approx(m["dividends_net"], 3.0 * sh, 1e-9)
    short = engine.backtest(engine.Strategy.from_code("def on_bar(ctx):\n    return -1.0\n"), ds,
                            engine.Config(slippage_bps=0, allow_short=True))
    owed = sum(-short.position[t - 1] * ds.dividends[t] for t in range(1, ds.n))   # shares held into each ex-date
    assert owed > 290 and approx(short.carry["dividends_paid"], owed, 1e-12)
    assert approx(short.equity[-1], 10000 - owed, 1e-9), "a short pays each dividend"
    off = engine.backtest(engine.Strategy.from_code("def on_bar(ctx):\n    return 1.0\n"), ds,
                          engine.Config(slippage_bps=0, total_return=False))
    assert approx(off.equity[-1], 10000, 1e-9) and off.bench_equity[-1] == 10000


def test_dividends_recovered_from_adjusted_close():
    closes = [50.0 + 0.1 * i for i in range(300)]
    truth = {40: 0.40, 103: 0.42, 166: 0.45, 229: 0.47}
    factor, adj = 1.0, [0.0] * 300
    for t in range(299, -1, -1):          # Yahoo's adjustment: step the factor down before each ex-date
        adj[t] = round(closes[t] * factor, 6)
        if t in truth:
            factor *= 1 - truth[t] / closes[t - 1]
    days = engine.session_days("2021-01-04", "2030-01-01")[:300]
    text = "Date,Open,High,Low,Close,Adj Close,Volume\n" + "".join(
        "%s,%.4f,%.4f,%.4f,%.4f,%.6f,1000\n" % (d, c, c, c, c, a) for d, c, a in zip(days, closes, adj))
    ds = engine.parse_csv_text(text, "YHOO")
    got = {i: v for i, v in enumerate(ds.dividends) if v}
    assert set(got) == set(truth) and all(abs(got[i] - truth[i]) < 1e-3 for i in truth), got
    assert ds.close == closes, "the raw close stays the trading price"
    only_adj = engine.parse_csv_text("Date,Adj Close\n2021-01-04,10\n2021-01-05,11\n", "X")
    assert only_adj.price_basis == "total_return" and only_adj.close == [10.0, 11.0]


def test_dividend_and_split_files():
    kind, ev = engine.parse_events_csv("Ex/EFF Date,Type,Cash Amount,Declaration Date\n03/20/2020,Cash,$1.25,03/01/2020\n")
    assert kind == "dividends" and ev == [("2020-03-20", 1.25)]
    kind, ev = engine.parse_events_csv("Date,Stock Splits\n2020-02-03,2:1\n")
    assert kind == "splits" and ev == [("2020-02-03", 2.0)]
    raw = synthetic("RAW", [100.0] * 21 + [50.0] * 20, dividends=[("2020-01-10", 1.0)])
    done = raw.apply_splits([("2020-02-03", "2:1")])
    assert done[0]["adjusted"] and raw.close[0] == 50.0 and raw.dividends[raw.index["2020-01-10"]] == 0.5
    already = synthetic("ADJ", [50.0] * 41)
    assert not already.apply_splits([("2020-02-03", 2)])[0]["adjusted"] and already.close[0] == 50.0


def test_cash_interest_act360_and_rate_series():
    ds = synthetic("CASHY", [100.0] * 253)
    flat = "def on_bar(ctx):\n    return 0.0\n"
    res = engine.backtest(engine.Strategy.from_code(flat), ds, engine.Config(cash_rate=0.05))
    want = 10000.0
    for a, b in zip(ds.days, ds.days[1:]):
        want *= 1 + 0.05 * (b - a).days / 360
    assert approx(res.equity[-1], want, 1e-12)
    assert approx(engine.Config(risk_free=0.02).cash_rate or 0.02, 0.02) and res.carry["interest_earned"] > 500
    rf = engine.backtest(engine.Strategy.from_code(flat), ds, engine.Config(risk_free=0.03))
    assert rf.equity[-1] > 10250, "cash earns the risk-free rate by default"
    fred = engine.parse_rate_csv("observation_date,DTB3\n2019-12-31,1.5\n2020-07-01,.\n2020-07-02,0.1\n")
    assert fred.at("2020-03-01") == 0.015 and fred.at("2020-12-01") == 0.001
    stepped = engine.backtest(engine.Strategy.from_code(flat), ds, engine.Config(), rates=fred)
    assert 10000 * 1.004 < stepped.equity[-1] < 10000 * 1.012


def test_margin_financing_and_borrow_fees():
    ds = synthetic("LEV", [100.0] * 253)
    lev = engine.backtest(engine.Strategy.from_code("def on_bar(ctx):\n    return 2.0\n"), ds,
                          engine.Config(max_leverage=2.0, cash_rate=0.04, margin_spread=0.01, slippage_bps=0))
    assert lev.carry["interest_paid"] > 0 and lev.equity[-1] < 10000 * (1 - 0.045)
    borrow = engine.backtest(engine.Strategy.from_code("def on_bar(ctx):\n    return -1.0\n"), ds,
                             engine.Config(allow_short=True, cash_rate=0.0, borrow_fee=0.05, slippage_bps=0))
    fees = sum(-borrow.position[t - 1] * 100.0 * 0.05 * (ds.days[t] - ds.days[t - 1]).days / 360
               for t in range(1, ds.n))
    assert approx(borrow.carry["borrow_paid"], fees, 1e-9) and fees > 480
    custom = engine.Config(allow_short=True, borrow_fee=0.01, borrow_fees={"lev": 0.5})
    assert custom.borrow_fees == {"LEV": 0.5}


def test_margin_call_cuts_positions():
    closes = [100.0 * (1.03 ** i) for i in range(120)]          # a short squeeze
    ds = synthetic("SQZ", closes)
    code = "def on_bar(ctx):\n    if ctx.bar == 0:\n        ctx.order_target_shares(-200)   # short once, then let it run\n"
    res = engine.backtest(engine.Strategy.from_code(code), ds,
                          engine.Config(allow_short=True, max_leverage=2.0, slippage_bps=0, maintenance_margin=0.3))
    assert res.margin_calls, "equity below 30% of gross exposure must trigger a call"
    assert any(f.get("reason") == "margin call" for f in res.fills)
    assert any("margin call" in n for n in res.notes)
    t = res.dates.index(res.margin_calls[0]["date"])
    assert res.equity[t] / (res.gross[t] * res.equity[t]) >= 0.3 - 1e-9
    off = engine.backtest(engine.Strategy.from_code(code), ds,
                          engine.Config(allow_short=True, max_leverage=2.0, slippage_bps=0, maintenance_margin=0.0))
    assert not off.margin_calls


# ------------------------------------------------- data quality -----

def test_quality_report_flags_problems_without_dropping_rows():
    days = engine.session_days("2023-01-03", "2023-12-29")
    closes = [100.0 + 0.2 * i for i in range(len(days))]
    for i in range(50, 57):
        closes[i] = closes[49]                       # stale run
    closes[120] = closes[119] * 1.5                  # bad print
    rows = list(zip(days, closes))
    del rows[80:82]                                  # two missing sessions
    rows.append(("2023-07-04", 999.0))               # a holiday bar
    rows.append(("2023-03-01", rows[38][1]))         # duplicate date
    text = "Date,Open,High,Low,Close,Volume\n" + "".join(
        "%s,%s,%s,%s,%s,1000\n" % (d, c, c - 1 if i == 5 else c, c, c, ) for i, (d, c) in enumerate(rows)) + "bad-date,1,1,1,1,1\n"
    ds = engine.parse_csv_text(text, "DIRTY")
    q = engine.quality_report(ds)["issues"]
    for kind in ("missing_days", "off_calendar", "stale", "jump", "duplicate", "dropped", "ohlc"):
        assert kind in q, (kind, sorted(q))
    assert q["missing_days"]["count"] == 1 and "2 sessions" in q["missing_days"]["examples"][0]["detail"]
    assert ds.n == len(rows) - 1, "only the duplicate collapses; nothing else is removed"
    clean = engine.quality_report(bundled("SPY"))["issues"]
    assert "missing_days" not in clean and "off_calendar" not in clean, "bundled SPY matches the NYSE calendar"


def test_nyse_holidays():
    h = engine.nyse_holidays(2024)
    for d in ("2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19",
              "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25"):
        assert d in h, d
    assert len(h) == 10
    assert "2022-12-26" in engine.nyse_holidays(2022) and "2021-12-31" not in engine.nyse_holidays(2021)
    assert "2027-06-18" in engine.nyse_holidays(2027), "Saturday Juneteenth closes Friday"
    assert len(engine.session_days("2024-01-01", "2024-12-31")) == 252


def test_portfolio_templates_run_and_are_causal():
    t = load_templates()
    ids = [k for k in t if k.startswith("pf_")]
    assert len(ids) == 5
    for tid in ids:
        strat = engine.Strategy.from_code(t[tid])
        assert strat.universe, tid
        sets = [bundled(tk) for tk in strat.universe]
        res = engine.backtest_portfolio(strat, sets, engine.Config(allow_short=True), bench=bundled("SPY"))
        m, _ = engine.summarize(res)
        assert m["final_equity"] > 0 and abs(engine.attribution(res)["check"]) < 1e-6, tid
        if res.signals is not None:
            chk = engine.lookahead_check(strat, res.panel, res.signals, samples=3)
            assert not chk["violations"], tid
    cheat = ("UNIVERSE = ['DEMO-STOCK', 'DEMO-BOND']\ndef signals(data):\n"
             "    c = data['DEMO-STOCK'].close\n"
             "    return [{'DEMO-STOCK': 1.0 if i + 1 < len(c) and c[i + 1] > c[i] else 0.0} for i in range(len(c))]\n")
    strat = engine.Strategy.from_code(cheat)
    res = engine.backtest_portfolio(strat, [DEMO, BOND])
    assert engine.lookahead_check(strat, res.panel, res.signals)["violations"], "portfolio look-ahead is caught too"


def test_portfolio_signal_formats_agree():
    sets = [DEMO, BOND]
    as_dict = ("def signals(data):\n    s = qt.sma(data['DEMO-STOCK'].close, 50)\n"
               "    w = [0.6 if c > m else 0.2 for c, m in zip(data['DEMO-STOCK'].close, s)]\n"
               "    return {'DEMO-STOCK': w, 'DEMO-BOND': [1 - x for x in w]}\n")
    rows = as_dict.replace("    return {'DEMO-STOCK': w, 'DEMO-BOND': [1 - x for x in w]}\n",
                           "    return [{'DEMO-STOCK': x, 'DEMO-BOND': 1 - x} for x in w]\n")
    results = [engine.backtest_portfolio(engine.Strategy.from_code(c), sets).equity for c in (as_dict, rows)]
    try:
        import numpy  # noqa: F401
        matrix = as_dict.replace("    return {'DEMO-STOCK': w, 'DEMO-BOND': [1 - x for x in w]}\n",
                                 "    import numpy as np\n    return np.column_stack([w, [1 - x for x in w]])\n")
        results.append(engine.backtest_portfolio(engine.Strategy.from_code(matrix), sets).equity)
    except ImportError:
        pass
    assert all(r == results[0] for r in results), "dict of lists, list of dicts and a matrix are the same weights"


def test_portfolio_series_numpy_and_python_agree():
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("skip  numpy/python series check (needs numpy)")
        return
    t = load_templates()
    sets = {tk: bundled(tk) for tk in ("AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM", "KO", "XLF", "XLE", "XLV", "XLK", "SPY", "TLT", "GLD")}
    for tid in ("pf_long_short", "pf_momentum", "pf_risk_parity"):
        strat = engine.Strategy.from_code(t[tid])
        data = [sets[x] for x in strat.universe if x in sets] or list(sets.values())
        cfg = engine.Config(allow_short=True, max_leverage=1.5)
        out = []
        for flag in (True, False):
            engine.USE_NUMPY = flag
            try:
                out.append(engine.backtest_portfolio(engine.Strategy.from_code(t[tid]), data, cfg))
            finally:
                engine.USE_NUMPY = True
        a, b = out
        assert a.equity == b.equity, tid
        for name in ("gross", "net", "long", "short"):
            assert max(abs(x - y) for x, y in zip(getattr(a, name), getattr(b, name))) < 1e-12, (tid, name)
        for tk in a.weights:
            assert max(abs(x - y) for x, y in zip(a.weights[tk], b.weights[tk])) < 1e-12
            assert max(abs(x - y) for x, y in zip(a.asset_pnl[tk], b.asset_pnl[tk])) < 1e-6


def test_vectorized_rebalance_is_bit_identical():
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("skip  vectorized rebalance check (needs numpy)")
        return
    rng = random.Random(9)
    days = engine.session_days("2019-01-02", "2021-12-31")
    sets = []
    for k in range(20):
        px, rows = 20 + 80 * rng.random(), []
        for i, d in enumerate(days):
            if k % 5 == 3 and i % 17 == 0:
                continue                                    # some instruments miss days
            nxt = px * (1 + rng.gauss(0.0002, 0.02))
            rows.append((d, px, max(px, nxt) * 1.01, min(px, nxt) * 0.99, nxt, rng.randint(20000, 500000)))
            px = nxt
        start = 40 * (k % 4)                                # and some list later
        d, o, h, l, c, v = zip(*rows[start:])
        sets.append(engine.DataSet("R%02d" % k, d, c, o, h, l, v))
    code = ("def on_bar(ctx):\n    r = random.random()\n"
            "    return {t: random.uniform(-0.12, 0.12) for t in ctx.tickers if ctx[t].listed and random.random() < 0.8}\n")
    configs = [dict(allow_short=True), dict(allow_short=True, max_leverage=2.0, spread_bps=8, impact_coef=1.0, commission_bps=3),
               dict(fractional=False, cash=250000.0, commission_fixed=1.0, rebalance_band=0.0),
               dict(allow_short=False, rebalance_band=0.05, slippage_bps=5, impact_coef=0.5, execution="close")]
    engine.VECTOR_MIN = 16                                  # exercise the NumPy path with 20 instruments
    for cfg in configs:
        out = []
        for flag in (True, False):
            engine.USE_NUMPY = flag
            try:
                out.append(engine.backtest_portfolio(engine.Strategy.from_code(code), sets, engine.Config(seed=3, **cfg)))
            finally:
                engine.USE_NUMPY = True
        a, b = out
        assert len(a.fills) > 500, cfg
        assert a.equity == b.equity, cfg
        assert [(f["i"], f["ticker"], f["qty"], f["price"]) for f in a.fills] == [(f["i"], f["ticker"], f["qty"], f["price"]) for f in b.fills], cfg
    engine.VECTOR_MIN = 200


def test_portfolio_errors_are_readable():
    for code, msg in (("def on_bar(ctx):\n    return {'NOPE': 1}\n", "not in the portfolio"),
                      ("def on_bar(ctx):\n    return [1, 2, 3]\n", "3 weights for 2 instruments"),
                      ("UNIVERSE = 'SPY'\ndef on_bar(ctx):\n    return None\n", "UNIVERSE must be a list")):
        try:
            engine.backtest_portfolio(engine.Strategy.from_code(code), [DEMO, BOND])
            raise AssertionError("expected an error for %r" % code)
        except engine.StrategyError as e:
            assert msg in e.info["message"], e.info["message"]


def browser_parse(texts):
    """Run index.html's parseCSV() under Node on each text; None if Node isn't installed."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        return None
    page = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
    a = page.index("function splitCSVLine(line) {")
    b = page.index("function tickerFromFilename(name) {")
    js = page[a:b] + "\nconst texts = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n" \
        "process.stdout.write(JSON.stringify(texts.map(t => { try { return parseCSV(t); } catch (e) { return {error: e.message}; } })));\n"
    out = subprocess.run([node, "-e", js], input=json.dumps(texts), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_browser_and_python_parse_csv_identically():
    yahoo = "Date,Open,High,Low,Close,Adj Close,Volume\n" + "".join(
        "2023-01-%02d,%d,%d,%d,%d.5,%s,100\n" % (d, 100 + d, 101 + d, 99 + d, 100 + d, "%.6f" % ((100.5 + d) * (0.99 if d < 15 else 1.0)))
        for d in range(3, 28) if d not in (7, 8, 14, 15, 21, 22))
    messy = ('"Date","Close/Last","Volume","Open","High","Low"\n'
             '01/05/2024 00:00,"$1,010.50",N/A,"$1,000.00",$1011,$990\n'
             '01/04/2024,$1005.25,"1,200",$1000,$1006,$999\n'
             '01/04/2024,$1004.75,1100,$1000,$1006,$999\n'
             '13/45/2024,$1,1,1,1,1\n'
             '01/03/2024,-5,1,1,1,1\n'
             '01/02/2024,$999.00,900,,,\n')
    adj_only = "Date,Adj Close\n2021-01-04,10\n2021-01-05,11\n2021-01-06,12.5\n"
    spy = open(os.path.join(ROOT, "data", "SPY.csv"), encoding="utf-8").read()
    texts = [yahoo, messy, adj_only, spy]
    parsed = browser_parse(texts)
    if parsed is None:
        print("skip  browser/Python CSV parity (needs node)")
        return
    for text, js in zip(texts, parsed):
        assert "error" not in js, js
        py = engine.parse_csv_text(text, "X")
        via_browser = engine.DataSet.from_dict("X", js)
        for col in ("dates", "open", "high", "low", "close", "volume", "dividends"):
            assert getattr(via_browser, col) == getattr(py, col), (col, text[:40])
        assert via_browser.price_basis == py.price_basis and via_browser.dividend_source == py.dividend_source
    assert parsed[1]["dates"] == ["2024-01-02", "2024-01-04", "2024-01-05"] and parsed[1]["close"][1] == 1004.75
    assert engine.parse_csv_text(yahoo, "Y").dividend_source == "adj_close"


def test_bundled_data_parses():
    import json
    man = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
    for m in man["instruments"]:
        ds = engine.load_csv(os.path.join(ROOT, "data", m["file"]), m["ticker"])
        assert ds.n > 1000, m["ticker"]
        assert all(l <= min(o, c) + 1e-9 and h >= max(o, c) - 1e-9 for o, h, l, c in zip(ds.open, ds.high, ds.low, ds.close)), m["ticker"]


def test_json_output_replaces_only_nonfinite_numbers():
    x = {"a": float("nan"), "b": [1.5, float("inf"), -float("inf")], 3: (1, 2),
         "log": 'ratio: NaN, "quoted" \\ Infinity -Infinity', "nested": {"z": [float("nan")]}}
    out = json.loads(engine.to_json(x))
    assert out == {"a": None, "b": [1.5, None, None], "3": [1, 2],
                   "log": 'ratio: NaN, "quoted" \\ Infinity -Infinity', "nested": {"z": [None]}}


def test_signal_cache_reuses_signals_when_only_settings_change():
    engine.DATA.clear()
    engine.SIGNAL_CACHE.clear()
    engine.DATA["DEMO-STOCK"] = DEMO
    code = ("def signals(data):\n    print('computing')\n    s = qt.sma(data.close, 50)\n"
            "    return [1.0 if c > m else 0.0 for c, m in zip(data.close, s)]\n")
    base = {"ticker": "DEMO-STOCK", "code": code}
    a = json.loads(engine.handle("run", json.dumps(dict(base, config={"slippage_bps": 0}))))["result"]
    n = len(engine.SIGNAL_CACHE)
    b = json.loads(engine.handle("run", json.dumps(dict(base, config={"slippage_bps": 20}))))["result"]
    assert len(engine.SIGNAL_CACHE) == n == 1, "same code, params & data: signals computed once"
    assert b["metrics"]["final_equity"] < a["metrics"]["final_equity"], "costs still apply on a cache hit"
    assert a["logs"].count("computing") == 1 and b["logs"].count("computing") == 1, "printed output replays"
    fresh = engine.backtest(engine.Strategy.from_code(code), DEMO, engine.Config(slippage_bps=20))
    assert abs(fresh.equity[-1] - b["metrics"]["final_equity"]) < 1e-6
    json.loads(engine.handle("run", json.dumps(dict(base, params={"x": 1}, config={"slippage_bps": 20}))))
    assert len(engine.SIGNAL_CACHE) == 2, "different parameters: new signals"
    json.loads(engine.handle("run", json.dumps(dict(base, config={"slippage_bps": 20, "seed": 7}))))
    assert len(engine.SIGNAL_CACHE) == 3, "a different random seed: new signals"


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
    prov = r["provenance"]
    assert prov["engine_version"] == engine.ENGINE_VERSION and len(prov["code_sha256"]) == 64
    assert prov["data"]["DEMO-STOCK"] == DEMO.fingerprint() and prov["config"]["cash"] == 10000
    again = json.loads(engine.handle("run", json.dumps({
        "ticker": "DEMO-STOCK", "benchmark": "DEMO-BOND", "code": load_templates()["sma_cross"], "config": {"cash": 10000}})))
    assert again["result"]["equity"] == r["equity"] and again["result"]["provenance"] == prov, "a run reproduces exactly"
    an = json.loads(engine.handle("analytics", json.dumps({"tickers": ["DEMO-STOCK", "DEMO-BOND"]})))
    assert an["ok"] and len(an["result"]["matrix"]) == 2
    assert all(m["tier"] in ("core", "std", "extra") for m in json.loads(engine.handle("metric_list", "{}"))["result"])
    base = {"ticker": "DEMO-STOCK", "benchmark": "DEMO-BOND", "code": load_templates()["sma_cross"]}
    for method, extra in (("subperiods", {"n": 3}), ("costs", {"levels": [0, 10]}), ("permutation", {"n_perm": 50})):
        out = json.loads(engine.handle(method, json.dumps(dict(base, **extra))))
        assert out["ok"], (method, out)


# ------------------------------------------------- modules & dev tools -----

MOD_STRAT = ("from my_signals import trend\n"
             "PARAMS = {'n': 50}\n"
             "def on_bar(ctx):\n"
             "    return trend(ctx, PARAMS['n'])\n")
MOD_SRC = "def trend(ctx, n):\n    return 1.0 if ctx.price > ctx.sma(n) else 0.0\n"
INLINE = "PARAMS = {'n': 50}\ndef on_bar(ctx):\n    return 1.0 if ctx.price > ctx.sma(PARAMS['n']) else 0.0\n"


def test_user_module_matches_inline_code_and_reloads_fresh():
    try:
        engine.set_modules({"my_signals": MOD_SRC})
        a, b = run(MOD_STRAT), run(INLINE)
        assert a.equity == b.equity and len(a.fills) == len(b.fills) > 0
        strat = engine.Strategy.from_code(MOD_STRAT)
        assert strat.modules == ["my_signals"]
        # a changed module is picked up on the next compile, without restarting Python
        engine.set_modules({"my_signals": MOD_SRC.replace("else 0.0", "else 0.5")})
        c = run(MOD_STRAT)
        assert any(0.45 < x < 0.55 for x in c.exposure) and max(c.exposure) < 1.01 and c.equity != a.equity
        # module-level state doesn't leak from one run into the next
        engine.set_modules({"counter": "calls = []\ndef hit():\n    calls.append(1)\n    return len(calls)\n"})
        code = "import counter\ndef on_bar(ctx):\n    ctx.state['n'] = counter.hit()\n    return 0.0\n"
        assert run(code).equity == run(code).equity
        n1 = engine.Strategy.from_code(code)
        import counter
        assert counter.calls == [], "each compile imports the module afresh"
    finally:
        engine.set_modules({})


def test_user_module_errors_point_at_the_module_and_bad_names_are_refused():
    engine.DATA["DEMO-STOCK"] = DEMO
    try:
        engine.set_modules({"helpers": "def f(x):\n    return x / 0\n"})
        out = json.loads(engine.handle("run", json.dumps({
            "ticker": "DEMO-STOCK", "code": "from helpers import f\ndef on_bar(ctx):\n    return f(1)\n",
            "modules": {"helpers": "def f(x):\n    return x / 0\n"}})))
        err = out["error"]
        assert not out["ok"] and err["file"] == "helpers" and err["line"] == 2 and "helpers.py, line 2" in err["trace"]
        syn = json.loads(engine.handle("inspect", json.dumps({"code": "import broken\ndef on_bar(ctx):\n    return 0\n",
                                                               "modules": {"broken": "def f(:\n"}})))
        assert syn["error"]["file"] == "broken" and syn["error"]["line"] == 1
        for bad in ("math", "numpy", "engine", "2fast", "my-lib", "class"):
            assert engine.module_name_problem(bad), bad
        assert engine.module_name_problem("my_lib") is None
        refused = json.loads(engine.handle("inspect", json.dumps({"code": INLINE, "modules": {"json": "x = 1\n"}})))
        assert not refused["ok"] and "can't be called json" in refused["error"]["message"]
        # a real package always wins over a module of the same name: the finder comes after Python's own
        import importlib.machinery
        mine = next(i for i, f in enumerate(sys.meta_path) if isinstance(f, engine._ModuleFinder))
        path = next(i for i, f in enumerate(sys.meta_path) if f is importlib.machinery.PathFinder)
        assert mine > path
    finally:
        engine.set_modules({})


def test_module_changes_invalidate_signal_cache_and_show_in_provenance():
    code = "import sig_lib\ndef signals(data):\n    return sig_lib.weights(data.close)\n"
    m1 = "def weights(c):\n    return [1.0] * len(c)\n"
    m2 = "def weights(c):\n    return [0.5] * len(c)\n"
    engine.DATA["DEMO-STOCK"] = DEMO
    try:
        base = {"ticker": "DEMO-STOCK", "code": code}
        r1 = json.loads(engine.handle("run", json.dumps(dict(base, modules={"sig_lib": m1}))))["result"]
        r2 = json.loads(engine.handle("run", json.dumps(dict(base, modules={"sig_lib": m2}))))["result"]
        assert max(r1["exposure"]) > 0.9 and max(r2["exposure"]) < 0.6, "the cache must not serve the old module's signals"
        h1, h2 = r1["provenance"]["modules"]["sig_lib"], r2["provenance"]["modules"]["sig_lib"]
        import hashlib
        assert h1 == hashlib.sha256(m1.encode()).hexdigest() and h1 != h2
        plain = json.loads(engine.handle("run", json.dumps({"ticker": "DEMO-STOCK", "code": INLINE, "modules": {"sig_lib": m1}})))
        assert plain["result"]["provenance"]["modules"] == {}, "only modules the strategy imports are recorded"
    finally:
        engine.set_modules({})


def test_lint_reports_undefined_names_unused_imports_and_ctx_typos():
    code = "import os\ndef on_bar(ctx):\n    if ctx.smaa(20) > 0:\n        return y\n    return ctx['SPY'].pricee\n"
    for use_pyflakes in (True, False):
        saved = sys.modules.get("pyflakes")
        if not use_pyflakes:
            sys.modules["pyflakes"] = None          # import pyflakes -> ImportError: the built-in check runs
        try:
            msgs = engine.lint(code)
        finally:
            if not use_pyflakes:
                if saved is None:
                    del sys.modules["pyflakes"]
                else:
                    sys.modules["pyflakes"] = saved
        got = {(m["line"], m["code"]) for m in msgs}
        assert (1, "UnusedImport") in got and (4, "UndefinedName") in got, (use_pyflakes, msgs)
        assert (3, "UnknownAttribute") in got and (5, "UnknownAttribute") in got
        typo = [m for m in msgs if m["line"] == 3][0]
        assert "did you mean sma" in typo["message"] and typo["col"] == 12
    assert engine.lint("def on_bar(ctx):\n    return ctx.sma(20) > ctx.price and ctx.state.get('x')\n") == []
    assert engine.lint("def on_bar(ctx:\n")[0]["code"] == "SyntaxError"
    assert engine.lint("x = 1\n")[0]["code"] == "NoEntryPoint"
    assert engine.lint("def helper(x):\n    return qt.sma(x, 5)\n", kind="module", name="m") == []
    for tid, tpl in load_templates().items():
        errors = [m for m in engine.lint(tpl) if m["severity"] == "error"]
        assert not errors, (tid, errors)


def test_strategy_tests_runner():
    code = (INLINE +
            "def test_long_in_uptrend():\n"
            "    res = run_strategy(make_data([100 + i for i in range(120)]))\n"
            "    assert res.exposure[-1] > 0.9\n"
            "def test_flat_in_downtrend():\n"
            "    res = run_strategy(make_data([200 - i for i in range(120)]), params={'n': 20})\n"
            "    assert max(res.exposure) == 0, 'should never buy'\n"
            "def test_fails():\n"
            "    print('checking')\n"
            "    assert_close(1.0, 1.1)\n"
            "def test_errors():\n"
            "    return {}['missing']\n")
    out = engine.run_tests(code)
    by = {t["name"]: t for t in out["tests"]}
    assert out["passed"] == 2 and out["failed"] == 2 and out["error"] is None
    assert by["test_long_in_uptrend"]["ok"] and by["test_flat_in_downtrend"]["ok"]
    f = by["test_fails"]
    assert not f["ok"] and "1.0 != 1.1" in f["message"] and f["output"] == "checking\n" and f["line"] == 12
    assert by["test_errors"]["message"].startswith("KeyError") and by["test_errors"]["line"] == 14
    only = engine.run_tests(code, only="test_fails")
    assert [t["name"] for t in only["tests"]] == ["test_fails"]
    broken = engine.run_tests("x = 1 / 0\n")
    assert broken["error"]["line"] == 1 and broken["tests"] == []
    mod = engine.run_tests("def double(x):\n    return 2 * x\ndef test_double():\n    assert double(2) == 5\n",
                           kind="module", name="mathx")
    assert mod["failed"] == 1 and "assert double(2) == 5" in mod["tests"][0]["message"]
    ds = engine.synthetic_data(300, seed=3)
    assert ds.n == 300 and ds.dates[0] == "2020-01-02" and all(h >= c >= l for h, c, l in zip(ds.high, ds.close, ds.low))
    assert engine.synthetic_data(300, seed=3).close == ds.close


def test_console_keeps_variables_and_sees_data_and_last_run():
    engine.DATA["DEMO-STOCK"] = DEMO
    engine.handle("run", json.dumps({"ticker": "DEMO-STOCK", "code": INLINE}))
    r = engine.console_run("x = 21\nx * 2", ["DEMO-STOCK"], reset=True)
    assert r["output"] == "42\n" and r["error"] is None and {"name": "x", "type": "int"} in r["vars"]
    r = engine.console_run("print(len(data.close), data.ticker)\nround(summarize(result)[0]['sharpe'], 6)", ["DEMO-STOCK"])
    assert r["output"].splitlines()[0] == "%d DEMO-STOCK" % DEMO.n
    assert abs(float(r["output"].splitlines()[1]) - engine.summarize(run(INLINE, slippage_bps=2.0))[0]["sharpe"]) < 1e-6
    assert engine.console_run("x + 1", ["DEMO-STOCK"])["output"] == "22\n", "variables persist"
    e = engine.console_run("def f():\n    return 1 / 0\nf()", ["DEMO-STOCK"])
    assert e["error"].endswith("ZeroDivisionError: division by zero") and "line 2, in f" in e["error"]
    assert engine.console_run("x", ["DEMO-STOCK"], reset=True)["error"].startswith("  console")
    assert "SyntaxError" in engine.console_run("for", [])["error"]
    assert "<Result DEMO-STOCK" in engine.console_run("result", [])["output"]


def test_long_output_is_kept_in_full():
    code = "def on_bar(ctx):\n    print('bar', ctx.i, 'x' * 40)\n    return 0.0\n"
    engine.DATA["DEMO-STOCK"] = DEMO
    out = json.loads(engine.handle("run", json.dumps({"ticker": "DEMO-STOCK", "code": code})))["result"]
    full = json.loads(engine.handle("rows", json.dumps({"kind": "logs"})))["result"]
    assert out["logs_total"] == len(full) > engine.LOG_SHOW
    assert full.count("\n") == DEMO.n - 1 and full.startswith("bar 0 ") and full.rstrip().endswith("x" * 40)
    assert len(out["logs"]) < engine.LOG_SHOW + 200 and "characters (" in out["logs"]
    assert out["logs"].startswith(full[:1000]) and out["logs"].endswith(full[-1000:])


def test_completions_describe_ctx_and_qt():
    c = engine.completions()
    ctx = {e["name"]: e for e in c["ctx"]}
    assert ctx["sma"]["kind"] == "method" and ctx["sma"]["sig"].startswith("(n")
    assert ctx["state"]["kind"] == "attribute" and ctx["price"]["kind"] == "property"
    assert "rsi" in {e["name"] for e in c["qt"]} and "isnan" in {e["name"] for e in c["globals"]}
    assert "weights" in {e["name"] for e in c["portfolio"]} and "close" in {e["name"] for e in c["data"]}


# ------------------------------------------------------- risk analytics -----

def test_special_functions_match_scipy():
    try:
        from scipy import stats, special
    except ImportError:
        stats = special = None
    for df in (1, 2, 3, 7.5, 30, 250):
        for t in (-4.0, -1.2, 0.0, 0.3, 1.96, 6.0):
            got = engine.t_sf(t, df)
            if stats:
                assert abs(got - stats.t.sf(t, df)) < 1e-12, (t, df, got, stats.t.sf(t, df))
            else:   # Simpson integration of the t density from t to a far tail
                c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
                f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)
                hi, n = 400.0, 200000
                h = (hi - t) / n
                ref = h / 3 * (f(t) + f(hi) + sum((4 if i % 2 else 2) * f(t + i * h) for i in range(1, n)))
                assert abs(got - ref) < 1e-6
    assert abs(engine.t_sf(2.0, 1e6) - (1 - engine.norm_cdf(2.0))) < 1e-6
    if special:
        for a, b, x in ((0.5, 0.5, 0.3), (5, 2, 0.9), (30, 40, 0.42), (2.5, 0.5, 0.01)):
            assert abs(engine.betainc(a, b, x) - special.betainc(a, b, x)) < 1e-12
        for df in (1, 2, 3, 5, 12):
            for x in (0.01, 0.5, 3.84, 10.0, 40.0):
                assert abs(engine.chi2_sf(x, df) - stats.chi2.sf(x, df)) < 1e-11, (x, df)


def _rand_regression(n=600, seed=4):
    rng = random.Random(seed)
    f1 = [rng.gauss(0, 0.01) for _ in range(n)]
    f2 = [rng.gauss(0, 0.008) for _ in range(n)]
    e, prev = [], 0.0
    for _ in range(n):                         # autocorrelated, heteroskedastic noise: HAC differs from OLS
        prev = 0.5 * prev + rng.gauss(0, 0.004) * (1 + abs(rng.gauss(0, 1)))
        e.append(prev)
    y = [0.0003 + 1.4 * a - 0.6 * b + u for a, b, u in zip(f1, f2, e)]
    return y, [[1.0, a, b] for a, b in zip(f1, f2)]


def test_ols_and_newey_west_match_numpy():
    import numpy as np
    y, X = _rand_regression()
    fit = engine.ols(y, X)
    Xa, ya = np.array(X), np.array(y)
    b = np.linalg.lstsq(Xa, ya, rcond=None)[0]
    assert np.allclose(fit["coef"], b, rtol=0, atol=1e-12)
    e = ya - Xa @ b
    n, k = Xa.shape
    inv = np.linalg.inv(Xa.T @ Xa)
    assert np.allclose(fit["se"], np.sqrt(np.diag(inv) * (e @ e) / (n - k)), rtol=1e-10)
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    S = (Xa * e[:, None]).T @ (Xa * e[:, None])
    for lag in range(1, L + 1):
        G = (Xa[lag:] * e[lag:, None]).T @ (Xa[:-lag] * e[:-lag, None])
        S += (1 - lag / (L + 1)) * (G + G.T)
    V = inv @ S @ inv
    assert fit["hac_lags"] == L and np.allclose(fit["se_hac"], np.sqrt(np.diag(V)), rtol=1e-9)
    assert fit["se_hac"][1] != fit["se"][1]
    assert abs(fit["coef"][1] - 1.4) < 0.1 and abs(fit["coef"][2] + 0.6) < 0.1
    assert abs(fit["r2"] - (1 - (e @ e) / ((ya - ya.mean()) @ (ya - ya.mean())))) < 1e-12


def test_factor_regression_decompositions_add_up():
    import numpy as np
    engine.DATA["DEMO-STOCK"], engine.DATA["DEMO-BOND"] = DEMO, BOND
    res = run(load_templates()["sma_cross"], risk_free=0.02)
    fr = engine.factor_regression(res, ["DEMO-STOCK", "DEMO-STOCK-DEMO-BOND"])
    assert fr["factors"] == ["DEMO-STOCK", "DEMO-STOCK-DEMO-BOND"]
    total = sum(a["return"] for a in fr["attribution"])
    assert abs(total - fr["mean_excess"]) < 1e-12, "alpha + sum(beta x factor mean) = mean excess return"
    assert abs(sum(v["share"] for v in fr["variance"]) - 1) < 1e-12
    assert abs(sum(v["share"] for v in fr["variance"][:-1]) - fr["r2"]) < 1e-9, "factor shares add up to R-squared"
    # independently: rebuild the regression with numpy on the same dates
    rf = 0.02 / 252
    rs, rb = engine.returns_by_date(DEMO), engine.returns_by_date(BOND)
    ys, xs = [], []
    for t in range(1, len(res.equity)):
        d = res.dates[t]
        if d in rs and d in rb:
            ys.append(res.equity[t] / res.equity[t - 1] - 1 - rf)
            xs.append([1.0, rs[d] - rf, rs[d] - rb[d]])
    b = np.linalg.lstsq(np.array(xs), np.array(ys), rcond=None)[0]
    assert fr["n"] == len(ys)
    assert abs(fr["rows"][0]["coef"] - b[0] * 252) < 1e-10 and abs(fr["rows"][1]["coef"] - b[1]) < 1e-10
    hold = run("def on_bar(ctx):\n    return 1.0\n", risk_free=0.02)
    hb = engine.factor_regression(hold, ["DEMO-STOCK"])
    assert abs(hb["rows"][1]["coef"] - 1) < 0.01 and hb["r2"] > 0.99, "buy and hold has beta 1 to itself"


def test_factor_csv_ken_french_format():
    text = ("This file was created by CMPT_ME_BEME_RETS using the 202401 CRSP database.\n\n"
            ",Mkt-RF,SMB,HML,RF\n"
            "20200102,  0.86, -0.97, -0.33, 0.006\n"
            "20200103, -0.67,  0.30,  0.00, 0.006\n"
            "20200106,  0.36, -0.21, -0.55, 0.006\n\n"
            " Annual Factors: January-December\n,Mkt-RF,SMB,HML,RF\n2020, 23.66, 13.0, -46.5, 0.45\n")
    t = engine.parse_factor_csv(text)
    assert t["dates"] == ["2020-01-02", "2020-01-03", "2020-01-06"] and t["percent"]
    assert list(t["columns"]) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert abs(t["columns"]["Mkt-RF"][0] - 0.0086) < 1e-15 and abs(t["columns"]["RF"][1] - 0.00006) < 1e-15
    plain = engine.parse_factor_csv("date,MOM\n2021-03-01,0.012\n2021-03-02,-0.004\n")
    assert not plain["percent"] and plain["columns"]["MOM"] == [0.012, -0.004]
    # an uploaded table with RF: the strategy's excess return uses that RF
    dates = [d for d in DEMO.dates[1:400]]
    rs = engine.returns_by_date(DEMO)
    engine.FACTORS["ff"] = {"dates": dates, "columns": {"MKT": [rs[d] - 0.0001 for d in dates], "RF": [0.0001] * len(dates)}}
    res = run("def on_bar(ctx):\n    return 1.0\n", risk_free=0.0)
    fr = engine.factor_regression(res, ["ff:MKT"])
    assert fr["rf_source"] == "uploaded RF column" and fr["n"] <= 399 and abs(fr["rows"][1]["coef"] - 1) < 0.01


def test_rolling_beta_matches_direct_computation():
    import numpy as np
    y, X = _rand_regression(400, seed=9)
    b = [r[1] for r in X]
    beta, corr = engine.rolling_beta(y, b, 60)
    assert all(math.isnan(v) for v in beta[:59])
    for t in (59, 150, 399):
        yy, bb = np.array(y[t - 59:t + 1]), np.array(b[t - 59:t + 1])
        c = np.cov(yy, bb)
        assert abs(beta[t] - c[0, 1] / c[1, 1]) < 1e-9 and abs(corr[t] - np.corrcoef(yy, bb)[0, 1]) < 1e-9


def _pf_result():
    engine.DATA["DEMO-STOCK"], engine.DATA["DEMO-BOND"] = DEMO, BOND
    code = "UNIVERSE = ['DEMO-STOCK', 'DEMO-BOND']\ndef on_bar(ctx):\n    return {'DEMO-STOCK': 0.6, 'DEMO-BOND': 0.3}\n"
    return engine.backtest_portfolio(engine.Strategy.from_code(code), [DEMO, BOND], engine.Config(slippage_bps=0), bench=DEMO)


def test_risk_decomposition_contributions_sum_to_vol():
    import numpy as np
    res = _pf_result()
    d = engine.risk_decomposition(res, 252)
    fin = d["final"]
    assert abs(sum(a["contribution"] for a in fin["assets"]) - fin["vol"]) < 1e-12
    assert abs(sum(a["share"] for a in fin["assets"]) - 1) < 1e-12
    R, _ = engine._asset_returns(res, 252)
    w = np.array(engine.current_weights(res))
    C = np.cov(np.array(R).T)
    assert abs(fin["vol"] - math.sqrt(w @ C @ w * 252)) < 1e-12
    assert abs(fin["assets"][0]["weight"] - 0.6) < 0.02 and fin["diversification"] >= 1 - 1e-12


def test_stress_tests_replay_holdings_and_window_returns():
    res = _pf_result()
    custom = ("Test window", res.dates[300], res.dates[340])
    st = engine.stress_tests(res, [custom], shocks=(-0.1,))
    row = [r for r in st["scenarios"] if r["name"] == "Test window"][0]
    i0, i1 = 299, 340       # the close before the window starts, and its last day
    assert abs(row["strategy"] - (res.equity[i1] / res.equity[i0] - 1)) < 1e-12
    w = engine.current_weights(res)
    trs, trb = DEMO.total_return_index(), BOND.total_return_index()
    js = lambda ds, d: ds.dates.index(d)
    exp = w[0] * (trs[js(DEMO, res.dates[i1])] / trs[js(DEMO, res.dates[i0])] - 1) + \
        w[1] * (trb[js(BOND, res.dates[i1])] / trb[js(BOND, res.dates[i0])] - 1)
    assert abs(row["holdings"] - exp) < 1e-12 and row["coverage"] == 1.0
    assert row["strategy_mdd"] <= 0 and row["benchmark"] is not None
    b = st["beta"]
    assert abs(b["betas"]["DEMO-STOCK"] - 1) < 1e-9, "the benchmark's beta to itself is 1"
    assert abs(st["shocks"][0]["pnl"] - b["portfolio_beta"] * -0.1) < 1e-15
    ew = engine.backtest_portfolio(engine.Strategy.from_code("UNIVERSE = ['DEMO-STOCK', 'DEMO-BOND']\ndef on_bar(ctx):\n    return {'DEMO-STOCK': 0.5, 'DEMO-BOND': 0.5}\n"),
                                   [DEMO, BOND], engine.Config(slippage_bps=0))
    assert ew.benchmark == "Equal weight"
    sw = engine.stress_tests(ew, [custom])
    r2 = [r for r in sw["scenarios"] if r["name"] == "Test window"][0]
    assert abs(r2["benchmark"] - (ew.bench_equity[i1] / ew.bench_equity[i0] - 1)) < 1e-12 and sw["shocks"], "equal-weight benchmark from the run"
    # a window before the data starts can't be replayed
    early = engine.stress_tests(res, [("Too early", "1990-01-02", "1990-02-01")])
    assert not any(r["name"] == "Too early" for r in early["scenarios"])


def test_portfolio_var_methods_and_kupiec():
    import numpy as np
    from scipy import stats
    res = _pf_result()
    v = engine.portfolio_var(res, 500, levels=(0.99,), horizon=10)
    R, _ = engine._asset_returns(res, 500)
    w = np.array(engine.current_weights(res))
    pnl = np.array(R) @ w
    row = v["levels"][0]
    assert abs(row["hist"] + engine.percentile(list(pnl), 1)) < 1e-15
    assert abs(row["hist"] + np.percentile(pnl, 1)) < 1e-12, "same as numpy's linear percentile"
    mu, sig, z = pnl.mean(), math.sqrt(w @ np.cov(np.array(R).T) @ w), stats.norm.ppf(0.01)
    assert abs(row["normal"] + (mu + z * sig)) < 1e-12
    assert abs(row["normal_es"] - (-(mu - sig * stats.norm.pdf(z) / 0.01))) < 1e-12
    S, K = stats.skew(pnl), stats.kurtosis(pnl)
    zcf = z + (z * z - 1) * S / 6 + (z ** 3 - 3 * z) * K / 24 - (2 * z ** 3 - 5 * z) * S * S / 36
    assert abs(row["cornish_fisher"] + (mu + zcf * sig)) < 1e-10
    assert abs(sum(c["var"] for c in row["components"]) - (-z * sig)) < 1e-12
    bt = v["backtest"][0]
    r = engine.daily_returns(res.equity)
    x = sum(1 for t in range(250, len(r)) if r[t] < engine.percentile(r[t - 250:t], 1))
    assert bt["breaches"] == x and bt["days"] == len(r) - 250
    n, p = bt["days"], 0.01
    lr = -2 * ((n - x) * math.log(1 - p) + x * math.log(p) - (n - x) * math.log(1 - x / n) - x * math.log(x / n))
    assert abs(bt["lr"] - lr) < 1e-9 and abs(bt["p_value"] - stats.chi2.sf(lr, 1)) < 1e-12
    assert engine.kupiec_pof(5, 500, 0.01)[1] > 0.99, "exactly the expected number of breaches"


def test_capacity_curve_degrades_with_size():
    engine.DATA["DEMO-STOCK"] = DEMO
    sets, cfg = [DEMO], engine.Config(slippage_bps=0, max_participation=0.1)
    strat = engine.Strategy.from_code(load_templates()["sma_cross"])
    cap = engine.capacity_curve(strat, sets, cfg, levels=[1e4, 1e7, 1e9, 1e11])
    rows = cap["rows"]
    assert cap["impact_forced"] and cap["impact_coef"] == 1.0
    assert rows[0]["cagr"] > rows[1]["cagr"] > rows[2]["cagr"] > rows[3]["cagr"] and rows[0]["costs_pct"] < rows[1]["costs_pct"] < rows[2]["costs_pct"]
    assert rows[3]["partial_fills"] > rows[2]["partial_fills"] > 0 == rows[0]["partial_fills"], "the cap spreads big orders over days"
    assert rows[0]["participation_median"] < rows[1]["participation_median"] < rows[2]["participation_median"]
    assert rows[-1]["participation_p95"] <= 0.1 + 1e-9, "the participation cap holds"
    s0 = rows[0]["sharpe"]
    assert rows[2]["sharpe"] >= 0.5 * s0 > rows[3]["sharpe"] and 1e9 < cap["capacity"] < 1e11, "Sharpe halves between 1e9 and 1e11"
    assert not cap["beyond_tested"]
    small = engine.capacity_curve(strat, sets, cfg, levels=[1e4, 1e5])
    assert small["capacity"] is None and small["beyond_tested"]
    out = json.loads(engine.handle("capacity", json.dumps({"ticker": "DEMO-STOCK", "code": load_templates()["sma_cross"], "levels": [1e4, 1e8]})))
    assert out["ok"] and len(out["result"]["rows"]) == 2


def test_risk_report_api():
    engine.DATA["DEMO-STOCK"], engine.DATA["DEMO-BOND"] = DEMO, BOND
    out = json.loads(engine.handle("risk", json.dumps({
        "ticker": "DEMO-STOCK", "benchmark": "DEMO-BOND", "code": load_templates()["sma_cross"],
        "factors": ["DEMO-STOCK", "NOPE"], "window": 63,
        "scenarios": [{"name": "Mine", "start": DEMO.dates[100], "end": DEMO.dates[150]}]})))
    assert out["ok"], out
    r = out["result"]
    assert "factors" in r["errors"] and "NOPE" in r["errors"]["factors"], "a bad factor is reported, the rest still runs"
    assert {"rolling", "decomposition", "stress", "var"} <= set(r)
    assert r["rolling"]["window"] == 63 and len(r["rolling"]["dates"]) == len(r["rolling"]["beta"])
    assert any(s["name"] == "Mine" for s in r["stress"]["scenarios"])


# --------------------------------------------------------- sizing & limits -----

def _limits(ds_list, **cfg):
    panel = engine.Panel(ds_list)
    return engine._Limits(engine.Config(**cfg), panel), panel


def test_vol_target_and_kelly_scale_from_trailing_returns():
    lim, panel = _limits([DEMO], sizing="vol_target", vol_target=0.05, sizing_lookback=63)
    t = 700
    tr = panel.aligned(0, "total_return")
    r = [tr[u] / tr[u - 1] - 1 for u in range(t - 62, t + 1)]
    sd = engine.stdev(r)
    w = lim.apply({0: 1.0}, t)
    assert abs(w[0] - 0.05 / (sd * math.sqrt(252))) < 1e-12
    assert lim.counts == {"vol_target": 1} and "x " in lim.log[0]["detail"]
    w2 = lim.apply({0: -0.5}, t)
    assert abs(w2[0] + 0.5 * 0.05 / (0.5 * sd * math.sqrt(252))) < 1e-12, "scale targets the weights' own volatility"
    kel, _ = _limits([DEMO], sizing="kelly", kelly_fraction=0.5, sizing_lookback=63, max_leverage=3.0)
    m = engine.mean(r)
    k = kel.apply({0: 1.0}, t)[0]
    assert abs(k - min(3.0, max(0.0, 0.5 * m / (sd * sd)))) < 1e-12
    # too little history: unchanged
    early, _ = _limits([DEMO], sizing="vol_target", vol_target=0.05)
    assert early.apply({0: 1.0}, 5) == {0: 1.0} and not early.log


def test_limits_use_only_past_data():
    cut = 800
    changed = engine.DataSet("DEMO-STOCK", DEMO.dates, DEMO.close[:cut + 1] + [c * 1.5 for c in DEMO.close[cut + 1:]],
                             DEMO.open[:cut + 1] + [c * 1.5 for c in DEMO.open[cut + 1:]], DEMO.high[:cut + 1] + [c * 1.5 for c in DEMO.high[cut + 1:]],
                             DEMO.low[:cut + 1] + [c * 1.5 for c in DEMO.low[cut + 1:]], DEMO.volume)
    a, _ = _limits([DEMO], sizing="vol_target", vol_target=0.1)
    b, _ = _limits([changed], sizing="vol_target", vol_target=0.1)
    assert a.apply({0: 1.0}, cut) == b.apply({0: 1.0}, cut)
    assert a.apply({0: 1.0}, cut + 5) != b.apply({0: 1.0}, cut + 5)


def test_position_net_and_gross_limits():
    lim, _ = _limits([DEMO, BOND], max_weight=0.6, max_net=0.3, max_gross=1.0, allow_short=True)
    w = lim.apply({0: 0.9, 1: -0.2}, 300)
    # cap: 0.9 -> 0.6; net 0.4 > 0.3: longs scaled to 0.3 + 0.2 = 0.5; gross 0.7 within 1.0
    assert abs(w[0] - 0.5) < 1e-12 and abs(w[1] + 0.2) < 1e-12
    assert lim.counts == {"position cap": 1, "net limit": 1}
    g = lim.apply({0: 0.6, 1: -0.6}, 301)
    assert abs(sum(abs(v) for v in g.values()) - 1.0) < 1e-12 and abs(g[0] + g[1]) < 1e-12 and lim.counts["gross limit"] == 1
    neg = lim.apply({0: 0.1, 1: -0.6}, 302)
    assert abs(sum(neg.values()) + 0.3) < 1e-12 and abs(neg[0] - 0.1) < 1e-12, "a net short is cut on the short side"


def test_drawdown_derisking_resizes_without_new_signals():
    code = "def on_bar(ctx):\n    return 1.0 if ctx.bar == 0 else None\n"
    base = run(code)
    dd = engine.drawdown(base.equity)
    assert min(dd) < -0.2, "the demo stock needs a real drawdown for this test"
    res = run(code, dd_limit=0.1, dd_scale=0.5, dd_resume=0.5)
    log = res.limits["log"]
    on = [x for x in log if "weights x 0.50" in x["detail"]]
    off = [x for x in log if "full size" in x["detail"]]
    assert on, log
    i = on[0]["i"] - res.start
    eq = base.equity
    peak = max(eq[:i + 1])
    assert eq[i] / peak - 1 <= -0.1 < max(e / max(eq[:j + 1]) - 1 for j, e in enumerate(eq[:i])) , "triggers on the first close 10% below the high"
    assert abs(res.exposure[i + 2] - 0.5) < 0.05, "half size from the next bar, with no new signal"
    if off:
        j = off[0]["i"] - res.start
        assert abs(res.exposure[j + 2] - 1.0) < 0.05
    assert 0 < res.limits["derisked_share"] < 1
    m, _ = engine.summarize(res)
    assert m["limit_events"] == res.limits["events"] == len(log) and m["derisked_share"] == res.limits["derisked_share"]
    assert min(engine.drawdown(res.equity)) > min(dd), "de-risking shallows the worst drawdown here"


def test_vol_target_brings_realized_vol_near_target():
    code = "def on_bar(ctx):\n    return 1.0\n"
    plain = run(code)
    tgt = run(code, sizing="vol_target", vol_target=0.08, sizing_lookback=63, max_leverage=2.0, rebalance_band=0.0)
    vol = lambda e: engine.stdev(engine.daily_returns(e)) * math.sqrt(252)
    assert abs(vol(tgt.equity) - 0.08) < abs(vol(plain.equity) - 0.08)
    assert 0.06 < vol(tgt.equity) < 0.10, vol(tgt.equity)
    assert tgt.limits["counts"].get("vol_target", 0) > 100 and not math.isnan(tgt.limits["avg_scale"])
    assert engine.summarize(plain)[0]["limit_events"] == 0 and plain.limits is None


def test_limit_settings_are_validated():
    for bad in ({"sizing": "magic"}, {"vol_target": 0}, {"max_weight": -1}, {"dd_limit": 1.5}, {"dd_scale": 2}):
        try:
            engine.Config(**bad)
        except ValueError:
            continue
        raise AssertionError("accepted %r" % bad)


# ---------------------------------------------------- calendars & intraday -----

def test_time_stamps_parse_dates_times_and_epochs():
    cases = {
        "2024-01-02": "2024-01-02", "01/02/2024": "2024-01-02", "20240102": "2024-01-02",
        "2024-01-02 09:30": "2024-01-02T09:30", "2024-01-02T09:30:00Z": "2024-01-02T09:30",
        "2024-01-02T09:30:15-05:00": "2024-01-02T09:30:15", "01/02/2024 1:05 PM": "2024-01-02T13:05",
        "01/02/2024 12:00 AM": "2024-01-02", "01/05/2024 00:00": "2024-01-05", "1704187800": "2024-01-02T09:30",
        "1704187800000": "2024-01-02T09:30", "2024-13-01": None, "2024-01-02 25:00": None,
    }
    for raw, want in cases.items():
        assert engine._iso_stamp(raw) == want, (raw, engine._iso_stamp(raw), want)


def _intraday(days=20, per_day=78, seed=1, ticker="BARS", start="2024-01-02"):
    rng = random.Random(seed)
    d0 = engine._dt.date.fromisoformat(start)
    stamps, close, px = [], [], 100.0
    day = d0
    while len({x[:10] for x in stamps}) < days:
        if day.weekday() < 5:
            for k in range(per_day):
                m = 9 * 60 + 30 + 5 * k
                stamps.append("%sT%02d:%02d" % (day.isoformat(), m // 60, m % 60))
                px *= 1 + rng.gauss(0, 0.002)
                close.append(px)
        day += engine._dt.timedelta(days=1)
    return engine.DataSet(ticker, stamps, close, volume=[1e5] * len(close))


def test_browser_and_python_parse_intraday_csv_identically():
    sep_cols = "Date,Time,Open,High,Low,Close,Volume\n" + "".join(
        "01/%02d/2024,%s,%d,%d,%d,%d.5,1000\n" % (d, t, 100 + d, 102 + d, 99 + d, 101 + d) for d in (2, 3) for t in ("9:30 AM", "12:00 PM", "3:55 PM"))
    iso = "timestamp,open,high,low,close,volume\n" + "".join(
        "2024-03-04T%02d:00:00Z,1,2,0.5,%d,10\n" % (h, h) for h in range(10, 16))
    epoch = "time,open,high,low,close,Volume\n" + "".join("%d,1,1,1,%d,5\n" % (1704187800 + 300 * i, 10 + i) for i in range(5))
    texts = [sep_cols, iso, epoch]
    parsed = browser_parse(texts)
    pys = [engine.parse_csv_text(t, "X") for t in texts]
    assert pys[0].dates[:3] == ["2024-01-02T09:30", "2024-01-02T12:00", "2024-01-02T15:55"]
    assert pys[1].dates[0] == "2024-03-04T10:00" and pys[2].dates[1] == "2024-01-02T09:35"
    if parsed is None:
        print("skip  browser/Python intraday parity (needs node)")
        return
    for text, js, py in zip(texts, parsed, pys):
        assert "error" not in js, js
        via = engine.DataSet.from_dict("X", js)
        for col in ("dates", "open", "high", "low", "close", "volume"):
            assert getattr(via, col) == getattr(py, col), (col, text[:30])


def test_calendars_and_periods_per_year():
    assert engine.guess_calendar(DEMO) == "weekdays" and engine.periods_per_year(DEMO) == 252, "weekday data keeps 252 unless chosen"
    spy = engine.load_csv(os.path.join(ROOT, "data", "SPY.csv"), "SPY")
    assert engine.guess_calendar(spy) == "NYSE" and engine.periods_per_year(spy) == 252
    d0 = engine._dt.date(2022, 1, 1)
    days = [(d0 + engine._dt.timedelta(days=i)).isoformat() for i in range(400)]
    btc = engine.DataSet("BTC", days, [100 * 1.001 ** i for i in range(400)])
    assert engine.guess_calendar(btc) == "24/7" and engine.periods_per_year(btc) == 365
    fx = engine.DataSet("EURUSD", DEMO.dates, DEMO.close)
    fx.calendar = "weekdays"
    assert engine.periods_per_year(fx) == 260
    res = engine.backtest(engine.Strategy.from_code("def on_bar(ctx):\n    return 1.0\n"), btc, engine.Config(slippage_bps=0))
    m, _ = engine.summarize(res)
    r = engine.daily_returns(res.equity)
    assert res.ann == 365 and approx(m["volatility"], engine.stdev(r) * math.sqrt(365))
    q = engine.quality_report(btc)
    assert q["calendar"] == "24/7" and "missing_days" not in q["issues"] and q["periods_per_year"] == 365


def test_intraday_backtest():
    ds = _intraday()
    assert engine.bars_per_day(ds) == 78 and engine.periods_per_year(ds) == 252 * 78
    code = "def on_bar(ctx):\n    if ctx.time == '09:30':\n        ctx.state.setdefault('times', []).append(ctx.time)\n    return 0.5 if ctx.time and ctx.time < '15:00' else 0.0\n"
    res = engine.backtest(engine.Strategy.from_code(code), ds, engine.Config(slippage_bps=0, cash_rate=0.036))
    m, _ = engine.summarize(res)
    r = engine.daily_returns(res.equity)
    assert approx(m["volatility"], engine.stdev(r) * math.sqrt(252 * 78))
    assert m["years"] > 0 and res.ann == 252 * 78
    # the strategy is flat at every close of the day, so only cash earns interest: once per calendar day
    days = [engine._dt.date.fromisoformat(d) for d in sorted({x[:10] for x in ds.dates})]
    assert res.carry["interest_earned"] > 0
    # positions open after 15:00 never: exposure at the 15:55 bars is 0
    assert all(abs(res.exposure[i]) < 1e-9 for i, d in enumerate(res.dates) if d.endswith("15:55"))
    # a dividend dated by day is paid on the first bar of that day
    ds2 = _intraday()
    ds2.set_dividends([("2024-01-10", 0.5)])
    j = ds2.dates.index("2024-01-10T09:30")
    assert ds2.dividends[j] == 0.5 and sum(ds2.dividends) == 0.5
    # a daily benchmark on intraday bars uses only days that have closed
    daily = engine.DataSet("DAY", sorted({x[:10] for x in ds.dates}), [100 + i for i in range(20)])
    res2 = engine.backtest(engine.Strategy.from_code("def on_bar(ctx):\n    return 0.0\n"), ds, engine.Config(), bench=daily)
    i = res2.dates.index("2024-01-04T12:00")
    k = res2.dates.index("2024-01-02T09:30")
    assert approx(res2.bench_equity[i] / res2.bench_equity[k], 101 / 100), "Jan 4 midday sees Jan 3's close, not Jan 4's"
    # end dates include the whole last day
    s, e = engine._resolve_range(ds.dates, "2024-01-03", "2024-01-04")
    assert ds.dates[s] == "2024-01-03T09:30" and ds.dates[e] == "2024-01-04T15:55"


# ----------------------------------------------- delisting & survivorship -----

def _cut(ds, n, ticker):
    return engine.DataSet(ticker, ds.dates[:n], ds.close[:n], ds.open[:n], ds.high[:n], ds.low[:n], ds.volume[:n])


def test_delisted_positions_are_closed_at_the_last_close():
    gone = _cut(BOND, 1000, "GONE")
    code = "def on_bar(ctx):\n    return {'DEMO-STOCK': 0.5, 'GONE': 0.5}\n"
    for ret in (0.0, -0.5):
        res = engine.backtest_portfolio(engine.Strategy.from_code(code), [DEMO, gone], engine.Config(slippage_bps=0, delisting_return=ret))
        d = [f for f in res.fills if f.get("kind") == "delisting"]
        assert len(d) == 1 and d[0]["ticker"] == "GONE" and d[0]["position"] == 0 and d[0]["commission"] == 0
        assert d[0]["date"] == DEMO.dates[1000], "closed on the first bar after its last one"
        assert abs(d[0]["price"] - gone.close[-1] * (1 + ret)) < 1e-9
        assert not [f for f in res.fills if f["ticker"] == "GONE" and f["i"] > d[0]["i"]], "never traded again"
        assert res.weights["GONE"][-1] == 0 and res.delisted[0]["ticker"] == "GONE"
        assert any("GONE has no data after %s" % gone.dates[-1] in n for n in res.notes)
        t = d[0]["i"] - res.start
        held = abs(d[0]["qty"]) * gone.close[-1]
        jump = res.equity[t] - res.equity[t - 1]
        demo_move = res.weights["DEMO-STOCK"][t - 1] * res.equity[t - 1] * (DEMO.close[1000] / DEMO.close[999] - 1)
        assert abs(jump - demo_move - ret * held) < 0.05 * held + 1, (jump, demo_move, ret * held)
    m, _ = engine.summarize(res)
    assert m["final_equity"] > 0


def test_survivorship_note():
    third = engine.DataSet("THIRD", DEMO.dates, [c * 1.01 for c in DEMO.close])
    code = "def on_bar(ctx):\n    return [1 / 3] * 3\n"
    res = engine.backtest_portfolio(engine.Strategy.from_code(code), [DEMO, BOND, third], engine.Config())
    assert any(n.startswith("Survivorship bias: all 3 instruments") for n in res.notes)
    res2 = engine.backtest_portfolio(engine.Strategy.from_code(code), [DEMO, BOND, _cut(third, 1500, "THIRD")], engine.Config())
    assert not any(n.startswith("Survivorship") for n in res2.notes), "a universe with a delisted name has no survivorship note"
    single = run("def on_bar(ctx):\n    return 1.0\n")
    assert not any(n.startswith("Survivorship") for n in single.notes)


# --------------------------------------------------- point-in-time fundamentals -----

def test_fundamentals_csv_and_point_in_time_access():
    text = ("Ticker,Period End,Filing Date,EPS,Revenue,Form\n"
            "DEMO-STOCK,2017-03-31,2017-05-02,1.10,100,10-Q\n"
            "DEMO-STOCK,2017-06-30,2017-08-01,1.25,110,10-Q\n"
            "DEMO-STOCK,2017-09-30,,1.40,120,10-Q\n"
            "DEMO-BOND,2017-03-31,2017-04-20,0.5,10,10-Q\n")
    assert engine.csv_kind(engine.read_csv_table(text)[0]) == "fundamentals"
    f = engine.parse_fundamentals_csv(text)
    assert f["fields"] == ["eps", "revenue"], "text columns like Form are not fields"
    eps = f["tickers"]["DEMO-STOCK"]["eps"]
    assert eps[:2] == [("2017-05-02", 1.1), ("2017-08-01", 1.25)]
    assert eps[2] == ("2017-11-14", 1.4) and "1 row had no filing date" in f["warnings"][0], "period end + 45 days"
    ds = engine.DataSet("DEMO-STOCK", DEMO.dates, DEMO.close, DEMO.open, DEMO.high, DEMO.low, DEMO.volume)
    fp = ds.fingerprint()
    ds.set_fundamentals(f["tickers"]["DEMO-STOCK"])
    assert ds.fingerprint() != fp, "fundamentals are part of the data a run used"
    ser = ds.fundamental_series("eps")
    i = ds.dates.index("2017-05-02")
    assert math.isnan(ser[i]) and ser[i + 1] == 1.1, "known from the bar after its filing date"
    code = ("def on_bar(ctx):\n"
            "    e = ctx.fundamental('eps')\n"
            "    ctx.state.setdefault('seen', {})[ctx.date] = e\n"
            "    return 1.0 if e > 1.2 else 0.0\n")
    strat = engine.Strategy.from_code(code)
    res = engine.backtest(strat, ds, engine.Config(slippage_bps=0))
    first_long = next(i for i, x in enumerate(res.exposure) if x > 0.5)
    known = ds.dates.index("2017-08-01") + 1 - res.start
    assert first_long == known + 1, "decided on the first bar after the filing, filled the next"
    sig = "def signals(data):\n    e = data.fundamental('eps')\n    return [1.0 if v > 1.2 else 0.0 for v in e]\n"
    r2 = engine.backtest(engine.Strategy.from_code(sig), ds, engine.Config(slippage_bps=0))
    assert r2.equity == res.equity
    look = engine.lookahead_check(engine.Strategy.from_code(sig), ds, r2.signals)
    assert not look["violations"], "point-in-time values pass the look-ahead check"
    # portfolio access, aligned to the shared calendar
    bond = engine.DataSet("DEMO-BOND", BOND.dates, BOND.close)
    bond.set_fundamentals(f["tickers"]["DEMO-BOND"])
    pcode = "def on_bar(ctx):\n    return {'DEMO-BOND': 1.0 if ctx['DEMO-BOND'].fundamental('eps') > 0 else 0.0}\n"
    pr = engine.backtest_portfolio(engine.Strategy.from_code(pcode), [ds, bond], engine.Config(slippage_bps=0))
    assert pr.weights["DEMO-BOND"][-1] > 0.9
    try:
        ds.fundamental_series("pe")
        raise AssertionError("unknown field accepted")
    except KeyError as e:
        assert "eps" in str(e)
    out = json.loads(engine.handle("parse_csv", json.dumps({"text": text})))
    assert out["ok"] and out["result"]["kind"] == "fundamentals"


# ------------------------------------------------------------- currencies -----

def test_foreign_currency_prices_convert_unhedged():
    n = 300
    dates = DEMO.dates[:n]
    eur = engine.DataSet("SAP", dates, [100.0] * n)            # flat in euros
    eur.currency = "EUR"
    fx = engine.DataSet("EURUSD", dates, [1.0 + 0.001 * i for i in range(n)])   # the euro rises 0.1 cents a day
    engine.DATA.update({"SAP": eur, "EURUSD": fx})
    try:
        conv = engine.in_base(eur, "USD")
        assert conv.currency == "USD" and conv.close[10] == 100.0 * fx.close[10] and conv.ticker == "SAP"
        assert engine.in_base(eur, "USD") is conv, "conversions are cached (sweeps reuse panels)"
        out = json.loads(engine.handle("run", json.dumps({"ticker": "SAP", "code": "def on_bar(ctx):\n    return 1.0\n",
                                                          "config": {"slippage_bps": 0}})))["result"]
        assert any("converted from EUR to USD with EURUSD" in x for x in out["notes"])
        held = out["equity"][-1] / out["equity"][2]
        assert abs(held - fx.close[-1] / fx.close[2]) < 1e-9, "a flat euro price earns the currency's move"
        # an inverted quote: USDEUR (euros per dollar)
        del engine.DATA["EURUSD"]
        engine.DATA["USDEUR"] = engine.DataSet("USDEUR", dates, [1 / (1.0 + 0.001 * i) for i in range(n)])
        c2 = engine.in_base(eur, "USD")
        assert abs(c2.close[50] - conv.close[50]) < 1e-9
        # base currency EUR: a USD instrument converts the other way
        usd = engine.DataSet("SPYX", dates, [100.0] * n)
        usd.currency = "USD"
        c3 = engine.in_base(usd, "EUR")
        assert abs(c3.close[50] - 100.0 * engine.DATA["USDEUR"].close[50]) < 1e-12
        del engine.DATA["USDEUR"]
        try:
            engine.in_base(eur, "USD")
            raise AssertionError("missing FX accepted")
        except ValueError as e:
            assert "EURUSD" in str(e)
        assert engine.in_base(DEMO, "USD") is DEMO, "no currency set: unchanged"
    finally:
        for t in ("SAP", "EURUSD", "USDEUR"):
            engine.DATA.pop(t, None)


def test_fx_for_intraday_bars_uses_the_previous_day():
    ds = _intraday(days=5)
    ds.currency = "EUR"
    days = sorted({d[:10] for d in ds.dates})
    fx = engine.DataSet("EURUSD", days, [1.1, 1.2, 1.3, 1.4, 1.5])
    c = engine.to_base_currency(ds, fx, False, "USD")
    i = c.dates.index(days[2] + "T12:00")
    assert abs(c.close[i] / ds.close[ds.dates.index(days[2] + "T12:00")] - 1.2) < 1e-12
    assert c.dates[0].startswith(days[1]), "bars before the first known FX rate are left out"


# ----------------------------------------------------------------- futures -----

def _contracts():
    d = DEMO.dates
    c1 = engine.DataSet("ESH", d[0:100], [100 + 0.5 * t for t in range(100)], volume=[1000 - 5 * t for t in range(100)])
    c2 = engine.DataSet("ESM", d[50:200], [105 + 0.5 * (t + 50) for t in range(150)], volume=[20 * t for t in range(150)])
    return c1, c2


def test_continuous_futures_roll_and_adjust():
    c1, c2 = _contracts()
    d = DEMO.dates
    for adjust in ("ratio", "difference", "none"):
        f = engine.continuous_futures([c1, c2], "ES", roll="days", days_before=5, adjust=adjust)
        assert f.kind == "future" and f.dates == d[:200] and len(f.rolls) == 1
        r = f.rolls[0]
        assert r["date"] == d[95] and r["from"] == "ESH" and r["to"] == "ESM" and abs(r["gap"] - 5) < 1e-12
        k = 95
        assert f.close[k:] == c2.close[45:], "after the roll: the new contract's own prices"
        if adjust == "ratio":
            # rolled at the roll day's close: that day's return is the old contract's, the next one's the new contract's
            assert abs(f.close[k] / f.close[k - 1] - c1.close[95] / c1.close[94]) < 1e-12
            assert abs(f.close[k + 1] / f.close[k] - c2.close[46] / c2.close[45]) < 1e-12
            assert abs(f.close[10] / f.close[9] - c1.close[10] / c1.close[9]) < 1e-12, "returns before the roll unchanged"
        elif adjust == "difference":
            assert abs((f.close[k] - f.close[k - 1]) - (c1.close[95] - c1.close[94])) < 1e-12
            assert abs(f.close[0] - (c1.close[0] + 5)) < 1e-12
        else:
            assert abs(f.close[k] - f.close[k - 1] - (c2.close[45] - c1.close[94])) < 1e-12, "raw splice keeps the gap"
    vol = engine.continuous_futures([c1, c2], "ES", roll="volume")
    j = next(t for t in range(50, 100) if c2.volume[t - 50] > c1.volume[t])
    assert vol.rolls[0]["date"] == d[j]
    try:
        engine.continuous_futures([c2, c1])
        raise AssertionError("out-of-order contracts accepted")
    except ValueError:
        pass


def test_futures_take_no_cash_and_pay_no_borrow():
    d = DEMO.dates[:300]
    flat = [100.0] * 300
    fut = engine.DataSet("FUT", d, flat)
    fut.kind = "future"
    stock = engine.DataSet("STK", d, flat)
    cfg = engine.Config(slippage_bps=0, cash_rate=0.05, allow_short=True, borrow_fee=0.02, margin_spread=0.01)
    long_code = "def on_bar(ctx):\n    return 1.0\n"
    rf = engine.backtest(engine.Strategy.from_code(long_code), fut, cfg)
    rs = engine.backtest(engine.Strategy.from_code(long_code), stock, cfg)
    days = (engine._dt.date.fromisoformat(d[-1]) - engine._dt.date.fromisoformat(d[0])).days
    assert rf.carry["interest_earned"] > 0.9 * 10000 * 0.05 * (days - 5) / 360, "a flat future earns interest on its collateral"
    assert rs.carry["interest_earned"] < 10, "a fully invested stock position has no cash left (only the first day's)"
    short_code = "def on_bar(ctx):\n    return -1.0\n"
    sf = engine.backtest(engine.Strategy.from_code(short_code), fut, cfg)
    ss = engine.backtest(engine.Strategy.from_code(short_code), stock, cfg)
    assert sf.carry["borrow_paid"] == 0 and ss.carry["borrow_paid"] > 0
    lev = engine.backtest(engine.Strategy.from_code("def on_bar(ctx):\n    return 2.0\n"), fut, engine.Config(slippage_bps=0, cash_rate=0.05, max_leverage=3, margin_spread=0.01))
    assert lev.carry["interest_paid"] == 0, "2x futures exposure borrows no cash"
    assert fut.fingerprint() != stock.fingerprint()


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
