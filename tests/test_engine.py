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
    assert all(m["tier"] in ("core", "std", "extra") for m in json.loads(engine.handle("metric_list", "{}"))["result"])
    base = {"ticker": "DEMO-STOCK", "benchmark": "DEMO-BOND", "code": load_templates()["sma_cross"]}
    for method, extra in (("subperiods", {"n": 3}), ("costs", {"levels": [0, 10]}), ("permutation", {"n_perm": 50})):
        out = json.loads(engine.handle(method, json.dumps(dict(base, **extra))))
        assert out["ok"], (method, out)


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
