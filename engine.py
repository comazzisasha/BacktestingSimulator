"""
QuantLab engine: a small, dependency-free backtesting library.

The same file runs in the browser (through Pyodide) and in regular CPython 3.9+,
so the website and the unit tests exercise exactly the same code.

Contents
  1. Math helpers
  2. qt: technical indicators that work on plain Python lists
  3. Data: DataSet (one instrument) and SeriesView (look-ahead-safe window)
  4. Context: the object passed to your on_bar(ctx)
  5. Strategy loading, configuration and the backtest loop
  6. Performance metrics
  7. Research tools: Monte Carlo, in/out-of-sample split, analytics, look-ahead check
  8. JSON API used by the web page (handle(method, payload_json))

Using it from plain Python:

    import engine
    ds = engine.load_csv("SPY.csv", "SPY")
    strat = engine.Strategy.from_code(open("my_strategy.py").read())
    res = engine.backtest(strat, ds, engine.Config(cash=10_000))
    print(res.metrics["sharpe"], res.metrics["max_drawdown"])
"""

import math
import json
import random
import statistics
import sys
import io
import traceback
import linecache
import datetime as _dt
from collections import deque
from types import SimpleNamespace

NAN = float("nan")
TRADING_DAYS = 252
STRATEGY_FILENAME = "<strategy>"


# =============================================================================
# 1. Math helpers
# =============================================================================

def isnan(x):
    """True for NaN or None (both mean 'no value yet')."""
    return x is None or x != x


def _finite(xs):
    return [x for x in xs if x is not None and x == x and not math.isinf(x)]


def mean(xs):
    return sum(xs) / len(xs) if xs else NAN


def stdev(xs, ddof=1):
    n = len(xs)
    if n - ddof <= 0:
        return NAN
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def percentile(xs, q):
    """Linear-interpolated percentile, q in [0, 100] (same as numpy's default)."""
    if not xs:
        return NAN
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def skewness(xs):
    """Population skewness (Fisher-Pearson g1)."""
    n = len(xs)
    if n < 3:
        return NAN
    m = sum(xs) / n
    m2 = sum((x - m) ** 2 for x in xs) / n
    if m2 == 0:
        return NAN
    m3 = sum((x - m) ** 3 for x in xs) / n
    return m3 / m2 ** 1.5


def kurtosis(xs):
    """Excess kurtosis (a normal distribution scores 0)."""
    n = len(xs)
    if n < 4:
        return NAN
    m = sum(xs) / n
    m2 = sum((x - m) ** 2 for x in xs) / n
    if m2 == 0:
        return NAN
    m4 = sum((x - m) ** 4 for x in xs) / n
    return m4 / (m2 * m2) - 3.0


def covariance(a, b, ddof=1):
    n = len(a)
    if n - ddof <= 0 or n != len(b):
        return NAN
    ma, mb = sum(a) / n, sum(b) / n
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (n - ddof)


def correlation(a, b):
    sa, sb = stdev(a), stdev(b)
    if isnan(sa) or isnan(sb) or sa == 0 or sb == 0:
        return NAN
    return covariance(a, b) / (sa * sb)


def _div(a, b):
    if b == 0 or isnan(a) or isnan(b):
        return NAN
    return a / b


# =============================================================================
# 2. qt: technical indicators
#
# Every function takes plain lists and returns a list of the same length.
# Values that cannot be computed yet (the warm-up period) are NaN.
# All indicators are causal: the value at bar i only uses bars 0..i.
# =============================================================================

def sma(x, n):
    """Simple moving average over n bars."""
    n = int(n)
    if n <= 0:
        raise ValueError("sma: n must be a positive integer")
    out = [NAN] * len(x)
    s = 0.0
    bad = 0
    for i, v in enumerate(x):
        if v != v:
            bad += 1
        else:
            s += v
        if i >= n:
            o = x[i - n]
            if o != o:
                bad -= 1
            else:
                s -= o
        if i >= n - 1 and bad == 0:
            out[i] = s / n
    return out


def _smooth(x, n, alpha):
    """Exponential smoothing seeded with the SMA of the first n valid values."""
    n = int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer")
    out = [NAN] * len(x)
    prev = None
    buf = 0.0
    count = 0
    for i, v in enumerate(x):
        if v != v:
            if prev is not None:
                out[i] = prev
            continue
        if prev is None:
            buf += v
            count += 1
            if count == n:
                prev = buf / n
                out[i] = prev
        else:
            prev = alpha * v + (1.0 - alpha) * prev
            out[i] = prev
    return out


def ema(x, n):
    """Exponential moving average, alpha = 2 / (n + 1)."""
    return _smooth(x, n, 2.0 / (int(n) + 1))


def wilder(x, n):
    """Wilder's smoothing (RMA), alpha = 1 / n. Used by RSI, ATR and ADX."""
    return _smooth(x, n, 1.0 / int(n))


def wma(x, n):
    """Linearly weighted moving average (most recent bar has weight n)."""
    n = int(n)
    out = [NAN] * len(x)
    denom = n * (n + 1) / 2.0
    for i in range(n - 1, len(x)):
        window = x[i - n + 1:i + 1]
        if any(v != v for v in window):
            continue
        out[i] = sum(w * v for w, v in zip(range(1, n + 1), window)) / denom
    return out


def rolling_std(x, n, ddof=0):
    """Rolling standard deviation over n bars."""
    n = int(n)
    out = [NAN] * len(x)
    if n - ddof <= 0:
        return out
    s = ss = 0.0
    bad = 0
    for i, v in enumerate(x):
        if v != v:
            bad += 1
        else:
            s += v
            ss += v * v
        if i >= n:
            o = x[i - n]
            if o != o:
                bad -= 1
            else:
                s -= o
                ss -= o * o
        if i >= n - 1 and bad == 0:
            var = (ss - s * s / n) / (n - ddof)
            out[i] = math.sqrt(var) if var > 0 else 0.0
    return out


def rolling_max(x, n):
    """Highest value of the last n bars (including the current one)."""
    n = int(n)
    out = [NAN] * len(x)
    dq = deque()
    for i, v in enumerate(x):
        while dq and x[dq[-1]] <= v:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - n:
            dq.popleft()
        if i >= n - 1:
            out[i] = x[dq[0]]
    return out


def rolling_min(x, n):
    """Lowest value of the last n bars (including the current one)."""
    n = int(n)
    out = [NAN] * len(x)
    dq = deque()
    for i, v in enumerate(x):
        while dq and x[dq[-1]] >= v:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - n:
            dq.popleft()
        if i >= n - 1:
            out[i] = x[dq[0]]
    return out


def returns(x):
    """Simple one-bar returns: x[i] / x[i-1] - 1 (first value NaN)."""
    out = [NAN] * len(x)
    for i in range(1, len(x)):
        out[i] = _div(x[i], x[i - 1]) - 1 if x[i - 1] else NAN
    return out


def log_returns(x):
    out = [NAN] * len(x)
    for i in range(1, len(x)):
        if x[i] > 0 and x[i - 1] > 0:
            out[i] = math.log(x[i] / x[i - 1])
    return out


def roc(x, n):
    """Rate of change over n bars: x[i] / x[i-n] - 1."""
    n = int(n)
    out = [NAN] * len(x)
    for i in range(n, len(x)):
        if x[i - n]:
            out[i] = x[i] / x[i - n] - 1
    return out


momentum = roc


def zscore(x, n):
    """(x - rolling mean) / rolling std over n bars."""
    m = sma(x, n)
    s = rolling_std(x, n, ddof=0)
    return [(_div(v - mu, sd) if not isnan(mu) else NAN) for v, mu, sd in zip(x, m, s)]


def realized_vol(x, n=20):
    """Annualized volatility of log returns over n bars."""
    lr = log_returns(x)
    sd = rolling_std(lr, n, ddof=1)
    return [v * math.sqrt(TRADING_DAYS) if not isnan(v) else NAN for v in sd]


def rsi(x, n=14):
    """Relative Strength Index with Wilder smoothing (0-100)."""
    n = int(n)
    m = len(x)
    out = [NAN] * m
    if m <= n:
        return out

    def value(ag, al):
        if al == 0:
            return 50.0 if ag == 0 else 100.0
        return 100.0 - 100.0 / (1.0 + ag / al)

    g = l = 0.0
    for i in range(1, n + 1):
        d = x[i] - x[i - 1]
        g += max(d, 0.0)
        l += max(-d, 0.0)
    ag, al = g / n, l / n
    out[n] = value(ag, al)
    for i in range(n + 1, m):
        d = x[i] - x[i - 1]
        ag = (ag * (n - 1) + max(d, 0.0)) / n
        al = (al * (n - 1) + max(-d, 0.0)) / n
        out[i] = value(ag, al)
    return out


def macd(x, fast=12, slow=26, signal=9):
    """MACD line, signal line and histogram."""
    f, s = ema(x, fast), ema(x, slow)
    line = [a - b if not (isnan(a) or isnan(b)) else NAN for a, b in zip(f, s)]
    sig = ema(line, signal)
    hist = [a - b if not (isnan(a) or isnan(b)) else NAN for a, b in zip(line, sig)]
    return line, sig, hist


def bbands(x, n=20, k=2.0):
    """Bollinger Bands: (middle, upper, lower) with population std."""
    mid = sma(x, n)
    sd = rolling_std(x, n, ddof=0)
    up = [m + k * s if not isnan(m) else NAN for m, s in zip(mid, sd)]
    lo = [m - k * s if not isnan(m) else NAN for m, s in zip(mid, sd)]
    return mid, up, lo


def true_range(high, low, close):
    out = [NAN] * len(close)
    for i in range(len(close)):
        if i == 0:
            out[i] = high[i] - low[i]
        else:
            pc = close[i - 1]
            out[i] = max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc))
    return out


def atr(high, low, close, n=14):
    """Average True Range (Wilder)."""
    return wilder(true_range(high, low, close), n)


def stochastic(high, low, close, k=14, d=3):
    """Stochastic oscillator: (%K, %D)."""
    hh, ll = rolling_max(high, k), rolling_min(low, k)
    kk = []
    for c, h, l in zip(close, hh, ll):
        if isnan(h):
            kk.append(NAN)
        elif h == l:
            kk.append(50.0)
        else:
            kk.append(100.0 * (c - l) / (h - l))
    return kk, sma(kk, d)


def williams_r(high, low, close, n=14):
    """Williams %R (-100 to 0)."""
    hh, ll = rolling_max(high, n), rolling_min(low, n)
    out = []
    for c, h, l in zip(close, hh, ll):
        if isnan(h) or h == l:
            out.append(NAN)
        else:
            out.append(-100.0 * (h - c) / (h - l))
    return out


def cci(high, low, close, n=20):
    """Commodity Channel Index."""
    n = int(n)
    tp = [(h + l + c) / 3.0 for h, l, c in zip(high, low, close)]
    m = sma(tp, n)
    out = [NAN] * len(tp)
    for i in range(n - 1, len(tp)):
        mu = m[i]
        md = sum(abs(v - mu) for v in tp[i - n + 1:i + 1]) / n
        out[i] = (tp[i] - mu) / (0.015 * md) if md > 0 else 0.0
    return out


def adx(high, low, close, n=14):
    """Average Directional Index: (adx, +DI, -DI)."""
    m = len(close)
    tr = [NAN] * m
    pdm = [NAN] * m
    mdm = [NAN] * m
    for i in range(1, m):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        pdm[i] = up if (up > down and up > 0) else 0.0
        mdm[i] = down if (down > up and down > 0) else 0.0
        pc = close[i - 1]
        tr[i] = max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc))
    atr_ = wilder(tr, n)
    sp, sm = wilder(pdm, n), wilder(mdm, n)
    pdi = [100 * _div(a, t) if not isnan(t) else NAN for a, t in zip(sp, atr_)]
    mdi = [100 * _div(a, t) if not isnan(t) else NAN for a, t in zip(sm, atr_)]
    dx = []
    for a, b in zip(pdi, mdi):
        if isnan(a) or isnan(b) or a + b == 0:
            dx.append(NAN if isnan(a) else 0.0)
        else:
            dx.append(100.0 * abs(a - b) / (a + b))
    return wilder(dx, n), pdi, mdi


def obv(close, volume):
    """On-Balance Volume."""
    out = [0.0] * len(close)
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
        else:
            out[i] = out[i - 1]
    return out


def donchian(high, low, n=20):
    """Donchian channel: (upper, lower, middle) over the last n bars."""
    up, lo = rolling_max(high, n), rolling_min(low, n)
    mid = [(a + b) / 2 if not isnan(a) else NAN for a, b in zip(up, lo)]
    return up, lo, mid


def keltner(high, low, close, n=20, mult=2.0):
    """Keltner channel: (middle EMA, upper, lower) using ATR."""
    mid = ema(close, n)
    a = atr(high, low, close, n)
    up = [m + mult * v if not (isnan(m) or isnan(v)) else NAN for m, v in zip(mid, a)]
    lo = [m - mult * v if not (isnan(m) or isnan(v)) else NAN for m, v in zip(mid, a)]
    return mid, up, lo


def shift(x, k=1):
    """Shift a series k bars later (k > 0) and pad with NaN."""
    k = int(k)
    if k <= 0:
        raise ValueError("shift: k must be positive (negative shifts look into the future)")
    return [NAN] * k + list(x[:-k]) if k < len(x) else [NAN] * len(x)


def crossover(a, b):
    """True at bars where series a crosses above series b."""
    out = [False] * len(a)
    for i in range(1, len(a)):
        if not any(isnan(v) for v in (a[i], b[i], a[i - 1], b[i - 1])):
            out[i] = a[i] > b[i] and a[i - 1] <= b[i - 1]
    return out


def crossunder(a, b):
    """True at bars where series a crosses below series b."""
    return crossover(b, a)


def drawdown(x):
    """Drawdown from the running peak at every bar (0 or negative)."""
    out = []
    peak = -math.inf
    for v in x:
        if v > peak:
            peak = v
        out.append(v / peak - 1 if peak > 0 else 0.0)
    return out


qt = SimpleNamespace(
    sma=sma, ema=ema, wma=wma, wilder=wilder, rolling_std=rolling_std,
    rolling_max=rolling_max, rolling_min=rolling_min, returns=returns,
    log_returns=log_returns, roc=roc, momentum=momentum, zscore=zscore,
    realized_vol=realized_vol, rsi=rsi, macd=macd, bbands=bbands,
    true_range=true_range, atr=atr, stochastic=stochastic, williams_r=williams_r,
    cci=cci, adx=adx, obv=obv, donchian=donchian, keltner=keltner, shift=shift,
    crossover=crossover, crossunder=crossunder, drawdown=drawdown,
    mean=mean, stdev=stdev, percentile=percentile, correlation=correlation,
    isnan=isnan, NAN=NAN,
)


# =============================================================================
# 3. Data
# =============================================================================

PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


class SeriesView:
    """Read-only view of a price column that hides future bars.

    Inside on_bar: ctx.close[-1] is today's close, ctx.close[-2] yesterday's,
    ctx.close[-20:] the last 20 closes, len(ctx.close) the bars seen so far.
    Asking for a bar after today raises IndexError.
    """

    __slots__ = ("_data", "_n")

    def __init__(self, data, n):
        self._data = data
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, k):
        if isinstance(k, slice):
            start, stop, step = k.indices(self._n)
            return self._data[start:stop:step]
        if k < 0:
            k += self._n
        if k < 0 or k >= self._n:
            raise IndexError(
                "bar index out of range: only past and current bars are visible "
                "(%d bars so far)" % self._n)
        return self._data[k]

    def __iter__(self):
        return iter(self._data[:self._n])

    def tolist(self):
        return self._data[:self._n]

    def __repr__(self):
        tail = ", ".join("%.2f" % v for v in self._data[max(0, self._n - 3):self._n])
        return "SeriesView(len=%d, last=[%s])" % (self._n, tail)


def _to_float(v, fallback):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return fallback
    if f != f or f <= 0:
        return fallback
    return f


class DataSet:
    """Daily price history for one instrument, oldest bar first."""

    def __init__(self, ticker, dates, close, open=None, high=None, low=None, volume=None):
        rows = []
        for j, d in enumerate(dates):
            c = _to_float(close[j], None)
            if c is None:
                continue
            o = _to_float(open[j], c) if open is not None else c
            h = _to_float(high[j], max(o, c)) if high is not None else max(o, c)
            l = _to_float(low[j], min(o, c)) if low is not None else min(o, c)
            vol = 0.0
            if volume is not None:
                try:
                    vol = float(volume[j])
                    if vol != vol:
                        vol = 0.0
                except (TypeError, ValueError):
                    vol = 0.0
            rows.append((str(d)[:10], o, max(h, o, c), min(l, o, c), c, vol))
        rows.sort(key=lambda r: r[0])
        dedup = []
        for r in rows:
            if dedup and dedup[-1][0] == r[0]:
                dedup[-1] = r
            else:
                dedup.append(r)
        if len(dedup) < 2:
            raise ValueError("%s: need at least 2 valid rows of price data" % ticker)
        self.ticker = ticker
        self.dates = [r[0] for r in dedup]
        self.open = [r[1] for r in dedup]
        self.high = [r[2] for r in dedup]
        self.low = [r[3] for r in dedup]
        self.close = [r[4] for r in dedup]
        self.volume = [r[5] for r in dedup]
        self.days = [_dt.date.fromisoformat(d) for d in self.dates]
        self.n = len(self.dates)
        self.index = {d: i for i, d in enumerate(self.dates)}
        self._cache = {}

    @classmethod
    def from_dict(cls, ticker, d):
        return cls(ticker, d["dates"], d["close"], d.get("open"), d.get("high"),
                   d.get("low"), d.get("volume"))

    def col(self, name):
        if name not in PRICE_COLUMNS:
            if name == "hlc3":
                return self.cached(("hlc3",), lambda: [(h + l + c) / 3 for h, l, c in
                                                      zip(self.high, self.low, self.close)])
            if name == "hl2":
                return self.cached(("hl2",), lambda: [(h + l) / 2 for h, l in zip(self.high, self.low)])
            raise ValueError("unknown price column %r (use open, high, low, close, volume, hl2, hlc3)" % name)
        return getattr(self, name)

    def cached(self, key, build):
        """Compute a full-length series once and reuse it across bars and runs."""
        v = self._cache.get(key)
        if v is None:
            v = build()
            self._cache[key] = v
        return v

    def view(self, n=None):
        return DataView(self, n)


class DataView:
    """Plain-list snapshot of the first n bars. Passed to signals(data)."""

    def __init__(self, ds, n=None):
        n = ds.n if n is None else n
        self.ticker = ds.ticker
        self.n = n
        self.dates = ds.dates[:n]
        self.days = ds.days[:n]
        self.open = ds.open[:n]
        self.high = ds.high[:n]
        self.low = ds.low[:n]
        self.close = ds.close[:n]
        self.volume = ds.volume[:n]

    def __len__(self):
        return self.n


def parse_csv_text(text, ticker="DATA"):
    """Parse a Nasdaq.com (or Yahoo-style) CSV export into a DataSet."""
    import csv
    reader = csv.reader(io.StringIO(text))
    header = [h.strip().lower() for h in next(reader)]

    def find(*names):
        for nm in names:
            if nm in header:
                return header.index(nm)
        return None

    idx = {
        "date": find("date"),
        "close": find("close/last", "close", "adj close"),
        "open": find("open"),
        "high": find("high"),
        "low": find("low"),
        "volume": find("volume"),
    }
    if idx["date"] is None or idx["close"] is None:
        raise ValueError("CSV needs a Date column and a Close or Close/Last column")
    cols = {k: [] for k in idx}
    for row in reader:
        if not row or len(row) <= idx["close"]:
            continue
        raw = row[idx["date"]].strip()
        try:
            if "/" in raw:
                m, d, y = raw.split("/")
                date = _dt.date(int(y), int(m), int(d)).isoformat()
            else:
                date = _dt.date.fromisoformat(raw[:10]).isoformat()
        except ValueError:
            continue
        cols["date"].append(date)
        for k in ("close", "open", "high", "low", "volume"):
            j = idx[k]
            val = row[j].strip().replace("$", "").replace(",", "") if j is not None and j < len(row) else ""
            cols[k].append(val)
    return DataSet(ticker, cols["date"], cols["close"],
                   cols["open"] if idx["open"] is not None else None,
                   cols["high"] if idx["high"] is not None else None,
                   cols["low"] if idx["low"] is not None else None,
                   cols["volume"] if idx["volume"] is not None else None)


def load_csv(path, ticker=None):
    import os
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    return parse_csv_text(text, ticker or os.path.splitext(os.path.basename(path))[0].upper())


# =============================================================================
# 4. Context: what on_bar(ctx) sees
# =============================================================================

class Context:
    """Everything a strategy may look at on the current bar.

    Prices: ctx.open/high/low/close/volume (SeriesView), ctx.price (today's close)
    Time:   ctx.i, ctx.date ('YYYY-MM-DD'), ctx.day (datetime.date), ctx.weekday (Mon=0)
    Account: ctx.cash, ctx.position (shares), ctx.equity, ctx.weight, ctx.is_flat/long/short
    Trade:  ctx.entry_price, ctx.entry_date, ctx.bars_in_trade
    Memory: ctx.state (dict kept between bars), ctx.params (the PARAMS dict)
    Orders: return a target weight from on_bar, or call ctx.order(qty) /
            ctx.order_target_shares(n). Everything fills on the next bar.
    Indicators: ctx.sma(50), ctx.rsi(14), ctx.bbands(20, 2) ... (see docs).
    Every indicator accepts ago=k to read the value k bars back.
    """

    def __init__(self, ds, account, params, start):
        self._ds = ds
        self._acct = account
        self.params = params
        self.state = {}
        self.ticker = ds.ticker
        self.start_index = start
        self.i = start
        self.open = SeriesView(ds.open, start + 1)
        self.high = SeriesView(ds.high, start + 1)
        self.low = SeriesView(ds.low, start + 1)
        self.close = SeriesView(ds.close, start + 1)
        self.volume = SeriesView(ds.volume, start + 1)
        self._views = (self.open, self.high, self.low, self.close, self.volume)
        self._order_delta = 0.0
        self._order_target = None

    def _advance(self, i):
        self.i = i
        n = i + 1
        for v in self._views:
            v._n = n
        self._order_delta = 0.0
        self._order_target = None

    # ---- time -------------------------------------------------------------
    @property
    def date(self):
        return self._ds.dates[self.i]

    @property
    def day(self):
        return self._ds.days[self.i]

    @property
    def weekday(self):
        return self._ds.days[self.i].weekday()

    @property
    def bar(self):
        """Bars since the backtest started (0 on the first bar)."""
        return self.i - self.start_index

    # ---- account ----------------------------------------------------------
    @property
    def price(self):
        return self._ds.close[self.i]

    @property
    def cash(self):
        return self._acct.cash

    @property
    def position(self):
        return self._acct.shares

    @property
    def equity(self):
        return self._acct.cash + self._acct.shares * self.price

    @property
    def weight(self):
        eq = self.equity
        return self._acct.shares * self.price / eq if eq > 0 else 0.0

    @property
    def is_flat(self):
        return self._acct.shares == 0

    @property
    def is_long(self):
        return self._acct.shares > 0

    @property
    def is_short(self):
        return self._acct.shares < 0

    @property
    def entry_price(self):
        t = self._acct.open_trade
        return t["entry_price"] if t else NAN

    @property
    def entry_date(self):
        t = self._acct.open_trade
        return self._ds.dates[t["entry_i"]] if t else None

    @property
    def bars_in_trade(self):
        t = self._acct.open_trade
        return self.i - t["entry_i"] if t else 0

    # ---- orders -----------------------------------------------------------
    def order(self, qty):
        """Buy (qty > 0) or sell (qty < 0) shares at the next bar."""
        self._order_delta += float(qty)

    def order_target_shares(self, n):
        """Trade whatever is needed to hold exactly n shares after the next fill."""
        self._order_target = float(n)
        self._order_delta = 0.0

    def log(self, *args):
        print(self.date, *args)

    # ---- indicators -------------------------------------------------------
    def _val(self, key, build, ago):
        ago = int(ago)
        if ago < 0:
            raise ValueError("ago must be 0 or positive (negative values would read future bars)")
        arr = self._ds.cached(key, build)
        j = self.i - ago
        if isinstance(arr, tuple):
            return tuple(a[j] if j >= 0 else NAN for a in arr)
        return arr[j] if j >= 0 else NAN

    def _c(self, src):
        return self._ds.col(src)

    def sma(self, n, ago=0, src="close"):
        return self._val(("sma", int(n), src), lambda: sma(self._c(src), n), ago)

    def ema(self, n, ago=0, src="close"):
        return self._val(("ema", int(n), src), lambda: ema(self._c(src), n), ago)

    def wma(self, n, ago=0, src="close"):
        return self._val(("wma", int(n), src), lambda: wma(self._c(src), n), ago)

    def rsi(self, n=14, ago=0, src="close"):
        return self._val(("rsi", int(n), src), lambda: rsi(self._c(src), n), ago)

    def macd(self, fast=12, slow=26, signal=9, ago=0):
        """Returns (macd_line, signal_line, histogram)."""
        return self._val(("macd", int(fast), int(slow), int(signal)),
                         lambda: tuple(macd(self._ds.close, fast, slow, signal)), ago)

    def bbands(self, n=20, k=2.0, ago=0):
        """Returns (middle, upper, lower)."""
        return self._val(("bb", int(n), float(k)), lambda: tuple(bbands(self._ds.close, n, k)), ago)

    def atr(self, n=14, ago=0):
        ds = self._ds
        return self._val(("atr", int(n)), lambda: atr(ds.high, ds.low, ds.close, n), ago)

    def stoch(self, k=14, d=3, ago=0):
        """Returns (%K, %D)."""
        ds = self._ds
        return self._val(("stoch", int(k), int(d)),
                         lambda: tuple(stochastic(ds.high, ds.low, ds.close, k, d)), ago)

    def highest(self, n, ago=0, src="high"):
        return self._val(("max", int(n), src), lambda: rolling_max(self._c(src), n), ago)

    def lowest(self, n, ago=0, src="low"):
        return self._val(("min", int(n), src), lambda: rolling_min(self._c(src), n), ago)

    def donchian(self, n=20, ago=0):
        """Returns (upper, lower, middle) of the last n bars."""
        ds = self._ds
        return self._val(("donchian", int(n)), lambda: tuple(donchian(ds.high, ds.low, n)), ago)

    def keltner(self, n=20, mult=2.0, ago=0):
        """Returns (middle, upper, lower)."""
        ds = self._ds
        return self._val(("keltner", int(n), float(mult)),
                         lambda: tuple(keltner(ds.high, ds.low, ds.close, n, mult)), ago)

    def roc(self, n, ago=0, src="close"):
        return self._val(("roc", int(n), src), lambda: roc(self._c(src), n), ago)

    def zscore(self, n, ago=0, src="close"):
        return self._val(("z", int(n), src), lambda: zscore(self._c(src), n), ago)

    def vol(self, n=20, ago=0):
        """Annualized realized volatility of daily log returns."""
        return self._val(("vol", int(n)), lambda: realized_vol(self._ds.close, n), ago)

    def cci(self, n=20, ago=0):
        ds = self._ds
        return self._val(("cci", int(n)), lambda: cci(ds.high, ds.low, ds.close, n), ago)

    def williams_r(self, n=14, ago=0):
        ds = self._ds
        return self._val(("wr", int(n)), lambda: williams_r(ds.high, ds.low, ds.close, n), ago)

    def adx(self, n=14, ago=0):
        """Returns (adx, plus_di, minus_di)."""
        ds = self._ds
        return self._val(("adx", int(n)), lambda: tuple(adx(ds.high, ds.low, ds.close, n)), ago)

    def obv(self, ago=0):
        ds = self._ds
        return self._val(("obv",), lambda: obv(ds.close, ds.volume), ago)

    def custom(self, name, fn, ago=0):
        """Cache your own indicator. fn(data) gets every bar and must return a
        list of the same length whose value at bar i uses only bars <= i."""
        def build():
            out = list(fn(DataView(self._ds)))
            if len(out) != self._ds.n:
                raise ValueError("custom indicator %r returned %d values, expected %d"
                                 % (name, len(out), self._ds.n))
            return out
        return self._val(("custom", name), build, ago)


# =============================================================================
# 5. Strategy, configuration and the backtest loop
# =============================================================================

class StrategyError(Exception):
    """An error raised by user strategy code, with a readable message."""

    def __init__(self, info):
        super().__init__(info.get("message", "strategy error"))
        self.info = info


def _namespace():
    return {
        "__name__": "strategy",
        "__builtins__": __builtins__,
        "math": math, "statistics": statistics, "random": random,
        "datetime": _dt, "qt": qt, "NAN": NAN, "isnan": isnan,
    }


def _register_source(code):
    lines = code.splitlines(True)
    linecache.cache[STRATEGY_FILENAME] = (len(code), None, lines, STRATEGY_FILENAME)


def format_error(exc, bar_date=None):
    """Turn an exception into {message, trace, line}, keeping only strategy frames."""
    if isinstance(exc, StrategyError):
        return exc.info
    if isinstance(exc, SyntaxError) and exc.filename == STRATEGY_FILENAME:
        text = (exc.text or "").rstrip()
        return {"message": "SyntaxError: %s (line %s)" % (exc.msg, exc.lineno),
                "trace": "  line %s\n    %s" % (exc.lineno, text.strip()),
                "line": exc.lineno, "user": True}
    frames = traceback.extract_tb(exc.__traceback__)
    user = [f for f in frames if f.filename == STRATEGY_FILENAME]
    msg = "%s: %s" % (type(exc).__name__, exc)
    if bar_date:
        msg += "  (on bar %s)" % bar_date
    if user:
        trace = "\n".join("  line %d, in %s\n    %s" % (f.lineno, f.name, (f.line or "").strip())
                          for f in user)
        return {"message": msg, "trace": trace, "line": user[-1].lineno, "user": True}
    return {"message": msg, "trace": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "line": None, "user": False}


class Strategy:
    """A strategy is either on_bar(ctx) -> target weight, or signals(data) -> list of weights."""

    def __init__(self, on_bar=None, signals=None, init=None, params=None, name="strategy"):
        if on_bar is None and signals is None:
            raise StrategyError({"message": "Define on_bar(ctx) or signals(data) in your strategy.",
                                 "trace": "", "line": None, "user": True})
        self.on_bar = on_bar
        self.signals = signals
        self.init = init
        self.params = params if params is not None else {}
        self.name = name

    @property
    def mode(self):
        return "on_bar" if self.on_bar is not None else "signals"

    @classmethod
    def from_code(cls, code, params_override=None):
        _register_source(code)
        ns = _namespace()
        try:
            compiled = compile(code, STRATEGY_FILENAME, "exec")
            exec(compiled, ns)
        except Exception as e:  # noqa: BLE001 - report anything the user wrote
            raise StrategyError(format_error(e))
        params = ns.get("PARAMS", {})
        if not isinstance(params, dict):
            raise StrategyError({"message": "PARAMS must be a dict, e.g. PARAMS = {'fast': 50}",
                                 "trace": "", "line": None, "user": True})
        if params_override:
            for k, v in params_override.items():
                if k in params and isinstance(params[k], bool):
                    v = bool(v)
                elif k in params and isinstance(params[k], int) and float(v) == int(float(v)):
                    v = int(float(v))
                params[k] = v
        ns["PARAMS"] = params
        on_bar = ns.get("on_bar")
        signals = ns.get("signals")
        init = ns.get("init")
        return cls(on_bar if callable(on_bar) else None,
                   signals if callable(signals) and not callable(on_bar) else None,
                   init if callable(init) else None, params)


class Config:
    """Backtest settings. Rates are fractions: 0.0002 is 2 basis points."""

    DEFAULTS = {
        "cash": 10000.0,          # starting cash
        "commission_bps": 0.0,    # commission as basis points of traded value
        "commission_fixed": 0.0,  # flat commission per fill, in dollars
        "slippage_bps": 2.0,      # price moves against you by this many bps on every fill
        "execution": "open",      # 'open' = next bar's open, 'close' = next bar's close
        "allow_short": False,
        "fractional": True,       # allow fractional shares
        "max_leverage": 1.0,      # largest |position value| / equity allowed
        "rebalance_band": 0.02,   # skip weight changes smaller than this share of equity
        "risk_free": 0.0,         # annual risk-free rate for Sharpe/Sortino/alpha (0.04 = 4%)
        "seed": 42,               # seed for the random module (reproducible runs)
    }

    def __init__(self, **kw):
        unknown = set(kw) - set(self.DEFAULTS)
        if unknown:
            raise ValueError("unknown config keys: %s" % ", ".join(sorted(unknown)))
        for k, default in self.DEFAULTS.items():
            v = kw.get(k, default)
            if v is None:
                v = default
            if isinstance(default, bool):
                v = bool(v)
            elif isinstance(default, float):
                v = float(v)
            elif isinstance(default, int):
                v = int(v)
            setattr(self, k, v)
        if self.execution not in ("open", "close"):
            raise ValueError("execution must be 'open' or 'close'")
        if self.cash <= 0:
            raise ValueError("starting cash must be positive")
        if self.max_leverage <= 0:
            raise ValueError("max leverage must be positive")

    def as_dict(self):
        return {k: getattr(self, k) for k in self.DEFAULTS}


class Account:
    """Cash, shares, fills and round-trip trades for a single instrument."""

    def __init__(self, cfg, ds):
        self.cfg = cfg
        self.ds = ds
        self.cash = cfg.cash
        self.shares = 0.0
        self.fills = []
        self.trades = []
        self.open_trade = None
        self.commission = 0.0
        self.slippage = 0.0
        self.traded_notional = 0.0
        self.clipped = 0

    def equity_at(self, price):
        return self.cash + self.shares * price

    # ---- order sizing -----------------------------------------------------
    def execute(self, kind, value, price, i):
        cfg = self.cfg
        sh = self.shares
        eq = self.equity_at(price)
        if eq <= 0:
            return
        slip = cfg.slippage_bps / 1e4
        cb = cfg.commission_bps / 1e4
        L = cfg.max_leverage
        if kind == "weight":
            w = value
            if isnan(w):
                return
            lo = -L if cfg.allow_short else 0.0
            w = min(max(w, lo), L)
            diff = w * eq - sh * price
            if w == 0:
                dq = -sh
            elif diff > 0:
                dq = max(0.0, (diff - cfg.commission_fixed) / (price * (1 + slip) * (1 + cb)))
            else:
                dq = diff / price
            target = sh + dq
            if (sh != 0 and target != 0 and (sh > 0) == (target > 0)
                    and abs(diff) < cfg.rebalance_band * eq):
                return
            report = False
        else:
            target = value if kind == "target" else sh + value
            report = True
        if not cfg.fractional:
            target = float(int(target))  # round toward zero
        dq = target - sh

        # Constraints: no shorting unless allowed, leverage cap, cash for buys.
        clipped = False
        if not cfg.allow_short and sh + dq < 0:
            dq = -sh
            clipped = True
        if dq > 0:
            unit = price * (1 + slip) * (1 + cb)
            room = (self.cash + (L - 1.0) * eq - cfg.commission_fixed) / unit
            cap = L * eq / price - sh  # keeps long exposure <= L x equity
            mx = max(0.0, min(room, cap) if sh >= 0 else room)
            if dq > mx + 1e-9:
                dq = mx
                clipped = True
        elif dq < 0 and cfg.allow_short and sh + dq < 0:
            mx = L * eq / price + min(sh, 0.0)  # extra shares we may short
            allowed = -(max(sh, 0.0) + max(0.0, mx))
            if dq < allowed - 1e-9:
                dq = allowed
                clipped = True
        if not cfg.fractional:
            dq = float(int(dq))
        if clipped and report:
            self.clipped += 1
        if abs(dq) < 1e-9:
            return
        self._fill(dq, price, i)

    # ---- fills and trade bookkeeping ---------------------------------------
    def _fill(self, q, price, i):
        cfg = self.cfg
        prev = self.shares
        slip = cfg.slippage_bps / 1e4
        px = price * (1 + slip) if q > 0 else price * (1 - slip)
        comm = abs(q) * px * cfg.commission_bps / 1e4 + cfg.commission_fixed
        eq_before = self.equity_at(price)
        self.cash -= q * px + comm
        new = prev + q
        if abs(new) < 1e-9:
            new = 0.0
        self.shares = new
        self.commission += comm
        self.slippage += abs(q) * price * slip
        self.traded_notional += abs(q) * price
        self.fills.append({"i": i, "date": self.ds.dates[i], "side": "buy" if q > 0 else "sell",
                           "qty": q, "price": px, "commission": comm, "position": new})
        eq_after = self.equity_at(price)

        flipped = prev != 0 and new != 0 and (prev > 0) != (new > 0)
        if prev != 0 and (new == 0 or flipped):
            self._close_trade(i, px, eq_after)
        if new != 0 and (prev == 0 or flipped):
            self.open_trade = {"entry_i": i, "entry_price": px, "entry_equity": eq_before if prev == 0 else eq_after,
                               "side": "long" if new > 0 else "short", "max_shares": abs(new)}
        elif new != 0 and self.open_trade is not None:
            t = self.open_trade
            if abs(new) > abs(prev):  # adding to the position: update average entry
                t["entry_price"] = (t["entry_price"] * abs(prev) + px * abs(q)) / abs(new)
            t["max_shares"] = max(t["max_shares"], abs(new))

    def _close_trade(self, i, px, eq_after, still_open=False):
        t = self.open_trade
        pnl = eq_after - t["entry_equity"]
        self.trades.append({
            "entry_date": self.ds.dates[t["entry_i"]], "exit_date": self.ds.dates[i],
            "entry_i": t["entry_i"], "exit_i": i, "side": t["side"],
            "entry_price": t["entry_price"], "exit_price": px, "shares": t["max_shares"],
            "pnl": pnl, "ret": pnl / t["entry_equity"] if t["entry_equity"] > 0 else NAN,
            "bars": i - t["entry_i"], "open": still_open,
        })
        self.open_trade = None


class Result:
    """Everything a backtest produced."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to_dict(self):
        return dict(self.__dict__)


def _resolve_range(ds, start=None, end=None):
    s, e = 0, ds.n - 1
    if start:
        while s < ds.n and ds.dates[s] < start[:10]:
            s += 1
    if end:
        while e >= 0 and ds.dates[e] > end[:10]:
            e -= 1
    if e - s < 1:
        raise ValueError("The chosen date range has fewer than 2 bars of %s data." % ds.ticker)
    return s, e


def align_benchmark(ds, bench, s, e):
    """Benchmark closes on the strategy's dates (forward-filled). Moves s forward if
    the benchmark history starts later. Returns (s, closes, note)."""
    note = None
    if bench is None or bench is ds:
        return s, ds.close[s:e + 1], note
    if bench.dates[0] > ds.dates[s]:
        new_s = s
        while new_s < e and ds.dates[new_s] < bench.dates[0]:
            new_s += 1
        note = "Start moved to %s, the first date with %s data." % (ds.dates[new_s], bench.ticker)
        s = new_s
    out = []
    last = None
    j = 0
    bd = bench.dates
    for i in range(s, e + 1):
        d = ds.dates[i]
        while j < bench.n and bd[j] <= d:
            last = bench.close[j]
            j += 1
        out.append(last)
    if any(v is None for v in out):
        raise ValueError("Benchmark %s has no data for the start of this range." % bench.ticker)
    if bench.dates[-1] < ds.dates[e]:
        note = (note + " " if note else "") + "%s data ends %s; it is held flat after that." % (
            bench.ticker, bench.dates[-1])
    return s, out, note


def backtest(strategy, ds, cfg=None, start=None, end=None, bench=None, capture_output=True):
    """Run a strategy over ds and return a Result.

    Timing: on every bar the strategy sees data up to and including that bar's
    close. Orders it places are filled on the NEXT bar (at the open, or at the
    close with execution='close'), so a strategy can never trade on information
    it could not have had.
    """
    if cfg is None:
        cfg = Config()
    elif isinstance(cfg, dict):
        cfg = Config(**cfg)
    s, e = _resolve_range(ds, start, end)
    s, bench_close, bench_note = align_benchmark(ds, bench, s, e)
    notes = [bench_note] if bench_note else []

    random.seed(cfg.seed)
    acct = Account(cfg, ds)
    ctx = Context(ds, acct, strategy.params, s)
    out_buf = io.StringIO()
    old_stdout = sys.stdout
    if capture_output:
        sys.stdout = out_buf
    bar_date = None
    try:
        sig = None
        if strategy.signals is not None:
            raw = strategy.signals(DataView(ds))
            sig = list(raw)
            if len(sig) != ds.n:
                raise StrategyError({"message": "signals(data) returned %d values but the data has %d bars."
                                     % (len(sig), ds.n), "trace": "", "line": None, "user": True})
        if strategy.init is not None:
            ctx._advance(s)
            bar_date = ds.dates[s]
            strategy.init(ctx)

        equity, position, exposure = [], [], []
        pending = None
        wiped = False
        for i in range(s, e + 1):
            c = ds.close[i]
            if pending is not None and not wiped:
                fp = ds.open[i] if cfg.execution == "open" else c
                acct.execute(pending[0], pending[1], fp, i)
                pending = None
            eq = acct.equity_at(c)
            if eq <= 0 and not wiped:
                wiped = True
                notes.append("Equity reached zero on %s; trading stopped." % ds.dates[i])
                if acct.shares != 0:
                    acct._fill(-acct.shares, c, i)
                acct.cash = 0.0
                eq = 0.0
            equity.append(eq)
            position.append(acct.shares)
            exposure.append(acct.shares * c / eq if eq > 0 else 0.0)
            if i == e or wiped:
                continue

            ctx._advance(i)
            bar_date = ds.dates[i]
            if sig is not None:
                w = sig[i]
                if w is not None and not isnan(w):
                    pending = ("weight", float(w))
            else:
                r = strategy.on_bar(ctx)
                if r is not None:
                    try:
                        w = float(r)
                    except (TypeError, ValueError):
                        raise TypeError("on_bar must return a number (target weight) or None, got %r" % (r,))
                    if not isnan(w):
                        pending = ("weight", w)
                elif ctx._order_target is not None:
                    pending = ("target", ctx._order_target)
                elif ctx._order_delta != 0:
                    pending = ("delta", ctx._order_delta)
    except StrategyError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StrategyError(format_error(exc, bar_date))
    finally:
        sys.stdout = old_stdout

    if acct.open_trade is not None:
        last = e
        acct._close_trade(last, ds.close[last], acct.equity_at(ds.close[last]), still_open=True)

    dates = ds.dates[s:e + 1]
    b0 = bench_close[0]
    bench_equity = [cfg.cash * v / b0 for v in bench_close]
    logs = out_buf.getvalue()
    if len(logs) > 20000:
        logs = logs[:20000] + "\n... output truncated ...\n"
    return Result(
        ticker=ds.ticker, benchmark=(bench.ticker if bench is not None else ds.ticker),
        start=s, end=e, dates=dates, close=ds.close[s:e + 1],
        equity=equity, position=position, exposure=exposure,
        bench_equity=bench_equity, fills=acct.fills, trades=acct.trades,
        commission=acct.commission, slippage=acct.slippage,
        traded_notional=acct.traded_notional, clipped=acct.clipped,
        logs=logs, notes=notes, config=cfg, mode=strategy.mode, params=dict(strategy.params),
    )


# =============================================================================
# 6. Performance metrics
# =============================================================================

METRICS = [
    # key, label, group, format, description
    ("total_return", "Total return", "Returns", "pct", "Final equity / starting equity - 1."),
    ("cagr", "CAGR", "Returns", "pct", "Compound annual growth rate: (final / start) ^ (1 / years) - 1."),
    ("final_equity", "Final equity", "Returns", "money", "Account value on the last bar."),
    ("best_day", "Best day", "Returns", "pct", "Largest one-day return."),
    ("worst_day", "Worst day", "Returns", "pct", "Largest one-day loss."),
    ("best_month", "Best month", "Returns", "pct", "Largest calendar-month return."),
    ("worst_month", "Worst month", "Returns", "pct", "Largest calendar-month loss."),
    ("best_year", "Best year", "Returns", "pct", "Largest calendar-year return (first and last years may be partial)."),
    ("worst_year", "Worst year", "Returns", "pct", "Worst calendar-year return."),
    ("pct_pos_days", "Positive days", "Returns", "pct", "Share of days with a gain."),
    ("pct_pos_months", "Positive months", "Returns", "pct", "Share of months with a gain."),

    ("volatility", "Volatility (ann.)", "Risk", "pct", "Standard deviation of daily returns x sqrt(252)."),
    ("downside_dev", "Downside deviation", "Risk", "pct", "Root-mean-square of returns below the risk-free rate, annualized."),
    ("max_drawdown", "Max drawdown", "Risk", "pct", "Largest peak-to-trough fall in equity."),
    ("max_dd_days", "Longest drawdown", "Risk", "days", "Most trading days spent below a previous equity high."),
    ("avg_drawdown", "Average drawdown", "Risk", "pct", "Mean depth of all drawdown episodes."),
    ("ulcer_index", "Ulcer index", "Risk", "pct", "sqrt(mean(drawdown^2)): punishes deep and long drawdowns."),
    ("var_95", "VaR 95% (1 day)", "Risk", "pct", "Historical 1-day Value at Risk: the loss exceeded on 5% of days."),
    ("cvar_95", "CVaR 95% (1 day)", "Risk", "pct", "Expected shortfall: the average loss on the worst 5% of days."),
    ("skew", "Skew", "Risk", "num", "Asymmetry of daily returns. Negative = fat left tail."),
    ("kurtosis", "Excess kurtosis", "Risk", "num", "Tail heaviness vs. a normal distribution (0 = normal)."),

    ("sharpe", "Sharpe ratio", "Risk-adjusted", "num", "(mean daily return - rf) / std x sqrt(252)."),
    ("sortino", "Sortino ratio", "Risk-adjusted", "num", "Like Sharpe but only penalizes downside volatility."),
    ("calmar", "Calmar ratio", "Risk-adjusted", "num", "CAGR / |max drawdown|."),
    ("omega", "Omega ratio", "Risk-adjusted", "num", "Sum of daily gains / sum of daily losses (threshold 0)."),
    ("tail_ratio", "Tail ratio", "Risk-adjusted", "num", "|95th percentile| / |5th percentile| of daily returns."),

    ("beta", "Beta", "Vs benchmark", "num", "Sensitivity to the benchmark: cov(r, rb) / var(rb)."),
    ("alpha", "Alpha (ann.)", "Vs benchmark", "pct", "Jensen's alpha: return not explained by beta x benchmark, annualized."),
    ("correlation", "Correlation", "Vs benchmark", "num", "Correlation of daily returns with the benchmark."),
    ("r_squared", "R-squared", "Vs benchmark", "num", "Share of return variance explained by the benchmark."),
    ("tracking_error", "Tracking error", "Vs benchmark", "pct", "Annualized std of (strategy - benchmark) daily returns."),
    ("information_ratio", "Information ratio", "Vs benchmark", "num", "Annualized excess return / tracking error."),
    ("up_capture", "Up capture", "Vs benchmark", "pct", "Average strategy return on benchmark up days / benchmark's."),
    ("down_capture", "Down capture", "Vs benchmark", "pct", "Average strategy return on benchmark down days / benchmark's. Lower is better."),
    ("excess_cagr", "Excess CAGR", "Vs benchmark", "pct", "Strategy CAGR - benchmark CAGR."),

    ("n_trades", "Trades", "Trading", "int", "Closed round trips (flat -> position -> flat)."),
    ("win_rate", "Win rate", "Trading", "pct", "Share of closed trades with a profit."),
    ("avg_trade", "Average trade", "Trading", "pct", "Mean return per closed trade."),
    ("avg_win", "Average win", "Trading", "pct", "Mean return of winning trades."),
    ("avg_loss", "Average loss", "Trading", "pct", "Mean return of losing trades."),
    ("best_trade", "Best trade", "Trading", "pct", "Largest single-trade return."),
    ("worst_trade", "Worst trade", "Trading", "pct", "Largest single-trade loss."),
    ("profit_factor", "Profit factor", "Trading", "num", "Gross profit / gross loss over closed trades."),
    ("expectancy", "Expectancy", "Trading", "money", "Average dollar profit per closed trade."),
    ("avg_bars", "Avg. bars held", "Trading", "num", "Mean length of a closed trade in bars."),
    ("max_consec_wins", "Max win streak", "Trading", "int", "Most winning trades in a row."),
    ("max_consec_losses", "Max loss streak", "Trading", "int", "Most losing trades in a row."),
    ("exposure", "Time in market", "Trading", "pct", "Share of bars with an open position."),
    ("avg_leverage", "Avg. exposure", "Trading", "pct", "Mean |position value| / equity."),
    ("turnover", "Turnover (ann.)", "Trading", "num", "Traded value / average equity per year (2.0 = account traded twice a year)."),
    ("n_fills", "Fills", "Trading", "int", "Number of executed orders."),
    ("total_costs", "Costs paid", "Trading", "money", "Commissions plus slippage."),
    ("orders_clipped", "Orders reduced", "Trading", "int", "Share orders cut down for lack of cash or the no-short rule."),
]


def daily_returns(equity):
    return [equity[i] / equity[i - 1] - 1 if equity[i - 1] > 0 else 0.0 for i in range(1, len(equity))]


def drawdown_stats(equity, dates):
    dd = drawdown(equity)
    mdd = min(dd) if dd else 0.0
    out = {"max_drawdown": mdd, "drawdown": dd}
    if mdd < 0:
        trough = dd.index(mdd)
        peak = trough
        while peak > 0 and dd[peak] < 0:
            peak -= 1
        rec = None
        for j in range(trough, len(dd)):
            if dd[j] >= 0:
                rec = j
                break
        out["dd_peak"] = dates[peak]
        out["dd_trough"] = dates[trough]
        out["dd_recovery"] = dates[rec] if rec is not None else None
    longest = cur = 0
    episodes = []
    depth = 0.0
    for v in dd:
        if v < 0:
            cur += 1
            depth = min(depth, v)
        else:
            if cur:
                episodes.append(depth)
            cur, depth = 0, 0.0
        longest = max(longest, cur)
    if cur:
        episodes.append(depth)
    out["max_dd_days"] = longest
    out["avg_drawdown"] = mean(episodes) if episodes else 0.0
    out["ulcer_index"] = math.sqrt(mean([v * v for v in dd])) if dd else 0.0
    return out


def period_returns(dates, equity, kind="month"):
    """[(period, return)] for calendar months ('YYYY-MM') or years ('YYYY')."""
    cut = 7 if kind == "month" else 4
    out = []
    prev_end = equity[0]
    cur_key = dates[0][:cut]
    last = equity[0]
    for d, v in zip(dates, equity):
        k = d[:cut]
        if k != cur_key:
            out.append((cur_key, last / prev_end - 1 if prev_end > 0 else 0.0))
            prev_end = last
            cur_key = k
        last = v
    out.append((cur_key, last / prev_end - 1 if prev_end > 0 else 0.0))
    return out


def trade_stats(trades):
    closed = [t for t in trades if not t["open"]]
    m = {"n_trades": len(closed)}
    if not closed:
        return m
    rets = [t["ret"] for t in closed]
    pnls = [t["pnl"] for t in closed]
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    m["win_rate"] = len(wins) / len(closed)
    m["avg_trade"] = mean(rets)
    m["avg_win"] = mean([t["ret"] for t in wins]) if wins else NAN
    m["avg_loss"] = mean([t["ret"] for t in losses]) if losses else NAN
    m["best_trade"] = max(rets)
    m["worst_trade"] = min(rets)
    gp = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    m["profit_factor"] = gp / gl if gl > 0 else (math.inf if gp > 0 else NAN)
    m["expectancy"] = mean(pnls)
    m["avg_bars"] = mean([t["bars"] for t in closed])
    best = worst = cw = cl = 0
    for p in pnls:
        if p > 0:
            cw, cl = cw + 1, 0
        else:
            cw, cl = 0, cl + 1
        best, worst = max(best, cw), max(worst, cl)
    m["max_consec_wins"] = best
    m["max_consec_losses"] = worst
    return m


def performance(dates, equity, rf=0.0, bench_equity=None, trades=None, exposure=None,
                traded_notional=None, costs=None, n_fills=None, clipped=None):
    """Compute every metric in METRICS (those that apply) for one equity curve."""
    m = {}
    n = len(equity)
    m["start"], m["end"], m["bars"] = dates[0], dates[-1], n
    e0, e1 = equity[0], equity[-1]
    years = max((_dt.date.fromisoformat(dates[-1]) - _dt.date.fromisoformat(dates[0])).days / 365.25, 1 / 365.25)
    m["years"] = years
    m["final_equity"] = e1
    m["total_return"] = e1 / e0 - 1
    m["cagr"] = (e1 / e0) ** (1 / years) - 1 if e1 > 0 else -1.0

    r = daily_returns(equity)
    rfd = (1 + rf) ** (1 / TRADING_DAYS) - 1
    ann = math.sqrt(TRADING_DAYS)
    if len(r) >= 2:
        mu, sd = mean(r), stdev(r)
        m["volatility"] = sd * ann
        m["sharpe"] = (mu - rfd) / sd * ann if sd > 0 else NAN
        dsd = math.sqrt(mean([min(x - rfd, 0.0) ** 2 for x in r]))
        m["downside_dev"] = dsd * ann
        m["sortino"] = (mu - rfd) / dsd * ann if dsd > 0 else NAN
        p5, p95 = percentile(r, 5), percentile(r, 95)
        m["var_95"] = -p5
        m["cvar_95"] = -mean([x for x in r if x <= p5])
        m["skew"] = skewness(r)
        m["kurtosis"] = kurtosis(r)
        m["best_day"], m["worst_day"] = max(r), min(r)
        m["pct_pos_days"] = sum(1 for x in r if x > 0) / len(r)
        gains = sum(x for x in r if x > 0)
        losses = -sum(x for x in r if x < 0)
        m["omega"] = gains / losses if losses > 0 else NAN
        m["tail_ratio"] = abs(p95 / p5) if p5 != 0 else NAN

    dd = drawdown_stats(equity, dates)
    m["max_drawdown"] = dd["max_drawdown"]
    m["max_dd_days"] = dd["max_dd_days"]
    m["avg_drawdown"] = dd["avg_drawdown"]
    m["ulcer_index"] = dd["ulcer_index"]
    for k in ("dd_peak", "dd_trough", "dd_recovery"):
        if k in dd:
            m[k] = dd[k]
    m["calmar"] = m["cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] < 0 else NAN

    months = [v for _, v in period_returns(dates, equity, "month")]
    yrs = [v for _, v in period_returns(dates, equity, "year")]
    if months:
        m["best_month"], m["worst_month"] = max(months), min(months)
        m["pct_pos_months"] = sum(1 for v in months if v > 0) / len(months)
    if yrs:
        m["best_year"], m["worst_year"] = max(yrs), min(yrs)

    if exposure is not None and exposure:
        m["exposure"] = sum(1 for w in exposure if abs(w) > 1e-9) / len(exposure)
        m["avg_leverage"] = mean([abs(w) for w in exposure])
    if traded_notional is not None:
        m["turnover"] = traded_notional / mean(equity) / years if mean(equity) > 0 else NAN
    if costs is not None:
        m["total_costs"] = costs
    if n_fills is not None:
        m["n_fills"] = n_fills
    if clipped is not None:
        m["orders_clipped"] = clipped
    if trades is not None:
        m.update(trade_stats(trades))

    if bench_equity is not None and len(bench_equity) == n and n > 2:
        rb = daily_returns(bench_equity)
        varb = stdev(rb) ** 2
        mub = mean(rb)
        beta = covariance(r, rb) / varb if varb > 0 else NAN
        m["beta"] = beta
        m["alpha"] = ((mean(r) - rfd) - beta * (mub - rfd)) * TRADING_DAYS if not isnan(beta) else NAN
        corr = correlation(r, rb)
        m["correlation"] = corr
        m["r_squared"] = corr * corr if not isnan(corr) else NAN
        diff = [a - b for a, b in zip(r, rb)]
        te = stdev(diff) * ann
        m["tracking_error"] = te
        m["information_ratio"] = mean(diff) * TRADING_DAYS / te if te > 0 else NAN
        up = [(a, b) for a, b in zip(r, rb) if b > 0]
        dn = [(a, b) for a, b in zip(r, rb) if b < 0]
        m["up_capture"] = _div(mean([a for a, _ in up]), mean([b for _, b in up])) if up else NAN
        m["down_capture"] = _div(mean([a for a, _ in dn]), mean([b for _, b in dn])) if dn else NAN
        b_cagr = (bench_equity[-1] / bench_equity[0]) ** (1 / years) - 1 if bench_equity[-1] > 0 else -1.0
        m["excess_cagr"] = m["cagr"] - b_cagr
    return m


def rolling_sharpe(equity, window=126, rf=0.0):
    r = daily_returns(equity)
    rfd = (1 + rf) ** (1 / TRADING_DAYS) - 1
    out = [NAN] * len(equity)
    if len(r) < window:
        return out
    s = sum(r[:window])
    ss = sum(x * x for x in r[:window])
    for t in range(window, len(r) + 1):
        if t > window:
            new, old = r[t - 1], r[t - 1 - window]
            s += new - old
            ss += new * new - old * old
        mu = s / window
        var = (ss - s * s / window) / (window - 1)
        out[t] = (mu - rfd) / math.sqrt(var) * math.sqrt(TRADING_DAYS) if var > 1e-18 else NAN
    return out


def histogram(series_list, bins=40):
    vals = [x for s in series_list for x in s]
    if not vals:
        return {"edges": [], "counts": [[] for _ in series_list]}
    lo, hi = percentile(vals, 0.5), percentile(vals, 99.5)
    if hi <= lo:
        lo, hi = min(vals), max(vals) + 1e-9
    width = (hi - lo) / bins
    edges = [lo + k * width for k in range(bins + 1)]
    counts = []
    for s in series_list:
        c = [0] * bins
        for x in s:
            k = int((x - lo) / width)
            c[min(max(k, 0), bins - 1)] += 1
        counts.append(c)
    return {"edges": edges, "counts": counts}


def monthly_table(dates, equity):
    months = dict(period_returns(dates, equity, "month"))
    years = dict(period_returns(dates, equity, "year"))
    rows = []
    for y in sorted(years):
        rows.append({"year": y, "months": [months.get("%s-%02d" % (y, k)) for k in range(1, 13)],
                     "total": years[y]})
    return rows


def summarize(res, rf=None):
    """Strategy and benchmark metrics for a Result."""
    rf = res.config.risk_free if rf is None else rf
    m = performance(res.dates, res.equity, rf, res.bench_equity, res.trades, res.exposure,
                    res.traded_notional, res.commission + res.slippage, len(res.fills), res.clipped)
    b = performance(res.dates, res.bench_equity, rf)
    return m, b


# =============================================================================
# 7. Research tools
# =============================================================================

def monte_carlo(returns_, n_paths=500, block=20, seed=7, points=120):
    """Circular block bootstrap of daily returns.

    Resamples blocks of `block` consecutive days (keeping short-term patterns
    such as volatility clustering) to build n_paths alternative histories of
    the same length. Shows how much of a result could be luck of the ordering.
    """
    R = list(returns_)
    T = len(R)
    if T < 20:
        raise ValueError("Need at least 20 daily returns for a Monte Carlo run.")
    n_paths = max(10, min(int(n_paths), 5000))
    block = max(1, min(int(block), T))
    rng = random.Random(seed)
    points = max(2, min(points, T))
    checkpoints = sorted(set(int(round(j * (T - 1) / (points - 1))) for j in range(points)))
    snaps = [[] for _ in checkpoints]
    years = T / TRADING_DAYS
    finals, cagrs, mdds = [], [], []
    for _ in range(n_paths):
        eq = peak = 1.0
        mdd = 0.0
        t = ci = 0
        while t < T:
            st = rng.randrange(T)
            for j in range(block):
                if t >= T:
                    break
                eq *= 1.0 + R[(st + j) % T]
                if eq > peak:
                    peak = eq
                elif eq / peak - 1 < mdd:
                    mdd = eq / peak - 1
                if ci < len(checkpoints) and t == checkpoints[ci]:
                    snaps[ci].append(eq)
                    ci += 1
                t += 1
        finals.append(eq - 1)
        cagrs.append(eq ** (1 / years) - 1 if eq > 0 else -1.0)
        mdds.append(mdd)
    actual = [1.0]
    for x in R:
        actual.append(actual[-1] * (1 + x))
    qs = (5, 25, 50, 75, 95)
    return {
        "n_paths": n_paths, "block": block, "checkpoints": checkpoints,
        "fan": {str(q): [percentile(s, q) for s in snaps] for q in qs},
        "actual": [actual[c + 1] for c in checkpoints],
        "final": {str(q): percentile(finals, q) for q in qs},
        "cagr": {str(q): percentile(cagrs, q) for q in qs},
        "max_drawdown": {str(q): percentile(mdds, q) for q in qs},
        "prob_loss": sum(1 for f in finals if f < 0) / n_paths,
        "actual_final": actual[-1] - 1,
        "actual_mdd": min(drawdown(actual)),
        "prob_worse_dd": sum(1 for d in mdds if d < min(drawdown(actual))) / n_paths,
        "cagr_hist": histogram([cagrs], 30),
    }


def split_metrics(res, split_index, rf=None):
    """Metrics for the in-sample (before split) and out-of-sample (after) parts of
    one continuous backtest. Positions and indicator warm-up carry across the split."""
    rf = res.config.risk_free if rf is None else rf
    n = len(res.dates)
    k = max(2, min(int(split_index), n - 2))

    def seg(a, b):
        d = res.dates[a:b]
        eq = res.equity[a:b]
        be = res.bench_equity[a:b]
        lo, hi = res.start + a, res.start + b - 1
        trades = [t for t in res.trades if lo <= t["entry_i"] <= hi]
        notional = sum(abs(f["qty"]) * f["price"] for f in res.fills if lo <= f["i"] <= hi)
        fills = sum(1 for f in res.fills if lo <= f["i"] <= hi)
        return {
            "start": d[0], "end": d[-1],
            "metrics": performance(d, eq, rf, be, trades, res.exposure[a:b], notional, None, fills),
            "bench": performance(d, be, rf),
        }

    return {"split_date": res.dates[k], "in_sample": seg(0, k + 1), "out_of_sample": seg(k, n)}


def lookahead_check(strategy, ds, full_signals, samples=8):
    """For signals(data) strategies: recompute the signal using only data up to bar k
    and compare with the full-history value. Any difference means the signal for bar k
    used later bars."""
    n = ds.n
    lo = max(1, n // 5)
    picks = sorted(set(lo + (n - 1 - lo) * j // max(1, samples - 1) for j in range(samples)))
    bad = []
    for k in picks:
        sub = list(strategy.signals(DataView(ds, k + 1)))
        a, b = sub[k] if len(sub) > k else None, full_signals[k]
        same = (isnan(a) and isnan(b)) or (not isnan(a) and not isnan(b) and abs(float(a) - float(b)) < 1e-9)
        if not same:
            bad.append({"date": ds.dates[k], "with_future": b, "without_future": a})
    return {"checked": len(picks), "violations": bad}


def instrument_stats(datasets, start=None, end=None, rf=0.0):
    """Per-instrument buy-and-hold stats plus a correlation matrix of daily returns
    on the dates all instruments share."""
    rows = []
    for ds in datasets:
        try:
            s, e = _resolve_range(ds, start, end)
        except ValueError:
            continue
        eq = ds.close[s:e + 1]
        m = performance(ds.dates[s:e + 1], eq, rf)
        rows.append({"ticker": ds.ticker, "start": ds.dates[s], "end": ds.dates[e], "bars": e - s + 1,
                     "last": eq[-1], "total_return": m["total_return"], "cagr": m["cagr"],
                     "volatility": m.get("volatility"), "sharpe": m.get("sharpe"),
                     "max_drawdown": m["max_drawdown"], "skew": m.get("skew"),
                     "best_day": m.get("best_day"), "worst_day": m.get("worst_day")})
    used = [ds for ds in datasets if any(r["ticker"] == ds.ticker for r in rows)]
    common = None
    for ds in used:
        s, e = _resolve_range(ds, start, end)
        ds_dates = set(ds.dates[s:e + 1])
        common = ds_dates if common is None else common & ds_dates
    common = sorted(common or [])
    tickers = [ds.ticker for ds in used]
    matrix, rebased = [], {}
    if len(common) >= 3:
        rets = {}
        for ds in used:
            closes = [ds.close[ds.index[d]] for d in common]
            rets[ds.ticker] = daily_returns(closes)
            step = max(1, len(common) // 600)
            rebased[ds.ticker] = [100 * closes[j] / closes[0] for j in range(0, len(common), step)]
        matrix = [[correlation(rets[a], rets[b]) if a != b else 1.0 for b in tickers] for a in tickers]
        step = max(1, len(common) // 600)
        rebased_dates = [common[j] for j in range(0, len(common), step)]
    else:
        rebased_dates = []
    return {"rows": rows, "tickers": tickers, "matrix": matrix, "common_start": common[0] if common else None,
            "common_end": common[-1] if common else None, "common_bars": len(common),
            "rebased": rebased, "rebased_dates": rebased_dates}


# =============================================================================
# 8. JSON API for the web page
# =============================================================================

DATA = {}


def _sanitize(x):
    if isinstance(x, float):
        return x if (x == x and not math.isinf(x)) else None
    if isinstance(x, dict):
        return {str(k): _sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sanitize(v) for v in x]
    return x


def _get(ticker):
    if ticker not in DATA:
        raise ValueError("No data loaded for %s." % ticker)
    return DATA[ticker]


def _prepare(p):
    ds = _get(p["ticker"])
    b = p.get("benchmark") or p["ticker"]
    bench = _get(b) if b != p["ticker"] else None
    cfg = Config(**(p.get("config") or {}))
    strat = Strategy.from_code(p["code"], p.get("params"))
    return ds, bench, cfg, strat


def api_load_data(p):
    ds = DataSet.from_dict(p["ticker"], p["data"])
    DATA[p["ticker"]] = ds
    return {"ticker": ds.ticker, "bars": ds.n, "start": ds.dates[0], "end": ds.dates[-1]}


def api_inspect(p):
    strat = Strategy.from_code(p["code"])
    return {"params": strat.params, "mode": strat.mode}


def api_run(p):
    ds, bench, cfg, strat = _prepare(p)
    res = backtest(strat, ds, cfg, p.get("start"), p.get("end"), bench)
    m, b = summarize(res)
    window = int(p.get("rolling_window") or 126)
    out = {
        "ticker": res.ticker, "benchmark": res.benchmark, "mode": res.mode, "params": res.params,
        "dates": res.dates, "close": res.close, "equity": res.equity, "bench_equity": res.bench_equity,
        "drawdown": drawdown(res.equity), "bench_drawdown": drawdown(res.bench_equity),
        "exposure": res.exposure, "position": res.position,
        "fills": [{"i": f["i"] - res.start, "date": f["date"], "side": f["side"], "qty": f["qty"],
                   "price": f["price"], "commission": f["commission"], "position": f["position"]}
                  for f in res.fills],
        "trades": res.trades, "metrics": m, "bench_metrics": b,
        "monthly": monthly_table(res.dates, res.equity),
        "bench_monthly": monthly_table(res.dates, res.bench_equity),
        "hist": histogram([daily_returns(res.equity), daily_returns(res.bench_equity)]),
        "rolling": {"window": window, "sharpe": rolling_sharpe(res.equity, window, cfg.risk_free),
                    "bench": rolling_sharpe(res.bench_equity, window, cfg.risk_free)},
        "logs": res.logs, "notes": res.notes, "config": cfg.as_dict(),
    }
    if strat.signals is not None:
        full = list(strat.signals(DataView(ds)))
        out["lookahead"] = lookahead_check(strat, ds, full)
    return out


def api_metrics(p):
    """Lightweight run for parameter sweeps: metrics only, optionally split."""
    ds, bench, cfg, strat = _prepare(p)
    res = backtest(strat, ds, cfg, p.get("start"), p.get("end"), bench)
    m, _ = summarize(res)
    out = {"params": res.params, "full": m}
    if p.get("split") is not None:
        k = int(round(float(p["split"]) * (len(res.dates) - 1)))
        sm = split_metrics(res, k)
        out["in_sample"] = sm["in_sample"]["metrics"]
        out["out_of_sample"] = sm["out_of_sample"]["metrics"]
        out["split_date"] = sm["split_date"]
    return out


def api_montecarlo(p):
    ds, bench, cfg, strat = _prepare(p)
    res = backtest(strat, ds, cfg, p.get("start"), p.get("end"), bench)
    mc = monte_carlo(daily_returns(res.equity), p.get("paths", 500), p.get("block", 20), p.get("seed", 7))
    mc["dates"] = [res.dates[c + 1] for c in mc["checkpoints"]]
    mc["start_date"] = res.dates[0]
    return mc


def api_split(p):
    ds, bench, cfg, strat = _prepare(p)
    res = backtest(strat, ds, cfg, p.get("start"), p.get("end"), bench)
    if p.get("split_date"):
        k = 0
        while k < len(res.dates) - 1 and res.dates[k] < p["split_date"]:
            k += 1
    else:
        k = int(round(float(p.get("split", 0.7)) * (len(res.dates) - 1)))
    out = split_metrics(res, k)
    out["dates"] = res.dates
    out["equity"] = res.equity
    out["bench_equity"] = res.bench_equity
    return out


def api_analytics(p):
    sets = [_get(t) for t in p["tickers"] if t in DATA]
    return instrument_stats(sets, p.get("start"), p.get("end"), float(p.get("rf", 0.0)))


def api_metric_list(p):
    return [{"key": k, "label": lab, "group": g, "fmt": f, "desc": d} for k, lab, g, f, d in METRICS]


API = {
    "load_data": api_load_data, "inspect": api_inspect, "run": api_run, "metrics": api_metrics,
    "montecarlo": api_montecarlo, "split": api_split, "analytics": api_analytics,
    "metric_list": api_metric_list,
}


def handle(method, payload_json):
    """Entry point for the web page: returns a JSON string {ok, result | error}."""
    try:
        p = json.loads(payload_json) if payload_json else {}
        if method not in API:
            raise ValueError("unknown method %r" % method)
        return json.dumps({"ok": True, "result": _sanitize(API[method](p))}, allow_nan=False)
    except StrategyError as e:
        return json.dumps({"ok": False, "error": _sanitize(e.info)})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"ok": False, "error": _sanitize(format_error(e))})
