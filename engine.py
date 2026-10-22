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
  7. Research tools: Monte Carlo, in/out-of-sample split, sub-periods, cost
     sensitivity, random-timing test, analytics, look-ahead check
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


def norm_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


Z_95 = 1.6448536269514722   # one-sided 95% quantile of the standard normal
Z_975 = 1.959963984540054   # two-sided 95%


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


def walk_forward(X, y, fit_predict, min_train=252, retrain_every=21, horizon=1, window=None, fill=NAN):
    """Out-of-sample model predictions for every bar, without look-ahead.

    X             one row of features per bar (list of lists or a 2-D NumPy array).
                  Rows containing NaN are skipped for training and get `fill`.
    y             one label per bar describing what happens AFTER that bar, e.g. the
                  next day's return. y[i] only becomes known `horizon` bars later.
    fit_predict   fit_predict(X_train, y_train, X_new) -> one prediction per row of X_new.
                  Gets NumPy arrays when X is one, else lists.

    Retrains every `retrain_every` bars, starting once `min_train` labelled rows
    exist. A model trained at bar t only sees rows j with j + horizon <= t (labels
    already known at t's close), the most recent `window` of them if set, and
    predicts bars t .. t + retrain_every - 1. Returns one value per bar.
    """
    n = len(X)
    if len(y) != n:
        raise ValueError("walk_forward: X has %d rows but y has %d" % (n, len(y)))
    retrain_every, horizon = max(1, int(retrain_every)), max(1, int(horizon))
    is_array = hasattr(X, "shape")
    take_x = (lambda idx: X[idx]) if is_array else (lambda idx: [X[j] for j in idx])
    y_array = hasattr(y, "shape")
    take_y = (lambda idx: y[idx]) if y_array else (lambda idx: [y[j] for j in idx])
    ok_x = [all(v == v for v in row) for row in X]
    ok_y = [not isnan(v) for v in y]
    out = [fill] * n
    known = []          # usable training rows whose labels are known by bar t
    t = 0
    started = False
    while t < n:
        j = t - horizon
        if j >= 0 and ok_x[j] and ok_y[j]:
            known.append(j)
        if not started and len(known) < min_train:
            t += 1
            continue
        started = True
        train = known[-int(window):] if window else known
        new = [k for k in range(t, min(t + retrain_every, n)) if ok_x[k]]
        if new and train:
            preds = fit_predict(take_x(train), take_y(train), take_x(new))
            for k, p in zip(new, preds):
                out[k] = float(p)
        # collect the labels that become known before the next retrain
        for t2 in range(t + 1, min(t + retrain_every, n)):
            j = t2 - horizon
            if j >= 0 and ok_x[j] and ok_y[j]:
                known.append(j)
        t += retrain_every
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


def _iso_date(raw):
    """'MM/DD/YYYY', 'YYYY-MM-DD' (optionally with a time) -> 'YYYY-MM-DD', else None."""
    raw = str(raw).strip().strip('"')
    try:
        if "/" in raw:
            m, d, y = raw.split("/")[:3]
            return _dt.date(int(y[:4]), int(m), int(d)).isoformat()
        return _dt.date.fromisoformat(raw[:10]).isoformat()
    except ValueError:
        return None


def _num(s):
    """Parse '$1,234.5', '1.2%', 'N/A' ... -> float or None."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s) if s == s else None
    s = str(s).strip().replace("$", "").replace(",", "").replace("%", "")
    if s in ("", ".", "-", "N/A", "n/a", "null", "None", "NaN", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


MAX_ISSUE_EXAMPLES = 8   # examples kept per kind of data-quality finding


class DataSet:
    """Daily price history for one instrument, oldest bar first.

    close is the price you trade at: split-adjusted but NOT dividend-adjusted
    (Nasdaq's Close/Last, Yahoo's Close). Cash dividends live in `dividends`
    (amount per share on the ex-date bar) and are paid into the account, so
    P&L is total return while signals see real prices. If only an adjusted
    close is available it is used as the price (`price_basis = 'total_return'`).
    """

    def __init__(self, ticker, dates, close, open=None, high=None, low=None, volume=None,
                 adj_close=None, dividends=None):
        self.issues = {}
        rows = []
        for j, d in enumerate(dates):
            day = _iso_date(d) if not isinstance(d, _dt.date) else d.isoformat()
            c = _to_float(close[j], None)
            if day is None or c is None:
                self._issue("dropped", day or str(d), "row without a valid date and positive close was skipped")
                continue
            o = _to_float(open[j], None) if open is not None else None
            h = _to_float(high[j], None) if high is not None else None
            l = _to_float(low[j], None) if low is not None else None
            if open is not None and o is None:
                self._issue("missing_ohlc", day, "open missing; the close was used")
            o = c if o is None else o
            h = max(o, c) if h is None else h
            l = min(o, c) if l is None else l
            if h < max(o, c) - 1e-9 or l > min(o, c) + 1e-9 or l > h:
                self._issue("ohlc", day, "high/low inconsistent with open/close (H %.4g L %.4g O %.4g C %.4g); widened to fit"
                            % (h, l, o, c))
            vol = _num(volume[j]) if volume is not None else None
            a = _to_float(adj_close[j], None) if adj_close is not None else None
            rows.append((day, o, max(h, o, c), min(l, o, c), c, vol if vol is not None and vol >= 0 else 0.0, a))
        rows.sort(key=lambda r: r[0])
        dedup = []
        for r in rows:
            if dedup and dedup[-1][0] == r[0]:
                self._issue("duplicate", r[0], "two rows for the same date; the later one was kept")
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
        self.dividends = [0.0] * self.n
        self.dividend_source = None      # None (price return only), 'adj_close' or 'events'
        self.price_basis = "price"       # 'price' or 'total_return' (only an adjusted close was given)
        self.splits_applied = []
        self.kind = "price"
        self._cache = {}
        adj = [r[6] for r in dedup]
        if adj_close is not None and all(a is not None for a in adj):
            self._dividends_from_adjusted(adj)
        if dividends:
            self.set_dividends(dividends)

    def _issue(self, kind, date, detail):
        rec = self.issues.setdefault(kind, {"count": 0, "examples": []})
        rec["count"] += 1
        if len(rec["examples"]) < MAX_ISSUE_EXAMPLES:
            rec["examples"].append({"date": date, "detail": detail})

    @classmethod
    def from_dict(cls, ticker, d):
        ds = cls(ticker, d["dates"], d["close"], d.get("open"), d.get("high"),
                 d.get("low"), d.get("volume"), d.get("adj_close"), d.get("dividends"))
        if d.get("price_basis") == "total_return":
            ds.price_basis = "total_return"
        return ds

    # ---- dividends and splits ----------------------------------------------
    def _dividends_from_adjusted(self, adj):
        """Recover cash dividends from an adjusted close (Yahoo's 'Adj Close').

        adj = close x F, where F only steps on ex-dates: F[t-1] / F[t] = 1 - D / close[t-1].
        A step larger than 25% of the price is a split, not a dividend, and is ignored.
        """
        f = [a / c for a, c in zip(adj, self.close)]
        found = 0
        for t in range(1, self.n):
            ratio = f[t - 1] / f[t]
            if ratio < 1 - 2e-5:
                d = self.close[t - 1] * (1 - ratio)
                if d / self.close[t - 1] > 0.25:
                    self._issue("adj_split", self.dates[t], "adjusted close steps by %.0f%%: treated as a split, not a dividend"
                                % (100 * (1 - ratio)))
                    continue
                self.dividends[t] += d
                found += 1
        self.dividend_source = "adj_close"
        self._cache.clear()
        return found

    def set_dividends(self, events):
        """Cash dividends per share as [(ex_date, amount), ...] or [{'date', 'amount'}].
        Each is booked on the first bar on or after its ex-date. Replaces any earlier ones."""
        self.dividends = [0.0] * self.n
        matched = 0
        for ev in events:
            day, amt = (ev.get("date"), ev.get("amount")) if isinstance(ev, dict) else ev
            day, amt = _iso_date(day), _num(amt)
            if day is None or amt is None or amt <= 0:
                continue
            j = _bisect_left(self.dates, day)
            if j >= self.n or (j == 0 and day < self.dates[0]):
                continue   # outside the price history
            self.dividends[j] += amt
            matched += 1
            if amt / self.close[max(0, j - 1)] > 0.10:
                self._issue("big_dividend", self.dates[j], "dividend of %.4g is %.0f%% of the price; check it isn't "
                            "unadjusted for a later split" % (amt, 100 * amt / self.close[max(0, j - 1)]))
        self.dividend_source = "events"
        self._cache.clear()
        return matched

    def apply_splits(self, events):
        """Split events [(date, ratio)] with ratio = new shares per old share (4.0 for 4:1).

        If the prices jump by about the ratio at the split, they are unadjusted: earlier
        prices and dividends are divided by the ratio (volume multiplied). If they don't,
        the prices are already split-adjusted and nothing changes. Returns what was done."""
        done = []
        for ev in events:
            day, ratio = (ev.get("date"), ev.get("ratio")) if isinstance(ev, dict) else ev
            day, ratio = _iso_date(day), _split_ratio(ratio)
            if day is None or ratio is None or ratio <= 0 or ratio == 1:
                continue
            j = _bisect_left(self.dates, day)
            if j <= 0 or j >= self.n:
                continue
            jump = self.close[j - 1] / self.close[j]
            if abs(math.log(jump / ratio)) < math.log(1.3):
                for i in range(j):
                    self.open[i] /= ratio
                    self.high[i] /= ratio
                    self.low[i] /= ratio
                    self.close[i] /= ratio
                    self.volume[i] *= ratio
                    self.dividends[i] /= ratio
                done.append({"date": self.dates[j], "ratio": ratio, "adjusted": True})
            else:
                done.append({"date": self.dates[j], "ratio": ratio, "adjusted": False})
        self.splits_applied += done
        self._cache.clear()
        return done

    def total_return_index(self):
        """Price index with dividends reinvested on their ex-dates, starting at close[0]."""
        def build():
            # close x cumulative reinvestment factor: identical to the closes when there are no dividends
            tr, f = [], 1.0
            for c, d in zip(self.close, self.dividends):
                if d:
                    f *= 1 + d / c
                tr.append(c * f)
            return tr
        return self.cached(("total_return",), build)

    def fingerprint(self):
        """Short hash of the prices and dividends, to tell exactly which data a run used."""
        import hashlib
        h = hashlib.sha256()
        for cols in (self.dates, self.open, self.high, self.low, self.close, self.volume, self.dividends):
            h.update(json.dumps(cols if cols is self.dates else [round(v, 8) for v in cols]).encode())
        return h.hexdigest()[:16]

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


def _bisect_left(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _split_ratio(v):
    """'4:1', '4-for-1', '4/1', 4.0 -> 4.0"""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().replace("for", ":").replace("-", "").replace(" ", "").replace("/", ":")
    if ":" in s:
        a, b = s.split(":")[:2]
        a, b = _num(a), _num(b)
        return a / b if a and b else None
    return _num(s)


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
        self.dividends = ds.dividends[:n]

    def __len__(self):
        return self.n


def read_csv_table(text):
    """CSV text -> (lower-case header, rows). Handles a UTF-8 BOM and quoted cells."""
    import csv
    reader = csv.reader(io.StringIO(text.lstrip("﻿")))
    rows = [r for r in reader if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError("The file is empty.")
    return [h.strip().lower() for h in rows[0]], rows[1:]


CSV_COLUMNS = {
    # the same names, in the same order of preference, as parseCSV() in index.html
    "date": ("date", "observation_date", "ex/eff date", "ex-date", "ex date", "ex-dividend date", "timestamp", "time"),
    "close": ("close/last", "close", "price", "last"),
    "adj_close": ("adj close", "adj. close", "adj_close", "adjusted close", "adjclose"),
    "open": ("open",), "high": ("high",), "low": ("low",), "volume": ("volume", "vol."),
    "dividend": ("dividends", "dividend", "cash amount", "amount", "dividend amount"),
    "split": ("stock splits", "split", "splits", "split ratio", "ratio"),
}


def csv_kind(header):
    """What a CSV holds: 'prices', 'dividends', 'splits' or 'rates' (a single value column, e.g. FRED)."""
    def has(key):
        return any(n in header for n in CSV_COLUMNS[key])
    if not has("date"):
        raise ValueError("CSV needs a Date column")
    if has("close") or has("adj_close"):
        return "prices"
    if has("dividend"):
        return "dividends"
    if has("split"):
        return "splits"
    if len(header) == 2:
        return "rates"
    raise ValueError("No Close, Dividends, Stock Splits or single value column found")


def _col(header, key):
    for n in CSV_COLUMNS[key]:
        if n in header:
            return header.index(n)
    return None


def parse_csv_text(text, ticker="DATA"):
    """Parse a price CSV (Nasdaq.com, Yahoo Finance, Stooq ...) into a DataSet.

    The raw close is the trading price; an 'Adj Close' column, if present, is
    used only to recover dividends. With an adjusted close alone, it becomes the price.
    """
    header, rows = read_csv_table(text)
    if csv_kind(header) != "prices":
        raise ValueError("This CSV holds %s, not prices" % csv_kind(header))
    idx = {k: _col(header, k) for k in ("date", "close", "adj_close", "open", "high", "low", "volume")}
    price_col = idx["close"] if idx["close"] is not None else idx["adj_close"]
    cols = {k: [] for k in idx}
    for row in rows:
        cols["date"].append(row[idx["date"]] if idx["date"] < len(row) else "")
        for k in ("close", "adj_close", "open", "high", "low", "volume"):
            j = price_col if k == "close" else idx[k]
            cols[k].append(_num(row[j]) if j is not None and j < len(row) else None)
    use_adj = idx["close"] is not None and idx["adj_close"] is not None
    ds = DataSet(ticker, cols["date"], cols["close"],
                 cols["open"] if idx["open"] is not None else None,
                 cols["high"] if idx["high"] is not None else None,
                 cols["low"] if idx["low"] is not None else None,
                 cols["volume"] if idx["volume"] is not None else None,
                 cols["adj_close"] if use_adj else None)
    if idx["close"] is None:
        ds.price_basis = "total_return"
    return ds


def parse_events_csv(text):
    """Dividend or split history CSV -> ('dividends' | 'splits', [(date, value), ...])."""
    header, rows = read_csv_table(text)
    kind = csv_kind(header)
    if kind not in ("dividends", "splits"):
        raise ValueError("Expected a dividend or split history, got %s" % kind)
    di, vi = _col(header, "date"), _col(header, "dividend" if kind == "dividends" else "split")
    out = []
    for row in rows:
        if max(di, vi) >= len(row):
            continue
        day = _iso_date(row[di])
        val = _num(row[vi]) if kind == "dividends" else _split_ratio(row[vi])
        if day and val:
            out.append((day, val))
    return kind, sorted(out)


class RateSeries:
    """An annual interest rate over time, in percent (e.g. FRED's DTB3 3-month T-bill)."""

    def __init__(self, ticker, dates, values):
        pts = sorted((d, v) for d, v in ((_iso_date(d), _num(v)) for d, v in zip(dates, values))
                     if d is not None and v is not None)
        if not pts:
            raise ValueError("%s: no valid rate observations" % ticker)
        self.ticker = ticker
        self.kind = "rate"
        self.dates = [d for d, _ in pts]
        self.values = [v for _, v in pts]

    def at(self, day):
        """Annual rate as a fraction on `day` (last observation on or before it; first one before the start)."""
        j = _bisect_left(self.dates, day)
        if j < len(self.dates) and self.dates[j] == day:
            return self.values[j] / 100.0
        return self.values[max(0, j - 1)] / 100.0


def parse_rate_csv(text, ticker="RATE"):
    header, rows = read_csv_table(text)
    di = _col(header, "date")
    vi = 1 - di if len(header) == 2 else None
    if di is None or vi is None:
        raise ValueError("A rate file needs a date column and one value column (like FRED's DATE,DTB3)")
    return RateSeries(ticker, [r[di] for r in rows if len(r) > vi], [r[vi] for r in rows if len(r) > vi])


def load_csv(path, ticker=None):
    import os
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    return parse_csv_text(text, ticker or os.path.splitext(os.path.basename(path))[0].upper())


# ---- exchange calendars and data-quality checks -----------------------------

def _easter(y):
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return _dt.date(y, month, day)


def _nth_weekday(y, month, weekday, nth):
    """nth (1-based; -1 = last) given weekday (Mon=0) of a month."""
    if nth > 0:
        d = _dt.date(y, month, 1)
        d += _dt.timedelta(days=(weekday - d.weekday()) % 7)
        return d + _dt.timedelta(weeks=nth - 1)
    d = _dt.date(y + (month == 12), month % 12 + 1, 1) - _dt.timedelta(days=1)
    return d - _dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d):
    """Saturday holidays close the Friday before, Sunday ones the Monday after."""
    if d.weekday() == 5:
        return d - _dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + _dt.timedelta(days=1)
    return d


NYSE_SPECIAL_CLOSURES = {
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",   # September 11
    "2004-06-11", "2007-01-02", "2018-12-05", "2025-01-09",   # national days of mourning
    "2012-10-29", "2012-10-30",                               # Hurricane Sandy
}


def nyse_holidays(year):
    """Full-day NYSE closures in a year (1990 onward), as ISO dates."""
    y = year
    days = {
        _nth_weekday(y, 2, 0, 3),                      # Washington's Birthday
        _easter(y) - _dt.timedelta(days=2),            # Good Friday
        _nth_weekday(y, 5, 0, -1),                     # Memorial Day
        _observed(_dt.date(y, 7, 4)),                  # Independence Day
        _nth_weekday(y, 9, 0, 1),                      # Labor Day
        _nth_weekday(y, 11, 3, 4),                     # Thanksgiving
        _observed(_dt.date(y, 12, 25)),                # Christmas
    }
    ny = _dt.date(y, 1, 1)
    if ny.weekday() != 5:                              # a Saturday New Year isn't moved to Dec 31
        days.add(_observed(ny))
    if y >= 1998:
        days.add(_nth_weekday(y, 1, 0, 3))             # Martin Luther King Jr. Day
    if y >= 2022:
        days.add(_observed(_dt.date(y, 6, 19)))        # Juneteenth
    out = {d.isoformat() for d in days if d.year == y}
    return out | {d for d in NYSE_SPECIAL_CLOSURES if d.startswith(str(y))}


CALENDARS = {
    # name: (description, days per year used to annualize)
    "NYSE": ("US equities: weekdays except NYSE holidays", 252),
    "weekdays": ("Every Monday to Friday (FX, many futures)", 260),
    "24/7": ("Every calendar day (crypto)", 365),
}


def session_days(start, end, calendar="NYSE"):
    """ISO dates the market is open from start to end inclusive."""
    d, last = _dt.date.fromisoformat(start), _dt.date.fromisoformat(end)
    hol = set()
    if calendar == "NYSE":
        for y in range(d.year, last.year + 1):
            hol |= nyse_holidays(y)
    out = []
    while d <= last:
        iso = d.isoformat()
        if calendar == "24/7" or (d.weekday() < 5 and iso not in hol):
            out.append(iso)
        d += _dt.timedelta(days=1)
    return out


def guess_calendar(ds):
    """'24/7' when the data trades on weekends, else 'NYSE'."""
    weekend = sum(1 for d in ds.days if d.weekday() >= 5)
    return "24/7" if weekend > 0.1 * ds.n else "NYSE"


def quality_report(ds, calendar=None, jump_sigma=8.0, min_jump=0.12, stale_run=5):
    """Data-quality findings for one instrument. Nothing is changed or dropped here.

    Returns {'calendar', 'issues': {kind: {'count', 'severity', 'title', 'examples': [...]}}},
    combining what was found while loading (dropped rows, repaired highs/lows, duplicates)
    with checks against the exchange calendar, stale prices and outsized jumps.
    """
    calendar = calendar or guess_calendar(ds)
    found = {k: dict(v) for k, v in ds.issues.items()}

    def add(kind, date, detail):
        rec = found.setdefault(kind, {"count": 0, "examples": []})
        rec["count"] += 1
        if len(rec["examples"]) < MAX_ISSUE_EXAMPLES:
            rec["examples"].append({"date": date, "detail": detail})

    expected = session_days(ds.dates[0], ds.dates[-1], calendar)
    have = set(ds.dates)
    missing = [d for d in expected if d not in have]
    run = []
    for d in missing + [None]:          # group consecutive missing sessions
        if run and (d is None or expected.index(d) != expected.index(run[-1]) + 1):
            add("missing_days", run[0], "%d session%s missing%s" % (
                len(run), "" if len(run) == 1 else "s", "" if len(run) == 1 else " (%s to %s)" % (run[0], run[-1])))
            run = []
        if d is not None:
            run.append(d)
    exp = set(expected)
    for d in ds.dates:
        if d not in exp:
            add("off_calendar", d, "bar on a day the %s calendar is closed" % calendar)
    same = 1
    for t in range(1, ds.n + 1):
        if t < ds.n and ds.close[t] == ds.close[t - 1]:
            same += 1
            continue
        if same >= stale_run:
            add("stale", ds.dates[t - same], "close unchanged for %d bars (%s to %s)" % (same, ds.dates[t - same], ds.dates[t - 1]))
        same = 1
    lr = [math.log(ds.close[t] / ds.close[t - 1]) for t in range(1, ds.n)]
    for t in range(1, len(lr)):
        window = lr[max(0, t - 60):t]
        sd = stdev(window) if len(window) >= 20 else NAN
        r = lr[t]
        if abs(r) > min_jump and (isnan(sd) or abs(r) > jump_sigma * sd):
            add("jump", ds.dates[t + 1], "%+.1f%% in one day%s; check for an unadjusted split or a bad print" % (
                100 * (math.exp(r) - 1), "" if isnan(sd) else " (%.0f sigma)" % (abs(r) / sd)))
    titles = {
        "dropped": ("Rows skipped", "error"), "ohlc": ("High/low repaired", "warn"),
        "missing_ohlc": ("Open/high/low missing", "info"), "duplicate": ("Duplicate dates", "warn"),
        "missing_days": ("Missing sessions", "warn"), "off_calendar": ("Bars on closed days", "info"),
        "stale": ("Stale prices", "warn"), "jump": ("Large one-day moves", "warn"),
        "adj_split": ("Splits in the adjusted close", "info"), "big_dividend": ("Unusually large dividends", "warn"),
    }
    for k, rec in found.items():
        rec["title"], rec["severity"] = titles.get(k, (k, "info"))
    return {"calendar": calendar, "issues": found, "bars": ds.n,
            "dividends": ds.dividend_source, "price_basis": ds.price_basis}


# =============================================================================
# 4. Panel and contexts: what signals(data) and on_bar(ctx) see
# =============================================================================

class Panel:
    """One or more instruments on a shared calendar: the union of their trading days.

    Each instrument keeps its own bars, so indicators never see filler values.
    nat[a][t] is instrument a's last bar on or before calendar day t (-1 before
    its first bar); live[a][t] is True when it has a bar on day t and can trade.
    """

    def __init__(self, datasets):
        self.sets = list(datasets)
        self.tickers = [ds.ticker for ds in self.sets]
        if not self.sets:
            raise ValueError("A backtest needs at least one instrument.")
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("Each instrument may appear only once in a portfolio.")
        self.k = len(self.sets)
        if self.k == 1:
            self.dates = list(self.sets[0].dates)
        else:
            self.dates = sorted(set().union(*(ds.dates for ds in self.sets)))
        self.n = len(self.dates)
        self.days = [_dt.date.fromisoformat(d) for d in self.dates]
        self.nat, self.live = [], []
        for ds in self.sets:
            if self.k == 1:
                self.nat.append(list(range(self.n)))
                self.live.append([True] * self.n)
                continue
            nat, live, j = [], [], -1
            for d in self.dates:
                if j + 1 < ds.n and ds.dates[j + 1] == d:
                    j += 1
                    live.append(True)
                else:
                    live.append(False)
                nat.append(j)
            self.nat.append(nat)
            self.live.append(live)
        self._aligned = {}

    def mark(self, a, t):
        """Instrument a's latest close on calendar day t (NaN before it starts trading)."""
        j = self.nat[a][t]
        return self.sets[a].close[j] if j >= 0 else NAN

    def aligned(self, a, col):
        """A column of instrument a on the shared calendar: forward-filled, NaN before
        its first bar. Volume and dividends are 0 on days without a bar."""
        key = (a, col)
        if key not in self._aligned:
            ds, nat, live = self.sets[a], self.nat[a], self.live[a]
            src = ds.total_return_index() if col == "total_return" else getattr(ds, col)
            flow = col in ("volume", "dividends")
            self._aligned[key] = [
                NAN if j < 0 else (src[j] if (live[t] or not flow) else 0.0)
                for t, j in enumerate(nat)]
        return self._aligned[key]


class AlignedView:
    """One instrument's columns on a portfolio's shared calendar (see PanelView)."""

    def __init__(self, panel, a, n):
        self.ticker = panel.tickers[a]
        self.n = n
        self.dates = panel.dates[:n]
        self.days = panel.days[:n]
        for col in ("open", "high", "low", "close", "volume", "dividends", "total_return"):
            setattr(self, col, panel.aligned(a, col)[:n])
        self.listed = panel.live[a][:n]

    def __len__(self):
        return self.n


class PanelView:
    """What signals(data) receives in a portfolio backtest: the first n calendar days.

    data.tickers, data.dates, data.days, len(data)
    data['SPY']            one instrument: .open .high .low .close .volume .dividends
                           .total_return .listed, as lists on the shared calendar
                           (forward-filled, NaN before its first bar); works with qt
    data.close['SPY']      the same lists by column (also open/high/low/volume/dividends)
    data.array('close')    NumPy array: one row per day, one column per ticker
    """

    def __init__(self, panel, n=None):
        self._panel = panel
        self.n = panel.n if n is None else n
        self.tickers = list(panel.tickers)
        self.dates = panel.dates[:self.n]
        self.days = panel.days[:self.n]
        self._views = {}

    def __len__(self):
        return self.n

    def __getitem__(self, ticker):
        if ticker not in self._views:
            if ticker not in self.tickers:
                raise KeyError("%r is not in this portfolio (%s)" % (ticker, ", ".join(self.tickers)))
            self._views[ticker] = AlignedView(self._panel, self.tickers.index(ticker), self.n)
        return self._views[ticker]

    def _col(self, col):
        return {t: getattr(self[t], col) for t in self.tickers}

    open = property(lambda self: self._col("open"))
    high = property(lambda self: self._col("high"))
    low = property(lambda self: self._col("low"))
    close = property(lambda self: self._col("close"))
    volume = property(lambda self: self._col("volume"))
    dividends = property(lambda self: self._col("dividends"))
    total_return = property(lambda self: self._col("total_return"))
    listed = property(lambda self: self._col("listed"))

    def array(self, col="close"):
        import numpy as np
        return np.array([getattr(self[t], col) for t in self.tickers], dtype=float).T


class AssetContext:
    """One instrument inside on_bar: its prices, indicators and position.

    In a single-instrument strategy this is ctx itself; in a portfolio strategy
    it is ctx['SPY']. Prices are the instrument's own bars up to today.
    """

    def __init__(self, panel, a, run):
        self._panel = panel
        self._a = a
        self._ds = panel.sets[a]
        self._run = run
        self.ticker = self._ds.ticker
        ds = self._ds
        self.open = SeriesView(ds.open, 0)
        self.high = SeriesView(ds.high, 0)
        self.low = SeriesView(ds.low, 0)
        self.close = SeriesView(ds.close, 0)
        self.volume = SeriesView(ds.volume, 0)
        self._views = (self.open, self.high, self.low, self.close, self.volume)
        self._j = -1

    def _sync(self, t):
        j = self._panel.nat[self._a][t]
        if j != self._j:
            self._j = j
            for v in self._views:
                v._n = j + 1

    @property
    def listed(self):
        """True once the instrument has started trading."""
        return self._j >= 0

    @property
    def trading(self):
        """True when the instrument has a bar today (it can be traded at the next bar)."""
        return self._panel.live[self._a][self._run.t]

    @property
    def price(self):
        return self._ds.close[self._j] if self._j >= 0 else NAN

    @property
    def dividend(self):
        """Cash dividend per share with today as ex-date (0 on most days)."""
        return self._ds.dividends[self._j] if self._j >= 0 and self.trading else 0.0

    @property
    def position(self):
        return self._run.acct.shares[self._a]

    @property
    def value(self):
        """Market value of the position at today's close."""
        sh = self._run.acct.shares[self._a]
        return sh * self.price if sh else 0.0

    @property
    def weight(self):
        eq = self._run.acct.equity_marked(self._run.t)
        sh = self._run.acct.shares[self._a]
        return sh * self.price / eq if eq > 0 and sh else 0.0

    @property
    def is_flat(self):
        return self._run.acct.shares[self._a] == 0

    @property
    def is_long(self):
        return self._run.acct.shares[self._a] > 0

    @property
    def is_short(self):
        return self._run.acct.shares[self._a] < 0

    @property
    def entry_price(self):
        tr = self._run.acct.open_trades[self._a]
        return tr["entry_price"] if tr else NAN

    @property
    def entry_date(self):
        tr = self._run.acct.open_trades[self._a]
        return self._panel.dates[tr["entry_i"]] if tr else None

    @property
    def bars_in_trade(self):
        tr = self._run.acct.open_trades[self._a]
        return self._run.t - tr["entry_i"] if tr else 0

    # ---- orders -----------------------------------------------------------
    def order(self, qty, **kw):
        """Buy (qty > 0) or sell (qty < 0) shares at the next bar."""
        self._run.add_order(self._a, "delta", float(qty), kw)

    def order_target_shares(self, n, **kw):
        """Trade whatever is needed to hold exactly n shares after the next fill."""
        self._run.add_order(self._a, "target", float(n), kw)

    # ---- indicators -------------------------------------------------------
    def _val(self, key, build, ago):
        ago = int(ago)
        if ago < 0:
            raise ValueError("ago must be 0 or positive (negative values would read future bars)")
        arr = self._ds.cached(key, build)
        j = self._j - ago
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


class _StrategyClock:
    """Calendar, memory and account fields shared by both kinds of ctx."""

    @property
    def date(self):
        return self._panel.dates[self._run.t]

    @property
    def day(self):
        return self._panel.days[self._run.t]

    @property
    def weekday(self):
        return self._panel.days[self._run.t].weekday()

    @property
    def i(self):
        return self._run.t

    @property
    def bar(self):
        """Bars since the backtest started (0 on the first bar)."""
        return self._run.t - self.start_index

    @property
    def cash(self):
        return self._run.acct.cash

    @property
    def equity(self):
        return self._run.acct.equity_marked(self._run.t)

    def log(self, *args):
        print(self.date, *args)


class Context(_StrategyClock, AssetContext):
    """Everything a single-instrument strategy may look at on the current bar.

    Prices: ctx.open/high/low/close/volume (SeriesView), ctx.price (today's close),
            ctx.dividend (cash dividend going ex today)
    Time:   ctx.i, ctx.date ('YYYY-MM-DD'), ctx.day (datetime.date), ctx.weekday (Mon=0)
    Account: ctx.cash, ctx.position (shares), ctx.equity, ctx.weight, ctx.is_flat/long/short
    Trade:  ctx.entry_price, ctx.entry_date, ctx.bars_in_trade
    Memory: ctx.state (dict kept between bars), ctx.params (the PARAMS dict)
    Orders: return a target weight from on_bar, or call ctx.order(qty) /
            ctx.order_target_shares(n). Everything fills on the next bar.
    Indicators: ctx.sma(50), ctx.rsi(14), ctx.bbands(20, 2) ... (see docs).
    Every indicator accepts ago=k to read the value k bars back.
    """

    def __init__(self, panel, run, params, start):
        AssetContext.__init__(self, panel, 0, run)
        self.params = params
        self.state = {}
        self.start_index = start


class PortfolioContext(_StrategyClock):
    """What on_bar(ctx) sees in a portfolio backtest.

    ctx.tickers               the instruments, in order
    ctx['SPY']                one instrument (prices, indicators, position: see Context)
    ctx.assets                {ticker: that instrument}
    ctx.date, ctx.day, ctx.weekday, ctx.i, ctx.bar
    ctx.cash, ctx.equity, ctx.weights, ctx.positions ({ticker: ...}), ctx.gross, ctx.net
    ctx.state, ctx.params, ctx.log(...)
    Orders: return {ticker: target weight} (tickers left out go to 0) or a list in
    ticker order, or None to keep everything; or call ctx.order(ticker, qty) /
    ctx.order_target_shares(ticker, n). Everything fills on the next bar.
    """

    def __init__(self, panel, run, params, start):
        self._panel = panel
        self._run = run
        self.params = params
        self.state = {}
        self.start_index = start
        self.tickers = list(panel.tickers)
        self.assets = {tk: AssetContext(panel, a, run) for a, tk in enumerate(panel.tickers)}

    def _sync(self, t):
        for a in self.assets.values():
            a._sync(t)

    def __getitem__(self, ticker):
        try:
            return self.assets[ticker]
        except KeyError:
            raise KeyError("%r is not in this portfolio (%s)" % (ticker, ", ".join(self.tickers)))

    def __iter__(self):
        return iter(self.tickers)

    def __len__(self):
        return len(self.tickers)

    @property
    def weights(self):
        eq = self.equity
        return {tk: (a.value / eq if eq > 0 else 0.0) for tk, a in self.assets.items()}

    @property
    def positions(self):
        return {tk: a.position for tk, a in self.assets.items()}

    @property
    def gross(self):
        eq = self.equity
        return sum(abs(a.value) for a in self.assets.values()) / eq if eq > 0 else 0.0

    @property
    def net(self):
        eq = self.equity
        return sum(a.value for a in self.assets.values()) / eq if eq > 0 else 0.0

    def order(self, ticker, qty, **kw):
        self[ticker].order(qty, **kw)

    def order_target_shares(self, ticker, n, **kw):
        self[ticker].order_target_shares(n, **kw)


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
        "datetime": _dt, "qt": qt, "NAN": NAN, "isnan": isnan, "walk_forward": walk_forward,
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
    if isinstance(exc, ModuleNotFoundError) and sys.platform == "emscripten":
        msg += (". In the browser, strategies can import NumPy, pandas, SciPy and scikit-learn plus the "
                "standard library. PyTorch, TensorFlow and other packages without a WebAssembly build "
                "only work when you run engine.py in regular Python.")
    if bar_date:
        msg += "  (on bar %s)" % bar_date
    if user:
        trace = "\n".join("  line %d, in %s\n    %s" % (f.lineno, f.name, (f.line or "").strip())
                          for f in user)
        return {"message": msg, "trace": trace, "line": user[-1].lineno, "user": True}
    return {"message": msg, "trace": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "line": None, "user": False}


def _user_error(message):
    return StrategyError({"message": message, "trace": "", "line": None, "user": True})


class Strategy:
    """A strategy is either on_bar(ctx) -> target weight(s), or signals(data) -> weights per bar.

    A portfolio strategy may set UNIVERSE = ['SPY', 'TLT', ...] as its default instruments.
    """

    def __init__(self, on_bar=None, signals=None, init=None, params=None, name="strategy", universe=None):
        if on_bar is None and signals is None:
            raise _user_error("Define on_bar(ctx) or signals(data) in your strategy.")
        self.on_bar = on_bar
        self.signals = signals
        self.init = init
        self.params = params if params is not None else {}
        self.name = name
        self.universe = universe

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
            raise _user_error("PARAMS must be a dict, e.g. PARAMS = {'fast': 50}")
        if params_override:
            for k, v in params_override.items():
                if k in params and isinstance(params[k], bool):
                    v = bool(v)
                elif k in params and isinstance(params[k], int) and float(v) == int(float(v)):
                    v = int(float(v))
                params[k] = v
        ns["PARAMS"] = params
        universe = ns.get("UNIVERSE")
        if universe is not None:
            if isinstance(universe, str) or not all(isinstance(t, str) for t in universe):
                raise _user_error("UNIVERSE must be a list of tickers, e.g. UNIVERSE = ['SPY', 'TLT', 'GLD']")
            universe = [t.upper() for t in universe]
        on_bar = ns.get("on_bar")
        signals = ns.get("signals")
        init = ns.get("init")
        return cls(on_bar if callable(on_bar) else None,
                   signals if callable(signals) and not callable(on_bar) else None,
                   init if callable(init) else None, params, universe=universe)


class Config:
    """Backtest settings. Rates are fractions: 0.0002 is 2 basis points, 0.04 is 4% a year."""

    DEFAULTS = {
        "cash": 10000.0,          # starting cash
        "commission_bps": 0.0,    # commission as basis points of traded value
        "commission_fixed": 0.0,  # flat commission per fill, in dollars
        "slippage_bps": 2.0,      # price moves against you by this many bps on every fill
        "execution": "open",      # 'open' = next bar's open, 'close' = next bar's close
        "allow_short": False,
        "fractional": True,       # allow fractional shares
        "max_leverage": 1.0,      # largest gross |position value| / equity allowed
        "rebalance_band": 0.02,   # skip weight changes smaller than this share of equity
        "risk_free": 0.0,         # annual risk-free rate for Sharpe/Sortino/alpha (0.04 = 4%)
        "seed": 42,               # seed for the random module (reproducible runs)
        # carry and financing (interest accrues ACT/360 between bars)
        "total_return": True,     # pay dividends into the account (False = price return only)
        "cash_rate": None,        # annual interest on cash; None = same as risk_free
        "rate_ticker": "",        # name of a loaded rate series (e.g. FRED DTB3) to use instead
        "margin_spread": 0.0,     # extra annual rate paid on borrowed cash (negative balance)
        "borrow_fee": 0.0,        # annual fee on the value of short positions
        "borrow_fees": {},        # per-ticker borrow fees overriding borrow_fee
        "maintenance_margin": 0.25,  # margin call when equity < this x gross exposure (0 = off)
    }

    def __init__(self, **kw):
        unknown = set(kw) - set(self.DEFAULTS)
        if unknown:
            raise ValueError("unknown config keys: %s" % ", ".join(sorted(unknown)))
        for k, default in self.DEFAULTS.items():
            v = kw.get(k, default)
            if v is None:
                v = default
            if v is None:
                pass
            elif isinstance(default, bool):
                v = bool(v)
            elif isinstance(default, float) or k == "cash_rate":
                v = float(v)
            elif isinstance(default, int):
                v = int(v)
            elif isinstance(default, str):
                v = str(v)
            elif isinstance(default, dict):
                v = {str(t).upper(): float(x) for t, x in dict(v).items()}
            setattr(self, k, v)
        if self.execution not in ("open", "close"):
            raise ValueError("execution must be 'open' or 'close'")
        if self.cash <= 0:
            raise ValueError("starting cash must be positive")
        if self.max_leverage <= 0:
            raise ValueError("max leverage must be positive")
        if not 0 <= self.maintenance_margin < 1:
            raise ValueError("maintenance margin must be from 0 to 1 (0.25 = 25%)")

    def as_dict(self):
        return {k: (dict(getattr(self, k)) if isinstance(getattr(self, k), dict) else getattr(self, k))
                for k in self.DEFAULTS}


class Account:
    """Cash, positions, fills and round-trip trades for one or more instruments.

    Every dollar is booked against the instrument that caused it (asset_cash:
    trades, costs, dividends, borrow fees), so per-instrument P&L adds up to the
    account's P&L except for interest on cash.
    """

    def __init__(self, cfg, panel, rates=None):
        self.cfg = cfg
        self.panel = panel
        self.k = k = panel.k
        self.rates = rates
        self.cash = cfg.cash
        self.shares = [0.0] * k
        self.fills = []
        self.trades = []
        self.open_trades = [None] * k
        self.commission = 0.0
        self.slippage = 0.0
        self.traded_notional = 0.0
        self.clipped = 0
        self.scaled = 0                      # rebalances scaled down to the leverage limit
        self.asset_cash = [0.0] * k
        self.asset_costs = [0.0] * k
        self.asset_dividends = [0.0] * k
        self.asset_notional = [0.0] * k
        self.carry = {"dividends_received": 0.0, "dividends_paid": 0.0, "interest_earned": 0.0,
                      "interest_paid": 0.0, "borrow_paid": 0.0}
        self.margin_calls = []
        self._val = [NAN] * k                # valuation prices during an execution

    # ---- valuation ----------------------------------------------------------
    def equity_marked(self, t):
        eq = self.cash
        for a, sh in enumerate(self.shares):
            if sh:
                eq += sh * self.panel.mark(a, t)
        return eq

    def _equity_val(self):
        eq = self.cash
        for a, sh in enumerate(self.shares):
            if sh:
                eq += sh * self._val[a]
        return eq

    def _set_val(self, px, t):
        for a in range(self.k):
            self._val[a] = px[a] if px[a] is not None else self.panel.mark(a, t)

    def asset_value(self, a, price):
        return self.asset_cash[a] + self.shares[a] * price

    # ---- carry: interest, borrow fees, dividends ----------------------------
    def rate_on(self, day):
        cfg = self.cfg
        if self.rates is not None:
            return self.rates.at(day)
        return cfg.risk_free if cfg.cash_rate is None else cfg.cash_rate

    def accrue(self, t_prev, t):
        """Interest on cash and fees on shorts for the calendar days between two bars (ACT/360)."""
        cfg, p = self.cfg, self.panel
        days = (p.days[t] - p.days[t_prev]).days
        if days <= 0:
            return
        rate = self.rate_on(p.dates[t_prev])
        if self.cash > 0:
            x = self.cash * rate * days / 360.0
            self.carry["interest_earned"] += x
        else:
            x = self.cash * (rate + cfg.margin_spread) * days / 360.0
            self.carry["interest_paid"] -= x
        fee_total = 0.0
        for a, sh in enumerate(self.shares):
            if sh < 0:
                fee = cfg.borrow_fees.get(p.tickers[a], cfg.borrow_fee)
                if fee:
                    f = -sh * p.mark(a, t_prev) * fee * days / 360.0
                    self.asset_cash[a] -= f
                    self.asset_costs[a] += f
                    fee_total += f
        self.carry["borrow_paid"] += fee_total
        self.cash += x - fee_total

    def pay_dividends(self, t):
        """Holders on the ex-date receive the dividend; short sellers pay it."""
        p = self.panel
        for a, sh in enumerate(self.shares):
            if sh and p.live[a][t]:
                d = p.sets[a].dividends[p.nat[a][t]]
                if d:
                    amt = sh * d
                    self.cash += amt
                    self.asset_cash[a] += amt
                    self.asset_dividends[a] += amt
                    self.carry["dividends_received" if amt > 0 else "dividends_paid"] += abs(amt)

    # ---- order sizing -------------------------------------------------------
    def rebalance(self, weights, px, t):
        """Trade toward target weights {asset: weight} at fill prices px (None = no bar).
        Returns the targets that couldn't trade today, to retry at the next bar."""
        cfg = self.cfg
        self._set_val(px, t)
        eq = self._equity_val()
        if eq <= 0:
            return {}
        L = cfg.max_leverage
        lo = -L if cfg.allow_short else 0.0
        w = {a: min(max(v, lo), L) for a, v in weights.items() if not isnan(v)}
        gross = sum(abs(v) for v in w.values())
        if gross > L * (1 + 1e-9):
            f = L / gross
            w = {a: v * f for a, v in w.items()}
            self.scaled += 1
        slip, cb, fixed = cfg.slippage_bps / 1e4, cfg.commission_bps / 1e4, cfg.commission_fixed
        plan, later = [], {}
        for a, wa in w.items():
            p = px[a]
            if p is None:
                if self.panel.nat[a][t] >= 0 or wa == 0:
                    later[a] = wa
                continue
            sh = self.shares[a]
            diff = wa * eq - sh * p
            if wa == 0:
                dq = -sh
            elif diff > 0:
                dq = max(0.0, (diff - fixed) / (p * (1 + slip) * (1 + cb)))
            else:
                dq = diff / p
            target = sh + dq
            if (sh != 0 and target != 0 and (sh > 0) == (target > 0)
                    and abs(diff) < cfg.rebalance_band * eq):
                continue
            if not cfg.fractional:
                target = float(int(target))  # round toward zero
            dq = target - sh
            if not cfg.allow_short and sh + dq < 0:
                dq = -sh
            plan.append([a, dq, p, False])
        self._execute_plan(plan, eq, t, report=False)
        return later

    def share_orders(self, orders, px, t):
        """Execute share orders {asset: ('delta' | 'target', value)}. Returns those
        for instruments without a bar today, to retry at the next bar."""
        cfg = self.cfg
        self._set_val(px, t)
        eq = self._equity_val()
        if eq <= 0:
            return {}
        plan, later = [], {}
        for a, (kind, v) in orders.items():
            p = px[a]
            if p is None:
                later[a] = (kind, v)
                continue
            sh = self.shares[a]
            target = v if kind == "target" else sh + v
            if not cfg.fractional:
                target = float(int(target))
            dq = target - sh
            clipped = False
            if not cfg.allow_short and sh + dq < 0:
                dq = -sh
                clipped = True
            plan.append([a, dq, p, clipped])
        self._execute_plan(plan, eq, t, report=True)
        return later

    def _execute_plan(self, plan, eq, t, report):
        """Sells first (they raise cash), then buys, which share the cash and leverage room."""
        cfg = self.cfg
        L = cfg.max_leverage
        slip, cb, fixed = cfg.slippage_bps / 1e4, cfg.commission_bps / 1e4, cfg.commission_fixed
        for item in plan:
            a, dq, p, clipped = item
            if dq < 0 and cfg.allow_short and self.shares[a] + dq < 0:
                sh = self.shares[a]
                mx = L * eq / p + min(sh, 0.0)  # extra shares we may short
                allowed = -(max(sh, 0.0) + max(0.0, mx))
                if dq < allowed - 1e-9:
                    item[1] = dq = allowed
                    item[3] = True
        for a, dq, p, clipped in plan:
            if dq >= 0:
                continue
            if not cfg.fractional:
                dq = float(int(dq))
            if clipped and report:
                self.clipped += 1
            if abs(dq) >= 1e-9:
                self._fill(a, dq, p, t)
        buys = [item for item in plan if item[1] > 0]
        if not buys:
            for a, dq, p, clipped in plan:
                if dq == 0 and clipped and report:
                    self.clipped += 1
            return
        for item in buys:
            a, dq, p, _ = item
            sh = self.shares[a]
            if sh >= 0:
                cap = L * eq / p - sh  # keeps this position <= L x equity
                if dq > cap + 1e-9:
                    item[1] = max(0.0, cap)
                    item[3] = True
        need = sum(dq * p * (1 + slip) * (1 + cb) for _, dq, p, _ in buys)
        room = self.cash + (L - 1.0) * eq - fixed * len(buys)
        if need > room + 1e-9 * max(1.0, need):
            f = max(0.0, room) / need if need > 0 else 0.0
            for item in buys:
                item[1] *= f
                item[3] = True
        for a, dq, p, clipped in buys:
            if not cfg.fractional:
                dq = float(int(dq))
            if clipped and report:
                self.clipped += 1
            if abs(dq) >= 1e-9:
                self._fill(a, dq, p, t)

    # ---- fills and trade bookkeeping ---------------------------------------
    def _fill(self, a, q, price, t, reason=None):
        cfg = self.cfg
        prev = self.shares[a]
        slip = cfg.slippage_bps / 1e4
        px = price * (1 + slip) if q > 0 else price * (1 - slip)
        comm = abs(q) * px * cfg.commission_bps / 1e4 + cfg.commission_fixed
        self._val[a] = price
        eq_before = self._equity_val()
        value_before = self.asset_value(a, price)
        self.cash -= q * px + comm
        self.asset_cash[a] -= q * px + comm
        new = prev + q
        if abs(new) < 1e-9:
            new = 0.0
        self.shares[a] = new
        cost = comm + abs(q) * price * slip
        self.commission += comm
        self.slippage += abs(q) * price * slip
        self.asset_costs[a] += cost
        self.traded_notional += abs(q) * price
        self.asset_notional[a] += abs(q) * price
        fill = {"i": t, "date": self.panel.dates[t], "ticker": self.panel.tickers[a],
                "side": "buy" if q > 0 else "sell", "qty": q, "price": px, "commission": comm, "position": new}
        if reason:
            fill["reason"] = reason
        self.fills.append(fill)
        eq_after = self._equity_val()
        value_after = self.asset_value(a, price)

        flipped = prev != 0 and new != 0 and (prev > 0) != (new > 0)
        if prev != 0 and (new == 0 or flipped):
            self._close_trade(a, t, px, value_after)
        if new != 0 and (prev == 0 or flipped):
            self.open_trades[a] = {"entry_i": t, "entry_price": px,
                                   "entry_value": value_before if prev == 0 else value_after,
                                   "entry_equity": eq_before if prev == 0 else eq_after,
                                   "side": "long" if new > 0 else "short", "max_shares": abs(new)}
        elif new != 0 and self.open_trades[a] is not None:
            tr = self.open_trades[a]
            if abs(new) > abs(prev):  # adding to the position: update average entry
                tr["entry_price"] = (tr["entry_price"] * abs(prev) + px * abs(q)) / abs(new)
            tr["max_shares"] = max(tr["max_shares"], abs(new))

    def _close_trade(self, a, t, px, value_after, still_open=False):
        tr = self.open_trades[a]
        pnl = value_after - tr["entry_value"]
        notional = tr["max_shares"] * tr["entry_price"]
        self.trades.append({
            "ticker": self.panel.tickers[a],
            "entry_date": self.panel.dates[tr["entry_i"]], "exit_date": self.panel.dates[t],
            "entry_i": tr["entry_i"], "exit_i": t, "side": tr["side"],
            "entry_price": tr["entry_price"], "exit_price": px, "shares": tr["max_shares"],
            "pnl": pnl, "ret": pnl / tr["entry_equity"] if tr["entry_equity"] > 0 else NAN,
            "ret_notional": pnl / notional if notional > 0 else NAN,
            "bars": t - tr["entry_i"], "open": still_open,
        })
        self.open_trades[a] = None

    def liquidate(self, t, fraction=1.0, reason=None):
        """Sell `fraction` of every position at today's close (instruments with a bar today)."""
        p = self.panel
        for a in range(self.k):
            sh = self.shares[a]
            if sh and (p.live[a][t] or fraction >= 1.0):
                self._set_val([None] * self.k, t)
                self._fill(a, -sh * fraction, p.mark(a, t), t, reason)


class Result:
    """Everything a backtest produced."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to_dict(self):
        return dict(self.__dict__)


def _resolve_range(dates, start=None, end=None, label="the data"):
    n = len(dates)
    s, e = 0, n - 1
    if start:
        while s < n and dates[s] < start[:10]:
            s += 1
    if end:
        while e >= 0 and dates[e] > end[:10]:
            e -= 1
    if e - s < 1:
        raise ValueError("The chosen date range has fewer than 2 bars of %s." % label)
    return s, e


def align_benchmark(dates, bench, s, e, total_return=True):
    """Benchmark values on the backtest's calendar (forward-filled). Moves s forward if
    the benchmark history starts later. Returns (s, values, notes)."""
    notes = []
    if bench.dates[0] > dates[s]:
        new_s = s
        while new_s < e and dates[new_s] < bench.dates[0]:
            new_s += 1
        notes.append("Start moved to %s, the first date with %s data." % (dates[new_s], bench.ticker))
        s = new_s
    src = bench.total_return_index() if total_return else bench.close
    out, last, j, gap, worst = [], None, 0, 0, (0, None)
    bd = bench.dates
    for i in range(s, e + 1):
        d = dates[i]
        hit = False
        while j < bench.n and bd[j] <= d:
            last = src[j]
            hit = bd[j] == d
            j += 1
        gap = 0 if hit else gap + 1
        if gap > worst[0]:
            worst = (gap, d)
        out.append(last)
    if any(v is None for v in out):
        raise ValueError("Benchmark %s has no data for the start of this range." % bench.ticker)
    if bench.dates[-1] < dates[e]:
        notes.append("%s data ends %s; it is held flat after that." % (bench.ticker, bench.dates[-1]))
    elif worst[0] >= 5:
        notes.append("%s has no prices for %d bars in a row up to %s; its last price was carried forward."
                     % (bench.ticker, worst[0], worst[1]))
    return s, out, notes


def _equal_weight_benchmark(panel, s, e, total_return=True):
    """Equal-weight portfolio of every listed instrument, rebalanced daily."""
    col = "total_return" if total_return else "close"
    series = [panel.aligned(a, col) for a in range(panel.k)]
    out = [1.0]
    for t in range(s + 1, e + 1):
        rets = [x[t] / x[t - 1] - 1 for x in series if not isnan(x[t - 1]) and not isnan(x[t])]
        out.append(out[-1] * (1 + (sum(rets) / len(rets) if rets else 0.0)))
    return out


def _weights_from(value, panel, where):
    """A strategy's target weights (dict by ticker, or a list in ticker order) -> {asset: weight}."""
    if value is None:
        return None
    if isinstance(value, dict):
        unknown = [tk for tk in value if tk not in panel.tickers]
        if unknown:
            raise _user_error("%s returned weights for %s, which %s not in the portfolio (%s)." % (
                where, ", ".join(map(str, unknown)), "is" if len(unknown) == 1 else "are", ", ".join(panel.tickers)))
        out = {}
        for a, tk in enumerate(panel.tickers):
            v = value.get(tk, 0.0)
            if v is not None:
                v = float(v)
                if not isnan(v):
                    out[a] = v
        return out
    try:
        vals = list(value)
    except TypeError:
        raise _user_error("%s must return {ticker: weight}, a list of %d weights, or None; got %r"
                          % (where, panel.k, value))
    if len(vals) != panel.k:
        raise _user_error("%s returned %d weights for %d instruments." % (where, len(vals), panel.k))
    out = {}
    for a, v in enumerate(vals):
        if v is not None and not isnan(float(v)):
            out[a] = float(v)
    return out


def signal_rows(raw, panel, portfolio, n=None):
    """Normalize what signals(data) returned to a function t -> {asset: weight} or None.
    n is the number of calendar bars the strategy was shown (default: all)."""
    n = panel.n if n is None else n
    if not portfolio:
        sig = list(raw)
        if len(sig) != n:
            raise _user_error("signals(data) returned %d values but the data has %d bars." % (len(sig), n))

        def row(t):
            w = sig[t]
            return {0: float(w)} if w is not None and not isnan(w) else None
        return row, sig
    if isinstance(raw, dict):
        cols = {}
        for tk, series in raw.items():
            if tk not in panel.tickers:
                raise _user_error("signals(data) returned weights for %r, which is not in the portfolio (%s)."
                                  % (tk, ", ".join(panel.tickers)))
            s = list(series)
            if len(s) != n:
                raise _user_error("signals(data)[%r] has %d values but the calendar has %d bars." % (tk, len(s), n))
            cols[panel.tickers.index(tk)] = s
        rows = [{a: cols[a][t] if a in cols else 0.0 for a in range(panel.k)} for t in range(n)]
    else:
        rows = list(raw)
        if len(rows) != n:
            raise _user_error("signals(data) returned %d rows but the calendar has %d bars." % (len(rows), n))
    norm = [_weights_from(r, panel, "signals(data)") for r in rows]
    norm = [r if r else None for r in norm]
    return (lambda t: norm[t]), norm


class _Run:
    """Mutable state shared by the account, the contexts and the loop."""

    def __init__(self, acct):
        self.acct = acct
        self.t = 0
        self.orders = {}

    def add_order(self, a, kind, value, kw):
        if kw:
            raise TypeError("order options %s are not supported" % ", ".join(sorted(kw)))
        cur = self.orders.get(a)
        if kind == "target":
            self.orders[a] = ("target", value)
        elif cur is None:
            self.orders[a] = ("delta", value)
        elif cur[0] == "delta":
            self.orders[a] = ("delta", cur[1] + value)
        # a delta after a target is ignored, as before: the target already fixes the size


def backtest(strategy, ds, cfg=None, start=None, end=None, bench=None, capture_output=True, rates=None):
    """Run a single-instrument strategy over ds and return a Result.

    Timing: on every bar the strategy sees data up to and including that bar's
    close. Orders it places are filled on the NEXT bar (at the open, or at the
    close with execution='close'), so a strategy can never trade on information
    it could not have had.
    """
    return _run_backtest(strategy, Panel([ds]), cfg, start, end, bench, False, capture_output, rates)


def backtest_portfolio(strategy, datasets, cfg=None, start=None, end=None, bench=None,
                       capture_output=True, rates=None):
    """Run a portfolio strategy over several instruments on their shared calendar.

    on_bar(ctx) returns {ticker: target weight}; signals(data) returns
    {ticker: [weight per bar]}, a list of such dicts, or an array with one column
    per ticker. Without a benchmark, the benchmark is an equal-weight portfolio of
    the same instruments rebalanced daily.
    """
    return _run_backtest(strategy, Panel(datasets), cfg, start, end, bench, True, capture_output, rates)


def _run_backtest(strategy, panel, cfg, start, end, bench, portfolio, capture_output, rates):
    if cfg is None:
        cfg = Config()
    elif isinstance(cfg, dict):
        cfg = Config(**cfg)
    if not cfg.total_return:
        panel = Panel([_price_only(ds) for ds in panel.sets])
    label = "%s data" % panel.tickers[0] if panel.k == 1 else "data for these instruments"
    s, e = _resolve_range(panel.dates, start, end, label)
    notes = []
    if bench is not None and not (panel.k == 1 and bench is panel.sets[0]):
        s, bench_vals, bnotes = align_benchmark(panel.dates, bench, s, e, cfg.total_return)
        notes += bnotes
        bench_name = bench.ticker
    elif portfolio:
        bench_vals = _equal_weight_benchmark(panel, s, e, cfg.total_return)
        bench_name = "Equal weight"
    else:
        src = panel.sets[0].total_return_index() if cfg.total_return else panel.sets[0].close
        bench_vals = src[s:e + 1]
        bench_name = panel.tickers[0]
    missing = [ds.ticker for ds in panel.sets if ds.dividend_source is None and ds.price_basis == "price"]
    if cfg.total_return and missing:
        names = ", ".join(missing[:6]) + (" and %d more" % (len(missing) - 6) if len(missing) > 6 else "")
        notes.append("No dividend data for %s: returns are price-only and understate total return. "
                     "Add a dividend history or an adjusted-close file on the Data tab." % names)

    random.seed(cfg.seed)
    k = panel.k
    acct = Account(cfg, panel, rates)
    run = _Run(acct)
    ctx = PortfolioContext(panel, run, strategy.params, s) if portfolio else Context(panel, run, strategy.params, s)
    out_buf = io.StringIO()
    old_stdout = sys.stdout
    if capture_output:
        sys.stdout = out_buf
    bar_date = None
    n = e - s + 1
    equity, exposure, gross_l, net_l, long_l, short_l = [], [], [], [], [], []
    position = [] if not portfolio else None
    weights = [[] for _ in range(k)]
    asset_pnl = [[] for _ in range(k)]
    try:
        row, sig = None, None
        if strategy.signals is not None:
            raw = strategy.signals(PanelView(panel) if portfolio else DataView(panel.sets[0]))
            row, sig = signal_rows(raw, panel, portfolio)
        if strategy.init is not None:
            run.t = s
            ctx._sync(s)
            bar_date = panel.dates[s]
            strategy.init(ctx)

        pending = None
        wiped = False
        exe_open = cfg.execution == "open"
        mm = cfg.maintenance_margin
        for t in range(s, e + 1):
            run.t = t
            if t > s and not wiped:
                acct.accrue(t - 1, t)
                if cfg.total_return:
                    acct.pay_dividends(t)
            if pending is not None and not wiped:
                px = [None] * k
                for a in range(k):
                    if panel.live[a][t]:
                        j = panel.nat[a][t]
                        px[a] = panel.sets[a].open[j] if exe_open else panel.sets[a].close[j]
                if pending[0] == "weights":
                    later = acct.rebalance(pending[1], px, t)
                else:
                    later = acct.share_orders(pending[1], px, t)
                pending = (pending[0], later) if later else None
            eq = acct.equity_marked(t)
            if eq <= 0 and not wiped:
                wiped = True
                notes.append("Equity reached zero on %s; trading stopped." % panel.dates[t])
                acct.liquidate(t)
                acct.cash = 0.0
                eq = 0.0
            marks = [panel.mark(a, t) for a in range(k)]
            g = sum(abs(sh * marks[a]) for a, sh in enumerate(acct.shares) if sh)
            if mm > 0 and not wiped and g > 0 and eq < mm * g:
                target = eq * min(cfg.max_leverage, 0.8 / mm)
                acct.margin_calls.append({"date": panel.dates[t], "equity": eq, "gross": g})
                acct.liquidate(t, 1 - target / g, "margin call")
                eq = acct.equity_marked(t)
                marks = [panel.mark(a, t) for a in range(k)]
                g = sum(abs(sh * marks[a]) for a, sh in enumerate(acct.shares) if sh)
            equity.append(eq)
            lv = sum(sh * marks[a] for a, sh in enumerate(acct.shares) if sh > 0)
            sv = sum(-sh * marks[a] for a, sh in enumerate(acct.shares) if sh < 0)
            gross_l.append(g / eq if eq > 0 else 0.0)
            net_l.append((lv - sv) / eq if eq > 0 else 0.0)
            long_l.append(lv / eq if eq > 0 else 0.0)
            short_l.append(sv / eq if eq > 0 else 0.0)
            for a in range(k):
                sh = acct.shares[a]
                weights[a].append(sh * marks[a] / eq if eq > 0 and sh else 0.0)
                asset_pnl[a].append(acct.asset_cash[a] + (sh * marks[a] if sh else 0.0))
            if portfolio:
                exposure.append(gross_l[-1])
            else:
                position.append(acct.shares[0])
                exposure.append(acct.shares[0] * marks[0] / eq if eq > 0 else 0.0)
            if t == e or wiped:
                continue

            ctx._sync(t)
            bar_date = panel.dates[t]
            if row is not None:
                w = row(t)
                if w is not None:
                    pending = ("weights", w)
                continue
            run.orders = {}
            r = strategy.on_bar(ctx)
            if r is not None:
                if portfolio:
                    w = _weights_from(r, panel, "on_bar")
                else:
                    try:
                        w = float(r)
                    except (TypeError, ValueError):
                        raise TypeError("on_bar must return a number (target weight) or None, got %r" % (r,))
                    w = None if isnan(w) else {0: w}
                if w is not None:
                    pending = ("weights", w)
            elif run.orders:
                pending = ("shares", dict(run.orders))
    except StrategyError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StrategyError(format_error(exc, bar_date))
    finally:
        sys.stdout = old_stdout

    for a in range(k):
        if acct.open_trades[a] is not None:
            m = panel.mark(a, e)
            acct._close_trade(a, e, m, acct.asset_value(a, m), still_open=True)
    acct.trades.sort(key=lambda tr: (tr["entry_i"], tr["ticker"]))

    dates = panel.dates[s:e + 1]
    b0 = bench_vals[0]
    bench_equity = [cfg.cash * v / b0 for v in bench_vals]
    logs = out_buf.getvalue()
    if len(logs) > 20000:
        logs = logs[:20000] + "\n... output truncated ...\n"
    if acct.margin_calls:
        notes.append("%d margin call%s (first on %s): positions were cut at the close to restore the margin."
                     % (len(acct.margin_calls), "" if len(acct.margin_calls) == 1 else "s", acct.margin_calls[0]["date"]))
    if acct.scaled:
        notes.append("%d rebalance%s asked for more than the %.1fx gross leverage limit and %s scaled down."
                     % (acct.scaled, "" if acct.scaled == 1 else "s", cfg.max_leverage, "was" if acct.scaled == 1 else "were"))
    return Result(
        ticker=panel.tickers[0] if not portfolio else "Portfolio", tickers=list(panel.tickers),
        benchmark=bench_name, portfolio=portfolio, panel=panel,
        start=s, end=e, dates=dates, close=None if portfolio else panel.sets[0].close[s:e + 1],
        equity=equity, position=position, exposure=exposure,
        gross=gross_l, net=net_l, long=long_l, short=short_l,
        weights={tk: weights[a] for a, tk in enumerate(panel.tickers)},
        asset_pnl={tk: asset_pnl[a] for a, tk in enumerate(panel.tickers)},
        asset_costs={tk: acct.asset_costs[a] for a, tk in enumerate(panel.tickers)},
        asset_dividends={tk: acct.asset_dividends[a] for a, tk in enumerate(panel.tickers)},
        asset_notional={tk: acct.asset_notional[a] for a, tk in enumerate(panel.tickers)},
        bench_equity=bench_equity, fills=acct.fills, trades=acct.trades,
        commission=acct.commission, slippage=acct.slippage,
        traded_notional=acct.traded_notional, clipped=acct.clipped,
        carry=dict(acct.carry), margin_calls=acct.margin_calls,
        logs=logs, notes=notes, config=cfg, mode=strategy.mode, params=dict(strategy.params),
        signals=sig,
    )


def _price_only(ds):
    """A copy of ds without dividends (for total_return=False runs)."""
    import copy
    c = copy.copy(ds)
    c.dividends = [0.0] * ds.n
    c.dividend_source = "off"
    c._cache = {}
    return c


# =============================================================================
# 6. Performance metrics
# =============================================================================

# Each statistic has a tier that decides whether the web page shows it:
#   core   always shown (the numbers almost everyone looks at)
#   std    shown by default, can be hidden
#   extra  hidden by default, can be switched on
METRICS = [
    # key, label, group, format, tier, description
    ("total_return", "Total return", "Returns", "pct", "core", "Final equity / starting equity - 1."),
    ("cagr", "CAGR", "Returns", "pct", "core", "Compound annual growth rate: (final / start) ^ (1 / years) - 1."),
    ("final_equity", "Final equity", "Returns", "money", "core", "Account value on the last bar."),
    ("ret_1y", "Last 12 months", "Returns", "pct", "extra", "Return over the final 365 calendar days of the backtest."),
    ("avg_month", "Average month", "Returns", "pct", "extra", "Mean calendar-month return."),
    ("best_day", "Best day", "Returns", "pct", "std", "Largest one-day return."),
    ("worst_day", "Worst day", "Returns", "pct", "std", "Largest one-day loss."),
    ("best_month", "Best month", "Returns", "pct", "std", "Largest calendar-month return."),
    ("worst_month", "Worst month", "Returns", "pct", "std", "Largest calendar-month loss."),
    ("best_year", "Best year", "Returns", "pct", "std", "Largest calendar-year return (first and last years may be partial)."),
    ("worst_year", "Worst year", "Returns", "pct", "std", "Worst calendar-year return."),
    ("pct_pos_days", "Positive days", "Returns", "pct", "std", "Share of days with a gain."),
    ("pct_pos_months", "Positive months", "Returns", "pct", "std", "Share of months with a gain."),
    ("pct_pos_years", "Positive years", "Returns", "pct", "extra", "Share of calendar years with a gain."),

    ("volatility", "Volatility (ann.)", "Risk", "pct", "core", "Standard deviation of daily returns x sqrt(252)."),
    ("downside_dev", "Downside deviation", "Risk", "pct", "std", "Root-mean-square of returns below the risk-free rate, annualized."),
    ("max_drawdown", "Max drawdown", "Risk", "pct", "core", "Largest peak-to-trough fall in equity."),
    ("max_dd_days", "Longest drawdown", "Risk", "days", "core", "Most trading days spent below a previous equity high."),
    ("avg_drawdown", "Average drawdown", "Risk", "pct", "std", "Mean depth of all drawdown episodes."),
    ("pain_index", "Pain index", "Risk", "pct", "extra", "Mean depth below the running equity high, over every day (0 on new highs)."),
    ("ulcer_index", "Ulcer index", "Risk", "pct", "std", "sqrt(mean(drawdown^2)): punishes deep and long drawdowns."),
    ("var_95", "VaR 95% (1 day)", "Risk", "pct", "std", "Historical 1-day Value at Risk: the loss exceeded on 5% of days."),
    ("cvar_95", "CVaR 95% (1 day)", "Risk", "pct", "std", "Expected shortfall: the average loss on the worst 5% of days."),
    ("var_99", "VaR 99% (1 day)", "Risk", "pct", "extra", "Historical 1-day Value at Risk: the loss exceeded on 1% of days."),
    ("cvar_99", "CVaR 99% (1 day)", "Risk", "pct", "extra", "Expected shortfall: the average loss on the worst 1% of days."),
    ("skew", "Skew", "Risk", "num", "std", "Asymmetry of daily returns. Negative = fat left tail."),
    ("kurtosis", "Excess kurtosis", "Risk", "num", "std", "Tail heaviness vs. a normal distribution (0 = normal)."),

    ("sharpe", "Sharpe ratio", "Risk-adjusted", "num", "core", "(mean daily return - rf) / std x sqrt(252)."),
    ("sortino", "Sortino ratio", "Risk-adjusted", "num", "core", "Like Sharpe but only penalizes downside volatility."),
    ("calmar", "Calmar ratio", "Risk-adjusted", "num", "core", "CAGR / |max drawdown|."),
    ("omega", "Omega ratio", "Risk-adjusted", "num", "std", "Sum of daily gains / sum of daily losses (threshold 0)."),
    ("tail_ratio", "Tail ratio", "Risk-adjusted", "num", "std", "|95th percentile| / |5th percentile| of daily returns."),
    ("martin", "Martin ratio", "Risk-adjusted", "num", "extra", "(CAGR - rf) / ulcer index. Rewards returns earned with shallow, short drawdowns."),
    ("pain_ratio", "Pain ratio", "Risk-adjusted", "num", "extra", "(CAGR - rf) / pain index."),
    ("recovery_factor", "Recovery factor", "Risk-adjusted", "num", "extra", "Total return / |max drawdown|: how many times the worst loss was earned back."),
    ("gain_to_pain", "Gain-to-pain ratio", "Risk-adjusted", "num", "extra", "Sum of monthly returns / |sum of losing months| (Schwager). Above 1 is good."),
    ("kelly", "Kelly fraction", "Risk-adjusted", "num", "extra", "(mean daily return - rf) / variance: the growth-optimal leverage if returns repeat. Most traders use half or less."),
    ("stability", "Stability (R\u00b2)", "Risk-adjusted", "num", "extra", "R-squared of a straight-line fit to log equity. 1 = perfectly steady growth."),

    ("beta", "Beta", "Vs benchmark", "num", "core", "Sensitivity to the benchmark: cov(r, rb) / var(rb)."),
    ("alpha", "Alpha (ann.)", "Vs benchmark", "pct", "core", "Jensen's alpha: return not explained by beta x benchmark, annualized."),
    ("correlation", "Correlation", "Vs benchmark", "num", "core", "Correlation of daily returns with the benchmark."),
    ("r_squared", "R-squared", "Vs benchmark", "num", "std", "Share of return variance explained by the benchmark."),
    ("tracking_error", "Tracking error", "Vs benchmark", "pct", "std", "Annualized std of (strategy - benchmark) daily returns."),
    ("information_ratio", "Information ratio", "Vs benchmark", "num", "std", "Annualized excess return / tracking error."),
    ("treynor", "Treynor ratio", "Vs benchmark", "num", "extra", "(CAGR - rf) / beta: return per unit of market risk."),
    ("up_capture", "Up capture", "Vs benchmark", "pct", "std", "Average strategy return on benchmark up days / benchmark's."),
    ("down_capture", "Down capture", "Vs benchmark", "pct", "std", "Average strategy return on benchmark down days / benchmark's. Lower is better."),
    ("beat_days", "Days beating benchmark", "Vs benchmark", "pct", "extra", "Share of days the strategy returned more than the benchmark."),
    ("beat_months", "Months beating benchmark", "Vs benchmark", "pct", "extra", "Share of calendar months the strategy returned more than the benchmark."),
    ("excess_cagr", "Excess CAGR", "Vs benchmark", "pct", "core", "Strategy CAGR - benchmark CAGR."),

    ("n_trades", "Trades", "Trading", "int", "core", "Closed round trips (flat -> position -> flat)."),
    ("trades_per_year", "Trades per year", "Trading", "num", "extra", "Closed trades / years in the backtest."),
    ("win_rate", "Win rate", "Trading", "pct", "core", "Share of closed trades with a profit."),
    ("avg_trade", "Average trade", "Trading", "pct", "core", "Mean return per closed trade."),
    ("avg_win", "Average win", "Trading", "pct", "std", "Mean return of winning trades."),
    ("avg_loss", "Average loss", "Trading", "pct", "std", "Mean return of losing trades."),
    ("payoff_ratio", "Payoff ratio", "Trading", "num", "extra", "Average win / |average loss|. A low win rate needs a high payoff ratio."),
    ("best_trade", "Best trade", "Trading", "pct", "std", "Largest single-trade return."),
    ("worst_trade", "Worst trade", "Trading", "pct", "std", "Largest single-trade loss."),
    ("profit_factor", "Profit factor", "Trading", "num", "core", "Gross profit / gross loss over closed trades."),
    ("expectancy", "Expectancy", "Trading", "money", "std", "Average dollar profit per closed trade."),
    ("sqn", "System quality (SQN)", "Trading", "num", "extra", "sqrt(trades) x mean trade return / std of trade returns (Van Tharp). Above 2 is decent, above 3 good."),
    ("avg_bars", "Avg. bars held", "Trading", "num", "std", "Mean length of a closed trade in bars."),
    ("avg_bars_win", "Avg. bars in winners", "Trading", "num", "extra", "Mean length of winning trades in bars."),
    ("avg_bars_loss", "Avg. bars in losers", "Trading", "num", "extra", "Mean length of losing trades in bars. Longer than winners can mean losers are held too long."),
    ("max_consec_wins", "Max win streak", "Trading", "int", "std", "Most winning trades in a row."),
    ("max_consec_losses", "Max loss streak", "Trading", "int", "std", "Most losing trades in a row."),
    ("exposure", "Time in market", "Trading", "pct", "core", "Share of bars with an open position."),
    ("avg_leverage", "Avg. exposure", "Trading", "pct", "std", "Mean |position value| / equity."),
    ("turnover", "Turnover (ann.)", "Trading", "num", "std", "Traded value / average equity per year (2.0 = account traded twice a year)."),
    ("n_fills", "Fills", "Trading", "int", "std", "Number of executed orders."),
    ("total_costs", "Costs paid", "Trading", "money", "core", "Commissions plus slippage."),
    ("orders_clipped", "Orders reduced", "Trading", "int", "std", "Share orders cut down for lack of cash or the no-short rule."),

    ("avg_gross", "Gross exposure (avg.)", "Carry & exposure", "pct", "std", "Mean of (long + short market value) / equity."),
    ("avg_net", "Net exposure (avg.)", "Carry & exposure", "pct", "std", "Mean of (long - short market value) / equity."),
    ("avg_long", "Long exposure (avg.)", "Carry & exposure", "pct", "extra", "Mean long market value / equity."),
    ("avg_short", "Short exposure (avg.)", "Carry & exposure", "pct", "extra", "Mean short market value / equity."),
    ("max_gross", "Gross exposure (max)", "Carry & exposure", "pct", "extra", "Highest gross exposure at any close."),
    ("dividends_net", "Dividends", "Carry & exposure", "money", "std", "Dividends received on longs minus dividends paid on shorts."),
    ("interest_net", "Interest on cash", "Carry & exposure", "money", "std", "Interest earned on cash minus interest paid on borrowed cash (ACT/360)."),
    ("borrow_paid", "Borrow fees", "Carry & exposure", "money", "extra", "Fees paid to borrow shares for short positions."),
    ("carry_total", "Net carry", "Carry & exposure", "money", "extra", "Dividends + interest - borrow fees."),
    ("margin_calls", "Margin calls", "Carry & exposure", "int", "std", "Times equity fell below the maintenance margin and positions were cut."),

    ("t_stat", "t-statistic (mean return)", "Statistical tests", "num", "extra", "Mean daily excess return / its standard error. Above 2 suggests the mean is not zero."),
    ("p_value", "p-value (mean return > 0)", "Statistical tests", "pval", "extra", "Chance of a t-statistic this high if the true mean return were zero (one-sided, normal approximation). Below 0.05 is the usual bar."),
    ("psr", "Probabilistic Sharpe", "Statistical tests", "pct", "extra", "Probability that the true Sharpe ratio is above 0, allowing for skew, fat tails and track length (Bailey & Lopez de Prado)."),
    ("sharpe_lo", "Sharpe 95% CI low", "Statistical tests", "num", "extra", "Lower end of a 95% confidence interval for the annualized Sharpe ratio."),
    ("sharpe_hi", "Sharpe 95% CI high", "Statistical tests", "num", "extra", "Upper end of a 95% confidence interval for the annualized Sharpe ratio."),
    ("min_trl", "Min. track record", "Statistical tests", "years", "extra", "Years of data needed to be 95% sure the Sharpe ratio is above 0. Longer than the backtest = not proven yet."),
    ("alpha_t", "Alpha t-statistic", "Statistical tests", "num", "extra", "Alpha / its standard error from the regression on the benchmark. Above 2 suggests real alpha."),
    ("alpha_p", "Alpha p-value", "Statistical tests", "pval", "extra", "Chance of an alpha t-statistic this high if true alpha were zero (one-sided, normal approximation)."),
    ("win_rate_p", "Win-rate p-value", "Statistical tests", "pval", "extra", "Chance of at least this many winning trades if each trade were a coin flip (one-sided binomial, normal approximation)."),
    ("autocorr", "Autocorrelation (1 day)", "Statistical tests", "num", "extra", "Correlation of each day's return with the previous day's. Far from 0 means returns are not independent."),
    ("autocorr_p", "Autocorrelation p-value", "Statistical tests", "pval", "extra", "Two-sided p-value that the 1-day autocorrelation is zero. Below 0.05 = significant serial correlation."),
    ("jb_p", "Normality p-value", "Statistical tests", "pval", "extra", "Jarque-Bera test that daily returns are normally distributed. Below 0.05 = fat tails or skew; treat Sharpe and VaR with care."),
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
    out["pain_index"] = -mean(dd) if dd else 0.0
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


def return_tests(r, rfd=0.0):
    """Significance tests on daily returns r (rfd = daily risk-free rate).

    The Sharpe ratio's standard error follows Lo (2002) / Mertens (2002), which
    widens it for negative skew and fat tails; PSR and the minimum track record
    length follow Bailey & Lopez de Prado (2012). p-values use the normal
    approximation, which is accurate for the hundreds of days a backtest has.
    """
    out = {}
    n = len(r)
    if n < 4:
        return out
    mu, sd = mean(r), stdev(r)
    g3, g4 = skewness(r), kurtosis(r)
    if sd > 0:
        sr = (mu - rfd) / sd
        out["t_stat"] = sr * math.sqrt(n)
        out["p_value"] = 1 - norm_cdf(out["t_stat"])
        if not isnan(g3) and not isnan(g4):
            spread = 1 - g3 * sr + (g4 + 2) / 4 * sr * sr   # (g4 + 2) = raw kurtosis - 1
            if spread > 0:
                se = math.sqrt(spread / (n - 1))
                ann = math.sqrt(TRADING_DAYS)
                out["psr"] = norm_cdf(sr / se)
                out["sharpe_lo"] = (sr - Z_975 * se) * ann
                out["sharpe_hi"] = (sr + Z_975 * se) * ann
                if sr > 0:
                    out["min_trl"] = (1 + spread * (Z_95 / sr) ** 2) / TRADING_DAYS
    if not isnan(g3) and not isnan(g4):
        jb = n / 6 * (g3 * g3 + g4 * g4 / 4)
        out["jb_p"] = math.exp(-jb / 2)   # chi-squared with 2 degrees of freedom
    rho = correlation(r[:-1], r[1:])
    if not isnan(rho):
        out["autocorr"] = rho
        out["autocorr_p"] = math.erfc(abs(rho) * math.sqrt(n) / math.sqrt(2))
    return out


def trade_stats(trades, ret_key="ret"):
    """ret_key 'ret' measures a trade against account equity at entry (single
    instrument); 'ret_notional' against the capital the position used (portfolios)."""
    closed = [t for t in trades if not t["open"]]
    m = {"n_trades": len(closed)}
    if not closed:
        return m
    rets = [t[ret_key] for t in closed]
    pnls = [t["pnl"] for t in closed]
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    m["win_rate"] = len(wins) / len(closed)
    m["avg_trade"] = mean(rets)
    m["avg_win"] = mean([t[ret_key] for t in wins]) if wins else NAN
    m["avg_loss"] = mean([t[ret_key] for t in losses]) if losses else NAN
    m["best_trade"] = max(rets)
    m["worst_trade"] = min(rets)
    gp = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    m["profit_factor"] = gp / gl if gl > 0 else (math.inf if gp > 0 else NAN)
    m["expectancy"] = mean(pnls)
    m["avg_bars"] = mean([t["bars"] for t in closed])
    m["avg_bars_win"] = mean([t["bars"] for t in wins]) if wins else NAN
    m["avg_bars_loss"] = mean([t["bars"] for t in losses]) if losses else NAN
    m["payoff_ratio"] = _div(m["avg_win"], abs(m["avg_loss"]))
    sd = stdev(rets)
    m["sqn"] = math.sqrt(len(rets)) * mean(rets) / sd if len(rets) >= 2 and sd > 0 else NAN
    # One-sided binomial test of the win rate against a coin flip, with continuity correction.
    n = len(closed)
    m["win_rate_p"] = 1 - norm_cdf((len(wins) - 0.5 - n / 2) / math.sqrt(n / 4))
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
                traded_notional=None, costs=None, n_fills=None, clipped=None, trade_ret="ret"):
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
        p1 = percentile(r, 1)
        m["var_99"] = -p1
        m["cvar_99"] = -mean([x for x in r if x <= p1])
        m["kelly"] = (mu - rfd) / (sd * sd) if sd > 0 else NAN
        m.update(return_tests(r, rfd))

    dd = drawdown_stats(equity, dates)
    m["max_drawdown"] = dd["max_drawdown"]
    m["max_dd_days"] = dd["max_dd_days"]
    m["avg_drawdown"] = dd["avg_drawdown"]
    m["ulcer_index"] = dd["ulcer_index"]
    m["pain_index"] = dd["pain_index"]
    for k in ("dd_peak", "dd_trough", "dd_recovery"):
        if k in dd:
            m[k] = dd[k]
    m["calmar"] = m["cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] < 0 else NAN
    m["recovery_factor"] = m["total_return"] / abs(m["max_drawdown"]) if m["max_drawdown"] < 0 else NAN
    m["martin"] = (m["cagr"] - rf) / m["ulcer_index"] if m["ulcer_index"] > 0 else NAN
    m["pain_ratio"] = (m["cagr"] - rf) / m["pain_index"] if m["pain_index"] > 0 else NAN
    if all(v > 0 for v in equity) and n >= 3:
        rho = correlation(list(range(n)), [math.log(v) for v in equity])
        m["stability"] = rho * rho if not isnan(rho) else NAN

    if years >= 1:
        cut = (_dt.date.fromisoformat(dates[-1]) - _dt.timedelta(days=365)).isoformat()
        k = next(i for i, d in enumerate(dates) if d >= cut)
        m["ret_1y"] = e1 / equity[k] - 1 if equity[k] > 0 else NAN

    months = [v for _, v in period_returns(dates, equity, "month")]
    yrs = [v for _, v in period_returns(dates, equity, "year")]
    if months:
        m["best_month"], m["worst_month"] = max(months), min(months)
        m["pct_pos_months"] = sum(1 for v in months if v > 0) / len(months)
        m["avg_month"] = mean(months)
        pain = -sum(v for v in months if v < 0)
        m["gain_to_pain"] = sum(months) / pain if pain > 0 else NAN
    if yrs:
        m["best_year"], m["worst_year"] = max(yrs), min(yrs)
        m["pct_pos_years"] = sum(1 for v in yrs if v > 0) / len(yrs)

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
        m.update(trade_stats(trades, trade_ret))
        m["trades_per_year"] = m["n_trades"] / years

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
        m["treynor"] = (m["cagr"] - rf) / beta if not isnan(beta) and abs(beta) > 1e-6 else NAN
        m["beat_days"] = sum(1 for a, b in zip(r, rb) if a > b) / len(r)
        sm = period_returns(dates, equity, "month")
        bm = period_returns(dates, bench_equity, "month")
        m["beat_months"] = sum(1 for (_, a), (_, b) in zip(sm, bm) if a > b) / len(sm)
        # Regression r - rf = a + beta (rb - rf) + e; the t-statistic of the intercept a.
        if not isnan(beta):
            x = [v - rfd for v in rb]
            y = [v - rfd for v in r]
            mx = mean(x)
            a = mean(y) - beta * mx
            sxx = sum((v - mx) ** 2 for v in x)
            s2 = sum((yv - a - beta * xv) ** 2 for xv, yv in zip(x, y)) / (len(x) - 2)
            se = math.sqrt(s2 * (1 / len(x) + mx * mx / sxx)) if sxx > 0 else 0.0
            if se > 1e-15:
                m["alpha_t"] = a / se
                m["alpha_p"] = 1 - norm_cdf(a / se)
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


def rolling_vol(equity, window=63):
    """Annualized volatility of the last `window` daily returns, aligned with equity."""
    r = daily_returns(equity)
    out = [NAN] * len(equity)
    ann = math.sqrt(TRADING_DAYS)
    for t, v in enumerate(rolling_std(r, window, ddof=1)):
        out[t + 1] = v * ann if not isnan(v) else NAN
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


def exposure_stats(res, a=0, b=None):
    """Average exposures and carry for bars a..b-1 of a Result."""
    b = len(res.equity) if b is None else b
    m = {}
    for key, series in (("avg_gross", res.gross), ("avg_net", res.net), ("avg_long", res.long), ("avg_short", res.short)):
        part = series[a:b]
        m[key] = mean(part) if part else NAN
    m["max_gross"] = max(res.gross[a:b]) if res.gross[a:b] else NAN
    return m


def summarize(res, rf=None):
    """Strategy and benchmark metrics for a Result."""
    rf = res.config.risk_free if rf is None else rf
    m = performance(res.dates, res.equity, rf, res.bench_equity, res.trades, res.exposure,
                    res.traded_notional, res.commission + res.slippage, len(res.fills), res.clipped,
                    "ret_notional" if res.portfolio else "ret")
    m.update(exposure_stats(res))
    c = res.carry
    m["dividends_net"] = c["dividends_received"] - c["dividends_paid"]
    m["interest_net"] = c["interest_earned"] - c["interest_paid"]
    m["borrow_paid"] = c["borrow_paid"]
    m["carry_total"] = m["dividends_net"] + m["interest_net"] - c["borrow_paid"]
    m["margin_calls"] = len(res.margin_calls)
    b = performance(res.dates, res.bench_equity, rf)
    return m, b


def attribution(res):
    """P&L by instrument: what each one contributed, plus interest on cash.
    The rows add up to the account's total P&L."""
    cash0 = res.config.cash
    years = max(len(res.dates) / TRADING_DAYS, 1e-9)
    avg_eq = mean(res.equity) or cash0
    rows = []
    for tk in res.tickers:
        w = res.weights[tk]
        tr = [t for t in res.trades if t["ticker"] == tk and not t["open"]]
        pnl = res.asset_pnl[tk][-1] if res.asset_pnl[tk] else 0.0
        rows.append({
            "ticker": tk, "pnl": pnl, "contribution": pnl / cash0,
            "avg_weight": mean(w), "avg_abs_weight": mean([abs(x) for x in w]),
            "time_held": sum(1 for x in w if x) / len(w),
            "trades": len(tr), "win_rate": (sum(1 for t in tr if t["pnl"] > 0) / len(tr)) if tr else NAN,
            "dividends": res.asset_dividends[tk], "costs": res.asset_costs[tk],
            "turnover": res.asset_notional[tk] / avg_eq / years,
        })
    rows.sort(key=lambda r: -r["pnl"])
    c = res.carry
    interest = c["interest_earned"] - c["interest_paid"]
    total = res.equity[-1] - cash0
    return {"rows": rows, "interest": interest, "total": total,
            "check": total - interest - sum(r["pnl"] for r in rows)}


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


def _segment(res, a, b, rf):
    """Metrics for bars a..b-1 of a Result (indexes relative to res.dates)."""
    d = res.dates[a:b]
    eq = res.equity[a:b]
    be = res.bench_equity[a:b]
    lo, hi = res.start + a, res.start + b - 1
    trades = [t for t in res.trades if lo <= t["entry_i"] <= hi]
    notional = sum(abs(f["qty"]) * f["price"] for f in res.fills if lo <= f["i"] <= hi)
    fills = sum(1 for f in res.fills if lo <= f["i"] <= hi)
    return {
        "start": d[0], "end": d[-1],
        "metrics": dict(performance(d, eq, rf, be, trades, res.exposure[a:b], notional, None, fills,
                                    "ret_notional" if res.portfolio else "ret"), **exposure_stats(res, a, b)),
        "bench": performance(d, be, rf),
    }


def split_metrics(res, split_index, rf=None):
    """Metrics for the in-sample (before split) and out-of-sample (after) parts of
    one continuous backtest. Positions and indicator warm-up carry across the split."""
    rf = res.config.risk_free if rf is None else rf
    n = len(res.dates)
    k = max(2, min(int(split_index), n - 2))
    return {"split_date": res.dates[k], "in_sample": _segment(res, 0, k + 1, rf),
            "out_of_sample": _segment(res, k, n, rf)}


def subperiod_metrics(res, n_periods=5, rf=None):
    """Cut one continuous backtest into n_periods back-to-back pieces of equal length
    and score each. A real edge should show up in most of them, not just one."""
    rf = res.config.risk_free if rf is None else rf
    n = len(res.dates)
    n_periods = max(2, min(int(n_periods), n // 20))
    if n_periods < 2:
        raise ValueError("Need at least 40 bars to compare sub-periods.")
    cuts = [int(round(j * (n - 1) / n_periods)) for j in range(n_periods + 1)]
    periods = [_segment(res, cuts[j], cuts[j + 1] + 1, rf) for j in range(n_periods)]
    pos = sum(1 for p in periods if p["metrics"]["total_return"] > 0)
    beat = sum(1 for p in periods if p["metrics"]["total_return"] > p["bench"]["total_return"])
    return {"periods": periods, "n": n_periods, "positive": pos, "beat": beat}


def _fresh(strategy):
    """Strategies may be passed as a Strategy or a function that builds a new one."""
    return strategy() if callable(strategy) and not isinstance(strategy, Strategy) else strategy


def cost_sensitivity(strategy, ds, cfg, levels, start=None, end=None, bench=None, rates=None):
    """Re-run the backtest at each cost level and report how the result decays.

    A level is the cost of every fill in basis points of traded value. It replaces
    the percentage commission and the slippage; the flat per-fill fee is kept.
    """
    levels = sorted(set(float(v) for v in levels))
    if not levels:
        raise ValueError("Give at least one cost level.")
    base = cfg.as_dict()
    rows, bench_m = [], None
    for lv in levels:
        c = Config(**dict(base, commission_bps=0.0, slippage_bps=lv))
        if isinstance(ds, (list, tuple)):
            res = backtest_portfolio(_fresh(strategy), ds, c, start, end, bench, rates=rates)
        else:
            res = backtest(_fresh(strategy), ds, c, start, end, bench, rates=rates)
        m, bench_m = summarize(res)
        rows.append({"bps": lv, "total_return": m["total_return"], "cagr": m["cagr"], "sharpe": m.get("sharpe"),
                     "max_drawdown": m["max_drawdown"], "costs": m.get("total_costs"), "turnover": m.get("turnover"),
                     "excess_cagr": m.get("excess_cagr")})

    def crossing(key, target):
        # First cost level (linearly interpolated) at which rows[key] falls to target.
        for a, b in zip(rows, rows[1:]):
            va, vb = a[key], b[key]
            if isnan(va) or isnan(vb):
                continue
            if va > target >= vb:
                return a["bps"] + (b["bps"] - a["bps"]) * (va - target) / (va - vb)
        return None

    return {"rows": rows, "bench_cagr": bench_m["cagr"], "current_bps": cfg.commission_bps + cfg.slippage_bps,
            "breakeven_zero": crossing("cagr", 0.0), "breakeven_bench": crossing("cagr", bench_m["cagr"])}


def permutation_test(res, n_perm=500, seed=7):
    """Random-timing test: is the strategy's timing better than chance?

    Shuffles the strategy's daily exposures (keeping how often and how heavily it
    is invested, but not WHEN) and pairs them with the asset's actual daily
    returns. If many shuffles score as well as the real ordering, the timing adds
    little. Scores are before costs, so they isolate timing skill.
    """
    panel, s = res.panel, res.start
    n = len(res.equity)
    if n < 30:
        raise ValueError("Need at least 30 bars for a random-timing test.")
    k = panel.k
    tr = [panel.aligned(a, "total_return")[s:s + n] for a in range(k)]
    r = [[(x[t] / x[t - 1] - 1) if not (isnan(x[t]) or isnan(x[t - 1])) else 0.0 for x in tr] for t in range(1, n)]
    W = [res.weights[tk] for tk in res.tickers]
    pos = [[W[a][t] for a in range(k)] for t in range(n - 1)]  # weights at bar t's close are held over bar t + 1
    ann = math.sqrt(TRADING_DAYS)
    m = len(r)

    def sharpe(p):
        s = ss = 0.0
        for wv, rv in zip(p, r):
            x = wv[0] * rv[0] if k == 1 else sum(a * b for a, b in zip(wv, rv))
            s += x
            ss += x * x
        var = (ss - s * s / m) / (m - 1)
        return s / m / math.sqrt(var) * ann if var > 1e-18 else 0.0

    actual = sharpe(pos)
    n_perm = max(50, min(int(n_perm), 5000))
    # Nearly the same exposure every day (e.g. buy and hold): shuffling changes nothing.
    tot = [sum(abs(v) for v in wv) for wv in pos]
    spread = max(percentile([wv[a] for wv in pos], 98) - percentile([wv[a] for wv in pos], 2) for a in range(k))
    constant = spread < 0.05 and percentile(tot, 98) - percentile(tot, 2) < 0.05
    rng = random.Random(seed)
    perm, scores = pos[:], []
    for _ in range(n_perm):
        rng.shuffle(perm)
        scores.append(sharpe(perm))
    better = sum(1 for s in scores if s >= actual - 1e-12)
    return {
        "actual": actual, "n_perm": n_perm, "constant": constant,
        "p_value": (1 + better) / (n_perm + 1),
        "percentile": sum(1 for s in scores if s < actual) / n_perm,
        "q": {str(q): percentile(scores, q) for q in (5, 50, 95)},
        "hist": histogram([scores], 40),
    }


def _same_weight(a, b):
    if isinstance(a, dict) or isinstance(b, dict):
        a, b = a or {}, b or {}
        return set(a) == set(b) and all(abs(a[k] - b[k]) < 1e-9 for k in a)
    return (isnan(a) and isnan(b)) or (not isnan(a) and not isnan(b) and abs(float(a) - float(b)) < 1e-9)


def lookahead_check(strategy, ds, full_signals, samples=8, budget=None):
    """For signals(data) strategies: recompute the signal using only data up to bar k
    and compare with the full-history value. Any difference means the signal for bar k
    used later bars. With a time budget in seconds (for slow strategies such as
    machine-learning models), stops after the budget once two samples are done.

    ds is a DataSet (single instrument) or a Panel (portfolio; full_signals are then
    the normalized rows from signal_rows)."""
    import time
    portfolio = isinstance(ds, Panel)
    n = ds.n
    lo = max(1, n // 5)
    picks = sorted(set(lo + (n - 1 - lo) * j // max(1, samples - 1) for j in range(samples)))
    # check the extremes first, so a cut-short check still covers early and late bars
    picks = [picks[0], picks[-1]] + picks[1:-1] if len(picks) > 2 else picks
    bad = []
    t0 = time.time()
    done = 0
    for k in picks:
        if budget is not None and done >= 2 and time.time() - t0 > budget:
            break
        done += 1
        if portfolio:
            _, sub = signal_rows(strategy.signals(PanelView(ds, k + 1)), ds, True, k + 1)
            name = lambda r: None if r is None else {ds.tickers[x]: v for x, v in r.items()}  # noqa: E731
            a, b = sub[k], full_signals[k]
        else:
            sub = list(strategy.signals(DataView(ds, k + 1)))
            a, b = sub[k] if len(sub) > k else None, full_signals[k]
        if not _same_weight(a, b):
            bad.append({"date": ds.dates[k], "with_future": name(b) if portfolio else b,
                        "without_future": name(a) if portfolio else a})
    bad.sort(key=lambda v: v["date"])
    return {"checked": done, "planned": len(picks), "violations": bad}


def instrument_stats(datasets, start=None, end=None, rf=0.0):
    """Per-instrument buy-and-hold stats plus a correlation matrix of daily returns
    on the dates all instruments share."""
    rows = []
    for ds in datasets:
        try:
            s, e = _resolve_range(ds.dates, start, end, "%s data" % ds.ticker)
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
        s, e = _resolve_range(ds.dates, start, end, "%s data" % ds.ticker)
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
    """Payload -> (datasets, benchmark, config, strategy, portfolio, rates).

    A single-instrument run sends 'ticker'; a portfolio run sends 'tickers' (or
    'portfolio': true with one ticker). 'benchmark' empty or equal to the single
    ticker means buy and hold of that ticker (single) or equal weight (portfolio)."""
    portfolio = bool(p.get("tickers")) or bool(p.get("portfolio"))
    tickers = list(p.get("tickers") or [p["ticker"]])
    sets = [_get(t) for t in tickers]
    b = p.get("benchmark") or ""
    bench = _get(b) if b and not (not portfolio and b == tickers[0]) else None
    cfg = Config(**(p.get("config") or {}))
    rates = None
    if cfg.rate_ticker:
        if cfg.rate_ticker not in RATES:
            raise ValueError("No rate series named %s is loaded. Add it on the Data tab." % cfg.rate_ticker)
        rates = RATES[cfg.rate_ticker]
    strat = Strategy.from_code(p["code"], p.get("params"))
    return sets, bench, cfg, strat, portfolio, rates


def _run_payload(p, strat=None, cfg=None):
    sets, bench, cfg0, strat0, portfolio, rates = _prepare(p)
    strat, cfg = strat or strat0, cfg or cfg0
    if portfolio:
        res = backtest_portfolio(strat, sets, cfg, p.get("start"), p.get("end"), bench, rates=rates)
    else:
        res = backtest(strat, sets[0], cfg, p.get("start"), p.get("end"), bench, rates=rates)
    return res, strat


def _downsample(n, limit=600):
    step = max(1, -(-n // limit))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def api_load_data(p):
    """Load prices (and optionally 'dividends' / 'splits' as [[date, value], ...])."""
    ds = DataSet.from_dict(p["ticker"], p["data"])
    if p.get("splits"):
        ds.apply_splits(p["splits"])
    if p.get("dividends"):
        ds.set_dividends(p["dividends"])
    DATA[p["ticker"]] = ds
    return dict(_data_info(ds), quality=quality_report(ds))


def _data_info(ds):
    return {"ticker": ds.ticker, "bars": ds.n, "start": ds.dates[0], "end": ds.dates[-1],
            "dividends": ds.dividend_source, "dividend_count": sum(1 for d in ds.dividends if d),
            "dividend_total": sum(ds.dividends), "price_basis": ds.price_basis,
            "splits": ds.splits_applied, "fingerprint": ds.fingerprint()}


def api_parse_csv(p):
    """Classify and parse any supported CSV: prices, dividends, splits or a rate series."""
    header, _ = read_csv_table(p["text"])
    kind = csv_kind(header)
    if kind == "prices":
        ds = parse_csv_text(p["text"], p.get("ticker") or "DATA")
        return {"kind": kind, "data": {"dates": ds.dates, "open": ds.open, "high": ds.high, "low": ds.low,
                                       "close": ds.close, "volume": ds.volume},
                "dividends": [[d, v] for d, v in zip(ds.dates, ds.dividends) if v],
                "price_basis": ds.price_basis, "quality": quality_report(ds)}
    if kind == "rates":
        r = parse_rate_csv(p["text"], p.get("ticker") or "RATE")
        return {"kind": kind, "dates": r.dates, "values": r.values}
    kind, events = parse_events_csv(p["text"])
    return {"kind": kind, "events": [[d, v] for d, v in events]}


def api_set_events(p):
    """Attach a dividend or split history to a loaded instrument."""
    ds = _get(p["ticker"])
    if p["kind"] == "splits":
        done = ds.apply_splits(p["events"])
        return dict(_data_info(ds), applied=done, quality=quality_report(ds))
    matched = ds.set_dividends(p["events"])
    return dict(_data_info(ds), matched=matched, quality=quality_report(ds))


RATES = {}


def api_load_rates(p):
    r = RateSeries(p["ticker"], p["dates"], p["values"])
    RATES[p["ticker"]] = r
    return {"ticker": r.ticker, "start": r.dates[0], "end": r.dates[-1], "last": r.values[-1], "points": len(r.dates)}


def api_quality(p):
    return quality_report(_get(p["ticker"]), p.get("calendar"))


def api_inspect(p):
    strat = Strategy.from_code(p["code"])
    return {"params": strat.params, "mode": strat.mode, "universe": strat.universe}


def api_run(p):
    res, strat = _run_payload(p)
    cfg = res.config
    m, b = summarize(res)
    window = int(p.get("rolling_window") or 126)
    out = {
        "ticker": res.ticker, "tickers": res.tickers, "portfolio": res.portfolio,
        "benchmark": res.benchmark, "mode": res.mode, "params": res.params,
        "dates": res.dates, "close": res.close, "equity": res.equity, "bench_equity": res.bench_equity,
        "drawdown": drawdown(res.equity), "bench_drawdown": drawdown(res.bench_equity),
        "exposure": res.exposure, "position": res.position,
        "gross": res.gross, "net": res.net, "long": res.long, "short": res.short,
        "fills": [{"i": f["i"] - res.start, "date": f["date"], "ticker": f["ticker"], "side": f["side"],
                   "qty": f["qty"], "price": f["price"], "commission": f["commission"],
                   "position": f["position"], "reason": f.get("reason")}
                  for f in res.fills],
        "trades": res.trades, "metrics": m, "bench_metrics": b,
        "carry": res.carry, "margin_calls": res.margin_calls[:50],
        "monthly": monthly_table(res.dates, res.equity),
        "bench_monthly": monthly_table(res.dates, res.bench_equity),
        "hist": histogram([daily_returns(res.equity), daily_returns(res.bench_equity)]),
        "rolling": {"window": window, "sharpe": rolling_sharpe(res.equity, window, cfg.risk_free),
                    "bench": rolling_sharpe(res.bench_equity, window, cfg.risk_free),
                    "vol_window": 63, "vol": rolling_vol(res.equity, 63), "bench_vol": rolling_vol(res.bench_equity, 63)},
        "annual": [{"year": y, "strategy": a, "bench": b} for (y, a), (_, b) in
                   zip(period_returns(res.dates, res.equity, "year"), period_returns(res.dates, res.bench_equity, "year"))],
        "logs": res.logs, "notes": res.notes, "config": cfg.as_dict(),
        "data": {tk: _data_info(ds) for tk, ds in zip(res.tickers, res.panel.sets)},
    }
    if res.portfolio:
        idx = _downsample(len(res.dates))
        out["attribution"] = attribution(res)
        out["weights"] = {"index": idx, "series": {tk: [res.weights[tk][i] for i in idx] for tk in res.tickers}}
    if res.signals is not None:
        data = res.panel if res.portfolio else res.panel.sets[0]
        out["lookahead"] = lookahead_check(strat, data, res.signals, budget=float(p.get("lookahead_budget", 4.0)))
    return out


def api_metrics(p):
    """Lightweight run for parameter sweeps: metrics only, optionally split."""
    res, _ = _run_payload(p)
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
    res, _ = _run_payload(p)
    mc = monte_carlo(daily_returns(res.equity), p.get("paths", 500), p.get("block", 20), p.get("seed", 7))
    mc["dates"] = [res.dates[c + 1] for c in mc["checkpoints"]]
    mc["start_date"] = res.dates[0]
    return mc


def api_split(p):
    res, _ = _run_payload(p)
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


def api_subperiods(p):
    res, _ = _run_payload(p)
    return subperiod_metrics(res, p.get("n", 5))


def api_costs(p):
    sets, bench, cfg, _, portfolio, rates = _prepare(p)
    return cost_sensitivity(lambda: Strategy.from_code(p["code"], p.get("params")), sets if portfolio else sets[0],
                            cfg, p.get("levels") or [0, 2, 5, 10, 20, 50], p.get("start"), p.get("end"), bench, rates)


def api_permutation(p):
    res, _ = _run_payload(p)
    return permutation_test(res, p.get("n_perm", 500), p.get("seed", 7))


def api_analytics(p):
    sets = [_get(t) for t in p["tickers"] if t in DATA]
    return instrument_stats(sets, p.get("start"), p.get("end"), float(p.get("rf", 0.0)))


def api_metric_list(p):
    return [{"key": k, "label": lab, "group": g, "fmt": f, "tier": t, "desc": d} for k, lab, g, f, t, d in METRICS]


API = {
    "load_data": api_load_data, "parse_csv": api_parse_csv, "set_events": api_set_events,
    "load_rates": api_load_rates, "quality": api_quality,
    "inspect": api_inspect, "run": api_run, "metrics": api_metrics,
    "montecarlo": api_montecarlo, "split": api_split, "analytics": api_analytics,
    "subperiods": api_subperiods, "costs": api_costs, "permutation": api_permutation,
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
