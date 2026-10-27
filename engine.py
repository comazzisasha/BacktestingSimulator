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

import csv
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

ENGINE_VERSION = "2.0"   # bump when a change alters results for the same inputs
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


def norm_ppf(p):
    """Inverse of the standard normal CDF (Acklam's rational approximation + one Newton step)."""
    if not 0 < p < 1:
        raise ValueError("norm_ppf needs 0 < p < 1")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = norm_cdf(x) - p                        # one Newton step polishes to ~1e-15
    return x - e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)


EULER_GAMMA = 0.5772156649015329
Z_95 = 1.6448536269514722   # one-sided 95% quantile of the standard normal
Z_975 = 1.959963984540054   # two-sided 95%


def _betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta function (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(ln) * _betacf(a, b, x) / a
    return 1.0 - math.exp(ln) * _betacf(b, a, 1 - x) / b


def t_sf(t, df):
    """P(T > t) for Student's t with df degrees of freedom (exact, via the incomplete beta)."""
    if isnan(t) or df <= 0:
        return NAN
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


def chi2_sf(x, df):
    """P(X > x) for a chi-squared variable (exact for 1 and 2 degrees of freedom, else via the
    regularized gamma function)."""
    if isnan(x):
        return NAN
    if x <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    if df == 2:
        return math.exp(-x / 2)
    return _gamma_q(df / 2.0, x / 2.0)


def _gamma_q(a, x):
    """Upper regularized incomplete gamma Q(a, x) (series below a + 1, continued fraction above)."""
    gl = math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - gl)
    tiny = 1e-300
    b, c, d = x + 1 - a, 1 / tiny, 1 / (x + 1 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1 / (d if abs(d) > tiny else tiny)
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - gl) * h


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


def walk_forward(X, y, fit_predict, min_train=252, retrain_every=21, horizon=1, window=None, fill=NAN, embargo=0):
    """Out-of-sample model predictions for every bar, without look-ahead.

    X             one row of features per bar (list of lists or a 2-D NumPy array).
                  Rows containing NaN are skipped for training and get `fill`.
    y             one label per bar describing what happens AFTER that bar, e.g. the
                  next day's return. y[i] only becomes known `horizon` bars later.
    fit_predict   fit_predict(X_train, y_train, X_new) -> one prediction per row of X_new.
                  Gets NumPy arrays when X is one, else lists.

    Retrains every `retrain_every` bars, starting once `min_train` labelled rows
    exist. A model trained at bar t only sees rows j with j + horizon + embargo <= t
    (labels already known at t's close, plus `embargo` bars of gap so training never
    ends right next to what it predicts), the most recent `window` of them if set,
    and predicts bars t .. t + retrain_every - 1. Returns one value per bar.
    """
    n = len(X)
    if len(y) != n:
        raise ValueError("walk_forward: X has %d rows but y has %d" % (n, len(y)))
    retrain_every, horizon = max(1, int(retrain_every)), max(1, int(horizon)) + max(0, int(embargo))
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


def _iso_stamp(raw):
    """A bar's time stamp: 'YYYY-MM-DD' for daily data, 'YYYY-MM-DDTHH:MM' (':SS' when the
    seconds aren't 0) for intraday bars, or None. Accepts what _iso_date does plus a time
    after a space or 'T' (24-hour or AM/PM; a time zone suffix is ignored), YYYYMMDD, and
    Unix seconds or milliseconds. Midnight counts as a plain date."""
    raw = str(raw).strip().strip('"')
    if raw.isdigit() and len(raw) in (9, 10, 12, 13):
        secs = int(raw) / (1000.0 if len(raw) >= 12 else 1.0)
        t = _dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=secs)
        return _stamp(t.date(), t.hour, t.minute, t.second)
    if raw.isdigit() and len(raw) == 8:
        return _iso_date("%s-%s-%s" % (raw[:4], raw[4:6], raw[6:]))
    sep = raw.find("T") if "T" in raw[8:11] else raw.find(" ")
    day = _iso_date(raw if sep < 0 else raw[:sep])
    if day is None or sep < 0:
        return day
    clock = raw[sep + 1:].strip().upper()
    pm, am = clock.endswith("PM"), clock.endswith("AM")
    clock = clock[:-2].strip() if pm or am else clock
    for z in ("Z", "+", "-"):
        k = clock.find(z, 1)
        if k > 0:
            clock = clock[:k]
    parts = clock.split(":")
    try:
        h, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        sec = int(float(parts[2])) if len(parts) > 2 and parts[2] else 0
    except ValueError:
        return day
    if pm and h < 12:
        h += 12
    if am and h == 12:
        h = 0
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= sec < 60):
        return None
    return _stamp(_dt.date.fromisoformat(day), h, m, sec)


def _stamp(day, h, m, sec):
    if h == m == sec == 0:
        return day.isoformat()
    return "%sT%02d:%02d%s" % (day.isoformat(), h, m, ":%02d" % sec if sec else "")


def _iso_date(raw):
    """'MM/DD/YYYY', 'YYYY-MM-DD' (optionally with a time) -> 'YYYY-MM-DD', else None."""
    raw = str(raw).strip().strip('"')
    try:
        if "/" in raw:
            m, d, y = raw.split("/")[:3]
            return _dt.date(int(y[:4]), int(m), int(d)).isoformat()
        if raw[:10].count(".") == 2:                  # DD.MM.YYYY (European)
            d, m, y = raw[:10].split(".")
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
            day = _iso_stamp(d) if not isinstance(d, _dt.date) else d.isoformat()
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
        self.days = [_dt.date.fromisoformat(d[:10]) for d in self.dates]
        self.n = len(self.dates)
        self.index = {d: i for i, d in enumerate(self.dates)}
        self.dividends = [0.0] * self.n
        self.dividend_source = None      # None (price return only), 'adj_close' or 'events'
        self.price_basis = "price"       # 'price' or 'total_return' (only an adjusted close was given)
        self.calendar = None             # 'NYSE', 'weekdays' or '24/7'; None = judged from the dates
        self.fundamentals = {}           # field -> ([dates the value became known], [values]), point in time
        self.currency = None             # e.g. 'EUR' for prices in euros; None = the base currency
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

    def set_fundamentals(self, table):
        """Point-in-time fundamentals: {field: [(date known, value), ...]}. A value counts
        from the first bar AFTER the date it became known (filings often come after the
        close), so a strategy never trades on it the same day. Replaces earlier ones."""
        out = {}
        for field, pts in (table or {}).items():
            clean = sorted((d, float(v)) for d, v in ((_iso_date(d), _num(v)) for d, v in pts) if d is not None and v is not None)
            if clean:
                out[str(field)] = ([d for d, _ in clean], [v for _, v in clean])
        self.fundamentals = out
        self._cache.clear()
        return {f: len(v[0]) for f, v in out.items()}

    def fundamental_series(self, field):
        """One value per bar: the latest `field` known before that bar's day (NaN before the first)."""
        if field not in self.fundamentals:
            raise KeyError("No fundamental %r for %s; loaded: %s" % (field, self.ticker, ", ".join(sorted(self.fundamentals)) or "none"))
        def build():
            dates, vals = self.fundamentals[field]
            out, j, cur = [], 0, NAN
            for d in self.dates:
                day = d[:10]
                while j < len(dates) and dates[j] < day:
                    cur = vals[j]
                    j += 1
                out.append(cur)
            return out
        return self.cached(("fundamental", field), build)

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
        """Short hash of the exact prices and dividends, to tell which data a run used."""
        def build():
            import hashlib
            import struct
            h = hashlib.sha256("\n".join(self.dates).encode())
            for col in (self.open, self.high, self.low, self.close, self.volume, self.dividends):
                h.update(struct.pack("<%dd" % len(col), *col))
            if self.kind == "future":                       # changes the accounting, so a different data fingerprint
                h.update(b"\0future")
            for field in sorted(self.fundamentals):          # only when there are any: older hashes stay the same
                dates, vals = self.fundamentals[field]
                h.update(("\0%s\0%s" % (field, "\n".join(dates))).encode())
                h.update(struct.pack("<%dd" % len(vals), *vals))
            return h.hexdigest()[:16]
        return self.cached(("fingerprint",), build)

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
        self._ds = ds

    def __len__(self):
        return self.n

    def fundamental(self, field):
        """Point-in-time values of a fundamental, one per bar (see DataSet.set_fundamentals)."""
        return self._ds.fundamental_series(field)[:self.n]

    @property
    def fundamental_fields(self):
        return sorted(self._ds.fundamentals)


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
    "date": ("date", "observation_date", "ex/eff date", "ex-date", "ex_date", "ex date", "ex-dividend date", "ex_dividend_date",
             "timestamp", "time"),
    "close": ("close/last", "close", "price", "last"),
    "adj_close": ("adj close", "adj. close", "adj_close", "adjusted close", "adjclose"),
    "open": ("open",), "high": ("high",), "low": ("low",), "volume": ("volume", "vol."),
    "dividend": ("dividends", "dividend", "cash amount", "amount", "dividend amount"),
    "split": ("stock splits", "split", "splits", "split ratio", "ratio"),
    "clock": ("time",),   # a time-of-day column next to a separate date column (intraday files)
    # fundamentals: when a value became known, the period it covers, and whose it is
    "available": ("available", "available date", "filing date", "filed", "filing_date", "date filed", "report date",
                  "reported", "announcement date", "acceptance date", "accepted", "knowledge date"),
    "period": ("period end", "period_end", "period ending", "fiscal period end", "fiscal date", "fiscal_date",
               "quarter end", "fiscal quarter end"),
    "ticker": ("ticker", "symbol"),
}


def csv_kind(header):
    """What a CSV holds: 'prices', 'dividends', 'splits' or 'rates' (a single value column, e.g. FRED)."""
    def has(key):
        return any(n in header for n in CSV_COLUMNS[key])
    if has("available") or has("period"):
        return "fundamentals"
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


# Generic tables: any delimiter, decimal commas, other column names (mapped to ours).
TABLE_ALIASES = {
    "date": ("date", "datetime", "timestamp", "time", "dt", "day", "trade date", "trading date", "date/time", "observation_date"),
    "time": ("time",),
    "open": ("open", "o", "open price", "opening price", "first"),
    "high": ("high", "h", "high price", "max"),
    "low": ("low", "l", "low price", "min"),
    "close": ("close", "c", "close/last", "close price", "closing price", "last", "last price", "price", "settle", "settlement"),
    "adj_close": ("adj close", "adj. close", "adj_close", "adjusted close", "adjclose"),
    "volume": ("volume", "v", "vol", "vol.", "volume traded", "qty", "quantity"),
}
TABLE_ORDER = ("date", "time", "open", "high", "low", "close", "adj_close", "volume")
TABLE_HEADERS = {"date": "Date", "time": "Time", "open": "Open", "high": "High", "low": "Low", "close": "Close",
                 "adj_close": "Adj Close", "volume": "Volume"}


def _split_line(line, sep):
    out, cur, q, i = [], "", False, 0
    while i < len(line):
        ch = line[i]
        if ch == '"':
            if q and i + 1 < len(line) and line[i + 1] == '"':
                cur += '"'
                i += 1
            else:
                q = not q
        elif ch == sep and not q:
            out.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    out.append(cur)
    return [c.strip() for c in out]


def sniff_delimiter(line):
    """The most frequent of , ; tab | outside quotes (',' on a tie)."""
    counts, q = {",": 0, ";": 0, "\t": 0, "|": 0}, False
    for ch in line:
        if ch == '"':
            q = not q
        elif not q and ch in counts:
            counts[ch] += 1
    best = max(counts.values())
    return "," if best == 0 or counts[","] == best else max(counts, key=lambda k: (counts[k], k == ";"))


def _is_decimal_comma(cells):
    import re
    hits = sum(1 for c in cells if re.fullmatch(r"-?\d{1,3}(\.\d{3})*,\d+|-?\d+,\d+", c.strip()))
    dots = sum(1 for c in cells if re.fullmatch(r"-?\d+\.\d+", c.strip()))
    return hits > 0 and hits >= dots


def guess_table_columns(header):
    """{our column: index} for the columns a header's names match (case-insensitive)."""
    low = [h.strip().strip('"').lower() for h in header]
    out = {}
    for key in TABLE_ORDER:
        for alias in TABLE_ALIASES[key]:
            if alias in low and low.index(alias) not in out.values():
                if key == "time" and out.get("date") == low.index(alias):
                    continue
                out[key] = low.index(alias)
                break
    return out


def normalize_table(text, mapping=None, delimiter=None, decimal_comma=None):
    """Any delimited price table -> a standard comma CSV with Date, (Time,) Open, High, Low,
    Close, (Adj Close,) Volume, which parse_csv_text reads. `mapping` is {our column: header
    name or index}; without it, column names are matched against TABLE_ALIASES. The
    delimiter (, ; tab |) and a decimal comma (1.234,56) are detected unless given."""
    lines = [ln for ln in text.lstrip("﻿").replace("\r", "").split("\n") if ln.strip()]
    if len(lines) < 2:
        raise ValueError("The file has no data rows.")
    sep = delimiter or sniff_delimiter(lines[0])
    header = _split_line(lines[0], sep)
    rows = [_split_line(ln, sep) for ln in lines[1:]]
    if mapping:
        low = [h.lower() for h in header]
        cols = {}
        for k, v in mapping.items():
            if v is None or v == "":
                continue
            cols[k] = int(v) if isinstance(v, int) or str(v).isdigit() else low.index(str(v).lower())
    else:
        cols = guess_table_columns(header)
    if "date" not in cols or ("close" not in cols and "adj_close" not in cols):
        raise ValueError("Couldn't tell which columns hold the date and the close: %s." % ", ".join(header))
    if decimal_comma is None:
        sample = [r[cols[k]] for r in rows[:200] for k in ("open", "high", "low", "close") if k in cols and cols[k] < len(r)]
        decimal_comma = sep != "," and _is_decimal_comma(sample)
    keys = [k for k in TABLE_ORDER if k in cols]

    def num(cell):
        c = cell.strip()
        if decimal_comma:
            c = c.replace(".", "").replace(" ", "").replace(",", ".")
        return c

    def q(cell):
        return '"%s"' % cell.replace('"', '""') if ("," in cell or '"' in cell) else cell

    out = [",".join(TABLE_HEADERS[k] for k in keys)]
    for r in rows:
        cells = []
        for k in keys:
            v = r[cols[k]] if cols[k] < len(r) else ""
            cells.append(q(v if k in ("date", "time") else num(v)))
        out.append(",".join(cells))
    return "\n".join(out) + "\n"


def parquet_to_csv(data):
    """A Parquet file's bytes -> CSV text (needs pandas and fastparquet or pyarrow). Date
    and datetime columns are written as ISO text; a named index becomes a column."""
    import pandas as pd
    df = pd.read_parquet(io.BytesIO(data))
    if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime"):
            ser = df[c]
            if getattr(ser.dt, "tz", None) is not None:
                ser = ser.dt.tz_localize(None)
            midnight = bool(((ser.dt.hour == 0) & (ser.dt.minute == 0) & (ser.dt.second == 0)).all())
            df[c] = ser.dt.strftime("%Y-%m-%d" if midnight else "%Y-%m-%dT%H:%M:%S")
    return df.to_csv(index=False)


def load_table(path, ticker=None, mapping=None, delimiter=None, decimal_comma=None):
    """Load a price file of any common layout: CSV/TSV/TXT with any delimiter and column
    names, or Parquet. Returns a DataSet."""
    import os
    ticker = ticker or os.path.splitext(os.path.basename(path))[0].upper()
    if path.lower().endswith((".parquet", ".pq")):
        with open(path, "rb") as f:
            text = parquet_to_csv(f.read())
    else:
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()
    return parse_csv_text(normalize_table(text, mapping, delimiter, decimal_comma), ticker)


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
    clock = _col(header, "clock")
    if clock == idx["date"]:
        clock = None
    for row in rows:
        d = row[idx["date"]] if idx["date"] < len(row) else ""
        if clock is not None and clock < len(row) and row[clock].strip():
            d = d.strip() + " " + row[clock].strip()
        cols["date"].append(d)
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
    ti = header.index("type") if kind == "dividends" and "type" in header else None   # Nasdaq: Cash, or stock & other kinds
    out = []
    for row in rows:
        if max(di, vi) >= len(row):
            continue
        if ti is not None and ti < len(row) and row[ti].strip().lower() not in ("cash", ""):
            continue
        day = _iso_date(row[di])
        val = _num(row[vi]) if kind == "dividends" else _split_ratio(row[vi])
        if day and val:
            out.append((day, val))
    return kind, sorted(out)


FUNDAMENTAL_LAG_DAYS = 45   # assumed delay from period end to publication when a file has no filing date


def parse_fundamentals_csv(text, lag_days=FUNDAMENTAL_LAG_DAYS):
    """A point-in-time fundamentals file: when each value became known (a filing,
    report or availability date) and one column per field (eps, revenue, book_value ...),
    optionally a Ticker column for several instruments in one file. Without a known
    date, the period end + lag_days is used, and a warning says so: values are rarely
    public on the last day of the quarter. Returns {'tickers': {TICKER or '': {field:
    [(date, value)]}}, 'fields', 'warnings'}."""
    header, rows = read_csv_table(text)
    ia, ip, it = _col(header, "available"), _col(header, "period"), _col(header, "ticker")
    if ia is None and ip is None:
        raise ValueError("A fundamentals file needs a filing/available date or a period end column.")
    skip = {i for i in (ia, ip, it, _col(header, "date")) if i is not None}
    fields = []
    for i, name in enumerate(header):
        if i in skip or not name:
            continue
        vals = [r[i] for r in rows if i < len(r) and r[i].strip()]
        nums = [v for v in vals if _num(v) is not None]
        if vals and len(nums) >= 0.5 * len(vals):
            fields.append((i, "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or "field%d" % i))
    if not fields:
        raise ValueError("No numeric columns (eps, revenue ...) found next to the dates.")
    lagged = 0
    out = {}
    for r in rows:
        when = _iso_date(r[ia]) if ia is not None and ia < len(r) and r[ia].strip() else None
        if when is None and ip is not None and ip < len(r):
            pe = _iso_date(r[ip])
            if pe:
                when = (_dt.date.fromisoformat(pe) + _dt.timedelta(days=int(lag_days))).isoformat()
                lagged += 1
        if when is None:
            continue
        tk = r[it].strip().upper() if it is not None and it < len(r) else ""
        tab = out.setdefault(tk, {})
        for i, name in fields:
            v = _num(r[i]) if i < len(r) else None
            if v is not None:
                tab.setdefault(name, []).append((when, v))
    warnings = []
    if lagged:
        warnings.append("%d row%s had no filing date, so each value counts from %d days after its period end. "
                        "Add a filing date column for exact point-in-time data." % (lagged, "" if lagged == 1 else "s", int(lag_days)))
    return {"tickers": {k: {f: sorted(v) for f, v in t.items()} for k, t in out.items()},
            "fields": [n for _, n in fields], "warnings": warnings}


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
    d, last = _dt.date.fromisoformat(start[:10]), _dt.date.fromisoformat(end[:10])
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
    """The calendar an instrument trades on, judged from its dates: '24/7' when a tenth or
    more of its bars fall on weekends (crypto), 'weekdays' when it trades on most NYSE
    holidays (FX, many futures), else 'NYSE'. An explicit ds.calendar wins."""
    if getattr(ds, "calendar", None):
        return ds.calendar
    return ds.cached(("guessed_calendar",), lambda: _guess_calendar(ds))


def _guess_calendar(ds):
    days = sorted(set(ds.days))
    weekend = sum(1 for d in days if d.weekday() >= 5)
    if weekend > 0.1 * len(days):
        return "24/7"
    if days:
        hol = set()
        for y in range(days[0].year, days[-1].year + 1):
            hol |= nyse_holidays(y)
        lo, hi = days[0].isoformat(), days[-1].isoformat()
        hol = {h for h in hol if lo <= h <= hi and _dt.date.fromisoformat(h).weekday() < 5}
        traded = sum(1 for d in days if d.isoformat() in hol)
        if hol and traded >= 0.5 * len(hol) and len(hol) >= 4:
            return "weekdays"
    return "NYSE"


def bars_per_day(ds):
    """1 for daily data; the median number of bars per date for intraday data."""
    def build():
        counts = {}
        for d in ds.dates:
            counts[d[:10]] = counts.get(d[:10], 0) + 1
        return max(1, int(round(percentile(list(counts.values()), 50)))) if counts else 1
    return ds.cached(("bars_per_day",), build)


def periods_per_year(ds):
    """Bars per year, to annualize this instrument's statistics: trading days x bars per
    day. Days are 365 for data that trades every day (crypto), the chosen calendar's when
    ds.calendar is set ('weekdays' = 260), and otherwise the usual 252."""
    cal = getattr(ds, "calendar", None) or ("24/7" if guess_calendar(ds) == "24/7" else "NYSE")
    return CALENDARS[cal][1] * bars_per_day(ds)


def panel_ann(panel):
    """Bars per year of a backtest's shared calendar: that of its busiest instrument."""
    if "ann" not in panel._aligned:
        panel._aligned["ann"] = max(periods_per_year(ds) for ds in panel.sets)
    return panel._aligned["ann"]


def res_ann(res):
    """Periods per year of a Result (252 for daily US equities)."""
    a = getattr(res, "ann", None)
    if a is None and getattr(res, "panel", None) is not None:
        a = panel_ann(res.panel)
    return a or TRADING_DAYS


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
    have = {d[:10] for d in ds.dates}
    per = "bar" if bars_per_day(ds) > 1 else "day"
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
    for d in sorted({d[:10] for d in ds.dates}):
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
            add("jump", ds.dates[t + 1], "%+.1f%% in one %s%s; check for an unadjusted split or a bad print" % (
                100 * (math.exp(r) - 1), per, "" if isnan(sd) else " (%.0f sigma)" % (abs(r) / sd)))
    titles = {
        "dropped": ("Rows skipped", "error"), "ohlc": ("High/low repaired", "warn"),
        "missing_ohlc": ("Open/high/low missing", "info"), "duplicate": ("Duplicate dates", "warn"),
        "missing_days": ("Missing sessions", "warn"), "off_calendar": ("Bars on closed days", "info"),
        "stale": ("Stale prices", "warn"), "jump": ("Large one-day moves", "warn"),
        "adj_split": ("Splits in the adjusted close", "info"), "big_dividend": ("Unusually large dividends", "warn"),
    }
    for k, rec in found.items():
        rec["title"], rec["severity"] = titles.get(k, (k, "info"))
    return {"calendar": calendar, "issues": found, "bars": ds.n, "bars_per_day": bars_per_day(ds),
            "periods_per_year": periods_per_year(ds),
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
        self.days = [_dt.date.fromisoformat(d[:10]) for d in self.dates]
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

    def impact_inputs(self, np):
        """20-day average volume and daily volatility before each bar, per instrument, as
        (calendar bars x instruments) arrays: what the square-root impact model needs."""
        if "impact" not in self._aligned:
            adv = np.full((self.n, self.k), np.nan)
            sig = np.full((self.n, self.k), np.nan)
            for a, ds in enumerate(self.sets):
                av = ds.cached(("adv", 20), lambda: _trailing_mean(ds.volume, 20))
                sv = ds.cached(("sigma", 20), lambda: _trailing_vol(ds.close, 20))
                for t, j in enumerate(self.nat[a]):
                    if j >= 0:
                        adv[t, a], sig[t, a] = av[j], sv[j]
            self._aligned["impact"] = (adv, sig)
        return self._aligned["impact"]

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
        self._panel, self._a = panel, a

    def __len__(self):
        return self.n

    def fundamental(self, field):
        """Point-in-time values of a fundamental on the shared calendar (NaN before the first)."""
        ser = self._panel.sets[self._a].fundamental_series(field)
        nat = self._panel.nat[self._a]
        return [ser[j] if j >= 0 else NAN for j in nat[:self.n]]

    @property
    def fundamental_fields(self):
        return sorted(self._panel.sets[self._a].fundamentals)


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
        self.options = OptionsView(run, a)
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

    def fundamental(self, field, default=NAN):
        """The latest value of a fundamental (e.g. 'eps') known before today: point in time,
        from a fundamentals file on the Data tab. `default` until the first one is known."""
        if self._j < 0:
            return default
        v = self._ds.fundamental_series(field)[self._j]
        return default if isnan(v) else v

    @property
    def fundamentals(self):
        """{field: latest value known before today} for every loaded fundamental."""
        return {f: self.fundamental(f) for f in sorted(self._ds.fundamentals)}

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
        """Buy (qty > 0) or sell (qty < 0) shares at the next bar.

        Options: limit=price, stop=price (both = stop-limit), tif='day' | 'gtc',
        at='open' | 'close' (market on open / on close), oco='name' (filling one
        order cancels the others with the same name). Returns the id of a resting order."""
        return self._run.add_order(self._a, "delta", float(qty), kw)

    def order_target_shares(self, n, **kw):
        """Trade whatever is needed to hold exactly n shares after the fill. Same options
        as order(); a target is sized when it fills, so exit orders never overshoot."""
        return self._run.add_order(self._a, "target", float(n), kw)

    @property
    def open_orders(self):
        """Resting limit/stop orders for this instrument, as dicts."""
        return self._run.open_orders(self._a)

    def cancel_all(self):
        """Cancel this instrument's resting orders. Returns how many were cancelled."""
        return self._run.cancel(a=self._a)

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
    def time(self):
        """The bar's time of day as 'HH:MM' (or 'HH:MM:SS') for intraday data, None for daily bars."""
        d = self._panel.dates[self._run.t]
        return d[11:] if len(d) > 10 else None

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

    def cancel(self, order_id):
        """Cancel one resting order by the id order() returned."""
        return self._run.cancel(oid=order_id)

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

    @property
    def options(self):
        raise AttributeError("in a portfolio strategy, use ctx['SPY'].options for the options on one instrument")

    def __init__(self, panel, run, params, start):
        self._panel = panel
        self._run = run
        self.params = params
        self.state = {}
        self.start_index = start
        self.tickers = list(panel.tickers)
        self._assets = {tk: AssetContext(panel, a, run) for a, tk in enumerate(panel.tickers)}
        self._all_synced = -1

    def _sync(self, t):
        pass                    # instruments catch up to today when the strategy looks at them

    @property
    def assets(self):
        t = self._run.t
        if self._all_synced != t:
            for a in self._assets.values():
                a._sync(t)
            self._all_synced = t
        return self._assets

    def __getitem__(self, ticker):
        try:
            a = self._assets[ticker]
        except KeyError:
            raise KeyError("%r is not in this portfolio (%s)" % (ticker, ", ".join(self.tickers)))
        a._sync(self._run.t)
        return a

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
        return self[ticker].order(qty, **kw)

    def order_target_shares(self, ticker, n, **kw):
        return self[ticker].order_target_shares(n, **kw)

    @property
    def open_orders(self):
        return self._run.open_orders()

    def cancel_all(self, ticker=None):
        return self._run.cancel(a=None if ticker is None else self.tickers.index(ticker))


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
        "purged_kfold": purged_kfold, "cpcv_splits": cpcv_splits, "cv_score": cv_score,
    }


def _register_source(code):
    lines = code.splitlines(True)
    linecache.cache[STRATEGY_FILENAME] = (len(code), None, lines, STRATEGY_FILENAME)


def format_error(exc, bar_date=None):
    """Turn an exception into {message, trace, line}, keeping only strategy frames."""
    if isinstance(exc, StrategyError):
        return exc.info
    if isinstance(exc, SyntaxError) and (exc.filename == STRATEGY_FILENAME or _module_of(exc.filename)):
        text = (exc.text or "").rstrip()
        mod = _module_of(exc.filename)
        return {"message": "SyntaxError: %s (%sline %s)" % (exc.msg, "%s.py, " % mod if mod else "", exc.lineno),
                "trace": "  line %s\n    %s" % (exc.lineno, text.strip()),
                "line": exc.lineno, "file": mod, "user": True}
    frames = traceback.extract_tb(exc.__traceback__)
    user = [f for f in frames if f.filename == STRATEGY_FILENAME or _module_of(f.filename)]
    msg = "%s: %s" % (type(exc).__name__, exc)
    if isinstance(exc, ModuleNotFoundError) and sys.platform == "emscripten":
        msg += (". In the browser, strategies can import NumPy, pandas, SciPy and scikit-learn plus the "
                "standard library. PyTorch, TensorFlow and other packages without a WebAssembly build "
                "only work when you run engine.py in regular Python.")
    if bar_date:
        msg += "  (on bar %s)" % bar_date
    if user:
        where = lambda f: ("%s.py, " % _module_of(f.filename)) if f.filename != STRATEGY_FILENAME else ""
        trace = "\n".join("  %sline %d, in %s\n    %s" % (where(f), f.lineno, f.name, (f.line or "").strip())
                          for f in user)
        return {"message": msg, "trace": trace, "line": user[-1].lineno,
                "file": _module_of(user[-1].filename), "user": True}
    return {"message": msg, "trace": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "line": None, "user": False}


def _user_error(message):
    return StrategyError({"message": message, "trace": "", "line": None, "user": True})


# User modules: Python files from the module library that strategies import like any
# package (import my_signals / from my_signals import zscore). Every strategy compile
# imports them afresh, so module-level state never carries from one run into the next.
MODULES = {}             # name -> source
_used_modules = []       # user modules imported since the last strategy compile, in order
_loaded_modules = set()  # user modules that may be sitting in sys.modules
_RESERVED = {"engine", "templates", "strategy", "qt", "numpy", "pandas", "scipy", "sklearn",
             "torch", "matplotlib", "micropip", "pyodide", "js", "pyflakes"}


def module_filename(name):
    return "<%s.py>" % name


def _module_of(filename):
    """'my_signals' for '<my_signals.py>'; None for the strategy or anything else."""
    if filename and filename.startswith("<") and filename.endswith(".py>"):
        name = filename[1:-4]
        return name if name in MODULES else None
    return None


def module_name_problem(name):
    """Why name can't be a user module, or None."""
    import keyword
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        return "%r isn't a valid module name: use letters, digits and _ and start with a letter, e.g. my_signals." % (name,)
    if name in _RESERVED or name in getattr(sys, "stdlib_module_names", ()) or name in sys.builtin_module_names:
        return "A module can't be called %s: Python or QuantLab already uses that name. Try %s_lib." % (name, name)
    return None


class _ModuleFinder:
    """Import hook for MODULES. It sits last on sys.meta_path, so it never hides a real package."""
    quantlab = True

    def find_spec(self, name, path=None, target=None):
        if path is None and name in MODULES:
            import importlib.util
            return importlib.util.spec_from_loader(name, self, origin=module_filename(name))
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__
        src, fn = MODULES[name], module_filename(name)
        linecache.cache[fn] = (len(src), None, src.splitlines(True), fn)
        for k, v in _namespace().items():     # the same helpers a strategy gets: qt, isnan, math ...
            if k != "__name__":
                module.__dict__.setdefault(k, v)
        module.__file__ = fn
        _loaded_modules.add(name)
        if name not in _used_modules:
            _used_modules.append(name)
        exec(compile(src, fn, "exec"), module.__dict__)


sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, "quantlab", False)] + [_ModuleFinder()]


def _forget_modules():
    """Drop imported user modules, so the next import runs their current code."""
    global _used_modules
    for name in _loaded_modules:
        sys.modules.pop(name, None)
    _loaded_modules.clear()
    _used_modules = []


def set_modules(mods):
    """Replace the module library with {name: source}."""
    mods = dict(mods or {})
    for name, src in mods.items():
        problem = module_name_problem(name)
        if problem:
            raise _user_error(problem)
        if not isinstance(src, str):
            raise _user_error("Module %s must be text." % name)
    MODULES.clear()
    MODULES.update(mods)
    _forget_modules()


def module_hashes(names):
    import hashlib
    return {n: hashlib.sha256(MODULES[n].encode("utf-8")).hexdigest() for n in names if n in MODULES}


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
        self.modules = []   # user modules the code imported (set by from_code)

    @property
    def mode(self):
        return "on_bar" if self.on_bar is not None else "signals"

    @classmethod
    def from_code(cls, code, params_override=None):
        _register_source(code)
        _forget_modules()
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
        strat = cls(on_bar if callable(on_bar) else None,
                    signals if callable(signals) and not callable(on_bar) else None,
                    init if callable(init) else None, params, universe=universe)
        strat.modules = _used_modules    # grows if the strategy imports a module later, inside on_bar
        return strat


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
        # execution realism
        "spread_bps": 0.0,        # quoted bid-ask spread; market fills pay half of it
        "impact_coef": 0.0,       # square-root impact: price moves impact_coef x daily vol x sqrt(qty / ADV)
        "max_participation": 0.0,  # most of a bar's volume one instrument may trade (0.1 = 10%; 0 = no limit)
        "fill_model": "standard",  # 'pessimistic': limits must trade through; stops hit before targets
        # sizing & limits, applied to target weights at the close they are decided
        "sizing": "none",         # 'vol_target': scale to vol_target; 'kelly': to kelly_fraction x the Kelly leverage
        "vol_target": 0.10,       # annual volatility aimed at with sizing='vol_target'
        "kelly_fraction": 0.5,    # share of the estimated Kelly leverage used with sizing='kelly'
        "sizing_lookback": 63,    # trading days of returns used to estimate volatility (and the mean, for Kelly)
        "max_weight": 0.0,        # largest |weight| in one instrument (0 = no cap)
        "max_gross": 0.0,         # largest sum of |weights| (0 = only max_leverage)
        "max_net": 0.0,           # largest |sum of weights| (0 = no limit)
        "dd_limit": 0.0,          # drawdown from the equity high that triggers de-risking (0.15 = 15%; 0 = off)
        "dd_scale": 0.5,          # weights are multiplied by this while de-risked
        "dd_resume": 0.5,         # full size again once the drawdown is back within dd_resume x dd_limit
        # data
        "delisting_return": 0.0,  # return on the last close when an instrument's data ends early (-0.3 = -30%)
        "base_currency": "USD",   # the account's currency; instruments priced in others are converted (unhedged)
        # options traded from a loaded option chain (ctx.options)
        "option_multiplier": 100.0,  # shares per contract
        "option_commission": 0.65,   # dollars per contract, each fill
        "option_fill": 1.0,          # share of the half-spread paid: 1 = buy at the ask, sell at the bid; 0 = the mid
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
        if self.fill_model not in ("standard", "pessimistic"):
            raise ValueError("fill_model must be 'standard' or 'pessimistic'")
        if self.spread_bps < 0 or self.impact_coef < 0 or not 0 <= self.max_participation <= 1:
            raise ValueError("spread, impact and participation must be non-negative (participation at most 1)")
        if self.sizing not in ("none", "vol_target", "kelly"):
            raise ValueError("sizing must be 'none', 'vol_target' or 'kelly'")
        if self.vol_target <= 0 or self.kelly_fraction <= 0 or self.sizing_lookback < 10:
            raise ValueError("vol_target and kelly_fraction must be positive and sizing_lookback at least 10 days")
        if min(self.max_weight, self.max_gross, self.max_net) < 0 or not 0 <= self.dd_limit < 1:
            raise ValueError("position, gross and net limits must be non-negative and dd_limit from 0 to 1")
        if not 0 <= self.dd_scale <= 1 or not 0 <= self.dd_resume <= 1:
            raise ValueError("dd_scale and dd_resume must be from 0 to 1")
        self.base_currency = self.base_currency.upper()
        if len(self.base_currency) != 3 or not self.base_currency.isalpha():
            raise ValueError("base_currency must be a 3-letter code such as USD")
        if not -1 <= self.delisting_return <= 1:
            raise ValueError("delisting_return must be from -1 to 1 (-0.3 = -30%)")

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
        self._futures = {a for a, ds in enumerate(panel.sets) if getattr(ds, "kind", "price") == "future"}
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
        self._eq = None                      # equity at those prices (None = recompute)
        self._closes = [panel.aligned(a, "close") for a in range(k)]   # = panel.mark(a, t)
        self._divs = {}                      # calendar bar -> [(asset, dividend per share)]
        index = {d: t for t, d in enumerate(panel.dates)}
        for a, ds in enumerate(panel.sets):
            for j, d in enumerate(ds.dividends):
                if d:
                    self._divs.setdefault(index[ds.dates[j]], []).append((a, d))
        self.spread_paid = 0.0
        self.impact_paid = 0.0
        self.partial_fills = 0               # market orders cut by the participation limit
        self._t = 0
        self._vol_left = None                # shares each instrument may still trade this bar
        self.arrival_t = None                # bar whose close is the arrival price of current fills
        self.opt_value = 0.0                 # marked value of option positions (OptionsDesk)

    def begin_bar(self, t):
        """Start a bar: the participation allowance is worked out when the first order needs it."""
        self._t = t
        self._vol_left = None

    def _allowance(self):
        cap, p, t = self.cfg.max_participation, self.panel, self._t
        self._vol_left = [cap * p.sets[a].volume[p.nat[a][t]] if p.live[a][t] and p.sets[a].volume[p.nat[a][t]] > 0
                          else math.inf for a in range(self.k)]

    def impact(self, a, qty):
        """Square-root market impact as a fraction of price for trading qty shares now."""
        cfg = self.cfg
        if not cfg.impact_coef or qty <= 0:
            return 0.0
        p = self.panel
        ds, j = p.sets[a], p.nat[a][self._t]
        adv = ds.cached(("adv", 20), lambda: _trailing_mean(ds.volume, 20))[j]
        sig = ds.cached(("sigma", 20), lambda: _trailing_vol(ds.close, 20))[j]
        if isnan(adv) or isnan(sig) or adv <= 0:
            return 0.0
        return cfg.impact_coef * sig * math.sqrt(qty / adv)

    def _unit(self, a, p, qty):
        """Estimated all-in price per share to buy qty shares (for sizing and the cash check)."""
        cfg = self.cfg
        return p * (1 + cfg.slippage_bps / 1e4 + cfg.spread_bps / 2e4 + self.impact(a, qty)) * (1 + cfg.commission_bps / 1e4)

    # ---- valuation ----------------------------------------------------------
    def equity_marked(self, t):
        eq = self.cash + self.opt_value
        for sh, cl in zip(self.shares, self._closes):
            if sh:
                eq += sh * cl[t]
        return eq

    def _equity_val(self):
        """Equity at the current valuation prices (kept up to date fill by fill)."""
        if self._eq is None:
            eq = self.cash + self.opt_value
            for a, sh in enumerate(self.shares):
                if sh:
                    eq += sh * self._val[a]
            self._eq = eq
        return self._eq

    def _set_val(self, px, t):
        for a in range(self.k):
            self._val[a] = px[a] if px[a] is not None else self.panel.mark(a, t)
        self._eq = None

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
        # Futures take no cash: the cash "spent" on them is still collateral and earns interest.
        bal = self.cash + sum(self.shares[a] * p.mark(a, t_prev) for a in self._futures if self.shares[a])
        if bal > 0:
            x = bal * rate * days / 360.0
            self.carry["interest_earned"] += x
        else:
            x = bal * (rate + cfg.margin_spread) * days / 360.0
            self.carry["interest_paid"] -= x
        fee_total = 0.0
        for a, sh in enumerate(self.shares if min(self.shares) < 0 else ()):
            if sh < 0 and a not in self._futures:    # shorting a future borrows nothing
                fee = cfg.borrow_fees.get(p.tickers[a], cfg.borrow_fee)
                if fee:
                    f = -sh * p.mark(a, t_prev) * fee * days / 360.0
                    self.asset_cash[a] -= f
                    self.asset_costs[a] += f
                    fee_total += f
        self.carry["borrow_paid"] += fee_total
        self.cash += x - fee_total
        self._eq = None

    def pay_dividends(self, t):
        """Holders on the ex-date receive the dividend; short sellers pay it."""
        for a, d in self._divs.get(t, ()):
            sh = self.shares[a]
            if sh:
                amt = sh * d
                self.cash += amt
                self._eq = None
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
        np = _numpy() if len(w) >= VECTOR_MIN else None
        if np is not None:
            plan, later = self._plan_vectorized(np, w, px, t, eq)
            rest = self._execute_plan(plan, eq, t, report=False)
            return later, rest
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
                dq = max(0.0, (diff - fixed) / self._unit(a, p, diff / p))
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
        rest = self._execute_plan(plan, eq, t, report=False)
        return later, rest

    def _plan_vectorized(self, np, w, px, t, eq):
        """The sizing loop of rebalance() over all instruments at once. Every element is
        computed with the same operations in the same order, so the plan is bit-identical."""
        cfg, panel = self.cfg, self.panel
        later = {}
        idx, wv = [], []
        for a, wa in w.items():
            if px[a] is None:
                if panel.nat[a][t] >= 0 or wa == 0:
                    later[a] = wa
            else:
                idx.append(a)
                wv.append(wa)
        if not idx:
            return [], later
        ia = np.array(idx)
        wa = np.array(wv, dtype=float)
        p = np.array([px[a] for a in idx], dtype=float)
        sh = np.array([self.shares[a] for a in idx], dtype=float)
        slip, half, cb, fixed = cfg.slippage_bps / 1e4, cfg.spread_bps / 2e4, cfg.commission_bps / 1e4, cfg.commission_fixed
        diff = wa * eq - sh * p
        with np.errstate(divide="ignore", invalid="ignore"):
            qty = diff / p
            if cfg.impact_coef:
                adv, sig = panel.impact_inputs(np)
                av, sg = adv[t, ia], sig[t, ia]
                ok = (qty > 0) & ~np.isnan(av) & ~np.isnan(sg) & (av > 0)
                imp = np.where(ok, cfg.impact_coef * sg * np.sqrt(np.where(ok, qty / np.where(ok, av, 1.0), 0.0)), 0.0)
            else:
                imp = np.zeros(len(idx))
            unit = p * (1 + slip + half + imp) * (1 + cb)
            buy = np.maximum(0.0, (diff - fixed) / unit)
        dq = np.where(wa == 0, -sh, np.where(diff > 0, buy, qty))
        target = sh + dq
        keep = ~((sh != 0) & (target != 0) & ((sh > 0) == (target > 0)) & (np.abs(diff) < cfg.rebalance_band * eq))
        if not cfg.fractional:
            target = np.trunc(target)
        dq = target - sh
        if not cfg.allow_short:
            dq = np.where(sh + dq < 0, -sh, dq)
        plan = [[a, float(d), float(x), False] for a, d, x, k in zip(idx, dq.tolist(), p.tolist(), keep.tolist()) if k]
        return plan, later

    def share_orders(self, orders, px, t, maker=False, kind="market"):
        """Execute share orders {asset: ('delta' | 'target', value)}. Returns those that
        couldn't fill today (no bar, or over the participation limit) to retry at the next bar.
        maker=True for limit fills: they provide liquidity and pay commission only."""
        cfg = self.cfg
        self._set_val(px, t)
        eq = self._equity_val()
        if eq <= 0:
            return {}
        plan, later = [], {}
        for a, (how, v) in orders.items():
            p = px[a]
            if p is None:
                later[a] = (how, v)
                continue
            sh = self.shares[a]
            target = v if how == "target" else sh + v
            if not cfg.fractional:
                target = float(int(target))
            dq = target - sh
            clipped = False
            if not cfg.allow_short and sh + dq < 0:
                dq = -sh
                clipped = True
            plan.append([a, dq, p, clipped])
        later.update(self._execute_plan(plan, eq, t, report=True, maker=maker, kind=kind))
        return later

    def _volume_cap(self, a, dq, rest):
        """Trim dq to what is left of this bar's participation allowance; the rest is
        recorded as a target for the next bar (a multi-day fill)."""
        if self.cfg.max_participation <= 0:
            return dq
        if self._vol_left is None:
            self._allowance()
        if abs(dq) <= self._vol_left[a] + 1e-9:
            if self._vol_left is not None:
                self._vol_left[a] = max(0.0, self._vol_left[a] - abs(dq))
            return dq
        allowed = math.copysign(self._vol_left[a], dq)
        rest[a] = ("target", self.shares[a] + dq)
        self.partial_fills += 1
        self._vol_left[a] = 0.0
        return allowed

    def _execute_plan(self, plan, eq, t, report, maker=False, kind="market"):
        """Sells first (they raise cash), then buys, which share the cash and leverage room.
        Returns targets left over by the participation limit."""
        cfg = self.cfg
        L = cfg.max_leverage
        fixed = cfg.commission_fixed
        rest = {}
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
            dq = self._volume_cap(a, dq, rest)
            if not cfg.fractional:
                dq = float(int(dq))
            if abs(dq) >= 1e-9:
                self._fill(a, dq, p, t, maker=maker, kind=kind)
        buys = [item for item in plan if item[1] > 0]
        if not buys:
            for a, dq, p, clipped in plan:
                if dq == 0 and clipped and report:
                    self.clipped += 1
            return rest
        for item in buys:
            a, dq, p, _ = item
            sh = self.shares[a]
            if sh >= 0:
                cap = L * eq / p - sh  # keeps this position <= L x equity
                if dq > cap + 1e-9:
                    item[1] = max(0.0, cap)
                    item[3] = True
        unit = {a: p * (1 + cfg.commission_bps / 1e4) if maker else self._unit(a, p, dq) for a, dq, p, _ in buys}
        need = sum(dq * unit[a] for a, dq, _, _ in buys)
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
            dq = self._volume_cap(a, dq, rest)
            if not cfg.fractional:
                dq = float(int(dq))
            if abs(dq) >= 1e-9:
                self._fill(a, dq, p, t, maker=maker, kind=kind)
        return rest

    # ---- fills and trade bookkeeping ---------------------------------------
    def _fill(self, a, q, price, t, reason=None, maker=False, kind="market", free=False):
        cfg = self.cfg
        prev = self.shares[a]
        if maker or free:              # a resting limit order: no spread, slippage or impact
            slip = half = imp = 0.0
        else:
            slip, half, imp = cfg.slippage_bps / 1e4, cfg.spread_bps / 2e4, self.impact(a, abs(q))
        adj = slip + half + imp
        px = price * (1 + adj) if q > 0 else price * (1 - adj)
        comm = 0.0 if free else abs(q) * px * cfg.commission_bps / 1e4 + cfg.commission_fixed
        if self._val[a] != price:
            self._val[a] = price
            self._eq = None
        eq_before = self._equity_val()
        value_before = self.asset_value(a, price)
        self.cash -= q * px + comm
        self.asset_cash[a] -= q * px + comm
        new = prev + q
        if abs(new) < 1e-9:
            new = 0.0
        self.shares[a] = new
        cost = comm + abs(q) * price * adj
        self.commission += comm
        self.slippage += abs(q) * price * slip
        self.spread_paid += abs(q) * price * half
        self.impact_paid += abs(q) * price * imp
        self.asset_costs[a] += cost
        self.traded_notional += abs(q) * price
        self.asset_notional[a] += abs(q) * price
        fill = {"i": t, "date": self.panel.dates[t], "ticker": self.panel.tickers[a],
                "side": "buy" if q > 0 else "sell", "qty": q, "price": px, "commission": comm, "position": new,
                "kind": kind}
        if self.arrival_t is not None:
            arr = self.panel.mark(a, self.arrival_t)
            if arr == arr and arr > 0:
                fill["arrival"] = arr      # close when the order was decided: shortfall = (price - arrival) x side
        if reason:
            fill["reason"] = reason
        self.fills.append(fill)
        self._eq = eq_before + q * price - q * px - comm   # the fill's only effect on equity
        eq_after = self._eq
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

    def delist(self, a, t, ret=0.0):
        """Close instrument a's position at its last close x (1 + ret), with no costs: its
        data has ended (delisted, merged, or a file that isn't up to date)."""
        sh = self.shares[a]
        if not sh:
            return None
        px = self.panel.mark(a, t) * (1 + ret)
        self.arrival_t = t
        self._set_val([None] * self.k, t)
        self._val[a] = px
        self._eq = None
        self._fill(a, -sh, px, t, "delisted", kind="delisting", free=True)
        return px

    def liquidate(self, t, fraction=1.0, reason=None):
        """Sell `fraction` of every position at today's close (instruments with a bar today)."""
        p = self.panel
        self.arrival_t = t
        for a in range(self.k):
            sh = self.shares[a]
            if sh and (p.live[a][t] or fraction >= 1.0):
                self._set_val([None] * self.k, t)
                self._fill(a, -sh * fraction, p.mark(a, t), t, reason)


def _trailing_mean(x, n):
    """Mean of the n values BEFORE each bar (so a fill never uses its own bar's volume)."""
    out, s = [NAN] * len(x), 0.0
    for i, v in enumerate(x):
        if i >= n:
            out[i] = s / n
            s -= x[i - n]
        s += v
    return out


def _trailing_vol(close, n):
    """Daily standard deviation of the n returns before each bar."""
    r = [NAN] + [close[i] / close[i - 1] - 1 for i in range(1, len(close))]
    sd = rolling_std(r[1:], n, ddof=1)       # sd[k] covers returns k-n+1..k (r index +1)
    out = [NAN] * len(close)
    for i in range(n + 1, len(close)):
        out[i] = sd[i - 2]
    return out


ORDER_FIELDS = ("limit", "stop", "tif", "at", "oco")


class Order:
    """A resting limit, stop or stop-limit order (market orders don't rest)."""

    __slots__ = ("id", "a", "kind", "value", "limit", "stop", "tif", "oco", "created", "triggered", "status")

    def __init__(self, oid, a, kind, value, limit, stop, tif, oco, created):
        self.id, self.a, self.kind, self.value = oid, a, kind, value
        self.limit, self.stop, self.tif, self.oco = limit, stop, tif, oco
        self.created, self.triggered, self.status = created, False, "open"

    @property
    def type(self):
        return "stop_limit" if self.limit is not None and self.stop is not None else "limit" if self.limit is not None else "stop"

    def as_dict(self, panel):
        return {"id": self.id, "ticker": panel.tickers[self.a], "type": self.type, self.kind: self.value,
                "limit": self.limit, "stop": self.stop, "tif": self.tif, "oco": self.oco, "triggered": self.triggered,
                "created": panel.dates[self.created]}


def _path(o, h, l, c, first_low):
    return (o, l, h, c) if first_low else (o, h, l, c)


def _cross(path, level, up, strict=False, after=0.0):
    """First moment in [after, 3] when a price path (open -> ... -> close, linear in between)
    reaches level going up (>=) or down (<=); strict needs it to go beyond. None if never."""
    def beyond(v):
        return (v > level if strict else v >= level) if up else (v < level if strict else v <= level)
    start = int(after)
    pos = path[start] + (path[start + 1] - path[start]) * (after - start) if start < 3 else path[3]
    if beyond(pos):
        return after
    for i in range(start, 3):
        a, b = (pos if i == start else path[i]), path[i + 1]
        if beyond(b) and not beyond(a):
            base = after if i == start else float(i)
            frac = (level - a) / (b - a) if b != a else 0.0
            return base + (i + 1 - base) * frac
    return None


def _book_fill(order, side, path, pessimistic):
    """(time, price, maker) if a resting order fills on this bar's path, else None."""
    buy = side > 0
    if order.stop is not None and not order.triggered:
        ts = _cross(path, order.stop, up=buy)
        if ts is None:
            return None
        trig_px = path[0] if ts == 0 else order.stop
        if order.limit is None:
            return ts, trig_px, False              # a stop becomes a market order
        order.triggered = True
        ok = trig_px <= order.limit if buy else trig_px >= order.limit
        if ok:
            return ts, trig_px, False
        if pessimistic:
            return None                            # gapped past the limit: wait for the next bar
        tl = _cross(path, order.limit, up=not buy, after=ts)
        return (tl, order.limit, True) if tl is not None else None
    # a limit (or a stop-limit that triggered on an earlier bar)
    tl = _cross(path, order.limit, up=not buy, strict=pessimistic)
    if tl is None:
        return None
    px = path[0] if tl == 0 and ((path[0] <= order.limit) if buy else (path[0] >= order.limit)) else order.limit
    return tl, px, True


class Result:
    """Everything a backtest produced."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to_dict(self):
        return dict(self.__dict__)

    def __repr__(self):
        eq = getattr(self, "equity", None) or [0.0]
        return ("<Result %s %s to %s: %d bars, equity %.2f -> %.2f, %d trades, %d fills; try .equity, .trades, "
                ".fills, .weights, summarize(result)>" % (getattr(self, "ticker", "?"), self.dates[0] if self.dates else "?",
                                                          self.dates[-1] if self.dates else "?", len(eq), eq[0], eq[-1],
                                                          len(getattr(self, "trades", [])), len(getattr(self, "fills", []))))


def _resolve_range(dates, start=None, end=None, label="the data"):
    n = len(dates)
    s, e = 0, n - 1
    if start:
        while s < n and dates[s] < start[:10]:
            s += 1
    if end:
        while e >= 0 and dates[e][:10] > end[:10]:   # an end date includes that day's intraday bars
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
    # A daily benchmark next to intraday bars counts only once its day has closed.
    daily_on_intraday = len(bd[-1]) == 10 and len(bd[0]) == 10 and any(len(x) > 10 for x in dates[s:min(e + 1, s + 500)])
    for i in range(s, e + 1):
        d = dates[i]
        hit = False
        while j < bench.n and (bd[j] < d[:10] if daily_on_intraday else bd[j] <= d):
            last = src[j]
            hit = bd[j] == d
            j += 1
        gap = 0 if hit else gap + 1
        if gap > worst[0]:
            worst = (gap, d)
        out.append(last)
    if daily_on_intraday and out and out[0] is None:   # before its first close: held at its first value
        first = next((v for v in out if v is not None), src[0])
        out = [first if v is None else v for v in out]
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
    key = ("equal_weight", s, e, total_return)
    if key in panel._aligned:
        return panel._aligned[key]
    col = "total_return" if total_return else "close"
    series = [panel.aligned(a, col) for a in range(panel.k)]
    out = [1.0]
    for t in range(s + 1, e + 1):
        rets = [x[t] / x[t - 1] - 1 for x in series if x[t - 1] == x[t - 1] and x[t] == x[t]]
        out.append(out[-1] * (1 + (sum(rets) / len(rets) if rets else 0.0)))
    panel._aligned[key] = out
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
        rows = [{tk: (cols[a][t] if a in cols else 0.0) for a, tk in enumerate(panel.tickers)} for t in range(n)]
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
        self.orders = {}          # market orders for the next bar: {(asset, 'open'|'close'): (kind, value)}
        self.book = []            # resting limit / stop orders
        self.next_id = 1
        self.book_stats = {"filled": 0, "expired": 0, "cancelled": 0, "placed": 0}
        self.desk = None          # OptionsDesk when an option chain is loaded for one of the instruments

    def add_order(self, a, kind, value, kw):
        bad = set(kw) - set(ORDER_FIELDS)
        if bad:
            raise TypeError("unknown order option %s (use limit=, stop=, tif=, at=, oco=)" % ", ".join(sorted(bad)))
        limit, stop = kw.get("limit"), kw.get("stop")
        tif, at, oco = kw.get("tif", "day"), kw.get("at"), kw.get("oco")
        if tif not in ("day", "gtc"):
            raise ValueError("tif must be 'day' (next bar only) or 'gtc' (until filled or cancelled)")
        if at not in (None, "open", "close"):
            raise ValueError("at must be 'open' or 'close'")
        for name, v in (("limit", limit), ("stop", stop)):
            if v is not None and not (float(v) > 0):
                raise ValueError("%s price must be positive, got %r" % (name, v))
        if limit is not None or stop is not None:
            o = Order(self.next_id, a, kind, value, None if limit is None else float(limit),
                      None if stop is None else float(stop), tif, oco, self.t)
            self.next_id += 1
            self.book.append(o)
            self.book_stats["placed"] += 1
            return o.id
        key = (a, at or self.acct.cfg.execution)
        cur = self.orders.get(key)
        if kind == "target":
            self.orders[key] = ("target", value)
        elif cur is None:
            self.orders[key] = ("delta", value)
        elif cur[0] == "delta":
            self.orders[key] = ("delta", cur[1] + value)
        # a delta after a target is ignored, as before: the target already fixes the size
        return None

    def cancel(self, oid=None, a=None):
        n = 0
        for o in self.book:
            if o.status == "open" and (oid is None or o.id == oid) and (a is None or o.a == a):
                o.status = "cancelled"
                n += 1
        self.book_stats["cancelled"] += n
        self.book = [o for o in self.book if o.status == "open"]
        return n

    def open_orders(self, a=None):
        return [o.as_dict(self.acct.panel) for o in self.book if o.status == "open" and (a is None or o.a == a)]

    def process_book(self, t):
        """Check resting orders against bar t (orders placed at t-1's close or earlier)."""
        acct, p = self.acct, self.acct.panel
        pessimistic = acct.cfg.fill_model == "pessimistic"
        live = [o for o in self.book if o.status == "open" and o.created < t]
        hits = []
        for o in live:
            if not p.live[o.a][t]:
                continue
            ds, j = p.sets[o.a], p.nat[o.a][t]
            sh = acct.shares[o.a]
            qty = o.value - sh if o.kind == "target" else o.value
            if abs(qty) < 1e-9:
                o.status = "done"          # e.g. a stop-loss on a position that is already closed
                continue
            if pessimistic and sh:
                first_low = sh > 0         # the adverse extreme comes first
            else:
                first_low = abs(ds.open[j] - ds.low[j]) <= abs(ds.high[j] - ds.open[j])
            hit = _book_fill(o, 1 if qty > 0 else -1, _path(ds.open[j], ds.high[j], ds.low[j], ds.close[j], first_low), pessimistic)
            if hit:
                hits.append((hit[0], o.id, o, hit[1], hit[2], 1 if qty > 0 else -1))
        hits.sort(key=lambda h: (h[0], h[1]))
        for _, _, o, price, maker, side in hits:
            if o.status != "open":
                continue
            sh = acct.shares[o.a]
            qty = o.value - sh if o.kind == "target" else o.value
            if abs(qty) < 1e-9 or (1 if qty > 0 else -1) != side:
                continue                   # an earlier fill this bar already changed the position
            px = [None] * acct.k
            px[o.a] = price
            acct.arrival_t = o.created
            acct.share_orders({o.a: ("delta", qty)}, px, t, maker=maker, kind=o.type)
            done = acct.shares[o.a] - sh
            if abs(done) < 1e-9:
                continue
            if o.kind == "delta":
                o.value -= done
            if abs((o.value - acct.shares[o.a]) if o.kind == "target" else o.value) < 1e-9:
                o.status = "filled"
                self.book_stats["filled"] += 1
            if o.oco is not None:
                for other in self.book:
                    if other is not o and other.status == "open" and other.oco == o.oco:
                        other.status = "cancelled"
                        self.book_stats["cancelled"] += 1
        for o in live:
            if o.status == "open" and o.tif == "day":
                o.status = "expired"
                self.book_stats["expired"] += 1
        self.book = [o for o in self.book if o.status == "open"]


def _carry(carry, t_dec, weights, w_at, orders):
    """Merge what couldn't fill today into the pending orders for the next bar."""
    if not weights and not orders:
        return carry
    if carry is None:
        carry = {"t": t_dec, "weights": None, "w_at": w_at, "orders": {}}
    if weights:
        carry["weights"] = {**(carry["weights"] or {}), **weights}
        carry["w_at"] = w_at
    carry["orders"].update(orders)
    return carry


LOG_SHOW = 20000          # characters of print() output sent with a run
LOG_KEEP = 5 * 10 ** 6    # characters kept in Result.full_logs (fetch them with api_rows kind='logs')


def _log_preview(text):
    """The start and end of long output, with how much was left out in between."""
    if len(text) <= LOG_SHOW:
        return text
    head, tail = text[:LOG_SHOW * 3 // 5], text[-(LOG_SHOW * 2 // 5):]
    cut = text[len(head):len(text) - len(tail)]
    return "%s\n... %s characters (%s lines) not shown ...\n%s" % (
        head, format(len(cut), ","), format(cut.count("\n"), ","), tail)


SIGNAL_CACHE = {}   # signals(data) results by (code, params, data, seed): reused when only settings change


def _remember(cache, key, value, limit=8):
    if len(cache) >= limit:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value


def backtest(strategy, ds, cfg=None, start=None, end=None, bench=None, capture_output=True, rates=None,
             signal_key=None, decide_last=False):
    """Run a single-instrument strategy over ds and return a Result.

    Timing: on every bar the strategy sees data up to and including that bar's
    close. Orders it places are filled on the NEXT bar (at the open, or at the
    close with execution='close'), so a strategy can never trade on information
    it could not have had. decide_last=True also asks the strategy on the final bar and
    puts the orders it would send for the next session in Result.next (see next_orders).
    """
    return _run_backtest(strategy, Panel([ds]), cfg, start, end, bench, False, capture_output, rates, signal_key, decide_last)


_PANELS = {}


def shared_panel(datasets):
    """A Panel for these instruments, reused while their data is unchanged (sweeps run
    the same instruments hundreds of times). Keyed by the data fingerprints."""
    key = tuple((id(ds), ds.fingerprint()) for ds in datasets)
    p = _PANELS.get(key)
    if p is None:
        if len(_PANELS) >= 8:
            _PANELS.pop(next(iter(_PANELS)))
        p = _PANELS[key] = Panel(datasets)
    return p


def backtest_portfolio(strategy, datasets, cfg=None, start=None, end=None, bench=None,
                       capture_output=True, rates=None, signal_key=None, decide_last=False):
    """Run a portfolio strategy over several instruments on their shared calendar.

    on_bar(ctx) returns {ticker: target weight}; signals(data) returns
    {ticker: [weight per bar]}, a list of such dicts, or an array with one column
    per ticker. Without a benchmark, the benchmark is an equal-weight portfolio of
    the same instruments rebalanced daily.
    """
    return _run_backtest(strategy, shared_panel(datasets), cfg, start, end, bench, True, capture_output, rates, signal_key, decide_last)


class _Limits:
    """Sizing and limits applied to every set of target weights, at the close they are
    decided, with only data up to that close. In order: volatility targeting or fractional
    Kelly (from what the proposed weights would have earned over the last sizing_lookback
    days), drawdown de-risking, per-instrument caps, the net limit and the gross limit.
    Every change is recorded in the constraint log. Share orders aren't resized."""

    LOG_MAX = 2000

    def __init__(self, cfg, panel):
        self.cfg, self.panel = cfg, panel
        self.ann = panel_ann(panel)
        self.on = (cfg.sizing != "none" or cfg.max_weight > 0 or cfg.max_gross > 0 or cfg.max_net > 0 or cfg.dd_limit > 0)
        self.log, self.counts = [], {}
        self.derisked, self.peak, self.seen = False, None, 0
        self.derisked_bars = 0
        self.last_raw = None
        self.scales = []
        self._r = None

    def _returns(self):
        if self._r is None:
            out = []
            for a in range(self.panel.k):
                tr = self.panel.aligned(a, "total_return")
                out.append([0.0] + [(tr[t] / tr[t - 1] - 1) if not (isnan(tr[t]) or isnan(tr[t - 1]) or tr[t - 1] <= 0) else 0.0
                                    for t in range(1, len(tr))])
            self._r = out
        return self._r

    def _note(self, t, rule, detail):
        self.counts[rule] = self.counts.get(rule, 0) + 1
        if len(self.log) < self.LOG_MAX:
            self.log.append({"i": t, "date": self.panel.dates[t], "rule": rule, "detail": detail})

    def observe(self, t, equity):
        """Track the equity high and switch de-risking on or off. True when the state changed,
        so the latest targets should be re-sized now."""
        for v in equity[self.seen:]:
            self.peak = v if self.peak is None else max(self.peak, v)
        self.seen = len(equity)
        cfg = self.cfg
        if not cfg.dd_limit or not equity:
            return False
        dd = equity[-1] / self.peak - 1 if self.peak else 0.0
        if self.derisked:
            self.derisked_bars += 1
        if not self.derisked and dd <= -cfg.dd_limit:
            self.derisked = True
            self._note(t, "drawdown", "drawdown %.1f%%: weights x %.2f until it recovers to %.1f%%"
                       % (100 * dd, cfg.dd_scale, -100 * cfg.dd_limit * cfg.dd_resume))
            return True
        if self.derisked and dd > -cfg.dd_limit * cfg.dd_resume:
            self.derisked = False
            self._note(t, "drawdown", "drawdown back to %.1f%%: full size again" % (100 * dd))
            return True
        return False

    def apply(self, w, t):
        cfg, tk = self.cfg, self.panel.tickers
        self.last_raw = dict(w)
        w = dict(w)
        gross = sum(abs(v) for v in w.values())
        if cfg.sizing != "none" and gross > 0:
            L = cfg.sizing_lookback
            lo = max(1, t - L + 1)
            if t - lo + 1 >= max(10, L // 3):
                R = self._returns()
                act = [(a, v) for a, v in w.items() if v]
                p = [sum(v * R[a][u] for a, v in act) for u in range(lo, t + 1)]
                m, sd = mean(p), stdev(p)
                if sd > 0:
                    if cfg.sizing == "vol_target":
                        scale = cfg.vol_target / (sd * math.sqrt(self.ann))
                        why = "estimated volatility %.1f%%" % (100 * sd * math.sqrt(self.ann))
                    else:
                        scale = max(0.0, cfg.kelly_fraction * m / (sd * sd))
                        why = "Kelly leverage %.2f" % (m / (sd * sd))
                    cap = cfg.max_leverage / gross
                    if scale > cap:
                        scale, why = cap, why + ", capped by max leverage"
                    self.scales.append(scale)
                    if abs(scale - 1) > 1e-9:
                        w = {a: v * scale for a, v in w.items()}
                        self._note(t, cfg.sizing, "weights x %.3f (%s)" % (scale, why))
        if self.derisked:
            w = {a: v * cfg.dd_scale for a, v in w.items()}
        if cfg.max_weight > 0:
            for a, v in list(w.items()):
                if abs(v) > cfg.max_weight + 1e-12:
                    w[a] = math.copysign(cfg.max_weight, v)
                    self._note(t, "position cap", "%s %.1f%% -> %.1f%%" % (tk[a], 100 * v, 100 * w[a]))
        if cfg.max_net > 0:
            net = sum(w.values())
            if abs(net) > cfg.max_net + 1e-12:
                longs = sum(v for v in w.values() if v > 0)
                shorts = -sum(v for v in w.values() if v < 0)
                if net > 0:
                    f = (cfg.max_net + shorts) / longs
                    w = {a: (v * f if v > 0 else v) for a, v in w.items()}
                else:
                    f = (cfg.max_net + longs) / shorts
                    w = {a: (v * f if v < 0 else v) for a, v in w.items()}
                self._note(t, "net limit", "net %.1f%% -> %s%.1f%% (the %s side scaled by %.3f)"
                           % (100 * net, "" if net > 0 else "-", 100 * cfg.max_net, "long" if net > 0 else "short", f))
        if cfg.max_gross > 0:
            g = sum(abs(v) for v in w.values())
            if g > cfg.max_gross + 1e-12:
                f = cfg.max_gross / g
                w = {a: v * f for a, v in w.items()}
                self._note(t, "gross limit", "gross %.1f%% -> %.1f%%" % (100 * g, 100 * cfg.max_gross))
        return w

    def summary(self, bars):
        return {"log": self.log, "counts": dict(self.counts), "events": sum(self.counts.values()),
                "derisked_share": self.derisked_bars / bars if bars else 0.0,
                "avg_scale": mean(self.scales) if self.scales else NAN, "truncated": sum(self.counts.values()) > len(self.log)}


def next_orders(acct, run, panel, e, pending, cfg):
    """The orders the strategy would send for the session after bar e, from its targets at
    e's close (or what is still unfilled), sized at that close like the backtest would."""
    marks = [panel.mark(a, e) for a in range(panel.k)]
    eq = acct.equity_marked(e)
    orders = []
    if pending is not None and pending.get("weights") is not None and eq > 0:
        for a in range(panel.k):
            w = pending["weights"].get(a)
            if w is None:
                continue
            m = marks[a]
            if isnan(m) or m <= 0:
                continue
            cur = acct.shares[a]
            if abs(w - cur * m / eq) < cfg.rebalance_band and w != 0:
                continue                              # inside the rebalance band: no trade
            target = w * eq / m
            if not cfg.fractional:
                target = float(int(target))
            dq = target - cur
            if abs(dq) > 1e-9:
                orders.append({"ticker": panel.tickers[a], "side": "buy" if dq > 0 else "sell", "qty": dq,
                               "target_shares": target, "target_weight": w, "ref_price": m, "value": dq * m,
                               "type": "market", "at": pending.get("w_at", cfg.execution)})
    if pending is not None:
        for (a, at), (kind, value) in sorted(pending.get("orders", {}).items()):
            dq = value - acct.shares[a] if kind == "target" else value
            if abs(dq) > 1e-9:
                orders.append({"ticker": panel.tickers[a], "side": "buy" if dq > 0 else "sell", "qty": dq,
                               "target_shares": acct.shares[a] + dq, "target_weight": None, "ref_price": marks[a],
                               "value": dq * marks[a] if not isnan(marks[a]) else NAN, "type": "market", "at": at})
    return {"date": panel.dates[e], "decided": pending is not None and pending.get("t") == e, "equity": eq,
            "cash": acct.cash, "positions": {tk: acct.shares[a] for a, tk in enumerate(panel.tickers) if acct.shares[a]},
            "orders": orders, "resting": run.open_orders()}


def _run_backtest(strategy, panel, cfg, start, end, bench, portfolio, capture_output, rates, signal_key=None, decide_last=False):
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
    desk = run.desk = OptionsDesk(cfg, panel, acct) if any(tk in OPTIONS for tk in panel.tickers) else None
    limits = _Limits(cfg, panel)
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
    shares_hist, cash_hist = [], []
    try:
        row, sig = None, None
        if strategy.signals is not None:
            key = None if signal_key is None else (signal_key, cfg.seed, cfg.total_return, portfolio,
                                                   tuple(ds.fingerprint() for ds in panel.sets))
            if key in SIGNAL_CACHE:
                row, sig, printed = SIGNAL_CACHE[key]
                out_buf.write(printed)
            else:
                mark = out_buf.tell()
                raw = strategy.signals(PanelView(panel) if portfolio else DataView(panel.sets[0]))
                row, sig = signal_rows(raw, panel, portfolio)
                if key is not None:
                    _remember(SIGNAL_CACHE, key, (row, sig, out_buf.getvalue()[mark:]))
        if strategy.init is not None:
            run.t = s
            ctx._sync(s)
            bar_date = panel.dates[s]
            strategy.init(ctx)

        pending = None
        wiped = False
        mm = cfg.maintenance_margin
        # Instruments whose data ends before the backtest does: their position is closed on the
        # first calendar bar after their last one.
        ends = {}
        if portfolio:
            import bisect
            for a, ds in enumerate(panel.sets):
                if ds.dates[-1] < panel.dates[e]:
                    ends.setdefault(max(s + 1, bisect.bisect_right(panel.dates, ds.dates[-1])), []).append(a)
        delisted = []
        for t in range(s, e + 1):
            run.t = t
            if t > s and not wiped:
                acct.accrue(t - 1, t)
                if cfg.total_return:
                    acct.pay_dividends(t)
            if t in ends and not wiped:
                for a in ends[t]:
                    sh = acct.shares[a]
                    px = acct.delist(a, t, cfg.delisting_return)
                    run.cancel(a=a)
                    delisted.append({"ticker": panel.tickers[a], "last_date": panel.sets[a].dates[-1], "date": panel.dates[t],
                                     "shares": sh, "price": px})
            if not wiped:
                acct.begin_bar(t)
                carry = None
                for phase in ("open", "close"):
                    if phase == "close" and run.book:
                        run.process_book(t)          # intrabar: between the open and the close
                    if pending is None:
                        continue
                    acct.arrival_t = pending["t"]
                    px = [None] * k
                    for a in range(k):
                        if panel.live[a][t]:
                            j = panel.nat[a][t]
                            px[a] = panel.sets[a].open[j] if phase == "open" else panel.sets[a].close[j]
                    if pending["weights"] is not None and pending["w_at"] == phase:
                        lw, ls = acct.rebalance(pending["weights"], px, t)
                        carry = _carry(carry, pending["t"], lw, phase, {(a, phase): v for a, v in ls.items()})
                    group = {a: v for (a, at), v in pending["orders"].items() if at == phase}
                    if group:
                        later = acct.share_orders(group, px, t)
                        carry = _carry(carry, pending["t"], None, phase, {(a, phase): v for a, v in later.items()})
                pending = carry
            if desk is not None and not wiped:
                desk.on_bar(t)                       # option fills at today's quotes, expiries, marks
            eq = acct.equity_marked(t)
            if eq <= 0 and not wiped:
                wiped = True
                notes.append("Equity reached zero on %s; trading stopped." % panel.dates[t])
                if desk is not None:
                    desk.liquidate()
                acct.liquidate(t)
                acct.cash = 0.0
                eq = 0.0
            if portfolio:
                # Series are computed once after the loop from these snapshots. A margin call
                # needs borrowed cash or a short, so the gross check is skipped without either.
                if mm > 0 and not wiped and (acct.cash < 0 or min(acct.shares) < 0):
                    g = sum(abs(sh * panel.mark(a, t)) for a, sh in enumerate(acct.shares) if sh)
                    if g > 0 and eq < mm * g:
                        target = eq * min(cfg.max_leverage, 0.8 / mm)
                        acct.margin_calls.append({"date": panel.dates[t], "equity": eq, "gross": g})
                        acct.liquidate(t, 1 - target / g, "margin call")
                        eq = acct.equity_marked(t)
                equity.append(eq)
                shares_hist.append(acct.shares[:])
                cash_hist.append(acct.asset_cash[:])
            else:
                c = panel.mark(0, t)
                sh = acct.shares[0]
                g = abs(sh * c) if sh else 0
                if mm > 0 and not wiped and g > 0 and eq < mm * g:
                    target = eq * min(cfg.max_leverage, 0.8 / mm)
                    acct.margin_calls.append({"date": panel.dates[t], "equity": eq, "gross": g})
                    acct.liquidate(t, 1 - target / g, "margin call")
                    eq = acct.equity_marked(t)
                    sh = acct.shares[0]
                    g = abs(sh * c) if sh else 0
                equity.append(eq)
                lv = sh * c if sh > 0 else 0
                sv = -sh * c if sh < 0 else 0
                gross_l.append(g / eq if eq > 0 else 0.0)
                net_l.append((lv - sv) / eq if eq > 0 else 0.0)
                long_l.append(lv / eq if eq > 0 else 0.0)
                short_l.append(sv / eq if eq > 0 else 0.0)
                weights[0].append(sh * c / eq if eq > 0 and sh else 0.0)
                asset_pnl[0].append(acct.asset_cash[0] + (sh * c if sh else 0.0))
                position.append(sh)
                exposure.append(sh * c / eq if eq > 0 else 0.0)
            if wiped or (t == e and not decide_last):
                continue

            ctx._sync(t)
            bar_date = panel.dates[t]
            resize = limits.on and limits.observe(t, equity)
            if row is not None:
                w = row(t)
                if w is None and resize and limits.last_raw is not None:
                    w = limits.last_raw
                if w is not None:
                    pending = {"t": t, "weights": limits.apply(w, t) if limits.on else w, "w_at": cfg.execution, "orders": {}}
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
            else:
                w = None
            if w is None and resize and limits.last_raw is not None and not run.orders:
                w = limits.last_raw           # de-risking switched: re-size the latest targets now
            if w is not None:
                if limits.on:
                    w = limits.apply(w, t)
                pending = {"t": t, "weights": w, "w_at": cfg.execution, "orders": {}}
            elif run.orders and r is None:
                pending = {"t": t, "weights": None, "w_at": cfg.execution, "orders": dict(run.orders)}
    except StrategyError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StrategyError(format_error(exc, bar_date))
    finally:
        sys.stdout = old_stdout

    nxt = next_orders(acct, run, panel, e, pending, cfg) if decide_last and not wiped else None
    if portfolio:
        weights, asset_pnl, gross_l, net_l, long_l, short_l = _portfolio_series(panel, s, shares_hist, cash_hist, equity)
        exposure = gross_l
    for a in range(k):
        if acct.open_trades[a] is not None:
            m = panel.mark(a, e)
            acct._close_trade(a, e, m, acct.asset_value(a, m), still_open=True)
    acct.trades.sort(key=lambda tr: (tr["entry_i"], tr["ticker"]))

    dates = panel.dates[s:e + 1]
    b0 = bench_vals[0]
    bench_equity = [cfg.cash * v / b0 for v in bench_vals]
    full_logs = out_buf.getvalue()
    if len(full_logs) > LOG_KEEP:
        full_logs = full_logs[:LOG_KEEP] + "\n... output stopped here: over %d MB ...\n" % (LOG_KEEP // 10 ** 6)
    logs = _log_preview(full_logs)
    for ds_ in list(panel.sets) + ([bench] if bench is not None else []):
        if getattr(ds_, "fx_note", None):
            notes.append(ds_.fx_note)
    for d in delisted:
        notes.append("%s has no data after %s: its position (%s shares) was closed at its last close%s on %s. If the "
                     "instrument still trades, the file is out of date; download a newer one."
                     % (d["ticker"], d["last_date"], format(round(d["shares"], 4), ","),
                        "" if not cfg.delisting_return else " %+.0f%% (delisting return)" % (100 * cfg.delisting_return), d["date"]))
    if portfolio and panel.k >= 3:
        span = (panel.days[e] - panel.days[s]).days / 365.25
        alive = all((panel.days[e] - _dt.date.fromisoformat(ds.dates[-1][:10])).days <= 10 for ds in panel.sets)
        if alive and span >= 3:
            notes.append("Survivorship bias: all %d instruments were still trading at the end of the test. A universe "
                         "chosen today leaves out companies that were delisted or dropped out of the index over these "
                         "%.0f years, which usually flatters the result. Add their histories (they are closed out when "
                         "their data ends) or test on an index fund." % (panel.k, span))
    opts = None
    if desk is not None and (desk.fills or desk.unfilled):
        opts = desk.summary(dates)
        if any(f["contracts"] < 0 and f["action"].endswith("to open") for f in desk.fills):
            notes.append("Short options aren't margined here: their marked value counts in equity, but the collateral a "
                         "broker would hold is not set aside, so check the cash a short call or put would need.")
        if desk.stale:
            notes.append("%d option marks reused an earlier price because the chain had no quote that day." % desk.stale)
        if desk.unfilled:
            notes.append("%d option order%s couldn't fill (no quote, bid or ask on the next day); see the Options panel."
                         % (len(desk.unfilled), "" if len(desk.unfilled) == 1 else "s"))
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
        spread_paid=acct.spread_paid, impact_paid=acct.impact_paid, partial_fills=acct.partial_fills,
        book_stats=dict(run.book_stats, open=len(run.book)),
        logs=logs, full_logs=full_logs, notes=notes, config=cfg, mode=strategy.mode, params=dict(strategy.params),
        signals=sig, signal_key=signal_key, limits=limits.summary(len(equity)) if limits.on else None, ann=panel_ann(panel),
        delisted=delisted, next=nxt, options=opts,
    )


USE_NUMPY = True   # tests switch this off to check the pure-Python path gives the same numbers
# Rebalances with at least this many targets size orders with NumPy. Below it, a plain loop
# is faster (NumPy calls carry a fixed overhead, large under WebAssembly). Results are identical.
VECTOR_MIN = 200


def _numpy():
    """NumPy if it is installed (and allowed), else None."""
    if not USE_NUMPY:
        return None
    try:
        import numpy
        return numpy
    except ImportError:
        return None


def _portfolio_series(panel, s, shares_hist, cash_hist, equity):
    """Per-instrument weights and P&L, and gross/net/long/short exposure, for every bar
    from the share and cash snapshots taken at each close."""
    n, k = len(equity), panel.k
    closes = [panel.aligned(a, "close")[s:s + n] for a in range(k)]
    np = _numpy()
    if np is not None:
        sh = np.array(shares_hist, dtype=float)
        mk = np.array(closes, dtype=float).T
        val = np.where(sh != 0, sh * np.nan_to_num(mk), 0.0)
        eq = np.array(equity, dtype=float)
        ok = eq > 0
        safe = np.where(ok, eq, 1.0)[:, None]
        w = np.where(ok[:, None] & (sh != 0), val / safe, 0.0)
        pnl = np.array(cash_hist, dtype=float) + val
        lv = np.where(val > 0, val, 0.0).sum(axis=1)
        sv = np.where(val < 0, -val, 0.0).sum(axis=1)
        div = np.where(ok, eq, np.inf)
        return ([w[:, a].tolist() for a in range(k)], [pnl[:, a].tolist() for a in range(k)],
                ((lv + sv) / div).tolist(), ((lv - sv) / div).tolist(), (lv / div).tolist(), (sv / div).tolist())
    weights, pnl = [[] for _ in range(k)], [[] for _ in range(k)]
    gross, net, long_, short = [], [], [], []
    for t in range(n):
        eq, row, cash = equity[t], shares_hist[t], cash_hist[t]
        lv = sv = 0.0
        for a in range(k):
            x = row[a]
            v = x * closes[a][t] if x else 0.0
            if v > 0:
                lv += v
            elif v < 0:
                sv -= v
            weights[a].append(v / eq if eq > 0 and x else 0.0)
            pnl[a].append(cash[a] + v)
        gross.append((lv + sv) / eq if eq > 0 else 0.0)
        net.append((lv - sv) / eq if eq > 0 else 0.0)
        long_.append(lv / eq if eq > 0 else 0.0)
        short.append(sv / eq if eq > 0 else 0.0)
    return weights, pnl, gross, net, long_, short


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
    ("dividends_received", "Dividends received", "Carry & exposure", "money", "std", "Cash dividends paid to the account on shares held at the close before each ex-date."),
    ("dividends_net", "Dividends (net)", "Carry & exposure", "money", "std", "Dividends received on longs minus dividends paid on shorts."),
    ("interest_net", "Interest on cash", "Carry & exposure", "money", "std", "Interest earned on cash minus interest paid on borrowed cash (ACT/360)."),
    ("borrow_paid", "Borrow fees", "Carry & exposure", "money", "extra", "Fees paid to borrow shares for short positions."),
    ("carry_total", "Net carry", "Carry & exposure", "money", "extra", "Dividends + interest - borrow fees."),
    ("margin_calls", "Margin calls", "Carry & exposure", "int", "std", "Times equity fell below the maintenance margin and positions were cut."),

    ("limit_events", "Sizing & limit changes", "Sizing & limits", "int", "extra", "Times the sizing rule, drawdown de-risking or a position, net or gross limit changed the strategy's target weights (see the constraint log)."),
    ("derisked_share", "Time de-risked", "Sizing & limits", "pct", "extra", "Share of bars spent scaled down after the drawdown limit was hit."),
    ("avg_scale", "Avg. sizing scale", "Sizing & limits", "num", "extra", "Mean factor the volatility-target or Kelly rule multiplied the weights by (1 = unchanged)."),

    ("is_bps", "Implementation shortfall", "Execution", "bps", "std", "Average fill price vs the close when the order was decided, in bps of traded value (side-adjusted; costs, delay and price moves until the fill)."),
    ("is_cost", "Shortfall ($)", "Execution", "money", "extra", "Implementation shortfall in dollars: what trading at the decision price would have saved, before commissions."),
    ("spread_paid", "Spread paid", "Execution", "money", "extra", "Half the bid-ask spread on every market fill."),
    ("impact_paid", "Market impact", "Execution", "money", "extra", "Square-root price impact of the fills, from their size vs average daily volume."),
    ("partial_fills", "Partial fills", "Execution", "int", "std", "Orders spread over several bars by the participation limit."),
    ("limit_filled", "Resting orders filled", "Execution", "int", "extra", "Limit, stop and stop-limit orders that filled."),
    ("limit_unfilled", "Resting orders unfilled", "Execution", "int", "extra", "Limit and stop orders that expired or were cancelled without filling completely."),

    ("t_stat", "t-statistic (mean return)", "Statistical tests", "num", "extra", "Mean daily excess return / its standard error. Above 2 suggests the mean is not zero."),
    ("p_value", "p-value (mean return > 0)", "Statistical tests", "pval", "extra", "Chance of a t-statistic this high if the true mean return were zero (one-sided, Student's t). Below 0.05 is the usual bar."),
    ("t_stat_hac", "t-statistic (Newey-West)", "Statistical tests", "num", "extra", "The t-statistic with Newey-West standard errors, valid when returns are autocorrelated or their volatility changes. Trust it over the plain one when they differ."),
    ("p_value_hac", "p-value (Newey-West)", "Statistical tests", "pval", "extra", "One-sided p-value of the Newey-West t-statistic."),
    ("sharpe_lo_adj", "Sharpe (Lo-adjusted)", "Statistical tests", "num", "extra", "Sharpe ratio annualized with the returns' autocorrelations (Lo 2002) instead of sqrt(252). Lower than the plain Sharpe when returns trend from day to day (smoothed or illiquid prices)."),
    ("n_eff", "Effective sample size", "Statistical tests", "int", "extra", "How many independent days of returns the autocorrelated returns are worth (Bartlett-weighted autocorrelations)."),
    ("eff_years", "Effective track record", "Statistical tests", "years", "extra", "Effective sample size in years. Compare with the minimum track record length."),
    ("psr", "Probabilistic Sharpe", "Statistical tests", "pct", "extra", "Probability that the true Sharpe ratio is above 0, allowing for skew, fat tails and track length (Bailey & Lopez de Prado)."),
    ("sharpe_lo", "Sharpe 95% CI low", "Statistical tests", "num", "extra", "Lower end of a 95% confidence interval for the annualized Sharpe ratio."),
    ("sharpe_hi", "Sharpe 95% CI high", "Statistical tests", "num", "extra", "Upper end of a 95% confidence interval for the annualized Sharpe ratio."),
    ("min_trl", "Min. track record", "Statistical tests", "years", "extra", "Years of data needed to be 95% sure the Sharpe ratio is above 0. Longer than the backtest = not proven yet."),
    ("alpha_t", "Alpha t-statistic", "Statistical tests", "num", "extra", "Alpha / its standard error from the regression on the benchmark. Above 2 suggests real alpha."),
    ("alpha_p", "Alpha p-value", "Statistical tests", "pval", "extra", "Chance of an alpha t-statistic this high if true alpha were zero (one-sided, Student's t)."),
    ("alpha_t_hac", "Alpha t (Newey-West)", "Statistical tests", "num", "extra", "Alpha's t-statistic with Newey-West standard errors."),
    ("alpha_p_hac", "Alpha p (Newey-West)", "Statistical tests", "pval", "extra", "One-sided p-value of alpha's Newey-West t-statistic."),
    ("win_rate_p", "Win-rate p-value", "Statistical tests", "pval", "extra", "Chance of at least this many winning trades if each trade were a coin flip (one-sided, exact binomial)."),
    ("autocorr", "Autocorrelation (1 day)", "Statistical tests", "num", "extra", "Correlation of each day's return with the previous day's. Far from 0 means returns are not independent."),
    ("autocorr_p", "Autocorrelation p-value", "Statistical tests", "pval", "extra", "Two-sided p-value that the 1-day autocorrelation is zero. Below 0.05 = significant serial correlation."),
    ("dsr", "Deflated Sharpe", "Statistical tests", "pct", "std", "Probability that the true Sharpe ratio beats the best you'd expect from this many trials with no skill (Bailey & Lopez de Prado). Below 95%: the result may be the luck of trying many versions."),
    ("sr_hurdle", "Sharpe hurdle (trials)", "Statistical tests", "num", "std", "The highest annualized Sharpe ratio expected by chance from the number of strategy variations tried so far."),
    ("trials", "Trials counted", "Statistical tests", "int", "std", "Distinct versions of this strategy (code, parameters, settings, market) run in this browser. Each one raises the bar."),
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


def return_tests(r, rfd=0.0, ann=None):
    """Significance tests on daily returns r (rfd = daily risk-free rate).

    The Sharpe ratio's standard error follows Lo (2002) / Mertens (2002), which
    widens it for negative skew and fat tails; PSR and the minimum track record
    length follow Bailey & Lopez de Prado (2012). The t-test's p-value is exact
    (Student's t, n - 1 degrees of freedom); its Newey-West version allows for
    autocorrelated and heteroskedastic returns. The Lo-adjusted Sharpe annualizes
    with the autocorrelations instead of sqrt(periods per year), and the effective
    sample size says how many independent observations the returns are worth.
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
        out["p_value"] = t_sf(out["t_stat"], n - 1)
        hac = newey_west_mean(r, rfd)
        if hac:
            out["t_stat_hac"], out["p_value_hac"] = hac
        P = ann or TRADING_DAYS
        rhos = autocorrelations(r, min(10, n // 4))
        q = int(round(P))
        denom = q + 2 * sum((q - k) * rho for k, rho in enumerate(rhos, 1) if k < q)
        if denom > 0:
            out["sharpe_lo_adj"] = sr * q / math.sqrt(denom)
        out["n_eff"] = effective_sample_size(r)
        out["eff_years"] = out["n_eff"] / P
        if not isnan(g3) and not isnan(g4):
            spread = 1 - g3 * sr + (g4 + 2) / 4 * sr * sr   # (g4 + 2) = raw kurtosis - 1
            if spread > 0:
                se = math.sqrt(spread / (n - 1))
                ann_sqrt = math.sqrt(ann or TRADING_DAYS)
                out["psr"] = norm_cdf(sr / se)
                out["sharpe_lo"] = (sr - Z_975 * se) * ann_sqrt
                out["sharpe_hi"] = (sr + Z_975 * se) * ann_sqrt
                if sr > 0:
                    out["min_trl"] = (1 + spread * (Z_95 / sr) ** 2) / (ann or TRADING_DAYS)
    if not isnan(g3) and not isnan(g4):
        jb = n / 6 * (g3 * g3 + g4 * g4 / 4)
        out["jb_p"] = math.exp(-jb / 2)   # chi-squared with 2 degrees of freedom
    rho = correlation(r[:-1], r[1:])
    if not isnan(rho):
        out["autocorr"] = rho
        out["autocorr_p"] = math.erfc(abs(rho) * math.sqrt(n) / math.sqrt(2))
    return out


def autocorrelations(r, max_lag):
    """Sample autocorrelations rho_1 .. rho_max_lag (the usual biased estimator)."""
    n = len(r)
    if n < 3 or max_lag < 1:
        return []
    m = sum(r) / n
    d = [x - m for x in r]
    c0 = sum(x * x for x in d)
    if c0 <= 0:
        return [0.0] * max_lag
    return [sum(d[t] * d[t - k] for t in range(k, n)) / c0 for k in range(1, max_lag + 1)]


def effective_sample_size(r, lags=None):
    """n / (1 + 2 sum of Bartlett-weighted autocorrelations), between 1 and n: how many
    independent observations autocorrelated returns are worth."""
    n = len(r)
    L = newey_west_lags(n) if lags is None else lags
    rho = autocorrelations(r, L)
    f = 1 + 2 * sum((1 - k / (L + 1.0)) * x for k, x in enumerate(rho, 1))
    return min(float(n), max(1.0, n / f)) if f > 0 else float(n)


def newey_west_mean(r, rfd=0.0):
    """(t-statistic, one-sided p-value) of the mean excess return with Newey-West errors."""
    n = len(r)
    if n < 10:
        return None
    try:
        fit = ols([x - rfd for x in r], [[1.0] for _ in r])
    except ValueError:
        return None
    t = fit["t_hac"][0]
    return (t, t_sf(t, n - 1)) if not isnan(t) else None


def binom_sf(k, n, p=0.5):
    """P(X >= k) for X ~ Binomial(n, p), exact (via the incomplete beta function)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return betainc(k, n - k + 1, p)


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
    m["win_rate_p"] = binom_sf(len(wins), n)
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
                traded_notional=None, costs=None, n_fills=None, clipped=None, trade_ret="ret", ann=None):
    """Compute every metric in METRICS (those that apply) for one equity curve.
    ann: return periods per year (252 for daily US equities; see periods_per_year)."""
    P = ann or TRADING_DAYS
    m = {}
    n = len(equity)
    m["start"], m["end"], m["bars"] = dates[0], dates[-1], n
    e0, e1 = equity[0], equity[-1]
    years = max((_dt.date.fromisoformat(dates[-1][:10]) - _dt.date.fromisoformat(dates[0][:10])).days / 365.25, 1 / 365.25)
    m["years"] = years
    m["final_equity"] = e1
    m["total_return"] = e1 / e0 - 1
    m["cagr"] = (e1 / e0) ** (1 / years) - 1 if e1 > 0 else -1.0

    r = daily_returns(equity)
    rfd = (1 + rf) ** (1 / P) - 1
    ann = math.sqrt(P)
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
        m.update(return_tests(r, rfd, P))

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
        cut = (_dt.date.fromisoformat(dates[-1][:10]) - _dt.timedelta(days=365)).isoformat()
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
        m["alpha"] = ((mean(r) - rfd) - beta * (mub - rfd)) * P if not isnan(beta) else NAN
        corr = correlation(r, rb)
        m["correlation"] = corr
        m["r_squared"] = corr * corr if not isnan(corr) else NAN
        diff = [a - b for a, b in zip(r, rb)]
        te = stdev(diff) * ann
        m["tracking_error"] = te
        m["information_ratio"] = mean(diff) * P / te if te > 0 else NAN
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
        if not isnan(beta) and len(r) > 2:
            x = [v - rfd for v in rb]
            y = [v - rfd for v in r]
            mx = mean(x)
            a = mean(y) - beta * mx
            sxx = sum((v - mx) ** 2 for v in x)
            s2 = sum((yv - a - beta * xv) ** 2 for xv, yv in zip(x, y)) / (len(x) - 2)
            se = math.sqrt(s2 * (1 / len(x) + mx * mx / sxx)) if sxx > 0 else 0.0
            if se > 1e-15:
                m["alpha_t"] = a / se
                m["alpha_p"] = t_sf(a / se, len(x) - 2)
            try:
                fit = ols(y, [[1.0, v] for v in x]) if len(x) > 10 else None
            except ValueError:            # the benchmark's returns don't vary
                fit = None
            if fit is not None:
                if not isnan(fit["t_hac"][0]):
                    m["alpha_t_hac"] = fit["t_hac"][0]
                    m["alpha_p_hac"] = t_sf(fit["t_hac"][0], len(x) - 2)
    return m


def rolling_sharpe(equity, window=126, rf=0.0, ann=None):
    r = daily_returns(equity)
    rfd = (1 + rf) ** (1 / (ann or TRADING_DAYS)) - 1
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
        out[t] = (mu - rfd) / math.sqrt(var) * math.sqrt((ann or TRADING_DAYS)) if var > 1e-18 else NAN
    return out


def rolling_vol(equity, window=63, ann=None):
    """Annualized volatility of the last `window` daily returns, aligned with equity."""
    r = daily_returns(equity)
    out = [NAN] * len(equity)
    ann = math.sqrt((ann or TRADING_DAYS))
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


def summarize(res, rf=None, trials=None):
    """Strategy and benchmark metrics for a Result. trials = {'n': count, 'var': variance
    of the annualized Sharpe ratios tried} feeds the Deflated Sharpe Ratio."""
    rf = res.config.risk_free if rf is None else rf
    m = performance(res.dates, res.equity, rf, res.bench_equity, res.trades, res.exposure,
                    res.traded_notional, res.commission + res.slippage + res.spread_paid + res.impact_paid,
                    len(res.fills), res.clipped, "ret_notional" if res.portfolio else "ret", res_ann(res))
    m.update(execution_stats(res))
    m.update(exposure_stats(res))
    c = res.carry
    m["dividends_received"] = c["dividends_received"]
    m["dividends_net"] = c["dividends_received"] - c["dividends_paid"]
    m["interest_net"] = c["interest_earned"] - c["interest_paid"]
    m["borrow_paid"] = c["borrow_paid"]
    m["carry_total"] = m["dividends_net"] + m["interest_net"] - c["borrow_paid"]
    m["margin_calls"] = len(res.margin_calls)
    lim = getattr(res, "limits", None) or {}
    m["limit_events"] = lim.get("events", 0)
    m["derisked_share"] = lim.get("derisked_share", 0.0)
    m["avg_scale"] = lim.get("avg_scale", NAN)
    t = trials or {}
    rfd = (1 + rf) ** (1 / res_ann(res)) - 1
    m.update(deflated_sharpe(daily_returns(res.equity), t.get("n", 1), t.get("var"), rfd, res_ann(res)))
    b = performance(res.dates, res.bench_equity, rf, ann=res_ann(res))
    return m, b


def execution_stats(res):
    """Transaction-cost analysis of the fills."""
    short_cost = notional = 0.0
    for f in res.fills:
        arr = f.get("arrival")
        if arr:
            side = 1 if f["qty"] > 0 else -1
            short_cost += side * (f["price"] - arr) * abs(f["qty"])
            notional += arr * abs(f["qty"])
    b = res.book_stats
    return {"is_bps": short_cost / notional * 1e4 if notional else NAN, "is_cost": short_cost,
            "spread_paid": res.spread_paid, "impact_paid": res.impact_paid, "partial_fills": res.partial_fills,
            "limit_filled": b["filled"], "limit_unfilled": b["expired"] + b["cancelled"]}


def attribution(res):
    """P&L by instrument: what each one contributed, plus interest on cash.
    The rows add up to the account's total P&L."""
    cash0 = res.config.cash
    years = max(len(res.dates) / res_ann(res), 1e-9)
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

def monte_carlo(returns_, n_paths=500, block=20, seed=7, points=120, ann=None, rfd=0.0):
    """Circular block bootstrap of daily returns.

    Resamples blocks of `block` consecutive days (keeping short-term patterns
    such as volatility clustering) to build n_paths alternative histories of
    the same length. Shows how much of a result could be luck of the ordering,
    and gives bootstrap 95% confidence intervals (2.5th to 97.5th percentiles)
    for the Sharpe ratio, volatility, CAGR and maximum drawdown.
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
    years = T / (ann or TRADING_DAYS)
    finals, cagrs, mdds, sharpes, vols = [], [], [], [], []
    A = ann or TRADING_DAYS
    for _ in range(n_paths):
        eq = peak = 1.0
        mdd = 0.0
        t = ci = 0
        s1 = s2 = 0.0
        while t < T:
            st = rng.randrange(T)
            for j in range(block):
                if t >= T:
                    break
                x = R[(st + j) % T]
                s1 += x
                s2 += x * x
                eq *= 1.0 + x
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
        var = (s2 - s1 * s1 / T) / (T - 1)
        sd = math.sqrt(var) if var > 0 else 0.0
        vols.append(sd * math.sqrt(A))
        sharpes.append((s1 / T - rfd) / sd * math.sqrt(A) if sd > 0 else NAN)
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
        "ci95": {k: [percentile(v, 2.5), percentile(v, 97.5)] for k, v in
                 (("sharpe", [x for x in sharpes if not isnan(x)]), ("volatility", vols), ("cagr", cagrs), ("max_drawdown", mdds))},
        "prob_sharpe_negative": sum(1 for x in sharpes if x < 0) / n_paths,
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
                                    trade_ret="ret_notional" if res.portfolio else "ret", ann=res_ann(res)), **exposure_stats(res, a, b)),
        "bench": performance(d, be, rf, ann=res_ann(res)),
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


SURFACE_STATS = {"total_return": "Total return", "cagr": "CAGR", "volatility": "Volatility", "sharpe": "Sharpe ratio",
                 "sortino": "Sortino ratio", "max_drawdown": "Max drawdown", "calmar": "Calmar ratio"}


def surface_grid(dates, step="quarter", start=None, end=None):
    """Window edges for a start/end surface: the first bar of every month, quarter or year
    from start to end, plus the last bar."""
    s, e = _resolve_range(dates, start, end)
    key = {"month": lambda d: d[:7], "quarter": lambda d: (d[:4], (int(d[5:7]) - 1) // 3),
           "year": lambda d: d[:4]}[step]
    grid, last = [], None
    for d in dates[s:e + 1]:
        k = key(d)
        if k != last:
            grid.append(d)
            last = k
    if grid[-1] != dates[e]:
        grid.append(dates[e])
    return grid


def prefix_stat(dates, equity, stat, ends, rf=0.0, ann=None):
    """`stat` of equity[0..k] for each bar index k in ends (ascending), exactly as
    performance() computes it on that slice, in one pass over the curve."""
    P = ann or TRADING_DAYS
    rfd = (1 + rf) ** (1 / P) - 1
    d0 = _dt.date.fromisoformat(dates[0][:10])
    out, ei = [], 0
    n = 0
    mu = m2 = down = 0.0          # Welford mean & sum of squared deviations of the returns; downside sum
    peak, mdd = -math.inf, 0.0
    for k, v in enumerate(equity):
        if k:
            r = v / equity[k - 1] - 1 if equity[k - 1] > 0 else 0.0
            n += 1
            d = r - mu
            mu += d / n
            m2 += d * (r - mu)
            down += min(r - rfd, 0.0) ** 2
        if v > peak:
            peak = v
        mdd = min(mdd, v / peak - 1 if peak > 0 else 0.0)
        while ei < len(ends) and ends[ei] == k:
            ei += 1
            years = max((_dt.date.fromisoformat(dates[k][:10]) - d0).days / 365.25, 1 / 365.25)
            cagr = (v / equity[0]) ** (1 / years) - 1 if v > 0 else -1.0
            sd = math.sqrt(m2 / (n - 1)) if n >= 2 else NAN
            if stat == "total_return":
                x = v / equity[0] - 1
            elif stat == "cagr":
                x = cagr
            elif stat == "volatility":
                x = sd * math.sqrt(P)
            elif stat == "sharpe":
                x = (mu - rfd) / sd * math.sqrt(P) if n >= 2 and sd > 0 else NAN
            elif stat == "sortino":
                dsd = math.sqrt(down / n) if n else 0.0
                x = (mu - rfd) / dsd * math.sqrt(P) if n >= 2 and dsd > 0 else NAN
            elif stat == "max_drawdown":
                x = mdd
            elif stat == "calmar":
                x = cagr / abs(mdd) if mdd < 0 else NAN
            else:
                raise ValueError("unknown statistic %r (use %s)" % (stat, ", ".join(SURFACE_STATS)))
            out.append(x)
    return out


def surface_row(make_strategy, sets, cfg, start, ends, stat, bench=None, portfolio=False, rates=None, signal_key=None):
    """One row of the start/end surfaces: the strategy backtested from `start` & the
    benchmark held from `start`, scored by `stat` on every window [start, e] for e in ends.

    A backtest from s to e is the backtest from s to the last end cut at e: the engine
    only ever uses data up to each bar, so one run per start gives the whole row."""
    run = backtest_portfolio if portfolio else backtest
    res = run(make_strategy(), sets if portfolio else sets[0], cfg, start, ends[-1], bench, rates=rates, signal_key=signal_key)
    import bisect
    idx = [bisect.bisect_right(res.dates, e) - 1 for e in ends]
    ok = [i for i, k in enumerate(idx) if k > 0]
    A, rf = res_ann(res), res.config.risk_free
    ks = [idx[i] for i in ok]
    S, B = prefix_stat(res.dates, res.equity, stat, ks, rf, A), prefix_stat(res.dates, res.bench_equity, stat, ks, rf, A)
    row_s, row_b = [None] * len(ends), [None] * len(ends)
    for j, i in enumerate(ok):
        row_s[i], row_b[i] = S[j], B[j]
    return {"start": res.dates[0], "requested": start, "S": row_s, "B": row_b, "ends": [res.dates[k] if k >= 0 else None for k in idx]}


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
    ann = math.sqrt(res_ann(res))
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


# ---- overfitting controls -----------------------------------------------------

def expected_max_sharpe(trials, sr_var):
    """Expected highest Sharpe ratio among `trials` strategies with no skill whose
    Sharpe ratios vary with variance sr_var (same units as the result)."""
    if trials <= 1 or sr_var <= 0:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(sr_var) * ((1 - g) * norm_ppf(1 - 1.0 / trials) + g * norm_ppf(1 - 1.0 / (trials * math.e)))


def deflated_sharpe(r, trials, sr_var_annual=None, rfd=0.0, ann=None):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) of daily returns r.

    The probability that the true Sharpe ratio beats the best one expected from
    `trials` attempts with no skill. sr_var_annual is the variance of the annualized
    Sharpe ratios across those trials; without it, the Sharpe estimate's own sampling
    variance is used. Returns {'dsr', 'sr_hurdle' (annualized), 'trials'}.
    """
    n = len(r)
    if n < 4:
        return {}
    mu, sd = mean(r), stdev(r)
    g3, g4 = skewness(r), kurtosis(r)
    if not sd > 0 or isnan(g3) or isnan(g4):
        return {}
    sr = (mu - rfd) / sd
    spread = 1 - g3 * sr + (g4 + 2) / 4 * sr * sr
    if spread <= 0:
        return {}
    trials = max(1, int(trials))
    var_d = sr_var_annual / (ann or TRADING_DAYS) if sr_var_annual else spread / (n - 1)
    sr0 = expected_max_sharpe(trials, var_d)
    return {"dsr": norm_cdf((sr - sr0) * math.sqrt(n - 1) / math.sqrt(spread)),
            "sr_hurdle": sr0 * math.sqrt((ann or TRADING_DAYS)), "trials": trials}


def block_stats(r, n_blocks):
    """Split daily returns into n_blocks consecutive blocks: [(sum, sum of squares, count)]."""
    n = len(r)
    cuts = [int(round(j * n / n_blocks)) for j in range(n_blocks + 1)]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        part = r[a:b]
        out.append((sum(part), sum(x * x for x in part), len(part)))
    return out


def pbo_cscv(blocks, labels=None, ann=None):
    """Probability of Backtest Overfitting by combinatorially symmetric cross-validation
    (Bailey, Borwein, Lopez de Prado & Zhu, 2015).

    blocks[c] holds block_stats() of configuration c (every configuration on the same
    dates and blocks). For every way to pick half the blocks as in-sample, the
    configuration with the best in-sample Sharpe is ranked out of sample. PBO is the
    share of splits where it lands at or below the median.
    """
    from itertools import combinations
    n_cfg = len(blocks)
    if n_cfg < 2:
        raise ValueError("PBO needs at least 2 configurations.")
    S = len(blocks[0])
    if S < 4 or S % 2 or any(len(b) != S for b in blocks):
        raise ValueError("PBO needs an even number of blocks (at least 4), the same for every configuration.")

    def sharpe(c, idx):
        s = ss = cnt = 0.0
        for i in idx:
            s += blocks[c][i][0]
            ss += blocks[c][i][1]
            cnt += blocks[c][i][2]
        var = (ss - s * s / cnt) / (cnt - 1) if cnt > 1 else 0.0
        return s / cnt / math.sqrt(var) * math.sqrt(ann or TRADING_DAYS) if var > 1e-18 else 0.0

    logits, is_best, oos_best, chosen = [], [], [], {}
    for comb in combinations(range(S), S // 2):
        oos = [i for i in range(S) if i not in comb]
        ins = [sharpe(c, comb) for c in range(n_cfg)]
        best = max(range(n_cfg), key=lambda c: (ins[c], -c))
        outs = [sharpe(c, oos) for c in range(n_cfg)]
        below = sum(1 for c in range(n_cfg) if outs[c] < outs[best])
        ties = sum(1 for c in range(n_cfg) if outs[c] == outs[best]) - 1
        w = (below + 0.5 * ties + 1) / (n_cfg + 1)       # relative rank of the in-sample winner
        logits.append(math.log(w / (1 - w)))
        is_best.append(ins[best])
        oos_best.append(outs[best])
        chosen[best] = chosen.get(best, 0) + 1
    m = len(logits)
    slope = covariance(is_best, oos_best) / stdev(is_best) ** 2 if m > 2 and stdev(is_best) > 0 else NAN
    top = sorted(chosen.items(), key=lambda kv: -kv[1])[:5]
    return {
        "pbo": sum(1 for x in logits if x <= 0) / m, "splits": m, "blocks": S, "configs": n_cfg,
        "logits": histogram([logits], 20), "prob_oos_loss": sum(1 for x in oos_best if x < 0) / m,
        "is_sharpe": mean(is_best), "oos_sharpe": mean(oos_best), "degradation_slope": slope,
        "pairs": [[a, b] for a, b in zip(is_best, oos_best)][:400],
        "most_chosen": [{"config": c, "label": labels[c] if labels else c, "share": k / m} for c, k in top],
    }


RANK_LOWER_BETTER = {"volatility", "downside_dev", "var_95", "cvar_95", "var_99", "cvar_99", "ulcer_index", "pain_index",
                     "max_dd_days", "tracking_error", "turnover", "costs_pct"}


def rank_strategies(runs, stat="sharpe", rf=0.0):
    """Rank finished runs by one statistic and say how much the winner owes to luck.

    runs: [{id, name, group?, metrics, returns?, dates?, blocks?, ann?}] or {id, name, error}.
    Ranks are 1 = best (ties share a rank; runs without the statistic come last, unranked).
    selection: the top run's Deflated Sharpe with trials = the number of runs ranked and the
    variance of their Sharpe ratios, and the PBO of picking by Sharpe among the runs on
    identical dates (the largest such group)."""
    lower = stat in RANK_LOWER_BETTER
    ok = [r for r in runs if not r.get("error")]
    failed = [{"id": r["id"], "name": r.get("name", r["id"]), "error": r["error"]} for r in runs if r.get("error")]
    val = lambda r: r["metrics"].get(stat)
    scored = [r for r in ok if val(r) is not None and not isnan(val(r))]
    unscored = [r for r in ok if r not in scored]
    scored.sort(key=lambda r: (val(r) if lower else -val(r), r["name"]))
    rows, prev, rank = [], None, 0
    for k, r in enumerate(scored):
        if val(r) != prev:
            rank, prev = k + 1, val(r)
        rows.append({"id": r["id"], "name": r.get("name", r["id"]), "group": r.get("group"), "rank": rank, "value": val(r), "metrics": r["metrics"]})
    rows += [{"id": r["id"], "name": r.get("name", r["id"]), "group": r.get("group"), "rank": None, "value": None, "metrics": r["metrics"]} for r in unscored]
    sel = {"trials": len(scored)}
    sharpes = [r["metrics"].get("sharpe") for r in scored]
    sharpes = [s for s in sharpes if s is not None and not isnan(s)]
    if scored and scored[0].get("returns") and len(sharpes) >= 2:
        top = scored[0]
        A = top.get("ann") or TRADING_DAYS
        rfd = (1 + rf) ** (1 / A) - 1
        d = deflated_sharpe(top["returns"], len(scored), stdev(sharpes) ** 2, rfd, A)
        if d:
            sel.update(best=top["id"], dsr=d["dsr"], sr_hurdle=d["sr_hurdle"], sharpe_variance=stdev(sharpes) ** 2)
    groups = {}
    for r in scored:
        if r.get("blocks") and r.get("dates"):
            groups.setdefault((r["dates"][0], r["dates"][-1], len(r["dates"])), []).append(r)
    if groups:
        same = max(groups.values(), key=len)
        if len(same) >= 2:
            try:
                p = pbo_cscv([r["blocks"] for r in same], [r["id"] for r in same], same[0].get("ann"))
                sel.update(pbo=p["pbo"], pbo_configs=p["configs"], pbo_oos_sharpe=p["oos_sharpe"], pbo_is_sharpe=p["is_sharpe"],
                           pbo_ids=[r["id"] for r in same])
            except ValueError:
                pass
    return {"stat": stat, "lower_better": lower, "rows": rows, "failed": failed, "selection": sel}


def score_returns(r, metric="sharpe", days=None, ann=None):
    """A score for a slice of daily returns: sharpe, sortino, cagr, total_return or calmar."""
    if len(r) < 2:
        return NAN
    if metric in ("sharpe", "sortino"):
        mu = mean(r)
        sd = stdev(r) if metric == "sharpe" else math.sqrt(mean([min(x, 0.0) ** 2 for x in r]))
        return mu / sd * math.sqrt((ann or TRADING_DAYS)) if sd > 0 else NAN
    eq, peak, mdd = 1.0, 1.0, 0.0
    for x in r:
        eq *= 1 + x
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    if metric == "total_return":
        return eq - 1
    years = (days if days else len(r)) / (365.25 if days else (ann or TRADING_DAYS))
    cagr = eq ** (1 / years) - 1 if eq > 0 and years > 0 else -1.0
    if metric == "cagr":
        return cagr
    if metric == "calmar":
        return cagr / abs(mdd) if mdd < 0 else NAN
    raise ValueError("unknown score %r (use sharpe, sortino, cagr, total_return or calmar)" % metric)


def walk_forward_optimization(dates, returns, labels, train, test, anchored=False, metric="sharpe",
                              cash=10000.0, rf=0.0, bench_returns=None):
    """Re-optimize on a moving training window and keep only what comes after it.

    returns[c] are the daily returns of configuration c (returns[c][i] is the return
    from dates[i] to dates[i + 1]). Every `test` bars, the configuration with the best
    `metric` over the previous `train` bars (or everything so far, if anchored) is
    chosen and held for the next `test` bars. The stitched out-of-sample returns are
    the honest result. Each configuration runs continuously, so switching between them
    at a fold boundary carries no extra trading cost: a mild simplification.
    """
    n_cfg, T = len(returns), len(returns[0])
    train, test = int(train), int(test)
    if n_cfg < 1 or T < train + test or train < 20 or test < 5:
        raise ValueError("Need at least %d bars for a %d-bar training and %d-bar test window." % (train + test + 1, train, test))
    folds, stitched, chosen_is, chosen_oos = [], [], [], []
    t0 = train
    while t0 < T:
        a = 0 if anchored else t0 - train
        b = min(T, t0 + test)
        scores = [score_returns(returns[c][a:t0], metric) for c in range(n_cfg)]
        best = max(range(n_cfg), key=lambda c: (-math.inf if isnan(scores[c]) else scores[c], -c))
        oos = returns[best][t0:b]
        stitched += oos
        folds.append({"train_start": dates[a], "train_end": dates[t0], "test_start": dates[t0], "test_end": dates[b],
                      "chosen": labels[best] if labels else best, "config": best,
                      "is_score": scores[best], "oos_score": score_returns(oos, metric)})
        chosen_is.append(score_returns(returns[best][a:t0], "cagr"))
        chosen_oos.append(score_returns(oos, "cagr"))
        t0 = b
    start = train
    eq = [cash]
    for x in stitched:
        eq.append(eq[-1] * (1 + x))
    d = dates[start:start + len(eq)]
    full = [score_returns(rr, metric) for rr in returns]
    hind = max(range(n_cfg), key=lambda c: (-math.inf if isnan(full[c]) else full[c], -c))
    hind_eq = [cash]
    for x in returns[hind][start:start + len(stitched)]:
        hind_eq.append(hind_eq[-1] * (1 + x))
    bench_eq = None
    if bench_returns is not None:
        bench_eq = [cash]
        for x in bench_returns[start:start + len(stitched)]:
            bench_eq.append(bench_eq[-1] * (1 + x))
    changes = sum(1 for f, g in zip(folds, folds[1:]) if f["config"] != g["config"])
    is_mean, oos_mean = mean(_finite(chosen_is)), mean(_finite(chosen_oos))
    return {
        "folds": folds, "dates": d, "equity": eq, "hindsight_equity": hind_eq, "bench_equity": bench_eq,
        "hindsight": labels[hind] if labels else hind, "metric": metric,
        "metrics": performance(d, eq, rf), "hindsight_metrics": performance(d, hind_eq, rf),
        "bench_metrics": performance(d, bench_eq, rf) if bench_eq else None,
        "efficiency": oos_mean / is_mean if is_mean and not isnan(is_mean) and is_mean > 0 else NAN,
        "param_changes": changes, "anchored": anchored, "train": train, "test": test,
    }


def purged_kfold(n, k=5, horizon=1, embargo=0):
    """Purged and embargoed k-fold splits for labels that look `horizon` bars ahead
    (Lopez de Prado, 2018). Yields (train_idx, test_idx) with contiguous test folds;
    training rows whose label window overlaps the test fold are purged, and the
    `embargo` rows right after it are dropped too. Use it to measure a model's skill:
    these folds train on the future as well, so never trade their predictions."""
    k = max(2, min(int(k), n))
    cuts = [int(round(j * n / k)) for j in range(k + 1)]
    for a, b in zip(cuts, cuts[1:]):
        lo, hi = a - horizon, b + embargo          # rows j in [lo, b) overlap; [b, hi) are embargoed
        train = [j for j in range(n) if j < lo or j >= hi]
        yield train, list(range(a, b))


def cpcv_splits(n, n_groups=6, n_test=2, horizon=1, embargo=0):
    """Combinatorial purged cross-validation: every choice of n_test groups out of
    n_groups is a test set, purged and embargoed like purged_kfold. Returns a list of
    (train_idx, test_idx, test_groups); each group is tested C(n_groups-1, n_test-1)
    times, which makes that many out-of-sample backtest paths."""
    from itertools import combinations
    cuts = [int(round(j * n / n_groups)) for j in range(n_groups + 1)]
    out = []
    for groups in combinations(range(n_groups), n_test):
        test, banned = [], set()
        for g in groups:
            a, b = cuts[g], cuts[g + 1]
            test += range(a, b)
            banned.update(range(max(0, a - horizon), min(n, b + embargo)))
        out.append(([j for j in range(n) if j not in banned], test, list(groups)))
    return out


def cv_score(X, y, fit_predict, score, k=5, horizon=1, embargo=0):
    """Average out-of-fold score of a model under purged k-fold cross-validation.
    score(y_true, y_pred) -> number. Rows with NaN features or labels are skipped."""
    ok = [all(v == v for v in row) and not isnan(t) for row, t in zip(X, y)]
    take_x = (lambda idx: X[idx]) if hasattr(X, "shape") else (lambda idx: [X[j] for j in idx])
    take_y = (lambda idx: y[idx]) if hasattr(y, "shape") else (lambda idx: [y[j] for j in idx])
    scores = []
    for train, test in purged_kfold(len(y), k, horizon, embargo):
        train = [j for j in train if ok[j]]
        test = [j for j in test if ok[j]]
        if train and test:
            pred = fit_predict(take_x(train), take_y(train), take_x(test))
            scores.append(score(take_y(test), pred))
    return {"mean": mean(scores), "folds": scores}


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
        quiet, sys.stdout = sys.stdout, io.StringIO()    # the re-runs' print() output isn't wanted
        try:
            sub_raw = strategy.signals(PanelView(ds, k + 1) if portfolio else DataView(ds, k + 1))
        finally:
            sys.stdout = quiet
        if portfolio:
            _, sub = signal_rows(sub_raw, ds, True, k + 1)
            name = lambda r: None if r is None else {ds.tickers[x]: v for x, v in r.items()}  # noqa: E731
            a, b = sub[k], full_signals[k]
        else:
            sub = list(sub_raw)
            a, b = sub[k] if len(sub) > k else None, full_signals[k]
        if not _same_weight(a, b):
            bad.append({"date": ds.dates[k], "with_future": name(b) if portfolio else b,
                        "without_future": name(a) if portfolio else a})
    bad.sort(key=lambda v: v["date"])
    return {"checked": done, "planned": len(picks), "violations": bad}


def instrument_stats(datasets, start=None, end=None, rf=0.0, total_return=True):
    """Per-instrument buy-and-hold stats plus a correlation matrix of daily returns
    on the dates all instruments share; dividends reinvested unless total_return is False."""
    rows = []
    series = {ds.ticker: ds.total_return_index() if total_return else ds.close for ds in datasets}
    for ds in datasets:
        try:
            s, e = _resolve_range(ds.dates, start, end, "%s data" % ds.ticker)
        except ValueError:
            continue
        eq = series[ds.ticker][s:e + 1]
        m = performance(ds.dates[s:e + 1], eq, rf)
        rows.append({"ticker": ds.ticker, "start": ds.dates[s], "end": ds.dates[e], "bars": e - s + 1,
                     "last": ds.close[e], "total_return": m["total_return"], "cagr": m["cagr"],
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
            closes = [series[ds.ticker][ds.index[d]] for d in common]
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
# 7b. Developer tools: lint, strategy tests, the console and editor completions
# =============================================================================

# pyflakes message classes that are mistakes rather than style; everything else is a warning.
LINT_ERRORS = {"UndefinedName", "UndefinedLocal", "UndefinedExport", "ReturnOutsideFunction",
               "YieldOutsideFunction", "ContinueOutsideLoop", "BreakOutsideLoop", "DuplicateArgument",
               "TwoStarredExpressions", "TooManyExpressionsInStarredAssignment", "DefaultExceptNotLast",
               "ForwardAnnotationSyntaxError", "PercentFormatInvalidFormat", "StringDotFormatInvalidFormat"}
# Attributes set on contexts in __init__ (found from the source when it is available).
_CTX_ATTRS = {"state", "params", "start_index", "ticker", "tickers", "open", "high", "low", "close", "volume", "options"}
_MEMBERS = {}


def _members(cls):
    """Public attribute names of a class and its instances."""
    if cls not in _MEMBERS:
        import inspect
        import re
        names = {n for n in dir(cls) if not n.startswith("_")} | (_CTX_ATTRS if "Context" in cls.__name__ else set())
        for klass in cls.__mro__:
            try:
                names |= set(re.findall(r"self\.([a-z]\w*)\s*=", inspect.getsource(klass)))
            except (OSError, TypeError):
                pass
        _MEMBERS[cls] = names
    return _MEMBERS[cls]


def _lint_builtins():
    return set(_namespace()) - {"__name__", "__builtins__"}


def lint(code, kind="strategy", name=None):
    """Static checks on strategy or module code: [{line, col, severity, message, code}].

    Uses pyflakes when it is installed (the web page loads a bundled copy), otherwise a
    smaller built-in check for syntax errors, undefined names and unused imports.
    Strategy code is also checked for a missing on_bar/signals and for ctx attributes
    that don't exist (ctx.smaa(20) -> "did you mean sma?")."""
    import ast
    fn = STRATEGY_FILENAME if kind == "strategy" else module_filename(name or "module")
    try:
        tree = ast.parse(code, fn)
    except SyntaxError as e:
        return [{"line": e.lineno or 1, "col": e.offset or 1, "severity": "error",
                 "message": "SyntaxError: %s" % e.msg, "code": "SyntaxError"}]
    try:
        from pyflakes import checker
    except ImportError:
        checker = None
    out = _pyflakes(tree, fn, checker) if checker else _basic_lint(tree)
    if kind == "strategy":
        out += _strategy_lint(tree)
    out.sort(key=lambda m: (m["line"], m["col"]))
    return out


def _pyflakes(tree, fn, checker):
    w = checker.Checker(tree, filename=fn, builtins=_lint_builtins())
    out = []
    for m in w.messages:
        kind = type(m).__name__
        out.append({"line": m.lineno, "col": (getattr(m, "col", 0) or 0) + 1,
                    "severity": "error" if kind in LINT_ERRORS else "warning",
                    "message": m.message % m.message_args, "code": kind})
    return out


def _basic_lint(tree):
    """Undefined names and unused imports, without pyflakes. Scope-blind: a name defined
    anywhere in the file counts as defined everywhere, so it misses some mistakes but
    doesn't invent any."""
    import ast
    import builtins
    known = set(dir(builtins)) | _lint_builtins() | {"__file__", "__name__", "__doc__"}
    defined, imported, loaded, star = set(), {}, set(), False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (loaded if isinstance(node.ctx, ast.Load) else defined).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name == "*":
                    star = True
                    continue
                nm = a.asname or a.name.split(".")[0]
                imported.setdefault(nm, (node.lineno, node.col_offset))
                defined.add(nm)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            defined.update(node.names)
        elif type(node).__name__ in ("MatchAs", "MatchStar") and getattr(node, "name", None):
            defined.add(node.name)
        elif type(node).__name__ == "MatchMapping" and getattr(node, "rest", None):
            defined.add(node.rest)
    out = []
    if not star:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in defined | known:
                out.append({"line": node.lineno, "col": node.col_offset + 1, "severity": "error",
                            "message": "undefined name %r" % node.id, "code": "UndefinedName"})
    for nm, (line, col) in imported.items():
        if nm not in loaded:
            out.append({"line": line, "col": col + 1, "severity": "warning",
                        "message": "%r imported but unused" % nm, "code": "UnusedImport"})
    return out


def _strategy_lint(tree):
    import ast
    import difflib
    out = []
    top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assigned = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    if not ({"on_bar", "signals"} & (set(top) | assigned)):
        out.append({"line": 1, "col": 1, "severity": "warning", "code": "NoEntryPoint",
                    "message": "Define on_bar(ctx) or signals(data): without one there is nothing to backtest."})
    ctx_names = _members(Context) | _members(PortfolioContext)
    asset_names = _members(AssetContext)
    set_on_ctx = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not fn.args.args or fn.args.args[0].arg != "ctx":
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load) or node.attr in set_on_ctx:
                continue
            v = node.value
            if isinstance(v, ast.Name) and v.id == "ctx":
                names, what = ctx_names, "ctx"
            elif isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name) and v.value.id == "ctx":
                names, what = asset_names, "ctx[...]"
            else:
                continue
            if node.attr not in names and not node.attr.startswith("_"):
                close = difflib.get_close_matches(node.attr, sorted(names), 1, 0.7)
                out.append({"line": node.lineno, "col": node.col_offset + len(what) + 2, "severity": "error",
                            "code": "UnknownAttribute",
                            "message": "%s has no attribute %r%s" % (what, node.attr, "; did you mean %s?" % close[0] if close else "")})
    return out


# ---- strategy tests ----------------------------------------------------------

def make_data(close, open=None, high=None, low=None, volume=None, start="2020-01-02", ticker="TEST"):
    """A DataSet for tests: these closes (and optional other columns) on consecutive weekdays from start."""
    d, dates = _dt.date.fromisoformat(start), []
    while len(dates) < len(close):
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += _dt.timedelta(days=1)
    return DataSet(ticker, dates, list(close), open, high, low, volume)


def synthetic_data(n=500, seed=0, drift=0.0003, vol=0.01, price=100.0, start="2020-01-02", ticker="TEST"):
    """A random-walk DataSet for tests: Gaussian daily returns, with opens, highs, lows and volume."""
    rng = random.Random(seed)
    o, h, lo, c = [], [], [], []
    px = float(price)
    for _ in range(n):
        op = px * (1 + rng.gauss(0, vol / 4))
        px = op * (1 + drift + rng.gauss(0, vol))
        o.append(op)
        c.append(px)
        h.append(max(op, px) * (1 + abs(rng.gauss(0, vol / 3))))
        lo.append(min(op, px) * (1 - abs(rng.gauss(0, vol / 3))))
    return make_data(c, o, h, lo, [1e6] * n, start, ticker)


def assert_close(actual, expected, tol=1e-9, msg=""):
    """Assert two numbers (or equal-length lists) agree within tol, relative to their size."""
    a = list(actual) if isinstance(actual, (list, tuple)) else [actual]
    b = list(expected) if isinstance(expected, (list, tuple)) else [expected]
    if len(a) != len(b):
        raise AssertionError("%slengths differ: %d vs %d" % (msg + ": " if msg else "", len(a), len(b)))
    for i, (x, y) in enumerate(zip(a, b)):
        if isnan(x) and isnan(y):
            continue
        if isnan(x) or isnan(y) or abs(x - y) > tol * max(1.0, abs(x), abs(y)):
            raise AssertionError("%s%r != %r%s (tolerance %g)" % (msg + ": " if msg else "", x, y,
                                                                 " at index %d" % i if len(a) > 1 else "", tol))


def _test_helpers(code, kind):
    def run_strategy(data, params=None, code=code if kind == "strategy" else None, start=None, end=None, **settings):
        """Backtest this file's strategy (or code=...) on a DataSet, or a list of them for a portfolio."""
        if code is None:
            raise ValueError("run_strategy in a module needs code=..., the strategy source to test.")
        strat = Strategy.from_code(code, params)
        cfg = Config(**settings)
        if isinstance(data, (list, tuple)):
            return backtest_portfolio(strat, list(data), cfg, start, end)
        return backtest(strat, data, cfg, start, end)

    return {"make_data": make_data, "synthetic_data": synthetic_data, "assert_close": assert_close,
            "run_strategy": run_strategy, "metrics": lambda res: summarize(res)[0],
            "DataSet": DataSet, "Config": Config, "DATA": DATA}


def _frames_in(exc, filenames):
    return [f for f in traceback.extract_tb(exc.__traceback__)
            if f.filename in filenames or _module_of(f.filename) or f.filename == STRATEGY_FILENAME]


def run_tests(code, kind="strategy", name=None, only=None):
    """Run the test_* functions in strategy or module code, like a tiny pytest.

    Tests take no arguments and use plain assert. Besides the strategy helpers they get
    make_data, synthetic_data, assert_close, metrics(res) and run_strategy(data, params,
    **settings), which backtests this file's strategy. Returns {tests: [{name, ok,
    message, line, trace, output, seconds}], passed, failed, error}."""
    import time
    fn = STRATEGY_FILENAME if kind == "strategy" else module_filename(name or "module")
    linecache.cache[fn] = (len(code), None, code.splitlines(True), fn)
    _forget_modules()
    ns = _namespace()
    ns["__name__"] = "strategy" if kind == "strategy" else (name or "module")
    ns.update(_test_helpers(code, kind))
    old_out, old_err = sys.stdout, sys.stderr
    buf = sys.stdout = sys.stderr = io.StringIO()
    try:
        exec(compile(code, fn, "exec"), ns)
    except Exception as e:  # noqa: BLE001
        return {"tests": [], "passed": 0, "failed": 0, "error": format_error(e)}
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    tests = [(k, v) for k, v in ns.items() if k.startswith("test") and callable(v) and (only is None or k == only)]
    results = []
    for k, f in tests:
        buf = io.StringIO()
        sys.stdout = sys.stderr = buf
        t0 = time.perf_counter()
        try:
            f()
            ok, msg, line, trace = True, "", getattr(getattr(f, "__code__", None), "co_firstlineno", None), ""
        except Exception as e:  # noqa: BLE001
            frames = _frames_in(e, {fn})
            here = [fr for fr in frames if fr.filename == fn]
            line = here[-1].lineno if here else getattr(f.__code__, "co_firstlineno", None)
            if isinstance(e, AssertionError):
                msg = str(e) or "assert failed: %s" % ((here[-1].line or "").strip() if here else "")
            else:
                msg = "%s: %s" % (type(e).__name__, e)
            trace = "\n".join("  %sline %d, in %s\n    %s" % (("%s.py, " % _module_of(fr.filename)) if _module_of(fr.filename) and fr.filename != fn else "",
                                                          fr.lineno, fr.name, (fr.line or "").strip()) for fr in frames)
            ok = False
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        results.append({"name": k, "ok": ok, "message": msg, "line": line, "trace": trace,
                        "output": _log_preview(buf.getvalue()), "seconds": time.perf_counter() - t0})
    passed = sum(1 for r in results if r["ok"])
    return {"tests": results, "passed": passed, "failed": len(results) - passed, "error": None}


# ---- the console --------------------------------------------------------------

CONSOLE = {"ns": None, "view": None, "n": 0}


def console_run(src, tickers=(), reset=False):
    """Run one console cell. Variables persist between cells until reset. Predefined:
    data (the selected instrument(s), as signals(data) sees them), result (the latest
    backtest), DATA, engine, summarize, backtest, the test helpers and the strategy
    helpers. A final expression is printed, as in the Python prompt, and kept in _."""
    import ast
    if reset or CONSOLE["ns"] is None:
        ns = _namespace()
        ns.update(_test_helpers(None, "module"))
        ns.update({"__name__": "__console__", "engine": sys.modules[__name__], "summarize": summarize,
                   "backtest": backtest, "backtest_portfolio": backtest_portfolio, "Strategy": Strategy})
        CONSOLE.update(ns=ns, view=None)
    ns = CONSOLE["ns"]
    tickers = tuple(t for t in tickers if t in DATA)
    if tickers != CONSOLE["view"]:
        if len(tickers) == 1:
            ns["data"] = DataView(DATA[tickers[0]])
        elif tickers:
            ns["data"] = PanelView(shared_panel([DATA[t] for t in tickers]))
        CONSOLE["view"] = tickers
    ns["result"] = LAST_RUN.get("res")
    CONSOLE["n"] += 1
    fn = "<console-%d>" % CONSOLE["n"]
    linecache.cache[fn] = (len(src), None, src.splitlines(True), fn)
    old_out, old_err = sys.stdout, sys.stderr
    buf = sys.stdout = sys.stderr = io.StringIO()
    error = None
    try:
        tree = ast.parse(src, fn, "exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, fn, "exec"), ns)
        if last is not None:
            value = eval(compile(last, fn, "eval"), ns)
            if value is not None:
                ns["_"] = value
                text = repr(value)
                print(text if len(text) <= 100000 else text[:100000] + " ... (%s characters)" % format(len(text), ","))
    except SyntaxError as e:
        error = "SyntaxError: %s (line %s)\n    %s" % (e.msg, e.lineno, (e.text or "").strip())
    except (Exception, SystemExit) as e:  # noqa: BLE001
        frames = [f for f in traceback.extract_tb(e.__traceback__)
                  if f.filename.startswith("<console") or f.filename == STRATEGY_FILENAME or _module_of(f.filename)]
        error = "".join("  %s, line %d, in %s\n    %s\n" % (f.filename.strip("<>"), f.lineno, f.name, (f.line or "").strip())
                        for f in frames) + "%s: %s" % (type(e).__name__, e)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    base = set(_namespace()) | set(_test_helpers(None, "module")) | {"engine", "summarize", "backtest", "backtest_portfolio",
                                                                    "Strategy", "data", "result", "_"}
    names = sorted(k for k in ns if not k.startswith("_") and k not in base)
    return {"output": _log_preview(buf.getvalue()), "error": error,
            "vars": [{"name": k, "type": type(ns[k]).__name__} for k in names][:200]}


# ---- completions ----------------------------------------------------------------

_ATTR_DOCS = {
    "state": "A dict kept from one bar to the next: remember anything here.",
    "params": "This strategy's PARAMS (with any values set by a sweep).",
    "start_index": "Index of the first bar of the backtest window.",
    "ticker": "The instrument's ticker.", "tickers": "The portfolio's tickers.",
    "open": "Opens up to today: open[-1] is today's.", "high": "Highs up to today.", "low": "Lows up to today.",
    "close": "Closes up to today: close[-1] is today's, close[-2] yesterday's.", "volume": "Volumes up to today.",
}


def completions():
    """Names the editor can complete after ctx., ctx['X']., qt., data. and at the top level."""
    import inspect

    def doc(o):
        return (inspect.getdoc(o) or "").split("\n\n")[0].replace("\n", " ")

    def sig(f):
        try:
            return str(inspect.signature(f)).replace("(self, ", "(").replace("(self)", "()")
        except (TypeError, ValueError):
            return "(...)"

    def entries(cls):
        out = []
        for n in sorted(_members(cls)):
            raw = next((k.__dict__[n] for k in cls.__mro__ if n in k.__dict__), None)
            if isinstance(raw, property):
                out.append({"name": n, "kind": "property", "doc": doc(raw.fget) or _ATTR_DOCS.get(n, "")})
            elif callable(raw):
                out.append({"name": n, "kind": "method", "sig": sig(raw), "doc": doc(raw)})
            else:
                out.append({"name": n, "kind": "attribute", "doc": _ATTR_DOCS.get(n, "")})
        return out

    def funcs(items):
        return [{"name": n, "kind": "function" if callable(f) else "constant", "sig": sig(f) if callable(f) else "",
                 "doc": doc(f) if callable(f) else ""} for n, f in sorted(items) if not n.startswith("_")]

    glob = {k: v for k, v in _namespace().items() if not k.startswith("_")}
    glob.update({k: v for k, v in _test_helpers(None, "module").items() if k in ("make_data", "synthetic_data", "assert_close")})
    return {"ctx": entries(Context), "portfolio": entries(PortfolioContext), "asset": entries(AssetContext),
            "qt": funcs(vars(qt).items()), "data": entries(DataView), "panel": entries(PanelView),
            "globals": funcs(glob.items())}


# =============================================================================
# 7c. Risk analytics: factor models, rolling beta, risk decomposition, stress
#     tests, portfolio VaR and capacity
# =============================================================================

def _solve(A, B):
    """Solve A X = B for square A (Gauss-Jordan with partial pivoting). B is a list of rows."""
    n = len(A)
    M = [list(A[i]) + list(B[i]) for i in range(n)]
    w = len(M[0])
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-14:
            raise ValueError("the factors are collinear (one is a combination of the others)")
        M[col], M[piv] = M[piv], M[col]
        inv = 1.0 / M[col][col]
        M[col] = [v * inv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col]:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    return [row[n:w] for row in M]


def _inverse(A):
    n = len(A)
    return _solve(A, [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])


def newey_west_lags(n):
    """The usual automatic lag length for Newey-West standard errors: floor(4 (n/100)^(2/9))."""
    return int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def ols(y, X, hac_lags=None):
    """Least squares y = X b + e, where each row of X starts with 1.0 for the intercept.

    Returns coefficients with ordinary standard errors and Newey-West (HAC) ones, which
    stay valid when residuals are autocorrelated or their variance changes over time
    (Bartlett weights, no small-sample correction). p-values are two-sided, from
    Student's t with n - k degrees of freedom."""
    n, k = len(y), len(X[0])
    if n <= k + 1:
        raise ValueError("Need more observations than factors.")
    XtX = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    Xty = [[sum(r[i] * v for r, v in zip(X, y))] for i in range(k)]
    inv = _inverse(XtX)
    b = [sum(inv[i][j] * Xty[j][0] for j in range(k)) for i in range(k)]
    fit = [sum(bi * xi for bi, xi in zip(b, r)) for r in X]
    e = [v - f for v, f in zip(y, fit)]
    sse = sum(x * x for x in e)
    my = sum(y) / n
    sst = sum((v - my) ** 2 for v in y)
    df = n - k
    s2 = sse / df
    se = [math.sqrt(max(0.0, s2 * inv[i][i])) for i in range(k)]
    L = newey_west_lags(n) if hac_lags is None else int(hac_lags)
    S = [[0.0] * k for _ in range(k)]
    for r, et in zip(X, e):
        for i in range(k):
            ri = r[i] * et * et
            row = S[i]
            for j in range(k):
                row[j] += ri * r[j]
    for lag in range(1, L + 1):
        wl = 1.0 - lag / (L + 1.0)
        for t in range(lag, n):
            ee = wl * e[t] * e[t - lag]
            xt, xs = X[t], X[t - lag]
            for i in range(k):
                for j in range(k):
                    S[i][j] += ee * (xt[i] * xs[j] + xs[i] * xt[j])
    tmp = [[sum(inv[i][m] * S[m][j] for m in range(k)) for j in range(k)] for i in range(k)]
    V = [[sum(tmp[i][m] * inv[m][j] for m in range(k)) for j in range(k)] for i in range(k)]
    se_hac = [math.sqrt(max(0.0, V[i][i])) for i in range(k)]
    tval = [bi / s if s > 0 else NAN for bi, s in zip(b, se)]
    thac = [bi / s if s > 0 else NAN for bi, s in zip(b, se_hac)]
    p2 = lambda t: NAN if isnan(t) else 2 * t_sf(abs(t), df)
    r2 = 1 - sse / sst if sst > 0 else NAN
    return {"coef": b, "se": se, "t": tval, "p": [p2(t) for t in tval], "se_hac": se_hac, "t_hac": thac,
            "p_hac": [p2(t) for t in thac], "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / df if not isnan(r2) else NAN,
            "resid": e, "fitted": fit, "n": n, "k": k, "df": df, "hac_lags": L}


def returns_by_date(ds, total_return=True):
    """{date: return} from each bar to the instrument's previous bar."""
    src = ds.total_return_index() if total_return else ds.close
    return {ds.dates[j]: src[j] / src[j - 1] - 1 for j in range(1, ds.n) if src[j - 1] > 0}


FACTORS = {}   # uploaded factor tables: name -> {"dates": [...], "columns": {col: [daily returns]}}


def parse_factor_csv(text, percent=None):
    """A table of daily factor returns, e.g. the Fama-French files from Ken French's data
    library: a date column (YYYYMMDD, YYYY-MM-DD or MM/DD/YYYY) and one column per factor.
    Lines before the header and after the data (notes, annual tables) are skipped. Values
    in percent (as in those files) are detected and divided by 100; pass percent=True or
    False to decide. A column named RF is the daily risk-free rate."""
    rows = [r for r in (line.strip() for line in text.replace("\r", "").split("\n"))]
    def as_date(x):
        x = x.strip().strip('"')
        if len(x) == 8 and x.isdigit():
            return "%s-%s-%s" % (x[:4], x[4:6], x[6:])
        return _iso_date(x)
    header, data = None, []
    for line in rows:
        cells = [c.strip().strip('"') for c in line.split(",")]
        if len(cells) < 2:
            if data:
                break
            continue
        d = as_date(cells[0])
        if d is None:
            if data:
                break                      # the daily table has ended (Ken French files append more)
            if any(c for c in cells[1:]):
                header = cells
            continue
        try:
            vals = [float(c) if c not in ("", "-99.99", "-999") else NAN for c in cells[1:]]
        except ValueError:
            if data:
                break
            continue
        data.append((d, vals))
    if not data:
        raise ValueError("No dated rows found. Expected a date column followed by factor returns.")
    k = len(data[0][1])
    names = [(header[i + 1] if header and i + 1 < len(header) and header[i + 1] else "F%d" % (i + 1)) for i in range(k)]
    data.sort()
    cols = {nm: [row[1][i] if i < len(row[1]) else NAN for row in data] for i, nm in enumerate(names)}
    if percent is None:
        vals = [abs(v) for c in cols.values() for v in c if not isnan(v)]
        percent = bool(vals) and percentile(vals, 50) > 0.02   # daily returns: a median above 2% means percent units
    if percent:
        cols = {nm: [v / 100.0 for v in c] for nm, c in cols.items()}
    return {"dates": [d for d, _ in data], "columns": cols, "percent": bool(percent)}


def _factor_series(spec, rf_daily):
    """(name, {date: return}) for one factor spec: a loaded ticker (excess of rf), 'A-B'
    (the return of A minus that of B), or 'set:column' from an uploaded table."""
    spec = spec.strip()
    if ":" in spec:
        name, col = spec.split(":", 1)
        tab = FACTORS.get(name)
        if tab is None or col not in tab["columns"]:
            raise ValueError("No uploaded factor %s." % spec)
        return col, dict(zip(tab["dates"], tab["columns"][col]))
    if "-" in spec and spec.upper() not in DATA:
        up = spec.upper()
        cuts = [(up[:i].strip(), up[i + 1:].strip()) for i, ch in enumerate(up) if ch == "-"]
        pair = next(((a, b) for a, b in cuts if a in DATA and b in DATA), None)   # tickers may contain '-' too
        if pair is None:
            raise ValueError("%s: use two loaded tickers, e.g. IWM-SPY for small minus large caps." % spec)
        a, b = pair
        ra, rb = returns_by_date(DATA[a]), returns_by_date(DATA[b])
        return "%s-%s" % (a, b), {d: ra[d] - rb[d] for d in ra if d in rb}
    t = spec.upper()
    if t not in DATA:
        raise ValueError("%s isn't loaded; add it on the Data tab to use it as a factor." % t)
    return t, {d: r - rf_daily.get(d, 0.0) for d, r in returns_by_date(DATA[t]).items()}


def factor_regression(res, specs, hac_lags=None):
    """Regress the strategy's daily excess returns on factors (see _factor_series).

    Reports alpha (annualized) and each factor's beta with ordinary and Newey-West
    t-statistics, R-squared, and two decompositions that add up exactly: mean return
    = alpha + sum(beta x mean factor return), and variance = each factor's share
    (beta x cov(factor, fitted)) + the idiosyncratic rest."""
    rf_const = res.config.risk_free / res_ann(res)
    rf = {}
    for spec in specs:
        if ":" in spec:
            tab = FACTORS.get(spec.split(":", 1)[0])
            if tab and "RF" in tab["columns"]:
                rf = dict(zip(tab["dates"], tab["columns"]["RF"]))
    rf_daily = rf or {d: rf_const for d in res.dates}
    named = [_factor_series(sp, rf_daily) for sp in specs if sp.strip()]
    if not named:
        raise ValueError("Pick at least one factor, e.g. SPY.")
    eq, dates = res.equity, res.dates
    rows, y = [], []
    for t in range(1, len(eq)):
        d = dates[t]
        if eq[t - 1] <= 0 or d not in rf_daily:
            continue
        fs = [f.get(d) for _, f in named]
        if any(v is None or isnan(v) for v in fs):
            continue
        y.append(eq[t] / eq[t - 1] - 1 - rf_daily[d])
        rows.append([1.0] + fs)
    if len(y) < 30:
        raise ValueError("Only %d days have the strategy and every factor; need at least 30." % len(y))
    fit = ols(y, rows, hac_lags)
    names = [n for n, _ in named]
    n = len(y)
    out_rows = []
    for i, nm in enumerate(["alpha"] + names):
        scale = res_ann(res) if i == 0 else 1.0
        out_rows.append({"name": nm, "coef": fit["coef"][i] * scale, "se": fit["se"][i] * scale, "t": fit["t"][i],
                         "p": fit["p"][i], "se_hac": fit["se_hac"][i] * scale, "t_hac": fit["t_hac"][i], "p_hac": fit["p_hac"][i]})
    means = [sum(r[i] for r in rows) / n for i in range(1, len(names) + 1)]
    attribution = [{"name": "alpha", "return": fit["coef"][0] * res_ann(res)}]
    attribution += [{"name": nm, "beta": fit["coef"][i + 1], "factor_return": means[i] * res_ann(res),
                     "return": fit["coef"][i + 1] * means[i] * res_ann(res)} for i, nm in enumerate(names)]
    var_y = sum((v - sum(y) / n) ** 2 for v in y) / (n - 1)
    fitted_sys = [f - fit["coef"][0] for f in fit["fitted"]]
    ms = sum(fitted_sys) / n
    variance = []
    for i, nm in enumerate(names):
        col = [r[i + 1] for r in rows]
        mc = sum(col) / n
        cov = sum((a - mc) * (b - ms) for a, b in zip(col, fitted_sys)) / (n - 1)
        variance.append({"name": nm, "share": fit["coef"][i + 1] * cov / var_y if var_y > 0 else NAN})
    variance.append({"name": "idiosyncratic", "share": 1 - sum(v["share"] for v in variance)})
    resid_vol = stdev(fit["resid"]) * math.sqrt(res_ann(res))
    return {"n": n, "factors": names, "rows": out_rows, "r2": fit["r2"], "adj_r2": fit["adj_r2"],
            "resid_vol": resid_vol, "hac_lags": fit["hac_lags"], "df": fit["df"],
            "mean_excess": sum(y) / n * res_ann(res), "attribution": attribution, "variance": variance,
            "rf_source": "uploaded RF column" if rf else "risk-free setting (%.2f%%)" % (100 * res.config.risk_free)}


def rolling_beta(r, b, window=126):
    """Rolling beta and correlation of r against b (same-length daily return lists).
    NaN until window observations are available."""
    n = len(r)
    beta, corr = [NAN] * n, [NAN] * n
    sx = sy = sxx = syy = sxy = 0.0
    for t in range(n):
        x, y = b[t], r[t]
        sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y
        if t >= window:
            x0, y0 = b[t - window], r[t - window]
            sx -= x0; sy -= y0; sxx -= x0 * x0; syy -= y0 * y0; sxy -= x0 * y0
        if t >= window - 1:
            m = float(window)
            vx, vy, cxy = sxx - sx * sx / m, syy - sy * sy / m, sxy - sx * sy / m
            if vx > 1e-18:
                beta[t] = cxy / vx
                if vy > 1e-18:
                    corr[t] = cxy / math.sqrt(vx * vy)
    return beta, corr


def _asset_returns(res, window):
    """Daily total returns of each instrument over the last `window` bars of the backtest
    (bars where it didn't trade count as 0) and their dates."""
    panel, s, n = res.panel, res.start, len(res.equity)
    lo = max(1, n - window)
    tr = [panel.aligned(a, "total_return")[s:s + n] for a in range(panel.k)]
    R = [[(x[t] / x[t - 1] - 1) if not (isnan(x[t]) or isnan(x[t - 1]) or x[t - 1] <= 0) else 0.0 for x in tr]
         for t in range(lo, n)]
    return R, res.dates[lo:n]


def _cov(R):
    n, k = len(R), len(R[0])
    m = [sum(r[i] for r in R) / n for i in range(k)]
    return [[sum((r[i] - m[i]) * (r[j] - m[j]) for r in R) / (n - 1) for j in range(k)] for i in range(k)], m


def current_weights(res):
    """Weights (position value / equity) held at the last close of the backtest."""
    return [res.weights[tk][-1] if res.portfolio else res.exposure[-1] for tk in res.tickers]


def risk_decomposition(res, window=252):
    """Ex-ante risk of the final holdings from the last `window` days of returns: each
    instrument's weight, volatility, marginal and total contribution to portfolio volatility
    (the contributions add up to it) and its share; plus the diversification ratio."""
    R, _ = _asset_returns(res, window)
    if len(R) < 20:
        raise ValueError("Need at least 20 days of returns for a risk decomposition.")
    C, _ = _cov(R)
    k = len(C)
    rows_out = []
    for label, w in (("final", current_weights(res)),
                     ("average", [mean(res.weights[tk]) if res.portfolio else mean(res.exposure) for tk in res.tickers])):
        Cw = [sum(C[i][j] * w[j] for j in range(k)) for i in range(k)]
        var = sum(w[i] * Cw[i] for i in range(k))
        vol = math.sqrt(max(var, 0.0))
        ann = math.sqrt(res_ann(res))
        assets = []
        for i, tk in enumerate(res.tickers):
            sd = math.sqrt(max(C[i][i], 0.0))
            mcr = Cw[i] / vol if vol > 0 else NAN
            assets.append({"ticker": tk, "weight": w[i], "vol": sd * ann, "marginal": mcr * ann if not isnan(mcr) else NAN,
                           "contribution": w[i] * mcr * ann if vol > 0 else 0.0,
                           "share": w[i] * Cw[i] / var if var > 0 else NAN})
        wsd = sum(abs(w[i]) * math.sqrt(max(C[i][i], 0.0)) for i in range(k))
        rows_out.append({"holdings": label, "vol": vol * ann, "assets": assets, "gross": sum(abs(x) for x in w),
                         "net": sum(w), "diversification": wsd / vol if vol > 0 else NAN})
    return {"window": len(R), "final": rows_out[0], "average": rows_out[1]}


STRESS_SCENARIOS = [
    # name, first day, last day (US equity peak-to-trough windows)
    ("Global financial crisis (Lehman to the low)", "2008-09-12", "2009-03-09"),
    ("2010 flash crash & euro crisis", "2010-04-23", "2010-07-02"),
    ("2011 US downgrade", "2011-07-22", "2011-10-03"),
    ("2015 China devaluation", "2015-08-17", "2015-08-25"),
    ("2018 volatility spike (Volmageddon)", "2018-01-26", "2018-02-08"),
    ("Q4 2018 sell-off", "2018-09-20", "2018-12-24"),
    ("COVID-19 crash", "2020-02-19", "2020-03-23"),
    ("2022 rate shock", "2022-01-03", "2022-10-12"),
    ("2023 regional bank stress", "2023-03-08", "2023-03-13"),
    ("Aug 2024 carry-trade unwind", "2024-07-16", "2024-08-05"),
]


def _window_return(ds_dates, values, start, end):
    """Return from the last value before `start` to the last value on or before `end`, or None
    if the series doesn't cover the window."""
    import bisect
    i0 = bisect.bisect_left(ds_dates, start) - 1
    i1 = bisect.bisect_right(ds_dates, end) - 1
    if i0 < 0 or i1 <= i0 or ds_dates[-1] < end:
        return None
    return values[i1] / values[i0] - 1


def stress_tests(res, scenarios=None, shocks=(-0.05, -0.10, -0.20), beta_window=252):
    """Historical stress windows and benchmark shocks.

    For each window: what the strategy actually did if it was running then; what the
    final holdings would have done (each instrument's total return over the window x its
    final weight, needing data for that window); and the benchmark. Shocks estimate the
    final holdings' one-day P&L from each instrument's beta to the benchmark."""
    scen = list(STRESS_SCENARIOS) + [tuple(x) for x in (scenarios or [])]
    w = current_weights(res)
    bench = DATA.get(res.benchmark) if res.benchmark in DATA else None
    # A benchmark that isn't a loaded instrument (equal weight) is known on the backtest's dates only.
    b_dates, b_vals = (bench.dates, bench.total_return_index()) if bench is not None else (res.dates, res.bench_equity)
    rows = []
    for name, a, b in scen:
        row = {"name": name, "start": a, "end": b}
        if res.dates[0] < a and res.dates[-1] >= b:
            row["strategy"] = _window_return(res.dates, res.equity, a, b)
            sub = [v for d, v in zip(res.dates, res.equity) if a <= d <= b]
            base = [v for d, v in zip(res.dates, res.equity) if d < a][-1]
            peak, mdd = base, 0.0
            for v in sub:
                peak = max(peak, v)
                mdd = min(mdd, v / peak - 1)
            row["strategy_mdd"] = mdd
        held, covered, total = 0.0, 0.0, sum(abs(x) for x in w)
        missing = []
        for tk, wt, ds in zip(res.tickers, w, res.panel.sets):
            if abs(wt) < 1e-12:
                continue
            rr = _window_return(ds.dates, ds.total_return_index(), a, b)
            if rr is None:
                missing.append(tk)
                continue
            held += wt * rr
            covered += abs(wt)
        row["holdings"] = held if covered > 0 or total == 0 else None
        row["coverage"] = covered / total if total > 0 else 1.0
        row["missing"] = missing
        if b_vals:
            row["benchmark"] = _window_return(b_dates, b_vals, a, b) if (bench is not None or res.dates[0] < a) else None
        if row.get("strategy") is not None or row["holdings"] is not None or row.get("benchmark") is not None:
            rows.append(row)
    beta_rows = []
    if b_vals:
        rb = returns_by_date(bench) if bench is not None else {
            res.dates[t]: res.bench_equity[t] / res.bench_equity[t - 1] - 1 for t in range(1, len(res.dates)) if res.bench_equity[t - 1] > 0}
        R, ds_dates = _asset_returns(res, beta_window)
        xs = [rb.get(d) for d in ds_dates]
        betas = []
        for i, tk in enumerate(res.tickers):
            pairs = [(rb_v, r[i]) for rb_v, r in zip(xs, R) if rb_v is not None]
            if len(pairs) < 20:
                betas.append(NAN)
                continue
            bx = [p_[0] for p_ in pairs]
            by = [p_[1] for p_ in pairs]
            vb = covariance(bx, bx)
            betas.append(covariance(bx, by) / vb if vb > 0 else NAN)
        port_beta = sum(wt * bt for wt, bt in zip(w, betas) if not isnan(bt))
        beta_rows = [{"shock": sh, "pnl": port_beta * sh, "pnl_money": port_beta * sh * res.equity[-1]} for sh in shocks]
        beta_info = {"portfolio_beta": port_beta, "betas": dict(zip(res.tickers, betas))}
    else:
        beta_info = None
    return {"scenarios": rows, "shocks": beta_rows, "beta": beta_info, "weights": dict(zip(res.tickers, w)),
            "benchmark": res.benchmark}


def kupiec_pof(exceptions, n, p):
    """Kupiec's proportion-of-failures test: are `exceptions` VaR breaches in n days
    consistent with a breach probability p? Returns (LR statistic, p-value)."""
    x = exceptions
    if n <= 0:
        return NAN, NAN
    phat = x / n
    def ll(q):
        out = 0.0
        if n - x:
            out += (n - x) * math.log(1 - q) if q < 1 else -math.inf
        if x:
            out += x * math.log(q) if q > 0 else -math.inf
        return out
    lr = -2 * (ll(p) - ll(phat))
    return lr, chi2_sf(lr, 1)


def portfolio_var(res, window=500, levels=(0.95, 0.99), horizon=10, backtest_window=250):
    """Value at Risk and expected shortfall of the final holdings, from the instruments'
    last `window` days of returns: historical (the holdings' P&L on each of those days),
    parametric (normal, from the covariance matrix) and Cornish-Fisher (adjusted for the
    skew and fat tails of the historical P&L). Horizon VaR: historical from overlapping
    `horizon`-day returns, parametric scaled by sqrt(horizon). Component VaR splits the
    parametric VaR by instrument. Also checks the strategy's own daily returns against a
    rolling historical VaR (Kupiec test): too many breaches means the VaR is too low."""
    R, _ = _asset_returns(res, window)
    if len(R) < 60:
        raise ValueError("Need at least 60 days of returns for VaR.")
    w = current_weights(res)
    k = len(w)
    pnl = [sum(wi * ri for wi, ri in zip(w, r)) for r in R]
    C, m = _cov(R)
    Cw = [sum(C[i][j] * w[j] for j in range(k)) for i in range(k)]
    sig = math.sqrt(max(sum(w[i] * Cw[i] for i in range(k)), 0.0))
    mu = sum(wi * mi for wi, mi in zip(w, m))
    S, K = skewness(pnl), kurtosis(pnl)
    S, K = (0.0 if isnan(S) else S), (0.0 if isnan(K) else K)
    h = int(horizon)
    multi = []
    for t in range(h, len(pnl) + 1):
        g = 1.0
        for x in pnl[t - h:t]:
            g *= 1 + x
        multi.append(g - 1)
    eq = res.equity[-1]
    out = []
    for q in levels:
        z = norm_ppf(1 - q)            # negative, e.g. -2.33 for 99%
        cut = percentile(pnl, 100 * (1 - q))
        tail = [x for x in pnl if x <= cut]
        zcf = z + (z * z - 1) * S / 6 + (z ** 3 - 3 * z) * K / 24 - (2 * z ** 3 - 5 * z) * S * S / 36
        row = {"level": q,
               "hist": -cut, "hist_es": -mean(tail) if tail else NAN,
               "normal": -(mu + z * sig), "normal_es": -(mu - sig * math.exp(-z * z / 2) / math.sqrt(2 * math.pi) / (1 - q)),
               "cornish_fisher": -(mu + zcf * sig),
               "hist_h": -percentile(multi, 100 * (1 - q)) if multi else NAN,
               "normal_h": -(mu * h + z * sig * math.sqrt(h))}
        row["money"] = {k_: v * eq for k_, v in row.items() if k_ != "level" and not isnan(v)}
        row["components"] = [{"ticker": tk, "var": -z * w[i] * Cw[i] / sig if sig > 0 else 0.0}
                             for i, tk in enumerate(res.tickers)]
        out.append(row)
    r = daily_returns(res.equity)
    bt = []
    for q in levels:
        x = nobs = 0
        for t in range(backtest_window, len(r)):
            v = -percentile(r[t - backtest_window:t], 100 * (1 - q))
            nobs += 1
            if r[t] < -v:
                x += 1
        lr, pv = kupiec_pof(x, nobs, 1 - q)
        bt.append({"level": q, "days": nobs, "breaches": x, "expected": nobs * (1 - q), "lr": lr, "p_value": pv})
    return {"window": len(R), "horizon": h, "equity": eq, "vol": sig * math.sqrt(res_ann(res)), "skew": S, "kurtosis": K,
            "levels": out, "backtest": bt, "backtest_window": backtest_window}


def risk_report(res, factors=None, window=126, var_window=500, scenarios=None):
    """Everything on the Risk Report: factor regression, rolling beta to the benchmark,
    risk decomposition, stress tests and portfolio VaR. A part that can't be computed
    reports its reason in 'errors' instead of failing the rest."""
    out, errors = {}, {}
    def attempt(key, fn):
        try:
            out[key] = fn()
        except (ValueError, ZeroDivisionError) as e:
            errors[key] = str(e)
    if factors:
        attempt("factors", lambda: factor_regression(res, factors))
    r, b = daily_returns(res.equity), daily_returns(res.bench_equity)
    def roll():
        beta, corr = rolling_beta(r, b, int(window))
        idx = _downsample(len(beta))
        return {"window": int(window), "dates": [res.dates[i + 1] for i in idx], "beta": [beta[i] for i in idx],
                "corr": [corr[i] for i in idx], "benchmark": res.benchmark}
    attempt("rolling", roll)
    attempt("decomposition", lambda: risk_decomposition(res))
    attempt("stress", lambda: stress_tests(res, scenarios))
    attempt("var", lambda: portfolio_var(res, int(var_window)))
    out["errors"] = errors
    return out


def capacity_curve(strategy, sets, cfg, levels=None, start=None, end=None, bench=None, portfolio=False, rates=None,
                   threshold=0.5, signal_key=None):
    """How results change with account size. Re-runs the backtest at each starting cash
    level with square-root market impact on (impact_coef 1 if it is off) and reports
    CAGR, Sharpe, costs and how much of each fill bar's volume the fills took.
    'capacity' is the largest size whose Sharpe stays above threshold x the Sharpe at
    the smallest size (interpolated on a log scale); 'beyond_tested' means it never fell
    that far within the sizes tested."""
    levels = sorted(levels or [1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    base = cfg.as_dict()
    coef = base.get("impact_coef") or 1.0
    if not any(v and not isnan(v) for ds in sets for v in ds.volume):
        raise ValueError("Capacity needs trading volume, and these instruments have none.")
    rows = []
    for cash in levels:
        c = Config(**dict(base, cash=float(cash), impact_coef=coef))
        if portfolio:
            res = backtest_portfolio(strategy, sets, c, start, end, bench, rates=rates, signal_key=signal_key)
        else:
            res = backtest(strategy, sets[0], c, start, end, bench, rates=rates, signal_key=signal_key)
        m, _ = summarize(res)
        parts = []
        for f in res.fills:
            a = res.tickers.index(f["ticker"]) if f.get("ticker") in res.tickers else 0
            ds = res.panel.sets[a]
            j = res.panel.nat[a][f["i"]]
            vol = ds.volume[j] if j >= 0 else NAN     # the fill bar's volume, as the participation limit uses
            if vol and not isnan(vol) and vol > 0:
                parts.append(abs(f["qty"]) / vol)
        rows.append({"cash": cash, "cagr": m["cagr"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                     "costs": m["total_costs"], "impact": m.get("impact_paid", NAN),
                     "costs_pct": m["total_costs"] / cash if cash else NAN,
                     "participation_median": percentile(parts, 50) if parts else NAN,
                     "participation_p95": percentile(parts, 95) if parts else NAN,
                     "partial_fills": m.get("partial_fills", 0), "fills": len(res.fills)})
    s0 = rows[0]["sharpe"]
    cap = None
    if s0 and not isnan(s0) and s0 > 0:
        bar = threshold * s0
        for lo, hi in zip(rows, rows[1:]):
            if lo["sharpe"] >= bar > hi["sharpe"]:
                f = (lo["sharpe"] - bar) / (lo["sharpe"] - hi["sharpe"])
                cap = math.exp(math.log(lo["cash"]) + f * (math.log(hi["cash"]) - math.log(lo["cash"])))
                break
    beyond = cap is None and bool(s0 and not isnan(s0) and s0 > 0) and rows[-1]["sharpe"] >= threshold * s0
    return {"rows": rows, "threshold": threshold, "capacity": cap, "beyond_tested": beyond, "impact_coef": coef,
            "impact_forced": not base.get("impact_coef"), "max_participation": base.get("max_participation")}


# ---- audit log of approved runs ------------------------------------------------
# Each entry is chained to the one before by SHA-256, so editing, removing or reordering
# any entry breaks every hash after it. The page writes the log; this verifies it.
AUDIT_FIELDS = ("seq", "at", "run_id", "run_digest", "decision", "reviewer", "comment", "prev_hash")
AUDIT_GENESIS = "0" * 64


def canonical_json(obj):
    """Deterministic JSON (sorted keys, no spaces, UTF-8): the same text the page hashes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def audit_hash(entry):
    """SHA-256 of an entry's fields (strings and integers only), chained via prev_hash."""
    import hashlib
    body = {k: entry[k] for k in AUDIT_FIELDS}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def verify_audit_log(entries):
    """Check an exported audit log: sequence numbers, the chain of prev_hash links and each
    entry's own hash. Returns {'ok', 'entries', 'error', 'at'}; 'at' is the first bad index."""
    prev = AUDIT_GENESIS
    for i, e in enumerate(entries):
        missing = [k for k in AUDIT_FIELDS + ("hash",) if k not in e]
        if missing:
            return {"ok": False, "entries": len(entries), "at": i, "error": "entry %d lacks %s" % (i, ", ".join(missing))}
        if e["seq"] != i:
            return {"ok": False, "entries": len(entries), "at": i, "error": "entry %d has sequence number %r" % (i, e["seq"])}
        if e["prev_hash"] != prev:
            return {"ok": False, "entries": len(entries), "at": i, "error": "entry %d doesn't link to the one before it" % i}
        if audit_hash(e) != e["hash"]:
            return {"ok": False, "entries": len(entries), "at": i, "error": "entry %d was changed after it was written" % i}
        prev = e["hash"]
    return {"ok": True, "entries": len(entries), "at": None, "error": None, "head": prev}


# =============================================================================
# 7d. Options overlays (modelled prices: Black-Scholes with an implied-volatility proxy)
# =============================================================================

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs(S, K, T, r, sigma, kind="call", q=0.0):
    """Black-Scholes price and Greeks of a European option: {price, delta, gamma, vega, theta}
    (theta per year). At or after expiry the price is the payoff."""
    if T <= 1e-12 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
        d = (1.0 if S > K else 0.0) if kind == "call" else (-1.0 if S < K else 0.0)
        return {"price": intrinsic, "delta": d, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    st = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / st
    d2 = d1 - st
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    if kind == "call":
        price = S * dq * norm_cdf(d1) - K * dr * norm_cdf(d2)
        delta = dq * norm_cdf(d1)
        theta = -S * dq * norm_pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * dr * norm_cdf(d2) + q * S * dq * norm_cdf(d1)
    else:
        price = K * dr * norm_cdf(-d2) - S * dq * norm_cdf(-d1)
        delta = -dq * norm_cdf(-d1)
        theta = -S * dq * norm_pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * dr * norm_cdf(-d2) - q * S * dq * norm_cdf(-d1)
    return {"price": price, "delta": delta, "gamma": dq * norm_pdf(d1) / (S * st), "vega": S * dq * norm_pdf(d1) * math.sqrt(T), "theta": theta}


def implied_vol_series(ds, source="realized", multiplier=1.1, window=21, ann=TRADING_DAYS):
    """Implied volatility per bar of ds: trailing realized volatility x multiplier (the usual
    premium of implied over realized), or the close of a loaded volatility index such as
    VIX (in percent), as of each bar."""
    if source == "realized":
        r = [NAN] + [ds.close[t] / ds.close[t - 1] - 1 for t in range(1, ds.n)]
        sd = rolling_std(r[1:], window, ddof=1)
        out = [NAN] + [v * math.sqrt(ann) * multiplier if not isnan(v) else NAN for v in sd]
        return out
    ix = DATA.get(source)
    if ix is None:
        raise ValueError("No volatility index %s is loaded." % source)
    out, j, cur = [], -1, NAN
    for d in ds.dates:
        while j + 1 < ix.n and ix.dates[j + 1] <= d:
            j += 1
            cur = ix.close[j] / 100.0
        out.append(cur * multiplier if not isnan(cur) else NAN)
    return out


def _legs(kind, otm, put_otm):
    """[(option type, strike as a multiple of spot, +1 bought / -1 sold)] for an overlay."""
    if kind == "covered_call":
        return [("call", 1 + otm, -1)]
    if kind == "protective_put":
        return [("put", 1 - put_otm, +1)]
    if kind == "collar":
        return [("call", 1 + otm, -1), ("put", 1 - put_otm, +1)]
    raise ValueError("overlay must be covered_call, protective_put or collar")


def options_overlay(res, kind="covered_call", otm=0.05, put_otm=0.05, days=21, iv="realized", iv_mult=1.1,
                    vol_window=21, cost=0.03):
    """Write calls on / buy puts under the shares a single-instrument strategy holds.

    Every `days` bars, options expiring `days` bars later are opened on the shares held at
    that close (long positions only), priced by Black-Scholes at the implied-volatility proxy,
    paying `cost` x premium to trade; they are marked daily and cash-settled at expiry. The
    overlay's P&L is added to the strategy's equity. Modelled prices, not market quotes."""
    if res.portfolio:
        raise ValueError("Options overlays work on single-instrument backtests.")
    ds, panel, s = res.panel.sets[0], res.panel, res.start
    A = res_ann(res)
    n = len(res.equity)
    ivs = implied_vol_series(ds, iv, iv_mult, vol_window, A)
    r = res.config.cash_rate if res.config.cash_rate is not None else res.config.risk_free
    legs = _legs(kind, otm, put_otm)
    cash = 0.0
    book = []          # open legs: {type, K, sign, qty, expiry_bar, open_i}
    pnl = [0.0] * n
    rolls = []
    for i in range(n):
        t = s + i
        S = ds.close[t]
        # expiries first
        if book and i >= book[0]["expiry"]:
            settle = 0.0
            for lg in book:
                pay = max(S - lg["K"], 0.0) if lg["type"] == "call" else max(lg["K"] - S, 0.0)
                settle += lg["sign"] * lg["qty"] * pay
            cash += settle
            rolls[-1].update(settle=settle, settle_price=S, assigned=any(
                (lg["type"] == "call" and S > lg["K"]) or (lg["type"] == "put" and S < lg["K"]) for lg in book))
            rolls[-1]["pnl"] = rolls[-1]["premium"] + settle - rolls[-1]["costs"]
            book = []
        if not book and i < n - 1 and i % days == 0:
            sh = res.position[i]
            sig = ivs[t]
            if sh > 0 and not isnan(sig) and sig > 0:
                T = days / A
                prem = costs = 0.0
                opened = []
                for typ, mult, sign in legs:
                    K = S * mult
                    px = bs(S, K, T, r, sig, typ)["price"]
                    prem += -sign * sh * px
                    costs += cost * sh * px
                    opened.append({"type": typ, "K": K, "sign": sign, "qty": sh, "expiry": min(i + days, n - 1)})
                cash += prem - costs
                book = opened
                rolls.append({"date": res.dates[i], "spot": S, "iv": sig, "shares": sh, "premium": prem, "costs": costs,
                              "strikes": [lg["K"] for lg in opened], "expiry": res.dates[opened[0]["expiry"]]})
        mark = 0.0
        for lg in book:
            T = max(lg["expiry"] - i, 0) / A
            sig = ivs[t] if not isnan(ivs[t]) else rolls[-1]["iv"]
            mark += lg["sign"] * lg["qty"] * bs(S, lg["K"], T, r, sig, lg["type"])["price"]
        pnl[i] = cash + mark
    equity = [e + x for e, x in zip(res.equity, pnl)]
    base = performance(res.dates, res.equity, res.config.risk_free, res.bench_equity, ann=A)
    over = performance(res.dates, equity, res.config.risk_free, res.bench_equity, ann=A)
    done = [x for x in rolls if "settle" in x]
    return {"kind": kind, "days": days, "iv_source": iv, "iv_mult": iv_mult, "equity": equity, "overlay_pnl": pnl,
            "rolls": rolls, "base": base, "with_overlay": over,
            "premium": sum(x["premium"] for x in rolls), "settlements": sum(x.get("settle", 0.0) for x in rolls),
            "costs": sum(x["costs"] for x in rolls), "assigned": sum(1 for x in done if x["assigned"]), "expired": len(done),
            "iv": [ivs[s + i] for i in range(n)]}


def delta_hedged(ds, start=None, end=None, kind="call", otm=0.0, days=21, iv="realized", iv_mult=1.1, vol_window=21,
                 r=0.0, notional=10000.0, ann=None, side=-1):
    """Sell (side=-1) or buy an option on `notional` of ds every `days` bars and delta-hedge it
    with the underlying at every close. Splits the P&L into the option's and the hedge's, and
    compares it with the gamma-theta estimate sum(0.5 x gamma x S^2 x (implied var x dt -
    realized return^2)): a short option earns when implied volatility exceeds realized."""
    A = ann or periods_per_year(ds)
    s, e = _resolve_range(ds.dates, start, end, ds.ticker)
    ivs = implied_vol_series(ds, iv, iv_mult, vol_window, A)
    opt_pnl = hedge_pnl = est = 0.0
    total, opt_series, hedge_series, est_series = [0.0], [0.0], [0.0], [0.0]
    leg = None
    periods = []
    for t in range(s, e + 1):
        S = ds.close[t]
        if leg is not None:
            T = max(leg["expiry"] - t, 0) / A
            q = bs(S, leg["K"], T, r, leg["iv"], kind)
            dv = side * leg["qty"] * (q["price"] - leg["value"])
            dh = -leg["hedge"] * (S - leg["S"])          # the hedge holds -side x delta x qty shares
            g = leg["greeks"]
            rr = S / leg["S"] - 1
            est_t = -side * leg["qty"] * 0.5 * g["gamma"] * leg["S"] ** 2 * (leg["iv"] ** 2 / A - rr * rr)
            opt_pnl += dv
            hedge_pnl += dh
            est += est_t
            leg["period"]["option"] += dv
            leg["period"]["hedge"] += dh
            leg["period"]["estimate"] += est_t
            leg.update(value=q["price"], S=S, greeks=q, hedge=side * q["delta"] * leg["qty"])
            if t >= leg["expiry"]:
                leg["period"]["realized_vol"] = stdev(leg["rets"] + [rr]) * math.sqrt(A) if len(leg["rets"]) > 1 else NAN
                leg = None
            else:
                leg["rets"].append(rr)
        if t > s:
            total.append(opt_pnl + hedge_pnl)
            opt_series.append(opt_pnl)
            hedge_series.append(hedge_pnl)
            est_series.append(est)
        if leg is None and t < e and (t - s) % days == 0 and not isnan(ivs[t]) and ivs[t] > 0:
            K = S * (1 + otm)
            q = bs(S, K, days / A, r, ivs[t], kind)
            qty = notional / S
            per = {"date": ds.dates[t], "spot": S, "strike": K, "iv": ivs[t], "premium": -side * qty * q["price"],
                   "option": 0.0, "hedge": 0.0, "estimate": 0.0}
            periods.append(per)
            leg = {"K": K, "iv": ivs[t], "qty": qty, "expiry": min(t + days, e), "value": q["price"], "S": S, "greeks": q,
                   "hedge": side * q["delta"] * qty, "period": per, "rets": []}
    for per in periods:
        per["total"] = per["option"] + per["hedge"]
    return {"dates": ds.dates[s:e + 1], "total": total, "option": opt_series, "hedge": hedge_series, "estimate": est_series,
            "periods": periods, "notional": notional, "side": "short" if side < 0 else "long", "kind": kind,
            "final": total[-1], "final_estimate": est_series[-1]}


# =============================================================================
# 3d. Option chains: end-of-day quotes from a data vendor, traded by strategies
# =============================================================================
#
# A chain holds one row per contract per day: date, expiry, strike, call/put, bid, ask and,
# when the file has them, last price, volume, open interest, implied volatility, delta and
# the underlying's price. Rows are kept in compact arrays sorted by (date, expiry, type,
# strike). Strategies see only the rows dated today (ctx.options) and trade at the next
# day's quotes, so a chain can never leak the future into a decision.

OPTIONS = {}            # underlying ticker -> OptionChain
OPTION_MAX_ROWS = 4_000_000

# Column names in the order they're preferred (matched after lower-casing and removing
# brackets & spaces). The same lists are in index.html (OPTION_COLUMNS), kept in step by a test.
OPTION_COLUMNS = {
    "date": ("quote_date", "trade_date", "tradedate", "date", "quotedate", "asof", "as_of", "data_date"),
    "expiry": ("expiration", "expire_date", "expiration_date", "expirdate", "exdate", "expiry", "expiry_date", "exp_date", "maturity"),
    "strike": ("strike", "strike_price"),
    "type": ("option_type", "cp_flag", "call_put", "put_call", "right", "type", "callput", "cp"),
    "bid": ("bid_eod", "bid", "best_bid", "bid_price", "bid_1545"),
    "ask": ("ask_eod", "ask", "best_offer", "ask_price", "offer", "ask_1545"),
    "last": ("last", "last_price", "close"),
    "volume": ("trade_volume", "volume"),
    "oi": ("open_interest", "openinterest"),
    "iv": ("implied_volatility_1545", "implied_volatility", "impl_volatility", "iv", "implied_vol"),
    "delta": ("delta_1545", "delta"),
    "under": ("underlying_last", "active_underlying_price_1545", "underlying_price", "stock_price", "stockprice", "spot", "underlying_close"),
    "symbol": ("underlying_symbol", "ticker", "underlying_ticker", "symbol", "root"),
}
# Wide files put a call and a put on each row: (call columns, put columns).
OPTION_WIDE = {
    "bid": (("c_bid", "callbidprice", "call_bid"), ("p_bid", "putbidprice", "put_bid")),
    "ask": (("c_ask", "callaskprice", "call_ask"), ("p_ask", "putaskprice", "put_ask")),
    "last": (("c_last", "call_last"), ("p_last", "put_last")),
    "volume": (("c_volume", "callvolume", "call_volume"), ("p_volume", "putvolume", "put_volume")),
    "oi": (("c_oi", "callopeninterest", "call_open_interest"), ("p_oi", "putopeninterest", "put_open_interest")),
    "iv": (("c_iv", "callmidiv", "call_iv"), ("p_iv", "putmidiv", "put_iv")),
    "delta": (("c_delta", "call_delta"), ("p_delta", "put_delta")),
}


def _opt_header(h):
    return h.strip().strip('"').strip().strip("[]").strip().lower().replace(" ", "_")


def _opt_date(raw):
    """Dates in option files: YYYY-MM-DD (optionally with a time), MM/DD/YYYY or YYYYMMDD."""
    raw = str(raw).strip().strip('"').strip()
    if len(raw) >= 8 and raw[:8].isdigit() and (len(raw) == 8 or not raw[8].isdigit()):
        try:
            return _dt.date(int(raw[:4]), int(raw[4:6]), int(raw[6:8])).isoformat()
        except ValueError:
            return None
    return _iso_date(raw)


def _opt_type(raw):
    v = str(raw).strip().strip('"').upper()
    return 1 if v in ("C", "CALL", "CALLS") else 0 if v in ("P", "PUT", "PUTS") else None


def option_file_layout(header):
    """Which columns of an option file hold what: {'wide': bool, 'cols': {field: index},
    'wide_cols': {field: (call index, put index)}, 'format': name, 'strike_scale': 1 or 1000}.
    Raises ValueError when the file isn't an option chain."""
    h = [_opt_header(x) for x in header]
    col = lambda names: next((h.index(n) for n in names if n in h), None)
    cols = {k: col(v) for k, v in OPTION_COLUMNS.items()}
    wide = {k: (col(c), col(pp)) for k, (c, pp) in OPTION_WIDE.items()}
    is_wide = wide["bid"][0] is not None and wide["bid"][1] is not None and cols["type"] is None
    if cols["date"] is None or cols["expiry"] is None or cols["strike"] is None:
        raise ValueError("not an option chain: needs a quote date, an expiration and a strike column")
    if not is_wide and (cols["type"] is None or cols["bid"] is None or cols["ask"] is None):
        raise ValueError("an option chain needs call/put, bid and ask columns (or C_BID/P_BID-style columns for calls & puts)")
    if is_wide and (wide["ask"][0] is None or wide["ask"][1] is None):
        raise ValueError("a wide option chain needs ask columns for calls and puts")
    fmt = ("OptionsDX" if "c_bid" in h else "ORATS" if "callbidprice" in h else "wide") if is_wide else \
        ("CBOE DataShop" if "underlying_symbol" in h else "OptionMetrics" if "exdate" in h and "best_bid" in h else "long")
    if is_wide and cols["delta"] is not None and wide["delta"][0] is None:
        wide["delta"] = (cols["delta"], None)            # ORATS: one delta column, the call's
    return {"wide": is_wide, "cols": cols, "wide_cols": wide, "format": fmt,
            "strike_scale": 1000.0 if h[cols["strike"]] == "strike_price" and fmt == "OptionMetrics" else 1.0}


def _third_friday(y, m):
    d = _dt.date(y, m, 15)
    return d + _dt.timedelta(days=(4 - d.weekday()) % 7)


def is_monthly_expiry(iso):
    """Standard monthly expiry: the third Friday, the Thursday before it (when Friday is a
    holiday) or the Saturday after it (how expirations were listed before 2015)."""
    d = _dt.date.fromisoformat(iso)
    return abs((d - _third_friday(d.year, d.month)).days) <= 1 and d.weekday() in (3, 4, 5)


def option_rows(text, max_dte=None, band=None, expiries="all", start=None, end=None, spot=None, underlying=None):
    """Parse an option chain file into normalized rows
    (date, expiry, strike, call, bid, ask, last, volume, oi, iv, delta, under, symbol),
    keeping contracts that pass the filters: max_dte calendar days to expiry, strikes
    within band (0.2 = ±20%) of the spot (the file's underlying price, else spot[date]),
    expiries 'all' | 'friday' | 'monthly', and dates from start to end.
    Returns (rows, info)."""
    first = text.split("\n", 1)[0]
    sep = "\t" if first.count("\t") > first.count(",") else ","
    lines = text.splitlines()
    header = next(csv.reader([lines[0]], delimiter=sep))
    lay = option_file_layout(header)
    c, w, sc = lay["cols"], lay["wide_cols"], lay["strike_scale"]
    out, dropped, bad, symbols, no_spot = [], 0, 0, set(), 0
    num = lambda row, i: _num(row[i]) if i is not None and i < len(row) else None
    for row in csv.reader(lines[1:], delimiter=sep):
        if not row or len(row) <= max(c["date"], c["expiry"], c["strike"]):
            continue
        d, x, k = _opt_date(row[c["date"]]), _opt_date(row[c["expiry"]]), _num(row[c["strike"]])
        if d is None or x is None or k is None or x < d:
            bad += 1
            continue
        k = k / sc
        sym = row[c["symbol"]].strip().strip('"').upper() if c["symbol"] is not None and c["symbol"] < len(row) else (underlying or "")
        if underlying and sym and sym != underlying.upper() and c["symbol"] is not None and lay["format"] != "OptionMetrics":
            dropped += 1
            continue
        symbols.add(sym)
        if (start and d < start) or (end and d > end):
            dropped += 1
            continue
        dte = (_dt.date.fromisoformat(x) - _dt.date.fromisoformat(d)).days
        if max_dte is not None and dte > max_dte:
            dropped += 1
            continue
        if expiries == "monthly" and not is_monthly_expiry(x) or expiries == "friday" and _dt.date.fromisoformat(x).weekday() not in (3, 4, 5):
            dropped += 1
            continue
        u = num(row, c["under"])
        if u is None and spot:
            u = spot.get(d)
        if band is not None:
            if u is None or u <= 0:
                no_spot += 1
            elif abs(k / u - 1) > band:
                dropped += 1
                continue
        if lay["wide"]:
            legs = []
            for side, call in ((0, 1), (1, 0)):
                cd = w["delta"][side]
                de = num(row, cd) if cd is not None else None
                if lay["format"] == "ORATS" and side == 1 and de is not None:
                    de = de - 1.0                      # ORATS gives the call's delta; the put's is delta - 1
                if lay["format"] == "ORATS" and side == 1 and w["delta"][1] is None and w["delta"][0] is not None:
                    cdv = num(row, w["delta"][0])
                    de = cdv - 1.0 if cdv is not None else None
                legs.append((call, num(row, w["bid"][side]), num(row, w["ask"][side]), num(row, w["last"][side]),
                             num(row, w["volume"][side]), num(row, w["oi"][side]), num(row, w["iv"][side]), de))
        else:
            t = _opt_type(row[c["type"]]) if c["type"] < len(row) else None
            if t is None:
                bad += 1
                continue
            legs = [(t, num(row, c["bid"]), num(row, c["ask"]), num(row, c["last"]), num(row, c["volume"]),
                     num(row, c["oi"]), num(row, c["iv"]), num(row, c["delta"]))]
        for call, b, a, l, v, oi, iv, de in legs:
            if a is None:                              # no ask, no quote (a zero bid with an ask is a real, far-out quote)
                continue
            out.append((d, x, k, call, b, a, l, v, oi, iv if iv is None or iv < 20 else iv / 100.0, de, u))
    info = {"format": lay["format"], "rows": len(out), "dropped": dropped, "unreadable": bad, "symbols": sorted(s for s in symbols if s),
            "no_spot": no_spot, "strike_scale": sc}
    return out, info


class OptionQuote:
    """One option on one day, as a strategy sees it. Prices per share (x the multiplier per
    contract). iv & Greeks are the file's when it has them, else implied from the mid with
    Black-Scholes (European, no dividends, the account's cash rate); theta is per calendar
    day and vega per volatility point (0.01)."""
    __slots__ = ("symbol", "underlying", "kind", "strike", "expiry", "dte", "bid", "ask", "mid", "last", "volume",
                 "open_interest", "spot", "_iv", "_delta", "_g", "_rate", "_row")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self._g = None

    def _greeks(self):
        if self._g is None:
            T = max(self.dte, 0) / 365.0
            iv = self._iv
            if (iv is None or isnan(iv) or iv <= 0) and self.mid > 0 and T > 0 and self.spot and self.spot > 0:
                iv = implied_vol(self.mid, self.spot, self.strike, T, self._rate, self.kind)
            if iv is None or isnan(iv) or not self.spot:
                self._g = {"iv": NAN, "delta": self._delta if self._delta is not None else NAN, "gamma": NAN, "theta": NAN, "vega": NAN}
            else:
                q = bs(self.spot, self.strike, T, self._rate, iv, self.kind)
                self._g = {"iv": iv, "delta": self._delta if self._delta is not None and not isnan(self._delta) else q["delta"],
                           "gamma": q["gamma"], "theta": q["theta"] / 365.0, "vega": q["vega"] / 100.0}
        return self._g

    iv = property(lambda self: self._greeks()["iv"])
    delta = property(lambda self: self._greeks()["delta"])
    gamma = property(lambda self: self._greeks()["gamma"])
    theta = property(lambda self: self._greeks()["theta"])
    vega = property(lambda self: self._greeks()["vega"])
    is_call = property(lambda self: self.kind == "call")
    is_put = property(lambda self: self.kind == "put")

    def __repr__(self):
        return "<%s %s %s %g, %d days: bid %.2f ask %.2f>" % (self.symbol, self.kind, self.expiry, self.strike, self.dte, self.bid, self.ask)


def implied_vol(price, S, K, T, r, kind):
    """Black-Scholes implied volatility of a European option price (NaN if none fits)."""
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return NAN
    disc = math.exp(-r * T)
    lo_price = max(S - K * disc, 0.0) if kind == "call" else max(K * disc - S, 0.0)
    hi_price = S if kind == "call" else K * disc
    if price <= lo_price + 1e-12 or price >= hi_price:
        return NAN
    lo, hi = 1e-4, 6.0
    sig = max(0.05, min(3.0, math.sqrt(2 * math.pi / T) * price / S))    # Brenner-Subrahmanyam start
    for _ in range(100):
        q = bs(S, K, T, r, sig, kind)
        diff = q["price"] - price
        if abs(diff) < 1e-10:
            return sig
        if diff > 0:
            hi = sig
        else:
            lo = sig
        step = sig - diff / q["vega"] if q["vega"] > 1e-12 else None
        sig = step if step is not None and lo < step < hi else (lo + hi) / 2
        if hi - lo < 1e-12:
            break
    return sig


def occ_symbol(root, expiry, call, strike):
    """OCC-style option symbol without padding: SPY240315C00500000."""
    return "%s%s%s%08d" % (root, expiry[2:4] + expiry[5:7] + expiry[8:10], "C" if call else "P", int(round(strike * 1000)))


def parse_occ(sym):
    """'SPY240315C00500000' -> (root, '2024-03-15', is_call, 500.0)."""
    sym = sym.replace(" ", "")
    tail = sym[-15:]
    return sym[:-15], "20%s-%s-%s" % (tail[:2], tail[2:4], tail[4:6]), tail[6] == "C", int(tail[7:]) / 1000.0


class OptionChain:
    """Every option row of one underlying, in compact arrays sorted by (date, expiry, type, strike)."""
    CODES = {"d": "i", "x": "i", "k": "d", "c": "b", "b": "f", "a": "f", "l": "f", "v": "f", "oi": "f", "iv": "f", "de": "f", "u": "f"}

    def __init__(self, underlying, dates, expiries, cols, fingerprint=None, info=None):
        from array import array
        self.underlying = underlying.upper()
        self.root = self.underlying.lstrip("^")
        self.dates, self.expiries = list(dates), list(expiries)
        self.date_index = {d: i for i, d in enumerate(self.dates)}
        self.expiry_index = {x: i for i, x in enumerate(self.expiries)}
        self.cols = {}
        n = None
        for key, code in self.CODES.items():
            src = cols.get(key)
            if src is None:
                arr = array(code, [0] * n) if code in "ib" else array(code, [NAN]) * n if n else array(code)
            elif isinstance(src, (bytes, bytearray)):
                arr = array(code)
                arr.frombytes(src)
            else:
                arr = array(code, [(NAN if v is None else v) for v in src]) if code in "fd" else array(code, [int(v) for v in src])
            if n is None:
                n = len(arr)
            if len(arr) != n:
                raise ValueError("option column %s has %d values, expected %d" % (key, len(arr), n))
            self.cols[key] = arr
        self.n = n or 0
        d, x, c, k = self.cols["d"], self.cols["x"], self.cols["c"], self.cols["k"]
        for i in range(1, self.n):
            if (d[i], x[i], c[i], k[i]) < (d[i - 1], x[i - 1], c[i - 1], k[i - 1]):
                raise ValueError("option rows must be sorted by date, expiry, type and strike")
        self.starts = [0] * (len(self.dates) + 1)
        j = 0
        for di in range(len(self.dates)):
            self.starts[di] = j
            while j < self.n and d[j] == di:
                j += 1
        self.starts[len(self.dates)] = self.n
        self.has_iv = any(not isnan(v) for v in self.cols["iv"][:2000])
        self.has_delta = any(not isnan(v) for v in self.cols["de"][:2000])
        self.info = info or {}
        self.fingerprint = fingerprint or self._fingerprint()

    def _fingerprint(self):
        import hashlib
        h = hashlib.sha256(self.underlying.encode())
        for key in ("d", "x", "k", "c", "b", "a"):
            h.update(self.cols[key].tobytes())
        h.update("|".join(self.dates + self.expiries).encode())
        return h.hexdigest()[:16]

    @classmethod
    def from_rows(cls, underlying, rows, info=None):
        """From option_rows() output; later duplicates of a (date, expiry, type, strike) win."""
        best = {}
        for r in rows:
            best[(r[0], r[1], r[3], r[2])] = r
        rows = sorted(best.values(), key=lambda r: (r[0], r[1], r[3], r[2]))
        dates = sorted({r[0] for r in rows})
        exps = sorted({r[1] for r in rows})
        di, xi = {d: i for i, d in enumerate(dates)}, {x: i for i, x in enumerate(exps)}
        cols = {"d": [di[r[0]] for r in rows], "x": [xi[r[1]] for r in rows], "k": [r[2] for r in rows], "c": [r[3] for r in rows],
                "b": [r[4] for r in rows], "a": [r[5] for r in rows], "l": [r[6] for r in rows], "v": [r[7] for r in rows],
                "oi": [r[8] for r in rows], "iv": [r[9] for r in rows], "de": [r[10] for r in rows], "u": [r[11] for r in rows]}
        return cls(underlying, dates, exps, cols, info=info)

    def block(self, day):
        di = self.date_index.get(day)
        return (0, 0) if di is None else (self.starts[di], self.starts[di + 1])

    def find_row(self, day, expiry, call, strike):
        """Row index of one contract on one day, or None."""
        s, e = self.block(day)
        xi = self.expiry_index.get(expiry)
        if xi is None or s == e:
            return None
        key = (xi, 1 if call else 0, strike)
        x, c, k = self.cols["x"], self.cols["c"], self.cols["k"]
        lo, hi = s, e
        while lo < hi:
            mid = (lo + hi) // 2
            if (x[mid], c[mid], k[mid]) < key:
                lo = mid + 1
            else:
                hi = mid
        if lo < e and x[lo] == xi and c[lo] == key[1] and abs(k[lo] - strike) < 1e-6:
            return lo
        return None

    def summary(self):
        return {"underlying": self.underlying, "rows": self.n, "start": self.dates[0] if self.dates else None,
                "end": self.dates[-1] if self.dates else None, "days": len(self.dates), "expiries": len(self.expiries),
                "has_iv": self.has_iv, "has_delta": self.has_delta, "fingerprint": self.fingerprint}


def _f(v, nd=4):
    v = float(v)
    return NAN if v != v else round(v, nd)


class OptionsDesk:
    """The account's options: orders placed on a bar fill at the next bar's end-of-day quotes
    (buy at the ask, sell at the bid; option_fill < 1 moves the price toward the mid), plus a
    commission per contract. Positions are marked at the mid every bar and settle in cash at
    their intrinsic value on the expiry date's close (no early exercise or assignment)."""

    def __init__(self, cfg, panel, acct):
        self.cfg, self.panel, self.acct = cfg, panel, acct
        self.chains = {a: OPTIONS[tk] for a, tk in enumerate(panel.tickers) if tk in OPTIONS}
        self.mult = cfg.option_multiplier
        self.pos = {}            # symbol -> {a, call, strike, expiry, qty, mark, cash (since this position opened), opened}
        self.pending = []        # [(symbol, a, contracts, placed date)]
        self.fills, self.settlements, self.unfilled, self.trades = [], [], [], []
        self.value = 0.0
        self.stale = 0
        self.stats = {"premium_paid": 0.0, "premium_received": 0.0, "commissions": 0.0, "contracts": 0, "settled_cash": 0.0}
        self.daily = {k: [] for k in ("value", "delta", "gamma", "theta", "vega", "long", "short")}
        self._cache = {}

    def chain_for(self, a):
        ch = self.chains.get(a)
        if ch is None:
            raise ValueError("No option chain is loaded for %s. Drop an option chain file on the Data tab (Docs: Options)." % self.panel.tickers[a])
        return ch

    def rate(self, t):
        return self.acct.rate_on(self.panel.dates[t])

    def quote(self, a, i, t):
        """OptionQuote for row i (on bar t), cached for the bar."""
        key = (i, t)
        q = self._cache.get(key)
        if q is None:
            ch = self.chains[a]
            col = ch.cols
            day = self.panel.dates[t][:10]
            b, ak = _f(col["b"][i]), _f(col["a"][i])
            b = 0.0 if isnan(b) else b
            ak = NAN if isnan(ak) else ak
            u = _f(col["u"][i])
            spot = u if not isnan(u) and u > 0 else self.panel.mark(a, t)
            exp = ch.expiries[col["x"][i]]
            call = col["c"][i] == 1
            iv, de = _f(col["iv"][i], 6), _f(col["de"][i], 6)
            q = OptionQuote(symbol=occ_symbol(ch.root, exp, call, col["k"][i]), underlying=ch.underlying, kind="call" if call else "put",
                            strike=col["k"][i], expiry=exp, dte=(_dt.date.fromisoformat(exp) - _dt.date.fromisoformat(day)).days,
                            bid=b, ask=ak, mid=(b + ak) / 2 if not isnan(ak) else NAN, last=_f(col["l"][i]),
                            volume=_f(col["v"][i], 0), open_interest=_f(col["oi"][i], 0), spot=spot,
                            _iv=None if isnan(iv) else iv, _delta=None if isnan(de) else de, _rate=self.rate(t), _row=i)
            if len(self._cache) > 20000:
                self._cache.clear()
            self._cache[key] = q
        return q

    # ---- the loop -------------------------------------------------------------
    def on_bar(self, t):
        day = self.panel.dates[t][:10]
        for sym, p in list(self.pos.items()):          # expired on a day without a bar: settle at the last close before
            if p["expiry"] < day:
                self._settle(sym, p, max(t - 1, 0), late=True)
        orders, self.pending = self.pending, []
        for o in orders:
            self._fill(o, t, day)
        for sym, p in list(self.pos.items()):
            if p["expiry"] == day:
                self._settle(sym, p, t)
        self._mark(t, day)

    def _fill(self, o, t, day):
        sym, a, qty, placed = o
        ch = self.chains[a]
        root, exp, call, strike = parse_occ(sym)
        i = ch.find_row(day, exp, call, strike)
        why = None
        if i is None:
            why = "no quote on %s" % day
        else:
            q = self.quote(a, i, t)
            share = self.cfg.option_fill
            if qty > 0:
                if not (q.ask > 0):
                    why = "no ask"
                else:
                    px = q.ask - (1 - share) * (q.ask - q.mid)
            else:
                if not (q.bid > 0):
                    why = "no bid"
                else:
                    px = q.bid + (1 - share) * (q.mid - q.bid)
        if why:
            self.unfilled.append({"placed": placed, "date": day, "symbol": sym, "contracts": qty, "reason": why})
            return
        px = round(px, 4)
        fee = self.cfg.option_commission * abs(qty)
        cash = -qty * px * self.mult - fee
        self.acct.cash += cash
        self.acct._eq = None
        self.stats["commissions"] += fee
        self.stats["contracts"] += abs(qty)
        self.stats["premium_paid" if qty > 0 else "premium_received"] += abs(qty) * px * self.mult
        p = self.pos.get(sym)
        before = p["qty"] if p else 0
        self.fills.append({"date": day, "placed": placed, "symbol": sym, "underlying": ch.underlying, "kind": "call" if call else "put",
                           "strike": strike, "expiry": exp, "contracts": qty, "price": px, "bid": q.bid, "ask": q.ask,
                           "commission": fee, "cash": round(cash, 4),
                           "action": ("buy" if qty > 0 else "sell") + (" to close" if before and (before > 0) != (qty > 0) else " to open")})
        remaining = qty
        if p and (before > 0) != (qty > 0):                 # the opposite side closes up to |before| contracts first
            n_close = min(abs(before), abs(qty))
            part = n_close / abs(qty)
            p["cash"] += cash * part
            p["qty"] += math.copysign(n_close, qty)
            remaining = qty - math.copysign(n_close, qty)
            cash *= 1 - part
            p["mark"] = q.mid if not isnan(q.mid) else px
            if abs(p["qty"]) < 1e-9:
                self._close_trade(sym, p, day, "closed")
                p = None
        if abs(remaining) > 1e-9:
            if p is None:
                p = self.pos[sym] = {"a": a, "call": call, "strike": strike, "expiry": exp, "qty": 0, "mark": px, "cash": 0.0,
                                     "opened": day, "max_qty": 0, "first_qty": remaining}
            p["qty"] += remaining
            p["cash"] += cash
            p["max_qty"] = max(p["max_qty"], abs(p["qty"]))
            p["mark"] = q.mid if not isnan(q.mid) else px

    def _settle(self, sym, p, t, late=False):
        S = self.panel.mark(p["a"], t)
        intrinsic = max(S - p["strike"], 0.0) if p["call"] else max(p["strike"] - S, 0.0)
        cash = p["qty"] * intrinsic * self.mult + 0.0      # + 0.0: no "-0.00" for a short that expired worthless
        self.acct.cash += cash
        self.acct._eq = None
        self.stats["settled_cash"] += cash
        p["cash"] += cash
        self.settlements.append({"date": p["expiry"], "settled_on": self.panel.dates[t][:10], "symbol": sym, "contracts": p["qty"],
                                 "underlying_close": S, "intrinsic": round(intrinsic, 4), "cash": round(cash, 4),
                                 "result": "in the money" if intrinsic > 0 else "expired worthless", "late": late})
        self._close_trade(sym, p, p["expiry"], "settled" if intrinsic > 0 else "expired")

    def _close_trade(self, sym, p, day, how):
        self.trades.append({"symbol": sym, "kind": "call" if p["call"] else "put", "strike": p["strike"], "expiry": p["expiry"],
                            "opened": p["opened"], "closed": day, "side": "long" if p["first_qty"] > 0 else "short",
                            "contracts": p["max_qty"], "pnl": round(p["cash"], 4), "how": how})
        self.pos.pop(sym, None)

    def _mark(self, t, day):
        val = de = ga = th = ve = 0.0
        lo = sh = 0
        for sym, p in self.pos.items():
            ch = self.chains[p["a"]]
            i = ch.find_row(day, p["expiry"], p["call"], p["strike"])
            if i is None:
                self.stale += 1
            else:
                q = self.quote(p["a"], i, t)
                if not isnan(q.mid):
                    p["mark"] = q.mid
                g = q._greeks()
                m = p["qty"] * self.mult
                if not isnan(g["delta"]):
                    de += m * g["delta"]
                if not isnan(g["gamma"]):
                    ga += m * g["gamma"]
                    th += m * g["theta"]
                    ve += m * g["vega"]
            val += p["qty"] * p["mark"] * self.mult
            lo += p["qty"] if p["qty"] > 0 else 0
            sh += -p["qty"] if p["qty"] < 0 else 0
        self.value = val
        self.acct.opt_value = val
        self.acct._eq = None
        for k, v in (("value", val), ("delta", de), ("gamma", ga), ("theta", th), ("vega", ve), ("long", lo), ("short", sh)):
            self.daily[k].append(v)

    def liquidate(self):
        """Equity hit zero: book the options at their marks and stop."""
        self.acct.cash += self.value
        self.pos, self.pending, self.value = {}, [], 0.0
        self.acct.opt_value = 0.0

    def summary(self, dates):
        open_value = self.value
        pnl = sum(f["cash"] for f in self.fills) + self.stats["settled_cash"] + open_value
        wins = [x for x in self.trades if x["pnl"] > 0]
        return {"pnl": pnl, "open_value": open_value, "open_positions": [{"symbol": s, "contracts": p["qty"], "mark": p["mark"], "expiry": p["expiry"]} for s, p in self.pos.items()],
                "fills": self.fills, "settlements": self.settlements, "unfilled": self.unfilled, "trades": self.trades,
                "n_trades": len(self.trades), "win_rate": len(wins) / len(self.trades) if self.trades else NAN,
                "expired_worthless": sum(1 for x in self.settlements if x["result"] == "expired worthless"),
                "in_the_money": sum(1 for x in self.settlements if x["result"] == "in the money"),
                "stale_marks": self.stale, "multiplier": self.mult, "daily": self.daily, "dates": dates,
                "chains": {self.panel.tickers[a]: ch.fingerprint for a, ch in self.chains.items()}, **self.stats}


class OptionsView:
    """ctx.options: the option chain of one underlying, as of today.

        ctx.options.expiries(min_dte=0, max_dte=None)        expiration dates listed today
        ctx.options.chain(expiry=None, kind=None, min_dte=None, max_dte=None)   today's quotes
        ctx.options.find('put', dte=30, delta=-0.25)        nearest contract (or strike=, moneyness=)
        ctx.options.quote(symbol)                            today's quote of one contract
        ctx.options.buy(option, contracts=1) / .sell(...)   fill at the next bar's quotes
        ctx.options.close(option=None)                       close one contract, or all
        ctx.options.positions                                {symbol: contracts}
        ctx.options.value, ctx.options.greeks                marked value; $ Greeks of the positions
    Every quote has symbol, kind, strike, expiry, dte, bid, ask, mid, last, volume,
    open_interest, spot, iv, delta, gamma, theta (per day), vega (per vol point)."""

    def __init__(self, run, a):
        self._run, self._a = run, a

    def _desk(self):
        d = self._run.desk
        if d is None:
            raise ValueError("No option chain is loaded for %s. Drop an option chain file on the Data tab (Docs: Options)."
                             % self._run.acct.panel.tickers[self._a])
        d.chain_for(self._a)
        return d

    @property
    def _day(self):
        return self._run.acct.panel.dates[self._run.t][:10]

    def chain(self, expiry=None, kind=None, min_dte=None, max_dte=None):
        d = self._desk()
        ch = d.chains[self._a]
        s, e = ch.block(self._day)
        if kind not in (None, "call", "put"):
            raise ValueError("kind must be 'call', 'put' or None")
        xi = ch.expiry_index.get(str(expiry)[:10]) if expiry is not None else None
        if expiry is not None and xi is None:
            return []
        out = []
        today = _dt.date.fromisoformat(self._day)
        col = ch.cols
        for i in range(s, e):
            if xi is not None and col["x"][i] != xi:
                continue
            if kind is not None and (col["c"][i] == 1) != (kind == "call"):
                continue
            if min_dte is not None or max_dte is not None:
                dte = (_dt.date.fromisoformat(ch.expiries[col["x"][i]]) - today).days
                if (min_dte is not None and dte < min_dte) or (max_dte is not None and dte > max_dte):
                    continue
            out.append(d.quote(self._a, i, self._run.t))
        return out

    def expiries(self, min_dte=0, max_dte=None):
        d = self._desk()
        ch = d.chains[self._a]
        s, e = ch.block(self._day)
        today = _dt.date.fromisoformat(self._day)
        xs = sorted({ch.expiries[ch.cols["x"][i]] for i in range(s, e)})
        return [x for x in xs if (_dt.date.fromisoformat(x) - today).days >= min_dte and
                (max_dte is None or (_dt.date.fromisoformat(x) - today).days <= max_dte)]

    def find(self, kind, dte=30, delta=None, strike=None, moneyness=None, min_dte=1):
        """The listed contract nearest the target: the expiry closest to `dte` calendar days
        (at least min_dte), then the strike nearest `delta` (puts negative; a positive delta
        for a put means its negative), `strike`, or spot x `moneyness` (at the money if none)."""
        if kind not in ("call", "put"):
            raise ValueError("kind must be 'call' or 'put'")
        exps = self.expiries(min_dte)
        if not exps:
            return None
        today = _dt.date.fromisoformat(self._day)
        exp = min(exps, key=lambda x: (abs((_dt.date.fromisoformat(x) - today).days - dte), -_dt.date.fromisoformat(x).toordinal()))
        qs = [q for q in self.chain(expiry=exp, kind=kind) if q.ask > 0]
        if not qs:
            return None
        if delta is not None:
            target = -abs(delta) if kind == "put" else abs(delta)
            qs = [q for q in qs if not isnan(q.delta)]
            return min(qs, key=lambda q: (abs(q.delta - target), q.strike)) if qs else None
        if strike is None:
            strike = qs[0].spot * (moneyness if moneyness is not None else 1.0)
        return min(qs, key=lambda q: (abs(q.strike - strike), q.strike))

    def quote(self, option):
        d = self._desk()
        sym = option.symbol if isinstance(option, OptionQuote) else str(option)
        root, exp, call, strike = parse_occ(sym)
        i = d.chains[self._a].find_row(self._day, exp, call, strike)
        return None if i is None else d.quote(self._a, i, self._run.t)

    def _order(self, option, contracts):
        d = self._desk()
        sym = option.symbol if isinstance(option, OptionQuote) else str(option)
        if not sym or len(sym) < 16:
            raise ValueError("pass an option from ctx.options.chain()/find() or its symbol, got %r" % (option,))
        if contracts != int(contracts) or contracts == 0:
            raise ValueError("options trade in whole contracts, got %r" % (contracts,))
        d.pending.append((sym, self._a, int(contracts), self._day))

    def buy(self, option, contracts=1):
        if contracts <= 0:
            raise ValueError("buy a positive number of contracts")
        self._order(option, contracts)

    def sell(self, option, contracts=1):
        if contracts <= 0:
            raise ValueError("sell a positive number of contracts (it opens a short if you hold none)")
        self._order(option, -contracts)

    def close(self, option=None):
        d = self._desk()
        syms = [option.symbol if isinstance(option, OptionQuote) else str(option)] if option is not None else \
            [s for s, p in d.pos.items() if p["a"] == self._a]
        for sym in syms:
            p = d.pos.get(sym)
            if p and p["qty"]:
                d.pending.append((sym, self._a, -int(p["qty"]), self._day))

    @property
    def positions(self):
        d = self._desk()
        return {s: p["qty"] for s, p in d.pos.items() if p["a"] == self._a}

    @property
    def pending(self):
        d = self._desk()
        return [{"symbol": s, "contracts": q} for s, a, q, _ in d.pending if a == self._a]

    @property
    def value(self):
        d = self._desk()
        return sum(p["qty"] * p["mark"] * d.mult for p in d.pos.values() if p["a"] == self._a)

    @property
    def greeks(self):
        """$ Greeks of today's positions: delta in shares of the underlying, gamma in shares
        per $1 move, theta in $ per day, vega in $ per volatility point."""
        d = self._desk()
        out = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        for s, p in d.pos.items():
            if p["a"] != self._a:
                continue
            q = self.quote(s)
            if q is None:
                continue
            for k in out:
                v = getattr(q, k)
                if not isnan(v):
                    out[k] += p["qty"] * d.mult * v
        return out

    @property
    def spot(self):
        return self._run.acct.panel.mark(self._a, self._run.t)

    def __repr__(self):
        return "<options on %s: %s>" % (self._run.acct.panel.tickers[self._a], "loaded" if self._run.desk and self._a in self._run.desk.chains else "no chain loaded")


# =============================================================================
# 7e. One strategy on many instruments; the stock dashboard
# =============================================================================

def strategy_breadth(rows, stat="sharpe"):
    """One strategy run on many instruments, each against its own benchmark (usually its
    buy & hold): rows [{ticker, metrics, bench}] or {ticker, error}.

    For `stat`: each instrument's strategy value, benchmark value and improvement (strategy
    minus benchmark, or the reverse for statistics where lower is better), ranked best
    first; how many instruments improved, a one-sided exact sign test of "improves on more
    than half" (ties left out) and a one-sided t-test of the mean improvement."""
    lower = stat in RANK_LOWER_BETTER
    fin = lambda v: v is not None and not isnan(v)
    items, failed, missing = [], [], []
    for r in rows:
        if r.get("error"):
            failed.append({"ticker": r["ticker"], "error": r["error"]})
            continue
        a, b = r["metrics"].get(stat), (r.get("bench") or {}).get(stat)
        if not (fin(a) and fin(b)):
            missing.append(r["ticker"])
            continue
        items.append({"ticker": r["ticker"], "value": a, "bench": b, "improvement": (b - a) if lower else (a - b), "metrics": r["metrics"]})
    items.sort(key=lambda x: (x["value"] if lower else -x["value"], x["ticker"]))
    for k, x in enumerate(items):
        x["rank"] = k + 1
    n = len(items)
    out = {"stat": stat, "lower_better": lower, "rows": items, "failed": failed, "missing": missing, "n": n}
    if not n:
        return out
    imp = [x["improvement"] for x in items]
    wins, losses = sum(1 for v in imp if v > 0), sum(1 for v in imp if v < 0)
    out.update(wins=wins, losses=losses, ties=n - wins - losses, share=wins / n,
               sign_p=binom_sf(wins, wins + losses, 0.5) if wins + losses else NAN,
               median_value=percentile([x["value"] for x in items], 50), median_bench=percentile([x["bench"] for x in items], 50),
               mean_improvement=mean(imp), median_improvement=percentile(imp, 50))
    if n >= 3:
        sd = stdev(imp)
        t = mean(imp) / (sd / math.sqrt(n)) if sd > 0 else NAN
        out.update(t_stat=t, t_p=t_sf(t, n - 1) if not isnan(t) else NAN)
    return out


DASH_EVENTS = [
    # key, label, side, what it means
    ("golden_cross", "Golden cross", "bull", "The 50-day average crosses above the 200-day."),
    ("death_cross", "Death cross", "bear", "The 50-day average crosses below the 200-day."),
    ("above_sma200", "Close crosses above 200-day", "bull", "The close moves from below to above its 200-day average."),
    ("below_sma200", "Close crosses below 200-day", "bear", "The close moves from above to below its 200-day average."),
    ("macd_bull", "MACD bullish cross", "bull", "MACD(12, 26) crosses above its 9-day signal line."),
    ("macd_bear", "MACD bearish cross", "bear", "MACD(12, 26) crosses below its 9-day signal line."),
    ("rsi_oversold", "RSI below 30", "bull", "RSI(14) falls below 30 (oversold)."),
    ("rsi_overbought", "RSI above 70", "bear", "RSI(14) rises above 70 (overbought)."),
    ("bb_break_down", "Close below lower Bollinger", "bull", "The close crosses below the lower Bollinger band (20, 2)."),
    ("bb_break_up", "Close above upper Bollinger", "bear", "The close crosses above the upper Bollinger band (20, 2)."),
    ("high_52w", "New 52-week high", "bull", "The close beats the highest close of the previous 252 bars, after not doing so the bar before."),
    ("low_52w", "New 52-week low", "bear", "The close falls below the lowest close of the previous 252 bars, after not doing so the bar before."),
]


def _clean(xs):
    return [None if (v is None or isnan(v)) else round(v, 6) for v in xs]


def dashboard(ds, start=None, end=None, sma_periods=(20, 50, 200), ema_periods=(12, 26), horizons=(5, 20, 60)):
    """Everything the stock dashboard shows for ds between start and end: prices, moving
    averages & channels, oscillators, pattern events, the latest readings and an event study.

    Indicators are computed on the whole history (so they are warmed up at `start`) and
    each uses only bars up to its own. An event on bar t depends only on bars t-1 and t.
    The event study measures, for each pattern, the return over the next h bars (dividends
    included when known) against the average over all bars in the range: a description
    of the past, with overlapping windows, not a test."""
    c, h, l, o, v = ds.close, ds.high, ds.low, ds.open, ds.volume
    n = ds.n
    s, e = _resolve_range(ds.dates, start, end, ds.ticker)
    sl = slice(s, e + 1)
    over = {}
    for k in sma_periods:
        over["sma_%d" % k] = sma(c, k)
    for k in ema_periods:
        over["ema_%d" % k] = ema(c, k)
    bb_mid, bb_up, bb_lo = bbands(c, 20, 2.0)
    kc_mid, kc_up, kc_lo = keltner(h, l, c, 20, 2.0)
    dc_up, dc_lo, dc_mid = donchian(h, l, 20)
    over.update(bb_mid=bb_mid, bb_up=bb_up, bb_lo=bb_lo, kc_mid=kc_mid, kc_up=kc_up, kc_lo=kc_lo, dc_up=dc_up, dc_lo=dc_lo, dc_mid=dc_mid)
    r14 = rsi(c, 14)
    m_line, m_sig, m_hist = macd(c, 12, 26, 9)
    st_k, st_d = stochastic(h, l, c, 14, 3)
    a_adx, a_pdi, a_mdi = adx(h, l, c, 14)
    a14 = atr(h, l, c, 14)
    panes = {"rsi": r14, "macd": m_line, "macd_signal": m_sig, "macd_hist": m_hist, "stoch_k": st_k, "stoch_d": st_d,
             "adx": a_adx, "plus_di": a_pdi, "minus_di": a_mdi, "atr": a14, "atr_pct": [_div(a, x) if not isnan(a) else NAN for a, x in zip(a14, c)],
             "obv": obv(c, v), "williams_r": williams_r(h, l, c, 14), "cci": cci(h, l, c, 20), "roc": roc(c, 20),
             "vol20": realized_vol(c, 20)}
    s50, s200 = sma(c, 50), sma(c, 200)
    lvl = lambda x: [x] * n
    prior_hi, prior_lo = shift(rolling_max(c, 252), 1), shift(rolling_min(c, 252), 1)
    flags = {
        "golden_cross": crossover(s50, s200), "death_cross": crossunder(s50, s200),
        "above_sma200": crossover(c, s200), "below_sma200": crossunder(c, s200),
        "macd_bull": crossover(m_line, m_sig), "macd_bear": crossunder(m_line, m_sig),
        "rsi_oversold": crossunder(r14, lvl(30.0)), "rsi_overbought": crossover(r14, lvl(70.0)),
        "bb_break_down": crossunder(c, bb_lo), "bb_break_up": crossover(c, bb_up),
        "high_52w": crossover(c, prior_hi), "low_52w": crossunder(c, prior_lo),
    }
    tr = ds.total_return_index()
    base = {}
    for hz in horizons:
        f = [tr[t + hz] / tr[t] - 1 for t in range(s, e + 1) if t + hz < n]
        base[hz] = {"n": len(f), "mean": mean(f) if f else NAN, "hit": sum(1 for x in f if x > 0) / len(f) if f else NAN}
    events = []
    for key, label, side, about in DASH_EVENTS:
        idx = [t for t in range(s, e + 1) if flags[key][t]]
        study = {}
        for hz in horizons:
            f = [tr[t + hz] / tr[t] - 1 for t in idx if t + hz < n]
            study[hz] = {"n": len(f), "mean": mean(f) if f else NAN, "median": percentile(f, 50) if f else NAN,
                         "hit": sum(1 for x in f if x > 0) / len(f) if f else NAN,
                         "vs_all": (mean(f) - base[hz]["mean"]) if f and not isnan(base[hz]["mean"]) else NAN}
        events.append({"key": key, "label": label, "side": side, "about": about, "count": len(idx),
                       "dates": [ds.dates[t] for t in idx], "index": [t - s for t in idx],
                       "last": ds.dates[idx[-1]] if idx else None, "study": study})
    # the latest readings
    t = e
    at = lambda xs: xs[t] if t < len(xs) else NAN
    year = max(s, t - 251)
    hi52, lo52 = max(c[year:t + 1]), min(c[year:t + 1])
    ago = lambda k: (tr[t] / tr[t - k] - 1) if t - k >= 0 else NAN
    read = {"date": ds.dates[t], "close": c[t], "sma50": at(s50), "sma200": at(s200),
            "pct_vs_sma50": _div(c[t], at(s50)) - 1 if not isnan(at(s50)) else NAN,
            "pct_vs_sma200": _div(c[t], at(s200)) - 1 if not isnan(at(s200)) else NAN,
            "trend": ("up" if at(s50) > at(s200) else "down") if not (isnan(at(s50)) or isnan(at(s200))) else None,
            "rsi": at(r14), "macd": at(m_line), "macd_signal": at(m_sig), "macd_hist": at(m_hist),
            "stoch_k": at(st_k), "stoch_d": at(st_d), "adx": at(a_adx), "plus_di": at(a_pdi), "minus_di": at(a_mdi),
            "atr": at(a14), "atr_pct": _div(at(a14), c[t]) if not isnan(at(a14)) else NAN,
            "bb_pct_b": _div(c[t] - at(bb_lo), at(bb_up) - at(bb_lo)) if not isnan(at(bb_up)) and at(bb_up) != at(bb_lo) else NAN,
            "bb_width": _div(at(bb_up) - at(bb_lo), at(bb_mid)) if not isnan(at(bb_mid)) else NAN,
            "high_52w": hi52, "low_52w": lo52, "pos_52w": _div(c[t] - lo52, hi52 - lo52) if hi52 > lo52 else NAN,
            "ret_1m": ago(21), "ret_3m": ago(63), "ret_6m": ago(126), "ret_1y": ago(252), "vol20": at(panes["vol20"]),
            "williams_r": at(panes["williams_r"]), "cci": at(panes["cci"])}
    return {"ticker": ds.ticker, "dates": ds.dates[sl], "open": _clean(o[sl]), "high": _clean(h[sl]), "low": _clean(l[sl]),
            "close": _clean(c[sl]), "volume": [x for x in v[sl]], "overlays": {k: _clean(x[sl]) for k, x in over.items()},
            "panes": {k: _clean(x[sl]) for k, x in panes.items()}, "events": events, "baseline": base, "readings": read,
            "horizons": list(horizons), "sma_periods": list(sma_periods), "ema_periods": list(ema_periods),
            "total_return": ds.dividend_source is not None or ds.price_basis == "total_return"}


# =============================================================================
# 8. JSON API for the web page
# =============================================================================

DATA = {}


def _sanitize(x):
    """Replace NaN and +/-Infinity with None (JSON has neither), keeping everything else.

    A list whose sum() is finite holds only finite numbers, and sum() runs in C, so
    long numeric series pass through without a Python-level loop."""
    t = type(x)
    if t is float:
        return x if x - x == 0 else None
    if t is list or t is tuple:
        try:
            if math.isfinite(sum(x)):
                return x
        except TypeError:
            pass
        return [_sanitize(v) for v in x]
    if t is dict:
        return {(k if type(k) is str else str(k)): _sanitize(v) for k, v in x.items()}
    return x


_BLENDS = {}


def parse_blend(spec):
    """'SPY 60, TLT 40' / 'SPY:0.6,TLT:0.4' / '60% SPY + 40% TLT' -> ([(ticker, weight)], rebalance).
    Weights are normalized to sum to 1; 'daily' or 'monthly' (default) may follow a '|'."""
    import re
    text, _, reb = str(spec).partition("|")
    reb = reb.strip().lower() or "monthly"
    if reb not in ("daily", "monthly", "quarterly", "none"):
        raise ValueError("Rebalance a blend daily, monthly, quarterly or none, not %r." % reb)
    parts = [x.strip() for x in re.split(r"[,+;/]", text) if x.strip()]
    out = []
    for part in parts:
        m = re.fullmatch(r"([A-Za-z0-9.^=\-]+)\s*[: ]\s*(-?[\d.]+)\s*%?|(-?[\d.]+)\s*%?\s+([A-Za-z0-9.^=\-]+)", part)
        if not m:
            raise ValueError("Couldn't read %r: write blends like 'SPY 60, TLT 40'." % part)
        tk, w = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        out.append((tk.upper(), float(w)))
    total = sum(w for _, w in out)
    if len(out) < 2 or total <= 0:
        raise ValueError("A blend needs two or more instruments with positive weights.")
    return [(t, w / total) for t, w in out], reb


def is_blend(spec):
    return bool(spec) and any(c in str(spec) for c in ",+;:|") and str(spec).upper() not in DATA


def blended_benchmark(sets, weights, rebalance="monthly", name=None):
    """A benchmark that holds fixed weights of several instruments (dividends reinvested),
    rebalanced back to them daily, monthly, quarterly or never. Returns a DataSet whose
    closes are the blend's value (its price basis is total return)."""
    key = (tuple((ds.ticker, ds.fingerprint()) for ds in sets), tuple(weights), rebalance)
    if key in _BLENDS:
        return _BLENDS[key]
    panel = Panel(sets)
    tr = [panel.aligned(a, "total_return") for a in range(panel.k)]
    start = next(t for t in range(panel.n) if all(not isnan(x[t]) for x in tr))   # every part has a price
    value, units, out, prev = 1.0, None, [], None
    for t in range(start, panel.n):
        px = [x[t] for x in tr]
        if units is not None:
            value = sum(u * p for u, p in zip(units, px))
        period = panel.dates[t][:7] if rebalance == "monthly" else (
            panel.dates[t][:4] + str((int(panel.dates[t][5:7]) - 1) // 3) if rebalance == "quarterly" else None)
        if units is None or rebalance == "daily" or (period is not None and period != prev):
            units = [w * value / p for w, p in zip(weights, px)]
        prev = period
        out.append(value * 100.0)
    label = name or " / ".join("%g%% %s" % (round(100 * w, 2), ds.ticker) for ds, w in zip(sets, weights))
    ds = DataSet(label, panel.dates[start:], out)
    ds.price_basis = "total_return"
    ds.blend = {"weights": {d.ticker: w for d, w in zip(sets, weights)}, "rebalance": rebalance}
    if len(_BLENDS) > 16:
        _BLENDS.pop(next(iter(_BLENDS)))
    _BLENDS[key] = ds
    return ds


def benchmark_from_spec(spec, base="USD"):
    """A loaded ticker, or a blend spec (see parse_blend), as a DataSet in the base currency."""
    if not spec:
        return None
    if is_blend(spec):
        parts, reb = parse_blend(spec)
        missing = [t for t, _ in parts if t not in DATA]
        if missing:
            raise ValueError("The benchmark blend uses %s, which %s not loaded." % (", ".join(missing), "is" if len(missing) == 1 else "are"))
        return blended_benchmark([in_base(DATA[t], base) for t, _ in parts], [w for _, w in parts], reb)
    return in_base(_get(str(spec).upper() if str(spec).upper() in DATA else spec), base)


def compare_benchmarks(res, specs):
    """Relative statistics of a Result against extra benchmarks: [{name, equity, metrics}]."""
    out = []
    keys = ("beta", "alpha", "correlation", "r_squared", "tracking_error", "information_ratio", "excess_cagr",
            "up_capture", "down_capture", "beat_months")
    for spec in specs or []:
        try:
            b = benchmark_from_spec(spec, res.config.base_currency)
            s0, vals, notes = align_benchmark(res.panel.dates, b, res.start, res.start + len(res.dates) - 1, res.config.total_return)
        except ValueError as e:
            out.append({"name": str(spec), "error": str(e)})
            continue
        off = s0 - res.start
        dates, eq = res.dates[off:], res.equity[off:]
        beq = [res.equity[off] * v / vals[0] for v in vals]
        m = performance(dates, eq, res.config.risk_free, beq, ann=res_ann(res))
        own = performance(dates, beq, res.config.risk_free, ann=res_ann(res))
        rel = [a / x for a, x in zip(eq, beq)]
        idx = _downsample(len(dates))
        out.append({"name": b.ticker, "spec": str(spec), "start": dates[0], "notes": notes,
                    "metrics": {k: m.get(k, NAN) for k in keys},
                    "own": {k: own.get(k, NAN) for k in ("total_return", "cagr", "volatility", "sharpe", "max_drawdown")},
                    "equity": [[i + off, beq[i]] for i in idx], "relative": [[i + off, rel[i] / rel[0]] for i in idx]})
    return out


def _get(ticker):
    if ticker not in DATA:
        raise ValueError("No data loaded for %s." % ticker)
    return DATA[ticker]


_FX_CACHE = {}


def fx_series(ccy, base):
    """The loaded FX rate that converts `ccy` into `base`: (DataSet, invert). EURUSD (dollars
    per euro) multiplies euro prices; USDEUR (euros per dollar) divides them. Yahoo-style
    names such as EURUSD=X work too."""
    ccy, base = ccy.upper(), base.upper()
    for name, inv in ((ccy + base, False), (base + ccy, True), (ccy + base + "=X", False), (base + ccy + "=X", True)):
        if name in DATA:
            return DATA[name], inv
    raise ValueError("Converting %s to %s needs an FX rate: load %s%s (or %s%s) prices on the Data tab."
                     % (ccy, base, ccy, base, base, ccy))


def to_base_currency(ds, fx, invert=False, base="USD"):
    """A copy of ds with prices and dividends in the base currency, unhedged: each bar is
    multiplied by the FX close known at that bar (the same day's for daily data; the
    previous day's for intraday bars with a daily FX series). Bars before the FX history
    starts are dropped. Volume stays in shares; fundamentals stay in their own currency."""
    key = (ds.ticker, ds.fingerprint(), fx.ticker, fx.fingerprint(), invert, base, ds.calendar)
    if key in _FX_CACHE:
        return _FX_CACHE[key]
    daily_fx = all(len(d) == 10 for d in fx.dates[:5])
    rows, j, rate = [], -1, None
    for i, d in enumerate(ds.dates):
        intraday = len(d) > 10
        while j + 1 < fx.n and (fx.dates[j + 1] < d[:10] if (intraday and daily_fx) else fx.dates[j + 1] <= d):
            j += 1
            rate = 1.0 / fx.close[j] if invert else fx.close[j]
        if rate is not None:
            rows.append((i, rate))
    if len(rows) < 2:
        raise ValueError("%s has no FX rates for %s's dates." % (fx.ticker, ds.ticker))
    idx = [i for i, _ in rows]
    r = [x for _, x in rows]
    out = DataSet(ds.ticker, [ds.dates[i] for i in idx], [ds.close[i] * x for i, x in zip(idx, r)],
                  [ds.open[i] * x for i, x in zip(idx, r)], [ds.high[i] * x for i, x in zip(idx, r)],
                  [ds.low[i] * x for i, x in zip(idx, r)], [ds.volume[i] for i in idx])
    out.dividends = [ds.dividends[i] * x for i, x in zip(idx, r)]
    out.dividend_source, out.price_basis, out.calendar = ds.dividend_source, ds.price_basis, ds.calendar
    out.fundamentals = dict(ds.fundamentals)
    out.issues = ds.issues
    out.currency = base
    out.source_currency = ds.currency
    out.fx_note = "%s prices converted from %s to %s with %s%s (unhedged: returns include the currency's moves)%s." % (
        ds.ticker, ds.currency, base, fx.ticker, " (inverted)" if invert else "",
        "" if idx[0] == 0 else "; bars before %s, the first FX rate, were left out" % fx.dates[0])
    out._cache.clear()
    if len(_FX_CACHE) > 32:
        _FX_CACHE.pop(next(iter(_FX_CACHE)))
    _FX_CACHE[key] = out
    return out


def in_base(ds, base):
    """ds converted to the base currency if it is priced in another one."""
    if ds is None or not ds.currency or ds.currency.upper() == base.upper():
        return ds
    fx, inv = fx_series(ds.currency, base)
    return to_base_currency(ds, fx, inv, base)


def continuous_futures(contracts, ticker="CONT", roll="days", days_before=5, adjust="ratio"):
    """Stitch single-contract histories (oldest expiry first) into one continuous series.

    roll='days' moves to the next contract `days_before` bars before the current one's last
    bar; roll='volume' moves on the first day the next contract trades more (and never back).
    adjust='ratio' multiplies all earlier prices by next/current at each roll (keeps returns
    exact and prices positive), 'difference' adds the gap (keeps price changes exact; long
    histories can go negative), 'none' splices raw prices (the roll gaps become fake
    returns). Returns a DataSet of kind 'future' (no cash outlay, see Account.accrue) with
    a .rolls list [{date, from, to, gap}]."""
    if len(contracts) < 1:
        raise ValueError("Give at least one contract.")
    if adjust not in ("ratio", "difference", "none") or roll not in ("days", "volume"):
        raise ValueError("adjust must be ratio, difference or none, and roll days or volume")
    segs, rolls = [], []                      # (contract index, [dates]) pieces in order
    cur, start = 0, None
    for i, ds in enumerate(contracts):
        if i and ds.dates[-1] <= contracts[i - 1].dates[-1]:
            raise ValueError("Contracts must be in order of expiry: %s ends before %s." % (ds.ticker, contracts[i - 1].ticker))
    for i, ds in enumerate(contracts):
        nxt = contracts[i + 1] if i + 1 < len(contracts) else None
        lo = 0 if start is None else _bisect_left(ds.dates, start)
        if nxt is None:
            hi = ds.n
        elif roll == "days":
            hi = max(lo + 1, ds.n - int(days_before))
            while hi < ds.n and ds.dates[hi] not in nxt.index:
                hi += 1                        # roll on a day both contracts trade
        else:
            hi = ds.n
            for j in range(lo + 1, ds.n):
                d = ds.dates[j]
                k = nxt.index.get(d)
                if k is not None and nxt.volume[k] > ds.volume[j]:
                    hi = j
                    break
        if hi <= lo:
            continue
        segs.append((i, lo, hi))
        if nxt is not None and hi < ds.n:
            start = ds.dates[hi]
        elif nxt is not None:
            start = ds.dates[-1]
    dates, o, h, l, c, v = [], [], [], [], [], []
    for n_seg, (i, lo, hi) in enumerate(segs):
        ds = contracts[i]
        if n_seg:                              # a roll: adjust everything before it
            d = ds.dates[lo]
            prev = contracts[segs[n_seg - 1][0]]
            pj = prev.index.get(d)
            old = prev.close[pj] if pj is not None else c[-1]
            new = ds.close[lo]
            gap = new - old
            rolls.append({"date": d, "from": prev.ticker, "to": ds.ticker, "gap": gap, "ratio": new / old if old else NAN})
            if adjust == "ratio" and old > 0:
                f = new / old
                o, h, l, c = [x * f for x in o], [x * f for x in h], [x * f for x in l], [x * f for x in c]
            elif adjust == "difference":
                o, h, l, c = [x + gap for x in o], [x + gap for x in h], [x + gap for x in l], [x + gap for x in c]
        for j in range(lo, hi):
            dates.append(ds.dates[j]); o.append(ds.open[j]); h.append(ds.high[j]); l.append(ds.low[j])
            c.append(ds.close[j]); v.append(ds.volume[j])
    bad = sum(1 for x in c if x <= 0)
    keep = [j for j in range(len(c)) if c[j] > 0]
    out = DataSet(ticker, [dates[j] for j in keep], [c[j] for j in keep], [o[j] for j in keep],
                  [h[j] for j in keep], [l[j] for j in keep], [v[j] for j in keep])
    out.kind = "future"
    out._cache.clear()
    out.rolls = rolls
    if bad:
        out._issue("dropped", dates[0], "%d bars went to zero or below with difference adjustment and were left out; "
                   "use ratio adjustment" % bad)
    return out


def _prepare(p):
    """Payload -> (datasets, benchmark, config, strategy, portfolio, rates).

    A single-instrument run sends 'ticker'; a portfolio run sends 'tickers' (or
    'portfolio': true with one ticker). 'benchmark' empty or equal to the single
    ticker means buy and hold of that ticker (single) or equal weight (portfolio)."""
    strat = Strategy.from_code(p["code"], p.get("params"))
    portfolio = bool(p.get("tickers")) or bool(p.get("portfolio")) or bool(strat.universe)
    tickers = list(p.get("tickers") or (strat.universe if strat.universe and not p.get("portfolio") else [p["ticker"]]))
    missing = [t for t in tickers if t not in DATA]
    if missing and strat.universe and not p.get("tickers"):
        raise ValueError("This strategy's UNIVERSE lists %s, which %s not loaded. Add %s on the Data tab."
                         % (", ".join(missing), "is" if len(missing) == 1 else "are", "it" if len(missing) == 1 else "them"))
    sets = [_get(t) for t in tickers]
    b = p.get("benchmark") or ""
    bench = None if not b or (not portfolio and b == tickers[0]) else ("blend" if is_blend(b) else _get(b))
    cfg = Config(**(p.get("config") or {}))
    sets = [in_base(ds, cfg.base_currency) for ds in sets]
    bench = benchmark_from_spec(b, cfg.base_currency) if bench == "blend" else in_base(bench, cfg.base_currency)
    rates = None
    if cfg.rate_ticker:
        if cfg.rate_ticker not in RATES:
            raise ValueError("No rate series named %s is loaded. Add it on the Data tab." % cfg.rate_ticker)
        rates = RATES[cfg.rate_ticker]
    return sets, bench, cfg, strat, portfolio, rates


def _signal_key(p, strat):
    """What signals(data) depends on besides the data: the code, its parameters and the modules it imported."""
    import hashlib
    mods = "".join("\0%s\0%s" % (n, MODULES.get(n, "")) for n in strat.modules)
    return hashlib.sha256((p["code"] + "\0" + json.dumps(p.get("params"), sort_keys=True) + mods).encode()).hexdigest()


def _run_payload(p, strat=None, cfg=None):
    sets, bench, cfg0, strat0, portfolio, rates = _prepare(p)
    strat, cfg = strat or strat0, cfg or cfg0
    key = _signal_key(p, strat)
    last = bool(p.get("decide_last"))
    if portfolio:
        res = backtest_portfolio(strat, sets, cfg, p.get("start"), p.get("end"), bench, rates=rates, signal_key=key, decide_last=last)
    else:
        res = backtest(strat, sets[0], cfg, p.get("start"), p.get("end"), bench, rates=rates, signal_key=key, decide_last=last)
    return res, strat


LOOKAHEAD_CACHE = {}


def _downsample(n, limit=600):
    step = max(1, -(-n // limit))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def api_load_data(p):
    """Load prices (and optionally 'dividends' / 'splits' as [[date, value], ...])."""
    ds = DataSet.from_dict(p["ticker"], p["data"])
    if p.get("asset_kind") == "future":
        ds.kind = "future"
        ds._cache.clear()
    if p.get("currency"):
        c = str(p["currency"]).upper()
        if len(c) != 3 or not c.isalpha():
            raise ValueError("Currency must be a 3-letter code such as EUR, not %r." % p["currency"])
        ds.currency = c
    if p.get("calendar"):
        if p["calendar"] not in CALENDARS:
            raise ValueError("Unknown calendar %r; use one of %s." % (p["calendar"], ", ".join(CALENDARS)))
        ds.calendar = p["calendar"]
    if p.get("splits"):
        ds.apply_splits(p["splits"])
    if p.get("dividends"):
        ds.set_dividends(p["dividends"])
    if p.get("fundamentals"):
        ds.set_fundamentals(p["fundamentals"])
    DATA[p["ticker"]] = ds
    return dict(_data_info(ds), quality=quality_report(ds))


def _data_info(ds):
    return {"ticker": ds.ticker, "bars": ds.n, "start": ds.dates[0], "end": ds.dates[-1],
            "dividends": ds.dividend_source, "dividend_count": sum(1 for d in ds.dividends if d),
            "dividend_total": sum(ds.dividends), "price_basis": ds.price_basis,
            "splits": ds.splits_applied, "fingerprint": ds.fingerprint(), "currency": ds.currency, "asset_kind": ds.kind,
            "fundamentals": {f: {"count": len(v[0]), "first": v[0][0], "last": v[0][-1]} for f, v in ds.fundamentals.items()}}


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
    if kind == "fundamentals":
        f = parse_fundamentals_csv(p["text"], p.get("lag_days", FUNDAMENTAL_LAG_DAYS))
        return {"kind": kind, **f}
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


LAST_RUN = {}      # the latest api_run Result, so exports can fetch every fill & trade
ROW_LIMIT = 5000   # fills and trades sent with a run; the rest stay here until asked for


def _fill_rows(res, fills):
    return [{"i": f["i"] - res.start, "date": f["date"], "ticker": f["ticker"], "side": f["side"],
             "qty": f["qty"], "price": f["price"], "commission": f["commission"],
             "position": f["position"], "reason": f.get("reason"), "kind": f.get("kind"),
             "arrival": f.get("arrival")} for f in fills]


def api_rows(p):
    """Every fill or trade of the latest run (the run output keeps only the last ROW_LIMIT)."""
    res = LAST_RUN.get("res")
    if res is None:
        raise ValueError("No run to export yet.")
    kind = p.get("kind")
    if kind == "logs":
        return res.full_logs
    return _fill_rows(res, res.fills) if kind == "fills" else res.trades


def api_run(p):
    import time
    t0 = time.perf_counter()
    res, strat = _run_payload(p)
    t1 = time.perf_counter()
    cfg = res.config
    m, b = summarize(res, trials=p.get("trials"))
    t2 = time.perf_counter()
    window = int(p.get("rolling_window") or 126)
    out = {
        "ticker": res.ticker, "tickers": res.tickers, "portfolio": res.portfolio,
        "benchmark": res.benchmark, "mode": res.mode, "params": res.params,
        "dates": res.dates, "close": res.close, "equity": res.equity, "bench_equity": res.bench_equity,
        "drawdown": drawdown(res.equity), "bench_drawdown": drawdown(res.bench_equity),
        "exposure": res.exposure, "position": res.position,
        "gross": res.gross, "net": res.net, "long": res.long, "short": res.short,
        "fills": _fill_rows(res, res.fills[-ROW_LIMIT:]), "fills_total": len(res.fills),
        "trades": res.trades[-ROW_LIMIT:], "trades_total": len(res.trades), "metrics": m, "bench_metrics": b,
        "carry": res.carry, "margin_calls": res.margin_calls[:50], "book": res.book_stats, "limits": res.limits, "next": res.next,
        "relative": [a / b / (res.equity[0] / res.bench_equity[0]) if b > 0 else NAN for a, b in zip(res.equity, res.bench_equity)],
        "benchmarks": compare_benchmarks(res, p.get("benchmarks")),
        "monthly": monthly_table(res.dates, res.equity),
        "bench_monthly": monthly_table(res.dates, res.bench_equity),
        "hist": histogram([daily_returns(res.equity), daily_returns(res.bench_equity)]),
        "rolling": {"window": window, "sharpe": rolling_sharpe(res.equity, window, cfg.risk_free, res_ann(res)),
                    "bench": rolling_sharpe(res.bench_equity, window, cfg.risk_free, res_ann(res)),
                    "vol_window": 63, "vol": rolling_vol(res.equity, 63, res_ann(res)), "bench_vol": rolling_vol(res.bench_equity, 63, res_ann(res))},
        "annual": [{"year": y, "strategy": a, "bench": b} for (y, a), (_, b) in
                   zip(period_returns(res.dates, res.equity, "year"), period_returns(res.dates, res.bench_equity, "year"))],
        "logs": res.logs, "logs_total": len(res.full_logs), "notes": res.notes, "config": cfg.as_dict(),
        "data": {tk: _data_info(ds) for tk, ds in zip(res.tickers, res.panel.sets)},
        "provenance": provenance(res, p["code"], strat.modules),
    }
    if res.options:
        o = res.options
        out["options"] = {k: v for k, v in o.items() if k not in ("fills", "settlements", "unfilled", "trades")}
        out["options"].update(fills=o["fills"][-ROW_LIMIT:], fills_total=len(o["fills"]), settlements=o["settlements"][-ROW_LIMIT:],
                              unfilled=o["unfilled"][-ROW_LIMIT:], trades=o["trades"][-ROW_LIMIT:], trades_total=len(o["trades"]))
    if res.portfolio:
        idx = _downsample(len(res.dates))
        out["attribution"] = attribution(res)
        out["weights"] = {"index": idx, "series": {tk: [res.weights[tk][i] for i in idx] for tk in res.tickers},
                          "full": {tk: [round(w, 6) for w in res.weights[tk]] for tk in res.tickers}}
    t3 = time.perf_counter()
    if res.signals is not None:
        data = res.panel if res.portfolio else res.panel.sets[0]
        lk = (res.signal_key, res.config.seed, res.portfolio, tuple(ds.fingerprint() for ds in res.panel.sets))
        out["lookahead"] = LOOKAHEAD_CACHE.get(lk) or _remember(LOOKAHEAD_CACHE, lk, lookahead_check(
            strat, data, res.signals, budget=float(p.get("lookahead_budget", 4.0))))
    out["timing"] = {"backtest": t1 - t0, "statistics": t2 - t1, "charts": t3 - t2, "lookahead": time.perf_counter() - t3}
    LAST_RUN["res"] = res
    return out


def provenance(res, code, modules=()):
    """What exactly produced a result: enough to reproduce it or tell why it changed."""
    import hashlib
    return {"engine_version": ENGINE_VERSION, "python": sys.version.split()[0],
            "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "modules": module_hashes(modules),
            "params": dict(res.params), "config": res.config.as_dict(),
            "data": {tk: ds.fingerprint() for tk, ds in zip(res.tickers, res.panel.sets)},
            "calendars": {tk: guess_calendar(ds) for tk, ds in zip(res.tickers, res.panel.sets)}, "periods_per_year": res_ann(res),
            "start": res.dates[0], "end": res.dates[-1], "bars": len(res.dates), "benchmark": res.benchmark,
            "option_chains": dict(res.options["chains"]) if getattr(res, "options", None) else {}}


def api_metrics(p):
    """Lightweight run for parameter sweeps: metrics only, optionally split. 'blocks': n
    adds block_stats for PBO; 'returns': true adds the daily returns (for walk-forward)."""
    res, _ = _run_payload(p)
    m, bm = summarize(res)
    out = {"params": res.params, "full": m, "ann": res_ann(res)}
    if p.get("bench_metrics"):
        out["bench"] = bm
        out["start"], out["end"], out["bars"] = res.dates[0], res.dates[-1], len(res.dates)
    r = daily_returns(res.equity)
    if p.get("blocks"):
        out["blocks"] = block_stats(r, int(p["blocks"]))
    if p.get("returns"):
        out["returns"] = [round(x, 10) for x in r]
        out["dates"] = res.dates
        out["bench_returns"] = [round(x, 10) for x in daily_returns(res.bench_equity)]
    if p.get("split") is not None:
        k = int(round(float(p["split"]) * (len(res.dates) - 1)))
        sm = split_metrics(res, k)
        out["in_sample"] = sm["in_sample"]["metrics"]
        out["out_of_sample"] = sm["out_of_sample"]["metrics"]
        out["split_date"] = sm["split_date"]
    return out


def api_montecarlo(p):
    res, _ = _run_payload(p)
    mc = monte_carlo(daily_returns(res.equity), p.get("paths", 500), p.get("block", 20), p.get("seed", 7), ann=res_ann(res),
                     rfd=(1 + res.config.risk_free) ** (1 / res_ann(res)) - 1)
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


def api_surface_grid(p):
    """Window edges for the start/end surfaces over the run's dates."""
    sets, _, _, _, portfolio, _ = _prepare(p)
    dates = shared_panel(sets).dates if portfolio else sets[0].dates
    grid = surface_grid(dates, p.get("step", "quarter"), p.get("start"), p.get("end"))
    return {"grid": grid, "stats": SURFACE_STATS}


def api_surface(p):
    """Rows of the start/end surfaces for the given start dates: {start, S, B, ends} each,
    with S & B the statistic of the strategy & the benchmark on [start, e] for every e in ends."""
    sets, bench, cfg, strat, portfolio, rates = _prepare(p)
    stat = p.get("stat", "sharpe")
    if stat not in SURFACE_STATS:
        raise ValueError("unknown statistic %r (use %s)" % (stat, ", ".join(SURFACE_STATS)))
    key = _signal_key(p, strat)
    ends = sorted(p["ends"])
    rows = []
    for st in p["starts"]:
        later = [e for e in ends if e > st]
        if not later:
            continue
        row = surface_row(lambda: Strategy.from_code(p["code"], p.get("params")), sets, cfg, st, later, stat, bench, portfolio, rates, key)
        full_s, full_b = [None] * len(ends), [None] * len(ends)
        for e, a, b in zip(later, row["S"], row["B"]):
            j = ends.index(e)
            full_s[j], full_b[j] = a, b
        rows.append({"start": st, "actual_start": row["start"], "S": full_s, "B": full_b})
    return {"stat": stat, "ends": ends, "rows": rows}


def api_costs(p):
    sets, bench, cfg, _, portfolio, rates = _prepare(p)
    return cost_sensitivity(lambda: Strategy.from_code(p["code"], p.get("params")), sets if portfolio else sets[0],
                            cfg, p.get("levels") or [0, 2, 5, 10, 20, 50], p.get("start"), p.get("end"), bench, rates)


def api_rank(p):
    """Rank finished strategy runs (see rank_strategies)."""
    return rank_strategies(p["runs"], p.get("stat", "sharpe"), float(p.get("rf") or 0.0))


def api_option_rows(p):
    """Parse an option chain file (see option_rows); for checks & small files."""
    rows, info = option_rows(p["text"], p.get("max_dte"), p.get("band"), p.get("expiries", "all"), p.get("start"), p.get("end"),
                             p.get("spot"), p.get("underlying"))
    return {"rows": rows[: int(p.get("limit", 1000))], "info": info}


def load_option_chain(underlying, dates, expiries, cols, fingerprint=None, info=None):
    """Keep an option chain for `underlying` (columns as lists or raw bytes; see OptionChain)."""
    ch = OptionChain(underlying, dates, expiries, cols, fingerprint, info)
    OPTIONS[ch.underlying] = ch
    return ch.summary()


def load_option_buffers(underlying, meta_json, buffers):
    """The page's binary path: buffers is a {column: typed array} proxy from JavaScript."""
    meta = json.loads(meta_json)
    has = getattr(buffers, "has", None) or buffers.__contains__        # a JS Map from the page, or a dict
    cols = {k: buffers.get(k).to_bytes() for k in OptionChain.CODES if has(k)}
    return json.dumps({"ok": True, "result": load_option_chain(underlying, meta["dates"], meta["expiries"], cols, meta.get("fingerprint"), meta.get("info"))})


def api_load_options(p):
    return load_option_chain(p["underlying"], p["dates"], p["expiries"], p["cols"], p.get("fingerprint"), p.get("info"))


def api_forget_options(p):
    OPTIONS.pop(str(p["underlying"]).upper(), None)
    return True


def api_breadth(p):
    """Summarize one strategy run on many instruments (see strategy_breadth)."""
    return strategy_breadth(p["rows"], p.get("stat", "sharpe"))


def _periods(v, default):
    if not v:
        return list(default)
    out = sorted({int(x) for x in v})
    if not out or out[0] < 2 or out[-1] > 1000 or len(out) > 6:
        raise ValueError("Moving-average periods must be 2 to 1000 bars, at most 6 of them.")
    return out


def api_dashboard(p):
    ds = _get(p["ticker"])
    return dashboard(ds, p.get("start"), p.get("end"), _periods(p.get("sma"), (20, 50, 200)), _periods(p.get("ema"), (12, 26)))


def api_pbo(p):
    A = float(p.get("ann") or TRADING_DAYS)
    out = pbo_cscv(p["blocks"], p.get("labels"), ann=A)
    sr = p.get("sharpes") or []
    if sr:
        best = max(range(len(sr)), key=lambda c: -math.inf if sr[c] is None else sr[c])
        v = stdev([x for x in sr if x is not None]) ** 2 if len(sr) > 2 else None
        out["best"] = best
        out["sr_hurdle"] = expected_max_sharpe(len(sr), v / A) * math.sqrt(A) if v else NAN
    return out


def api_wfo(p):
    return walk_forward_optimization(p["dates"], p["returns"], p.get("labels"), p["train"], p["test"],
                                     bool(p.get("anchored")), p.get("metric", "sharpe"), float(p.get("cash", 10000)),
                                     float(p.get("rf", 0.0)), p.get("bench_returns"))


def api_permutation(p):
    res, _ = _run_payload(p)
    return permutation_test(res, p.get("n_perm", 500), p.get("seed", 7))


def api_analytics(p):
    sets = [_get(t) for t in p["tickers"] if t in DATA]
    return instrument_stats(sets, p.get("start"), p.get("end"), float(p.get("rf", 0.0)), p.get("total_return", True) is not False)


def api_metric_list(p):
    return [{"key": k, "label": lab, "group": g, "fmt": f, "tier": t, "desc": d} for k, lab, g, f, t, d in METRICS]


def api_risk(p):
    res, _ = _run_payload(p)
    scen = [(x["name"], x["start"], x["end"]) for x in (p.get("scenarios") or []) if x.get("start") and x.get("end")]
    return risk_report(res, p.get("factors") or [], int(p.get("window") or 126), int(p.get("var_window") or 500), scen)


def api_capacity(p):
    sets, bench, cfg, strat, portfolio, rates = _prepare(p)
    return capacity_curve(strat, sets, cfg, p.get("levels"), p.get("start"), p.get("end"), bench, portfolio, rates,
                          float(p.get("threshold") or 0.5), _signal_key(p, strat))


def api_parse_factors(p):
    return parse_factor_csv(p["text"], p.get("percent"))


def api_load_factors(p):
    """Keep an uploaded factor table under a name; returns its columns and date range."""
    name = p["name"]
    FACTORS[name] = {"dates": list(p["dates"]), "columns": {k: [NAN if v is None else float(v) for v in vals]
                                                           for k, vals in p["columns"].items()}}
    return {"name": name, "columns": list(p["columns"]), "start": p["dates"][0], "end": p["dates"][-1], "n": len(p["dates"])}


def api_lint(p):
    return lint(p["code"], p.get("kind", "strategy"), p.get("name"))


def api_tests(p):
    return run_tests(p["code"], p.get("kind", "strategy"), p.get("name"), p.get("only"))


def api_console(p):
    tickers = p.get("tickers") or ([p["ticker"]] if p.get("ticker") else [])
    return console_run(p.get("code", ""), tickers, bool(p.get("reset")))


def api_completions(p):
    return completions()


def api_build_futures(p):
    """Build a continuous contract from loaded single-contract instruments (oldest first)."""
    sets = [_get(t) for t in p["contracts"]]
    ds = continuous_futures(sets, p.get("ticker") or "CONT", p.get("roll", "days"), int(p.get("days_before", 5)), p.get("adjust", "ratio"))
    return {"data": {"dates": ds.dates, "open": ds.open, "high": ds.high, "low": ds.low, "close": ds.close, "volume": ds.volume},
            "rolls": ds.rolls, "issues": ds.issues}


def api_normalize_table(p):
    return {"text": normalize_table(p["text"], p.get("mapping"), p.get("delimiter"), p.get("decimal_comma"))}


def api_read_parquet(p):
    import base64
    return {"text": parquet_to_csv(base64.b64decode(p["data"]))}


def api_verify_audit(p):
    return verify_audit_log(p["entries"])


def api_options(p):
    """Options overlay on the strategy, and a delta-hedged option on its instrument."""
    res, _ = _run_payload(p)
    o = p.get("options") or {}
    kw = dict(days=int(o.get("days", 21)), iv=o.get("iv", "realized"), iv_mult=float(o.get("iv_mult", 1.1)),
              vol_window=int(o.get("vol_window", 21)))
    over = options_overlay(res, o.get("kind", "covered_call"), float(o.get("otm", 0.05)), float(o.get("put_otm", 0.05)),
                           cost=float(o.get("cost", 0.03)), **kw)
    idx = _downsample(len(res.dates))
    out = {k: v for k, v in over.items() if k not in ("equity", "overlay_pnl", "iv")}
    out.update(dates=[res.dates[i] for i in idx], equity=[over["equity"][i] for i in idx], base_equity=[res.equity[i] for i in idx],
               iv=[over["iv"][i] for i in idx], rolls=over["rolls"][-500:])
    ds = res.panel.sets[0]
    r = res.config.cash_rate if res.config.cash_rate is not None else res.config.risk_free
    dh = delta_hedged(ds, res.dates[0], res.dates[-1], o.get("hedge_kind", "call"), float(o.get("hedge_otm", 0.0)),
                      r=r, notional=res.config.cash, ann=res_ann(res), **kw)
    j = _downsample(len(dh["dates"]))
    out["delta_hedged"] = {k: ([dh[k][i] for i in j] if isinstance(dh[k], list) and k != "periods" else dh[k]) for k in dh}
    out["delta_hedged"]["periods"] = dh["periods"][-500:]
    return out


def api_check_module(p):
    return {"problem": module_name_problem(p.get("name"))}


API = {
    "lint": api_lint, "tests": api_tests, "console": api_console, "completions": api_completions,
    "check_module": api_check_module, "build_futures": api_build_futures,
    "normalize_table": api_normalize_table, "read_parquet": api_read_parquet, "verify_audit": api_verify_audit,
    "options": api_options, "surface_grid": api_surface_grid, "surface": api_surface, "rank": api_rank, "breadth": api_breadth, "dashboard": api_dashboard,
    "option_rows": api_option_rows, "load_options": api_load_options, "forget_options": api_forget_options, "risk": api_risk, "capacity": api_capacity,
    "parse_factors": api_parse_factors, "load_factors": api_load_factors,
    "load_data": api_load_data, "parse_csv": api_parse_csv, "set_events": api_set_events,
    "load_rates": api_load_rates, "quality": api_quality,
    "inspect": api_inspect, "run": api_run, "metrics": api_metrics,
    "montecarlo": api_montecarlo, "split": api_split, "analytics": api_analytics,
    "subperiods": api_subperiods, "costs": api_costs, "permutation": api_permutation,
    "pbo": api_pbo, "wfo": api_wfo, "rows": api_rows,
    "metric_list": api_metric_list,
}


def to_json(x):
    """Compact JSON text with NaN and +/-Infinity as null (JavaScript can't parse them)."""
    return json.dumps(_sanitize(x), allow_nan=False, separators=(",", ":"))


def handle(method, payload_json):
    """Entry point for the web page: returns a JSON string {ok, result | error}."""
    try:
        import time
        p = json.loads(payload_json) if payload_json else {}
        if isinstance(p, dict) and "modules" in p:
            set_modules(p["modules"])
        if method not in API:
            raise ValueError("unknown method %r" % method)
        result = API[method](p)
        t0 = time.perf_counter()
        text = to_json({"ok": True, "result": result})
        if isinstance(result, dict) and "timing" in result:   # how long the hand-over to the page took
            text = text[:-2] + ',"_encode":%.6f}}' % (time.perf_counter() - t0)
        return text
    except StrategyError as e:
        return json.dumps({"ok": False, "error": _sanitize(e.info)})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"ok": False, "error": _sanitize(format_error(e))})
