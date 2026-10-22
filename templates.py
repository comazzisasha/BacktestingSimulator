#%% id: buy_hold
#%% name: Buy & Hold
#%% group: Baselines
#%% about: Buy on the first bar and never sell. Every strategy should be compared with this.
# Buy and hold: the yardstick.
# on_bar returns a target weight: the share of equity to hold in the asset.
#   1.0 = fully invested, 0.5 = half, 0 = all cash, -1.0 = fully short.

PARAMS = {}

def on_bar(ctx):
    return 1.0

#%% id: coin_flip
#%% name: Coin Flip (Random)
#%% group: Baselines
#%% about: Every Monday, flip a coin to be invested or in cash. Run it to see what pure luck looks like.
# Random baseline. If your strategy can't beat this, it may be noise.
# Change 'seed' in the settings to see other random outcomes.

PARAMS = {"p_long": 0.5}

def on_bar(ctx):
    if ctx.weekday == 0:  # Monday
        return 1.0 if random.random() < PARAMS["p_long"] else 0.0
    return None  # None = keep the current position

#%% id: sma_cross
#%% name: SMA Crossover (Golden Cross)
#%% group: Trend Following
#%% about: Long when the 50-day average is above the 200-day average, in cash otherwise.
# Moving-average crossover.
# The fast average rising above the slow one is a "golden cross";
# falling below is a "death cross".

PARAMS = {"fast": 50, "slow": 200}

def on_bar(ctx):
    fast = ctx.sma(PARAMS["fast"])
    slow = ctx.sma(PARAMS["slow"])
    if isnan(slow):        # not enough history yet
        return 0.0
    return 1.0 if fast > slow else 0.0

#%% id: ema_cross
#%% name: EMA Crossover
#%% group: Trend Following
#%% about: Faster-reacting crossover using exponential moving averages (12/26 by default).

PARAMS = {"fast": 12, "slow": 26}

def on_bar(ctx):
    fast = ctx.ema(PARAMS["fast"])
    slow = ctx.ema(PARAMS["slow"])
    if isnan(slow):
        return 0.0
    return 1.0 if fast > slow else 0.0

#%% id: trend_filter
#%% name: Price Above 200-Day Average
#%% group: Trend Following
#%% about: Hold the asset only while it closes above its long moving average (Faber's timing model).

PARAMS = {"n": 200}

def on_bar(ctx):
    avg = ctx.sma(PARAMS["n"])
    if isnan(avg):
        return 0.0
    return 1.0 if ctx.price > avg else 0.0

#%% id: tsmom
#%% name: Time-Series Momentum
#%% group: Trend Following
#%% about: Once a month, go long if the past 12 months’ return is positive; otherwise hold cash.

PARAMS = {"lookback": 252}

def on_bar(ctx):
    month = ctx.day.month
    if ctx.state.get("month") == month:
        return None                      # only rebalance when the month changes
    ctx.state["month"] = month
    past = ctx.roc(PARAMS["lookback"])   # return over the lookback window
    if isnan(past):
        return 0.0
    return 1.0 if past > 0 else 0.0

#%% id: donchian
#%% name: Donchian Breakout (Turtle)
#%% group: Trend Following
#%% about: Buy a close above the prior 55-day high; exit on a close below the prior 20-day low.
# ago=1 compares today's close with the channel up to YESTERDAY,
# otherwise today's high would always be part of the "highest high".

PARAMS = {"entry": 55, "exit": 20}

def on_bar(ctx):
    upper = ctx.highest(PARAMS["entry"], ago=1)
    lower = ctx.lowest(PARAMS["exit"], ago=1)
    if isnan(upper) or isnan(lower):
        return 0.0
    if ctx.is_flat and ctx.price > upper:
        return 1.0
    if ctx.is_long and ctx.price < lower:
        return 0.0
    return None

#%% id: macd
#%% name: MACD Crossover
#%% group: Trend Following
#%% about: Long while the MACD line is above its signal line.

PARAMS = {"fast": 12, "slow": 26, "signal": 9}

def on_bar(ctx):
    line, signal, hist = ctx.macd(PARAMS["fast"], PARAMS["slow"], PARAMS["signal"])
    if isnan(signal):
        return 0.0
    return 1.0 if line > signal else 0.0

#%% id: adx_trend
#%% name: ADX Trend Strength
#%% group: Trend Following
#%% about: Long only when a trend is strong (ADX above 25) and pointing up (+DI above -DI).

PARAMS = {"n": 14, "threshold": 25}

def on_bar(ctx):
    adx, plus_di, minus_di = ctx.adx(PARAMS["n"])
    if isnan(adx):
        return 0.0
    if adx > PARAMS["threshold"] and plus_di > minus_di:
        return 1.0
    if plus_di < minus_di:
        return 0.0
    return None

#%% id: atr_trailing
#%% name: Trend with ATR Trailing Stop
#%% group: Trend Following
#%% about: Enter above the 100-day average; exit when price falls 3 ATRs below its highest close since entry.

PARAMS = {"trend": 100, "atr_n": 14, "atr_mult": 3.0}

def on_bar(ctx):
    avg = ctx.sma(PARAMS["trend"])
    a = ctx.atr(PARAMS["atr_n"])
    if isnan(avg) or isnan(a):
        return 0.0
    if ctx.is_flat:
        if ctx.price > avg:
            ctx.state["peak"] = ctx.price
            return 1.0
        return 0.0
    ctx.state["peak"] = max(ctx.state["peak"], ctx.price)
    stop = ctx.state["peak"] - PARAMS["atr_mult"] * a
    return 0.0 if ctx.price < stop else None

#%% id: rsi_reversion
#%% name: RSI Oversold
#%% group: Mean Reversion
#%% about: Buy when RSI(14) drops below 30, sell when it recovers above 50.

PARAMS = {"n": 14, "buy_below": 30, "sell_above": 50}

def on_bar(ctx):
    r = ctx.rsi(PARAMS["n"])
    if isnan(r):
        return 0.0
    if ctx.is_flat and r < PARAMS["buy_below"]:
        return 1.0
    if ctx.is_long and r > PARAMS["sell_above"]:
        return 0.0
    return None

#%% id: rsi2
#%% name: RSI(2) Pullback in an Uptrend
#%% group: Mean Reversion
#%% about: Connors-style: in an uptrend (above the 200-day average), buy when RSI(2) < 10, exit above the 5-day average.

PARAMS = {"rsi_n": 2, "entry": 10, "trend": 200, "exit_ma": 5}

def on_bar(ctx):
    r = ctx.rsi(PARAMS["rsi_n"])
    trend = ctx.sma(PARAMS["trend"])
    exit_ma = ctx.sma(PARAMS["exit_ma"])
    if isnan(trend):
        return 0.0
    if ctx.is_flat and ctx.price > trend and r < PARAMS["entry"]:
        return 1.0
    if ctx.is_long and ctx.price > exit_ma:
        return 0.0
    return None

#%% id: bb_reversion
#%% name: Bollinger Band Reversion
#%% group: Mean Reversion
#%% about: Buy a close below the lower band, sell when price returns to the middle band.

PARAMS = {"n": 20, "k": 2.0}

def on_bar(ctx):
    mid, upper, lower = ctx.bbands(PARAMS["n"], PARAMS["k"])
    if isnan(mid):
        return 0.0
    if ctx.is_flat and ctx.price < lower:
        return 1.0
    if ctx.is_long and ctx.price > mid:
        return 0.0
    return None

#%% id: zscore_ls
#%% name: Z-Score Long/Short
#%% group: Mean Reversion
#%% about: Long when price is 2 std below its 20-day mean, short when 2 above, flat near the mean. Turn on shorting in settings.
# With shorting off, the short side is clipped to 0 and this becomes long-only.

PARAMS = {"n": 20, "entry_z": 2.0, "exit_z": 0.5}

def on_bar(ctx):
    z = ctx.zscore(PARAMS["n"])
    if isnan(z):
        return 0.0
    if z < -PARAMS["entry_z"]:
        return 1.0
    if z > PARAMS["entry_z"]:
        return -1.0
    if abs(z) < PARAMS["exit_z"]:
        return 0.0
    return None

#%% id: stoch
#%% name: Stochastic Oscillator
#%% group: Mean Reversion
#%% about: Buy when %K crosses above %D below 20 (oversold); exit when %K rises above 80.

PARAMS = {"k": 14, "d": 3, "low": 20, "high": 80}

def on_bar(ctx):
    k, d = ctx.stoch(PARAMS["k"], PARAMS["d"])
    k_prev, d_prev = ctx.stoch(PARAMS["k"], PARAMS["d"], ago=1)
    if isnan(d_prev):
        return 0.0
    crossed_up = k > d and k_prev <= d_prev
    if ctx.is_flat and crossed_up and k < PARAMS["low"]:
        return 1.0
    if ctx.is_long and k > PARAMS["high"]:
        return 0.0
    return None

#%% id: bb_breakout
#%% name: Bollinger Breakout
#%% group: Breakout
#%% about: Buy a close above the upper band (volatility expansion); exit below the middle band.

PARAMS = {"n": 20, "k": 2.0}

def on_bar(ctx):
    mid, upper, lower = ctx.bbands(PARAMS["n"], PARAMS["k"])
    if isnan(mid):
        return 0.0
    if ctx.is_flat and ctx.price > upper:
        return 1.0
    if ctx.is_long and ctx.price < mid:
        return 0.0
    return None

#%% id: keltner
#%% name: Keltner Channel Breakout
#%% group: Breakout
#%% about: Buy a close above the upper Keltner channel (EMA + 2 ATR); exit below the middle line.

PARAMS = {"n": 20, "mult": 2.0}

def on_bar(ctx):
    mid, upper, lower = ctx.keltner(PARAMS["n"], PARAMS["mult"])
    if isnan(mid):
        return 0.0
    if ctx.is_flat and ctx.price > upper:
        return 1.0
    if ctx.is_long and ctx.price < mid:
        return 0.0
    return None

#%% id: vol_target
#%% name: Volatility Targeting
#%% group: Risk Management
#%% about: Always invested, but scale the position so its volatility stays near 15% a year.
# Exposure = target vol / recent vol, capped at max_weight.
# Raise "Max leverage" in settings to let max_weight go above 1.

PARAMS = {"target_vol": 0.15, "lookback": 20, "max_weight": 1.0}

def on_bar(ctx):
    v = ctx.vol(PARAMS["lookback"])
    if isnan(v) or v == 0:
        return 0.0
    w = min(PARAMS["max_weight"], PARAMS["target_vol"] / v)
    return round(w, 1)   # round to avoid tiny daily trades

#%% id: turn_of_month
#%% name: Turn of the Month
#%% group: Calendar
#%% about: Hold only around month-end (from the 25th through the 3rd); historically a strong window for stocks.

PARAMS = {"from_day": 25, "to_day": 3}

def on_bar(ctx):
    d = ctx.day.day
    in_window = d >= PARAMS["from_day"] or d < PARAMS["to_day"]
    return 1.0 if in_window else 0.0

#%% id: day_of_week
#%% name: Buy Mon–Thu, Sell Friday
#%% group: Calendar
#%% about: Your original idea with share orders: buy 1 share each Monday–Thursday & sell everything on Friday.
# Share orders instead of weights:
#   ctx.order(n)               buy n shares (negative n sells)
#   ctx.order_target_shares(n) end up holding exactly n shares
# Orders placed today are filled at the next bar's open.

PARAMS = {"shares_per_day": 1}

def on_bar(ctx):
    if ctx.weekday == 4:                 # Friday
        ctx.order_target_shares(0)       # sell everything
    elif ctx.weekday < 4:                # Monday-Thursday
        ctx.order(PARAMS["shares_per_day"])

#%% id: sma_cross_vector
#%% name: SMA Crossover (Vectorized)
#%% group: Vectorized
#%% about: The same golden cross written as signals(data), which builds every bar's weight at once.
# signals(data) gets the whole history as lists and returns one target
# weight per bar. Weight i is decided at bar i's close and filled on bar i+1,
# so the value at i must only use data up to i. QuantLab re-checks this
# automatically and warns you if a signal peeks at the future.

PARAMS = {"fast": 50, "slow": 200}

def signals(data):
    fast = qt.sma(data.close, PARAMS["fast"])
    slow = qt.sma(data.close, PARAMS["slow"])
    return [1.0 if f > s else 0.0 for f, s in zip(fast, slow)]

#%% id: ml_logistic
#%% name: Logistic Regression (NumPy)
#%% group: Machine Learning
#%% about: Predicts whether tomorrow is up from momentum, RSI, trend & volatility; retrained every quarter on past data only. Uses NumPy.
# A machine-learning strategy in three steps:
#   1. features(data): one row of inputs per bar, built from causal indicators
#   2. walk_forward(): trains on the past & predicts the next bars, retraining on a
#      schedule; it never trains on a label the model couldn't have known yet
#   3. turn each predicted probability into a position
import numpy as np

PARAMS = {"min_train": 504, "retrain": 63, "threshold": 0.53, "l2": 0.01}

def features(data):
    c = data.close
    trend = [p / s - 1 for p, s in zip(c, qt.sma(c, 50))]
    return np.column_stack([
        qt.roc(c, 1), qt.roc(c, 5), qt.roc(c, 20),       # momentum: 1 day, 1 week, 1 month
        np.array(qt.rsi(c, 14)) / 100 - 0.5,              # overbought / oversold
        trend,                                            # distance from the 50-day average
        qt.realized_vol(c, 20),                           # recent volatility
    ])

def fit_predict(X, y, X_new):
    mu, sd = X.mean(0), X.std(0) + 1e-12     # scale with training rows only
    X, X_new = (X - mu) / sd, (X_new - mu) / sd
    up = (y > 0).astype(float)
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(200):                     # gradient descent on log-loss + L2 penalty
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= 0.5 * (X.T @ (p - up) / len(up) + PARAMS["l2"] * w)
        b -= 0.5 * (p - up).mean()
    return 1 / (1 + np.exp(-(X_new @ w + b)))  # probability that the next day is up

def signals(data):
    c = data.close
    # label for bar i = the NEXT day's return, known one bar later (horizon=1)
    y = np.array([c[i + 1] / c[i] - 1 if i + 1 < len(c) else NAN for i in range(len(c))])
    prob = walk_forward(features(data), y, fit_predict, min_train=PARAMS["min_train"],
                        retrain_every=PARAMS["retrain"], horizon=1)
    return [1.0 if p > PARAMS["threshold"] else 0.0 for p in prob]   # NaN compares False -> cash

#%% id: ml_mlp
#%% name: Neural Network (NumPy)
#%% group: Machine Learning
#%% about: A small two-layer neural network trained with Adam in plain NumPy, retrained twice a year: the same training loop you would write in PyTorch.
# PyTorch can't run in the browser, but a small network is only a few lines of NumPy.
# The forward pass, backward pass & Adam update below are what loss.backward() and
# optimizer.step() do for you in PyTorch. Training is seeded, so runs are repeatable.
import numpy as np

PARAMS = {"min_train": 504, "retrain": 126, "hidden": 16, "epochs": 100, "lr": 0.02, "threshold": 0.53}

def features(data):
    c = data.close
    trend = [p / s - 1 for p, s in zip(c, qt.sma(c, 50))]
    return np.column_stack([qt.roc(c, 1), qt.roc(c, 5), qt.roc(c, 20),
                            np.array(qt.rsi(c, 14)) / 100 - 0.5, trend, qt.realized_vol(c, 20)])

def fit_predict(X, y, X_new):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    X, X_new = (X - mu) / sd, (X_new - mu) / sd
    t = (y > 0).astype(float)[:, None]
    rng = np.random.default_rng(0)
    h = PARAMS["hidden"]
    params = [rng.normal(0, 1 / np.sqrt(X.shape[1]), (X.shape[1], h)), np.zeros(h),
              rng.normal(0, 1 / np.sqrt(h), (h, 1)), np.zeros(1)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    for step in range(1, PARAMS["epochs"] + 1):
        W1, b1, W2, b2 = params
        a = np.tanh(X @ W1 + b1)                        # forward
        p = 1 / (1 + np.exp(-(a @ W2 + b2)))
        d2 = (p - t) / len(t)                           # backward (log-loss gradient)
        da = (d2 @ W2.T) * (1 - a ** 2)
        grads = [X.T @ da, da.sum(0), a.T @ d2, d2.sum(0)]
        for k, g in enumerate(grads):                   # Adam update
            m[k] = 0.9 * m[k] + 0.1 * g
            v[k] = 0.999 * v[k] + 0.001 * g * g
            params[k] -= PARAMS["lr"] * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
    W1, b1, W2, b2 = params
    return (1 / (1 + np.exp(-(np.tanh(X_new @ W1 + b1) @ W2 + b2))))[:, 0]

def signals(data):
    c = data.close
    y = np.array([c[i + 1] / c[i] - 1 if i + 1 < len(c) else NAN for i in range(len(c))])
    prob = walk_forward(features(data), y, fit_predict, min_train=PARAMS["min_train"],
                        retrain_every=PARAMS["retrain"], horizon=1)
    return [1.0 if p > PARAMS["threshold"] else 0.0 for p in prob]

#%% id: ml_forest
#%% name: Random Forest (scikit-learn)
#%% group: Machine Learning
#%% about: A random forest predicts whether the next week is up, retrained twice a year. The first run downloads scikit-learn (about 75 MB, cached afterwards).
# Any scikit-learn model works the same way: swap RandomForestClassifier for
# GradientBoostingClassifier, LogisticRegression, MLPClassifier, SVC, ...
import numpy as np
from sklearn.ensemble import RandomForestClassifier

PARAMS = {"min_train": 504, "retrain": 126, "horizon": 5, "trees": 40, "depth": 4, "threshold": 0.55}

def features(data):
    c = data.close
    trend = [p / s - 1 for p, s in zip(c, qt.sma(c, 50))]
    return np.column_stack([qt.roc(c, 5), qt.roc(c, 20), qt.roc(c, 60),
                            qt.rsi(c, 14), trend, qt.realized_vol(c, 20)])

def fit_predict(X, y, X_new):
    model = RandomForestClassifier(n_estimators=PARAMS["trees"], max_depth=PARAMS["depth"],
                                   min_samples_leaf=20, random_state=0)
    model.fit(X, (y > 0).astype(int))
    return model.predict_proba(X_new)[:, 1]        # probability of class 1 = "up"

def signals(data):
    c, h = data.close, PARAMS["horizon"]
    # label = return over the next h days, so it is only known h bars later
    y = np.array([c[i + h] / c[i] - 1 if i + h < len(c) else NAN for i in range(len(c))])
    prob = walk_forward(features(data), y, fit_predict, min_train=PARAMS["min_train"],
                        retrain_every=PARAMS["retrain"], horizon=h)
    return [1.0 if p > PARAMS["threshold"] else 0.0 for p in prob]
