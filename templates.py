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

#%% id: bracket
#%% name: Breakout with Stop-Loss & Target
#%% group: Breakout
#%% about: Enter on a stop order just above the 20-day high, then exit with a bracket: a stop-loss 2 ATRs below the entry & a take-profit 3 ATRs above, whichever comes first.
# Resting orders fill inside the next bar when its range reaches them:
#   ctx.order(qty, stop=price)                buy when the price rises to `price`
#   ctx.order_target_shares(0, limit=price)   sell everything at `price` or better
# tif="gtc" keeps an order until it fills or is cancelled (the default lasts one bar),
# and orders with the same oco name cancel each other when one of them fills.

PARAMS = {"n": 20, "atr_n": 14, "stop_atr": 2.0, "target_atr": 3.0, "buffer": 0.005}

def on_bar(ctx):
    high = ctx.highest(PARAMS["n"])
    a = ctx.atr(PARAMS["atr_n"])
    if isnan(high) or isnan(a):
        return None
    if ctx.is_flat:
        ctx.cancel_all()                                   # clear any leftover exit order
        trigger = high * (1 + PARAMS["buffer"])
        ctx.order(ctx.equity * 0.98 / trigger, stop=trigger)  # a day order, placed again each bar
    elif not ctx.open_orders:                              # just filled: place the bracket once
        entry = ctx.entry_price
        ctx.order_target_shares(0, stop=entry - PARAMS["stop_atr"] * a, tif="gtc", oco="exit")
        ctx.order_target_shares(0, limit=entry + PARAMS["target_atr"] * a, tif="gtc", oco="exit")
    return None

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

#%% id: nn_template
#%% name: Neural Network Template (nn)
#%% group: Machine Learning
#%% about: Describe a network as a list of layers and nn.signals() does the rest: features, labels, walk-forward retraining, scaling, early stopping, positions and an out-of-sample report in the Log.
# The built-in nn module does the boilerplate, so a deep-learning strategy is a list of layers.
# Layers: nn.Dense(units, activation), nn.Dropout(rate), nn.LSTM(units). Activations: relu,
# leaky_relu, elu, gelu, tanh, sigmoid, linear. End with Dense(1, "sigmoid") to predict
# P(up) or Dense(1, "linear") with target="return" to predict the return itself.
#
# Every model is trained only on bars whose outcome was already known, then predicts the
# bars until the next retrain. The Log shows how it did out of sample: compare the hit rate
# with the base rate before trusting the equity curve.
#
# To change what happens behind the scenes (features, training loop, how predictions become
# positions), open Modules above the editor and choose Edit a Copy next to nn.py.
import nn

PARAMS = {"epochs": 60, "lr": 0.001, "dropout": 0.1, "retrain": 126, "threshold": 0.55}

def layers():
    return [
        nn.Dense(32, "relu"),
        nn.Dropout(PARAMS["dropout"]),
        nn.Dense(16, "relu"),
        nn.Dense(1, "sigmoid"),        # P(the next bar is up)
    ]

def signals(data):
    return nn.signals(
        data, layers(),
        target="direction",            # or "return" (with a linear output)
        horizon=1,                     # bars ahead the prediction is about
        min_train=504,                 # labelled bars before the first model
        retrain=PARAMS["retrain"],     # bars between retrains
        epochs=PARAMS["epochs"], lr=PARAMS["lr"], batch_size=256, patience=8,
        position="long_flat",          # long_flat, long_short or sized
        threshold=PARAMS["threshold"],
        ensemble=1,                    # >1 averages networks trained with different seeds
    )

#%% id: nn_lstm
#%% name: LSTM Sequence Model (nn)
#%% group: Machine Learning
#%% about: A recurrent network (LSTM) reads the last 10 bars of features as a sequence and predicts whether the next week is up. Slower to train than a plain network: it retrains once a year.
# lookback=10 gives the network the last 10 bars of every feature, in order, instead of one
# bar. An LSTM layer reads that sequence and passes its final state on. Training takes longer
# than a dense network, so this retrains once a year and stops at 20 epochs.
import nn

PARAMS = {"lookback": 10, "units": 12, "horizon": 5, "epochs": 20, "threshold": 0.55}

def signals(data):
    return nn.signals(
        data,
        [nn.LSTM(PARAMS["units"]), nn.Dense(1, "sigmoid")],
        lookback=PARAMS["lookback"], horizon=PARAMS["horizon"],
        retrain=252, epochs=PARAMS["epochs"], lr=0.003, patience=5,
        threshold=PARAMS["threshold"],
    )

#%% id: nn_custom
#%% name: Neural Network, Your Own Pieces (nn)
#%% group: Machine Learning
#%% about: The same building blocks with the boilerplate opened up: your own features, a return-predicting network trained with nn.Network inside walk_forward, and position sizing you write.
# nn.signals() is a shortcut. Here its steps are written out, so each one can change:
#   1. features: a dict of name -> function of the data (each may only look backwards)
#   2. labels: what happens after each bar (here: the return over the next 5 bars)
#   3. fit_predict: scale with the training rows' statistics, train a network, predict
#   4. walk_forward: calls fit_predict on a schedule, never training on unknown labels
#   5. positions: from predictions to target weights
import nn

PARAMS = {"horizon": 5, "epochs": 40, "scale": 0.02}

FEATURES = {
    "ret_5d": lambda d: qt.roc(d.close, 5),
    "ret_20d": lambda d: qt.roc(d.close, 20),
    "ret_60d": lambda d: qt.roc(d.close, 60),
    "rsi_14": lambda d: [x / 100 - 0.5 for x in qt.rsi(d.close, 14)],
    "vol_20": lambda d: qt.realized_vol(d.close, 20),
    "trend_200": lambda d: [c / s - 1 if s == s else NAN for c, s in zip(d.close, qt.sma(d.close, 200))],
}

LAYERS = [nn.Dense(24, "gelu"), nn.Dropout(0.2), nn.Dense(1, "linear")]

def fit_predict(X, y, X_new):
    scaler = nn.Scaler().fit(X)
    mu, sd = y.mean(), y.std() or 1.0
    net = nn.Network(LAYERS, loss="huber", seed=0)
    hist = net.fit(scaler.transform(X), (y - mu) / sd, epochs=PARAMS["epochs"], lr=0.003, weight_decay=1e-3, val=0.2, patience=6)
    print("trained on %d rows: %d epochs, best %s" % (len(X), hist["epochs"], hist["best_epoch"]))
    return net.predict(scaler.transform(X_new)) * sd + mu        # predicted 5-bar return

def signals(data):
    X, names = nn.build_features(data, FEATURES)
    y = nn.labels(data, "return", PARAMS["horizon"])
    pred = walk_forward(X, y, fit_predict, min_train=504, retrain_every=126, horizon=PARAMS["horizon"])
    # Size by conviction: full long at a predicted +2% (PARAMS["scale"]), flat at 0 or below.
    return [0.0 if p != p else max(0.0, min(1.0, p / PARAMS["scale"])) for p in pred]

#%% id: pf_equal_weight
#%% name: Equal Weight, Monthly Rebalance
#%% group: Portfolio
#%% about: Hold every instrument in the universe at the same weight and rebalance on the first day of each month. The portfolio baseline.
# A portfolio strategy lists its instruments in UNIVERSE (change them in the Market panel)
# and returns {ticker: target weight}. Tickers left out of the dict go to zero.
# ctx['SPY'] is one instrument: ctx['SPY'].close, ctx['SPY'].sma(50), ctx['SPY'].position ...

UNIVERSE = ["SPY", "EFA", "EEM", "TLT", "GLD"]
PARAMS = {}

def on_bar(ctx):
    if ctx.state.get("month") == ctx.day.month:
        return None                              # None = keep the current holdings
    ctx.state["month"] = ctx.day.month
    live = [t for t in ctx.tickers if ctx[t].listed]
    return {t: 1 / len(live) for t in live}

#%% id: pf_momentum
#%% name: Cross-Sectional Momentum (Top N)
#%% group: Portfolio
#%% about: Each month, rank the universe by its 12-month return (skipping the last month) and hold the best 3 equally weighted.
# Classic "12-1" momentum: the most recent month is skipped because short-term returns
# tend to reverse. Instruments without a full year of history are left out of the ranking.

UNIVERSE = ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV", "USO", "XLK", "XLF", "XLE", "XLV"]
PARAMS = {"lookback": 252, "skip": 21, "top": 3}

def on_bar(ctx):
    if ctx.state.get("month") == ctx.day.month:
        return None
    ctx.state["month"] = ctx.day.month
    look, skip = PARAMS["lookback"], PARAMS["skip"]
    scores = {}
    for t in ctx.tickers:
        c = ctx[t].close
        if len(c) > look:
            scores[t] = c[-1 - skip] / c[-1 - look] - 1
    if len(scores) < PARAMS["top"]:
        return {}                                # not enough history yet: hold cash
    best = sorted(scores, key=scores.get, reverse=True)[:PARAMS["top"]]
    return {t: 1 / len(best) for t in best}

#%% id: pf_risk_parity
#%% name: Inverse-Volatility Risk Parity
#%% group: Portfolio
#%% about: Weight each instrument by 1 / its recent volatility so each contributes roughly the same risk; rebalanced monthly.
# Bonds are much calmer than stocks, so they get bigger weights. This ignores
# correlations: it's the simple "naive risk parity" most funds start from.

UNIVERSE = ["SPY", "EFA", "TLT", "IEF", "GLD"]
PARAMS = {"lookback": 63}

def on_bar(ctx):
    if ctx.state.get("month") == ctx.day.month:
        return None
    ctx.state["month"] = ctx.day.month
    inv = {}
    for t in ctx.tickers:
        v = ctx[t].vol(PARAMS["lookback"])
        if not isnan(v) and v > 0:
            inv[t] = 1 / v
    total = sum(inv.values())
    return {t: x / total for t, x in inv.items()} if total else {}

#%% id: pf_pairs
#%% name: Pairs Trade (Stat Arb)
#%% group: Portfolio
#%% about: Trade the spread between XOM & the energy ETF XLE: when it is 2 std away from normal, short the rich leg & buy the cheap one. Turn on shorting in settings.
# The spread is the log price ratio. Its z-score over the lookback says how stretched it is.
# Entry at |z| > entry, exit when |z| < exit. Each leg gets half of the gross exposure.

UNIVERSE = ["XOM", "XLE"]
PARAMS = {"lookback": 60, "entry": 2.0, "exit": 0.5, "gross": 1.0}

def signals(data):
    a, b = UNIVERSE
    pa, pb = data[a].close, data[b].close
    spread = [math.log(x / y) if x == x and y == y else NAN for x, y in zip(pa, pb)]
    z = qt.zscore(spread, PARAMS["lookback"])
    half = PARAMS["gross"] / 2
    rows, side = [], 0
    for zi in z:
        if isnan(zi):
            side = 0
        elif side == 0 and zi > PARAMS["entry"]:
            side = -1                           # spread rich: short a, buy b
        elif side == 0 and zi < -PARAMS["entry"]:
            side = 1                            # spread cheap: buy a, short b
        elif side != 0 and abs(zi) < PARAMS["exit"]:
            side = 0
        rows.append({a: side * half, b: -side * half})
    return rows

#%% id: pf_long_short
#%% name: Long/Short Momentum (Dollar Neutral)
#%% group: Portfolio
#%% about: Each month buy the 3 strongest stocks & short the 3 weakest by 6-month return, with equal long & short dollars. Turn on shorting in settings.
# Market-neutral: the long and short books are the same size, so the market's
# direction mostly cancels and what's left is the spread between winners and losers.

UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM", "KO", "XLF", "XLE", "XLV", "XLK"]
PARAMS = {"lookback": 126, "n": 3, "book": 0.5}

def on_bar(ctx):
    if ctx.state.get("month") == ctx.day.month:
        return None
    ctx.state["month"] = ctx.day.month
    look, n = PARAMS["lookback"], PARAMS["n"]
    scores = {t: ctx[t].roc(look) for t in ctx.tickers if not isnan(ctx[t].roc(look))}
    if len(scores) < 2 * n:
        return {}
    ranked = sorted(scores, key=scores.get)
    w = PARAMS["book"] / n
    out = {t: w for t in ranked[-n:]}            # strongest: long
    out.update({t: -w for t in ranked[:n]})      # weakest: short
    return out

#%% id: opt_covered_call
#%% name: Covered Call (Option Data)
#%% group: Options (Needs Option Data)
#%% about: Hold the stock and sell one call per 100 shares about a month out, near 0.30 delta; sell the next one when it expires. Needs an option chain for the instrument (Data tab).
# Trades real option quotes from a chain you loaded on the Data tab.
# ctx.options.find() picks a listed contract; sell() fills at the next day's bid,
# and a call left to expire settles at its intrinsic value on the expiry's close.

PARAMS = {"dte": 30, "delta": 0.30}

def on_bar(ctx):
    o = ctx.options
    lots = int(ctx.position // 100)              # one contract covers 100 shares
    if not o.positions and not o.pending and lots > 0:
        call = o.find("call", dte=PARAMS["dte"], delta=PARAMS["delta"])
        if call is not None and call.bid > 0:
            o.sell(call, lots)
    return 1.0                                  # stay fully invested in the shares

#%% id: opt_cash_secured_put
#%% name: Cash-Secured Put (Option Data)
#%% group: Options (Needs Option Data)
#%% about: Sell puts about a month out near 0.25 delta, as many as the cash covers at the strike; buy them back at half their price, else let them expire. Needs an option chain (Data tab).
# Short puts collect premium and lose when the price falls through the strike.
# The cash stays in the account as cover; QuantLab doesn't hold it aside, so
# size with the strike: contracts x strike x 100 <= cash.

PARAMS = {"dte": 30, "delta": 0.25, "take_profit": 0.5}

def on_bar(ctx):
    o = ctx.options
    for sym, n in o.positions.items():
        q = o.quote(sym)
        sold = ctx.state.get(sym)
        if q is not None and sold and n < 0 and q.ask > 0 and q.ask <= sold * (1 - PARAMS["take_profit"]):
            o.close(sym)                         # most of the premium is in: take it
    if not o.positions and not o.pending:
        put = o.find("put", dte=PARAMS["dte"], delta=PARAMS["delta"])
        if put is not None and put.bid > 0:
            n = int(ctx.cash // (put.strike * 100))
            if n > 0:
                ctx.state[put.symbol] = put.bid
                o.sell(put, n)
    return 0.0                                  # no shares: the cash backs the puts

#%% id: opt_protective_put
#%% name: Protective Put (Option Data)
#%% group: Options (Needs Option Data)
#%% about: Hold the stock and keep a put 5% below the price about two months out, rolling it when it has 20 days left. Insurance against crashes, paid for every month. Needs an option chain (Data tab).
# Long puts lose their premium in calm markets and pay off in sharp falls.

PARAMS = {"dte": 60, "otm": 0.05, "roll_dte": 20}

def on_bar(ctx):
    o = ctx.options
    if o.pending:                               # wait for yesterday's orders to fill
        return 1.0
    lots = int(ctx.position // 100)             # one put covers 100 shares
    covered = 0
    for sym, n in o.positions.items():
        q = o.quote(sym)
        if q is None or n <= 0:
            continue
        if q.dte <= PARAMS["roll_dte"]:
            if q.bid > 0:
                o.close(sym)                     # roll: sell what's left of the old put
        else:                                    # (one with no bid is worthless: let it expire)
            covered += n
    if lots > covered:
        put = o.find("put", dte=PARAMS["dte"], moneyness=1 - PARAMS["otm"])
        if put is not None:
            o.buy(put, lots - covered)
    return 1.0
