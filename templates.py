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
