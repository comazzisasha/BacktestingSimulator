"""nn: neural networks for QuantLab strategies, in plain NumPy.

The one-line version, in a strategy:

    import nn
    LAYERS = [nn.Dense(32, "relu"), nn.Dropout(0.1), nn.Dense(16, "relu"), nn.Dense(1, "sigmoid")]

    def signals(data):
        return nn.signals(data, LAYERS)

nn.signals() does the boilerplate: it builds causal features, labels each bar with what
happens next, retrains the network on a walk-forward schedule (it only ever trains on labels
that were already known), scales inputs with the training rows' statistics, stops training
early on a time-ordered validation slice, averages an optional ensemble of seeds, turns the
predictions into positions and prints an out-of-sample report. Every step is a function below,
so you can call them yourself, or open this module from the module library ("Edit a Copy")
and change anything. Your copy replaces this built-in one for your strategies.

PyTorch can't run in the browser, so the layers here are written out: forward(), backward()
and an Adam update, the same computations PyTorch does for you. Training is seeded, so the
same code and data always give the same result.
"""
import copy
import math

import numpy as np


# ---------------------------------------------------------------- activations
_K = math.sqrt(2 / math.pi)


def _act(name, z):
    if name == "linear":
        return z
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "leaky_relu":
        return np.where(z > 0, z, 0.01 * z)
    if name == "elu":
        return np.where(z > 0, z, np.expm1(np.minimum(z, 0.0)))
    if name == "tanh":
        return np.tanh(z)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    if name == "gelu":                                  # the tanh approximation, as in GPT-2
        return 0.5 * z * (1 + np.tanh(_K * (z + 0.044715 * z ** 3)))
    raise ValueError("unknown activation %r (use %s)" % (name, ", ".join(ACTIVATIONS)))


def _dact(name, z, a):
    """d activation / dz, given z and the activation a."""
    if name == "linear":
        return np.ones_like(z)
    if name == "relu":
        return (z > 0).astype(z.dtype)
    if name == "leaky_relu":
        return np.where(z > 0, 1.0, 0.01)
    if name == "elu":
        return np.where(z > 0, 1.0, a + 1.0)
    if name == "tanh":
        return 1 - a * a
    if name == "sigmoid":
        return a * (1 - a)
    if name == "gelu":
        t = np.tanh(_K * (z + 0.044715 * z ** 3))
        return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * _K * (1 + 3 * 0.044715 * z * z)
    raise ValueError(name)


ACTIVATIONS = ("linear", "relu", "leaky_relu", "elu", "tanh", "sigmoid", "gelu")


# --------------------------------------------------------------------- layers
class Dense:
    """A fully connected layer: activation(x @ W + b). Takes (rows, features); a sequence
    (rows, steps, features) is flattened first."""

    def __init__(self, units, activation="relu"):
        if activation not in ACTIVATIONS:
            raise ValueError("unknown activation %r (use %s)" % (activation, ", ".join(ACTIVATIONS)))
        self.units, self.activation = int(units), activation

    def build(self, shape, rng):
        n_in = int(np.prod(shape))
        gain = 2.0 if self.activation in ("relu", "leaky_relu", "elu", "gelu") else 1.0   # He / Glorot-style scale
        self.W = rng.normal(0.0, math.sqrt(gain / n_in), (n_in, self.units))
        self.b = np.zeros(self.units)
        return (self.units,)

    def params(self):
        return [self.W, self.b]

    def forward(self, x, train=False, rng=None):
        self._shape = x.shape
        x = x.reshape(len(x), -1)
        self._x = x
        self._z = x @ self.W + self.b
        self._a = _act(self.activation, self._z)
        return self._a

    def backward(self, da, dz=None):
        """da: d loss / d output. dz (instead) skips the activation (sigmoid + cross-entropy)."""
        if dz is None:
            dz = da * _dact(self.activation, self._z, self._a)
        self.grads = [self._x.T @ dz, dz.sum(0)]
        return (dz @ self.W.T).reshape(self._shape)

    def __repr__(self):
        return "Dense(%d, %s)" % (self.units, self.activation)


class Dropout:
    """Sets a share `rate` of its inputs to zero while training (scaling the rest up), so the
    network can't lean on any one input; does nothing when predicting."""

    def __init__(self, rate=0.1):
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be from 0 to below 1")
        self.rate = float(rate)

    def build(self, shape, rng):
        return shape

    def params(self):
        return []

    def forward(self, x, train=False, rng=None):
        if not train or self.rate == 0:
            self._mask = None
            return x
        self._mask = (rng.random(x.shape) >= self.rate) / (1 - self.rate)
        return x * self._mask

    def backward(self, da, dz=None):
        self.grads = []
        return da if self._mask is None else da * self._mask

    def __repr__(self):
        return "Dropout(%g)" % self.rate


class LSTM:
    """A long short-term memory layer over a sequence (rows, steps, features), returning the
    last step's hidden state (rows, units). Use it with lookback > 1 in nn.signals()."""

    def __init__(self, units):
        self.units = int(units)

    def build(self, shape, rng):
        if len(shape) != 2:
            raise ValueError("LSTM needs sequences: use lookback > 1 (or put it first)")
        F, H = shape[1], self.units
        self.W = rng.normal(0.0, math.sqrt(1.0 / F), (F, 4 * H))
        self.U = rng.normal(0.0, math.sqrt(1.0 / H), (H, 4 * H))
        self.b = np.zeros(4 * H)
        self.b[H:2 * H] = 1.0                            # forget gate starts open
        return (H,)

    def params(self):
        return [self.W, self.U, self.b]

    def forward(self, x, train=False, rng=None):
        n, T, _ = x.shape
        H = self.units
        h, c = np.zeros((n, H)), np.zeros((n, H))
        self._cache = []
        for t in range(T):
            z = x[:, t] @ self.W + h @ self.U + self.b
            i = _act("sigmoid", z[:, :H])
            f = _act("sigmoid", z[:, H:2 * H])
            g = np.tanh(z[:, 2 * H:3 * H])
            o = _act("sigmoid", z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            self._cache.append((x[:, t], h, c, i, f, g, o, tc))
            h, c = o * tc, c_new
        return h

    def backward(self, dh, dz=None):
        dW, dU, db = np.zeros_like(self.W), np.zeros_like(self.U), np.zeros_like(self.b)
        T = len(self._cache)
        dx = np.zeros((dh.shape[0], T, self.W.shape[0]))
        dc_next = np.zeros_like(dh)
        for t in range(T - 1, -1, -1):
            xt, h_prev, c_prev, i, f, g, o, tc = self._cache[t]
            do = dh * tc
            dc = dc_next + dh * o * (1 - tc * tc)
            dzt = np.concatenate([dc * g * i * (1 - i), dc * c_prev * f * (1 - f), dc * i * (1 - g * g), do * o * (1 - o)], axis=1)
            dW += xt.T @ dzt
            dU += h_prev.T @ dzt
            db += dzt.sum(0)
            dx[:, t] = dzt @ self.W.T
            dh = dzt @ self.U.T
            dc_next = dc * f
        self.grads = [dW, dU, db]
        return dx

    def __repr__(self):
        return "LSTM(%d)" % self.units


# --------------------------------------------------------------------- losses
LOSSES = ("bce", "mse", "huber")


def _loss(name, p, y):
    """Mean loss and d loss / d prediction."""
    n = len(y)
    if name == "bce":
        q = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))), (q - y) / (q * (1 - q)) / n
    if name == "mse":
        r = p - y
        return float(np.mean(r * r)), 2 * r / n
    if name == "huber":                                 # quadratic within 1 (scaled target units), linear beyond
        r = p - y
        a = np.abs(r)
        return float(np.mean(np.where(a <= 1, 0.5 * r * r, a - 0.5))), np.clip(r, -1, 1) / n
    raise ValueError("unknown loss %r (use %s)" % (name, ", ".join(LOSSES)))


# -------------------------------------------------------------------- network
class Network:
    """Layers in order, trained with Adam (decoupled weight decay on the weights), mini-batches
    and early stopping. Layers are copied, so one list of layers can build many networks."""

    def __init__(self, layers, loss=None, seed=0):
        if not layers:
            raise ValueError("a network needs at least one layer")
        self.layers = [copy.deepcopy(layer) for layer in layers]
        last = self.layers[-1]
        out_act = getattr(last, "activation", None)
        self.loss = loss or ("bce" if out_act == "sigmoid" else "mse")
        if self.loss not in LOSSES:
            raise ValueError("unknown loss %r (use %s)" % (self.loss, ", ".join(LOSSES)))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.built = False

    def build(self, input_shape):
        shape = tuple(input_shape)
        for layer in self.layers:
            shape = layer.build(shape, self.rng)
        self.built = True
        return self

    def _params(self):
        return [p for layer in self.layers for p in layer.params()]

    def n_params(self):
        return int(sum(p.size for p in self._params()))

    def forward(self, X, train=False):
        a = X
        for layer in self.layers:
            a = layer.forward(a, train, self.rng)
        return a

    def _backward(self, p, y):
        loss, dp = _loss(self.loss, p, y)
        last = self.layers[-1]
        if self.loss == "bce" and getattr(last, "activation", None) == "sigmoid":
            g = last.backward(None, dz=(p - y) / len(y))      # sigmoid + cross-entropy: the gradient is p - y
        else:
            g = last.backward(dp)
        for layer in reversed(self.layers[:-1]):
            g = layer.backward(g)
        return loss

    def fit(self, X, y, epochs=60, batch_size=256, lr=1e-3, weight_decay=1e-4, val=0.2, patience=8, clip=5.0, shuffle=True):
        """Train on rows in time order. The last `val` share of them is held out to stop
        training once its loss stops improving for `patience` epochs (the best weights are
        kept). Returns the history: train & validation loss per epoch, best epoch."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(len(X), -1)
        if not self.built:
            self.build(X.shape[1:])
        n_val = int(len(X) * val) if val and len(X) >= 20 else 0
        Xt, yt = X[:len(X) - n_val], y[:len(X) - n_val]
        Xv, yv = X[len(X) - n_val:], y[len(X) - n_val:]
        params = self._params()
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        decay = [p.ndim > 1 for p in params]               # weights, not biases
        hist = {"loss": [], "val_loss": [], "best_epoch": None, "epochs": 0}
        best, best_val, wait, step = None, math.inf, 0, 0
        for epoch in range(1, int(epochs) + 1):
            order = self.rng.permutation(len(Xt)) if shuffle else np.arange(len(Xt))
            total = 0.0
            for s in range(0, len(Xt), int(batch_size)):
                idx = order[s:s + int(batch_size)]
                p = self.forward(Xt[idx], train=True)
                total += self._backward(p, yt[idx]) * len(idx)
                grads = [g for layer in self.layers for g in layer.grads]
                norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
                scale = clip / norm if clip and norm > clip else 1.0
                step += 1
                for k, (prm, g) in enumerate(zip(params, grads)):
                    g = g * scale
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.999 * v[k] + 0.001 * g * g
                    upd = (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
                    if decay[k] and weight_decay:
                        upd = upd + weight_decay * prm
                    prm -= lr * upd
            hist["loss"].append(total / max(1, len(Xt)))
            hist["epochs"] = epoch
            if n_val:
                vl = _loss(self.loss, self.forward(Xv), yv)[0]
                hist["val_loss"].append(vl)
                if vl < best_val - 1e-9:
                    best_val, best, wait, hist["best_epoch"] = vl, [p.copy() for p in params], 0, epoch
                else:
                    wait += 1
                    if patience and wait >= patience:
                        break
        if best is not None:
            for prm, b in zip(params, best):
                prm[...] = b
        return hist

    def predict(self, X):
        out = self.forward(np.asarray(X, dtype=float))
        return out[:, 0] if out.shape[1] == 1 else out

    def summary(self):
        return " → ".join(repr(layer) for layer in self.layers) + " · %s parameters" % format(self.n_params(), ",")


# ------------------------------------------------------- features & labels
def _ratio(a, b):
    return [x / y - 1 if (x == x and y == y and y) else float("nan") for x, y in zip(a, b)]


def _range_pos(c, n=252):
    hi, lo = qt.rolling_max(c, n), qt.rolling_min(c, n)
    return [(x - l) / (h - l) - 0.5 if (h == h and h > l) else float("nan") for x, h, l in zip(c, hi, lo)]


def _volume_z(v, n=20):
    lv = [math.log(x) if x and x > 0 else float("nan") for x in v]
    if all(x != x for x in lv):
        return [0.0] * len(v)
    return [0.0 if z != z else max(-5.0, min(5.0, z)) for z in qt.zscore(lv, n)]


# Each feature is a function of the data that only looks at bars up to its own.
DEFAULT_FEATURES = {
    "ret_1d": lambda d: qt.roc(d.close, 1),
    "ret_5d": lambda d: qt.roc(d.close, 5),
    "ret_20d": lambda d: qt.roc(d.close, 20),
    "ret_60d": lambda d: qt.roc(d.close, 60),
    "rsi_14": lambda d: [x / 100 - 0.5 for x in qt.rsi(d.close, 14)],
    "trend_50": lambda d: _ratio(d.close, qt.sma(d.close, 50)),
    "trend_200": lambda d: _ratio(d.close, qt.sma(d.close, 200)),
    "vol_20": lambda d: qt.realized_vol(d.close, 20),
    "vol_ratio": lambda d: _ratio(qt.realized_vol(d.close, 20), qt.realized_vol(d.close, 60)),
    "range_52w": lambda d: _range_pos(d.close),
    "volume_z": lambda d: _volume_z(d.volume),
}


def build_features(data, features=None):
    """One row per bar: X (bars, features) and the feature names. `features` maps a name to
    a function of the data, defaulting to DEFAULT_FEATURES."""
    spec = DEFAULT_FEATURES if features is None else features
    if not spec:
        raise ValueError("no features")
    cols = []
    for name, fn in spec.items():
        col = np.asarray(fn(data), dtype=float)
        if col.shape != (len(data.close),):
            raise ValueError("feature %s gave %s values for %d bars" % (name, col.shape, len(data.close)))
        cols.append(col)
    return np.column_stack(cols), list(spec)


def stack(X, lookback):
    """The last `lookback` rows ending at each bar: (bars, lookback, features); NaN until enough history."""
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    out = np.full((n, lookback, k), np.nan)
    for t in range(lookback - 1, n):
        out[t] = X[t - lookback + 1:t + 1]
    return out


def labels(data, target="direction", horizon=1):
    """What happens after each bar: the return over the next `horizon` bars (NaN where it
    isn't known yet). target='direction' (up or not) or 'return' use the same numbers;
    a function of the data gives your own labels (they must describe the future only)."""
    if callable(target):
        return np.asarray(target(data), dtype=float)
    if target not in ("direction", "return"):
        raise ValueError("target must be 'direction', 'return' or a function of the data")
    c = data.close
    h = int(horizon)
    return np.array([c[i + h] / c[i] - 1 if i + h < len(c) and c[i] else float("nan") for i in range(len(c))])


class Scaler:
    """Standardize each feature with the training rows' mean & standard deviation (clipped to ±5)."""

    def fit(self, X):
        axes = tuple(range(X.ndim - 1))
        self.mu = np.nanmean(X, axis=axes)
        self.sd = np.nanstd(X, axis=axes)
        self.sd[~(self.sd > 1e-12)] = 1.0
        return self

    def transform(self, X):
        return np.clip(np.nan_to_num((X - self.mu) / self.sd), -5, 5)


# -------------------------------------------------------------------- strategy
def default_layers(target="direction"):
    out = Dense(1, "sigmoid") if target == "direction" else Dense(1, "linear")
    return [Dense(32, "relu"), Dropout(0.1), Dense(16, "relu"), out]


LAST = {}   # the latest nn.signals() run: report text, retrains, features, predictions


def to_positions(pred, classify=True, position="long_flat", threshold=None):
    """Predictions -> target weights. Classification (P(up)): long_flat holds when P > threshold
    (default 0.55); long_short also shorts when P < 1 - threshold; sized scales from 0 at 0.5 to
    ±1 at the threshold. Regression (predicted return): threshold is the smallest return to act on
    (default 0), sized scales by the typical size of the predictions so far."""
    thr = (0.55 if classify else 0.0) if threshold is None else float(threshold)
    out, seen = [], []
    for p in pred:
        if p != p:
            out.append(0.0)
            continue
        if classify:
            if position == "long_flat":
                w = 1.0 if p > thr else 0.0
            elif position == "long_short":
                w = 1.0 if p > thr else -1.0 if p < 1 - thr else 0.0
            elif position == "sized":
                w = max(-1.0, min(1.0, (p - 0.5) / max(thr - 0.5, 1e-9)))
            else:
                raise ValueError("position must be long_flat, long_short or sized")
        else:
            seen.append(abs(p))
            if position == "long_flat":
                w = 1.0 if p > thr else 0.0
            elif position == "long_short":
                w = 1.0 if p > thr else -1.0 if p < -thr else 0.0
            elif position == "sized":
                scale = float(np.median(seen)) * 2 or 1.0
                w = max(-1.0, min(1.0, p / scale))
            else:
                raise ValueError("position must be long_flat, long_short or sized")
        out.append(w)
    return out


def _auc(score, label):
    pos = score[label == 1]
    neg = score[label == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def signals(data, layers=None, *, features=None, target="direction", horizon=1, lookback=1, min_train=504, retrain=126,
            window=None, epochs=60, batch_size=256, lr=1e-3, weight_decay=1e-4, val=0.2, patience=8, ensemble=1,
            loss=None, position="long_flat", threshold=None, seed=0, report=True):
    """Target weights for every bar from a neural network retrained walk-forward.

    layers    the network (default: Dense 32 → Dropout → Dense 16 → output)
    features  {name: function of the data} (default DEFAULT_FEATURES)
    target    'direction' (classify up / not, sigmoid output) or 'return' (regression)
    horizon   bars ahead the label looks; lookback: bars of features per input (sequences for LSTM)
    min_train labelled rows before the first model; retrain: bars between retrains; window: train on the latest rows only
    epochs, batch_size, lr, weight_decay, val, patience: training (see Network.fit)
    ensemble  networks with different seeds averaged; position, threshold: see to_positions()
    """
    layers = list(layers) if layers else default_layers(target)
    classify = target == "direction"
    X, names = build_features(data, features)
    y = labels(data, target, horizon)
    n, k = X.shape
    seq = any(isinstance(layer, LSTM) for layer in layers)
    lookback = max(1, int(lookback))
    W = stack(X, lookback) if lookback > 1 else X[:, None, :]
    flat = W.reshape(n, lookback * k)
    shape = (lookback, k) if seq else (lookback * k,)
    log = []

    def fit_predict(Xtr, ytr, Xnew):
        Xtr, Xnew = Xtr.reshape((len(Xtr),) + shape), Xnew.reshape((len(Xnew),) + shape)
        sc = Scaler().fit(Xtr)
        Xtr, Xnew = sc.transform(Xtr), sc.transform(Xnew)
        if classify:
            t = (ytr > 0).astype(float)
        else:
            y_mu, y_sd = float(np.mean(ytr)), float(np.std(ytr)) or 1.0
            t = (ytr - y_mu) / y_sd
        preds, hists = [], []
        for e in range(int(ensemble)):
            net = Network(layers, loss=loss or (None if classify else "huber"), seed=seed + 1000 * e)
            hists.append(net.fit(Xtr, t, epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay, val=val, patience=patience))
            preds.append(net.predict(Xnew))
        p = np.mean(preds, axis=0)
        log.append({"rows": len(Xtr), "epochs": [h["epochs"] for h in hists], "best_epoch": [h["best_epoch"] for h in hists],
                    "loss": [h["loss"][-1] for h in hists], "val_loss": [min(h["val_loss"]) if h["val_loss"] else None for h in hists],
                    "summary": net.summary()})
        return p if classify else p * y_sd + y_mu

    pred = walk_forward(flat, y, fit_predict, min_train=min_train, retrain_every=retrain, horizon=horizon, window=window)
    weights = to_positions(pred, classify, position, threshold)
    rep = _report(pred, y, log, names, lookback, classify, horizon)
    LAST.clear()
    LAST.update(report=rep, retrains=log, features=names, predictions=pred, weights=weights)
    if report:
        print(rep)
    return weights


def _report(pred, y, log, names, lookback, classify, horizon):
    pred = np.asarray(pred, dtype=float)
    ok = ~np.isnan(pred) & ~np.isnan(y)
    lines = []
    if not log:
        return "nn: no model was trained: fewer than min_train labelled rows (lower min_train or use more data)."
    lines.append("nn: %s · %d feature%s × lookback %d" % (log[-1]["summary"], len(names), "" if len(names) == 1 else "s", lookback))
    eps = [e for r in log for e in r["epochs"]]
    vls = [v for r in log for v in r["val_loss"] if v is not None]
    lines.append("Retrained %d times on %s–%s rows: median %d epochs%s." % (
        len(log), format(log[0]["rows"], ","), format(log[-1]["rows"], ","), int(np.median(eps)),
        ", median validation loss %.4f" % float(np.median(vls)) if vls else ""))
    if ok.sum():
        p, r = pred[ok], y[ok]
        up = (r > 0).astype(int)
        if classify:
            hit = float(np.mean((p > 0.5) == (up == 1)))
            base = max(float(up.mean()), 1 - float(up.mean()))
            lines.append("Out of sample (%s bars, %d-bar horizon): hit rate %.1f%%, against %.1f%% for always guessing the more common side." % (
                format(int(ok.sum()), ","), horizon, 100 * hit, 100 * base))
            lines.append("AUC %.3f (0.5 = no skill); mean predicted P(up) %.3f." % (_auc(p, up), float(p.mean())))
        else:
            ic = float(np.corrcoef(p, r)[0, 1]) if np.std(p) > 0 else float("nan")
            hit = float(np.mean(np.sign(p) == np.sign(r)))
            lines.append("Out of sample (%s bars, %d-bar horizon): correlation of predicted with actual returns %.3f." % (
                format(int(ok.sum()), ","), horizon, ic))
            lines.append("Sign right %.1f%% of the time; the price rose over %.1f%% of them." % (100 * hit, 100 * float(up.mean())))
    return "\n".join(lines)
