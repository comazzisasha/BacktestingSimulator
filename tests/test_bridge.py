"""Tests for the optional broker bridge (tools/bridge/broker_bridge.py).

Run with:  python tests/test_bridge.py   (or python -m pytest tests)
"""
import io
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "bridge"))

import broker_bridge as bb  # noqa: E402


def test_orders_translate_to_alpaca():
    assert bb.to_alpaca({"ticker": "spy", "qty": -12, "side": "sell", "at": "open"}) == \
        {"symbol": "SPY", "qty": "12", "side": "sell", "type": "market", "time_in_force": "opg"}
    frac = bb.to_alpaca({"ticker": "QQQ", "qty": 1.25, "at": "open"})
    assert frac["qty"] == "1.25" and frac["side"] == "buy" and frac["time_in_force"] == "day", "fractional: day orders only"
    assert bb.to_alpaca({"ticker": "TLT", "qty": 3, "side": "buy", "at": "close"})["time_in_force"] == "cls"
    for bad in ({"ticker": "X", "qty": 0}, {"ticker": "X", "qty": 1, "side": "hold"}):
        try:
            bb.to_alpaca(bad)
            raise AssertionError("accepted %r" % bad)
        except ValueError:
            pass


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_send_alpaca_posts_each_order_with_keys():
    seen = []

    def opener(req, timeout=None):
        seen.append((req.full_url, req.get_header("Apca-api-key-id"), json.loads(req.data)))
        return FakeResponse(b'{"id": "abc", "status": "accepted"}')
    out = bb.send_alpaca([{"symbol": "SPY", "qty": "1", "side": "buy", "type": "market", "time_in_force": "day"}], "paper",
                         opener, {"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"})
    assert seen[0][0] == "https://paper-api.alpaca.markets/v2/orders" and seen[0][1] == "k" and seen[0][2]["symbol"] == "SPY"
    assert out[0]["status"] == 200 and out[0]["body"]["id"] == "abc"
    try:
        bb.send_alpaca([], "paper", opener, {})
        raise AssertionError("missing keys accepted")
    except RuntimeError:
        pass


def _serve(mode="dry-run", opener=None):
    from http.server import ThreadingHTTPServer
    log = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl").name
    srv = ThreadingHTTPServer(("127.0.0.1", 0), bb.make_handler("s3cret", mode, "http://localhost:8000", log, opener))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d/orders" % srv.server_address[1], log


def _post(url, body, secret="s3cret"):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-QuantLab-Secret": secret})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # straight to localhost
    try:
        with opener.open(req, timeout=10) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)


def test_bridge_server_checks_the_secret_and_logs_dry_runs():
    srv, url, log = _serve()
    try:
        body = {"account": "Trend", "decided": "2026-10-21", "orders": [{"ticker": "SPY", "qty": 5, "side": "buy", "at": "open"}]}
        code, out, headers = _post(url, body)
        assert code == 200 and out["dry_run"] and out["would_send"][0]["symbol"] == "SPY"
        assert headers.get("Access-Control-Allow-Origin") == "http://localhost:8000"
        assert _post(url, body, "wrong")[0] == 401
        assert _post(url, {"orders": "nope"})[0] == 400
        rec = [json.loads(line) for line in open(log)]
        assert len(rec) == 1 and rec[0]["account"] == "Trend" and rec[0]["mode"] == "dry-run"
    finally:
        srv.shutdown()


def test_bridge_server_forwards_to_alpaca_paper():
    calls = []

    def opener(req, timeout=None):
        calls.append(req.full_url)
        return FakeResponse(b'{"id": "1"}')
    os.environ.setdefault("APCA_API_KEY_ID", "k")
    os.environ.setdefault("APCA_API_SECRET_KEY", "s")
    srv, url, _ = _serve("paper", opener)
    try:
        code, out, _ = _post(url, {"orders": [{"ticker": "SPY", "qty": 1, "side": "buy"}, {"ticker": "TLT", "qty": -2, "side": "sell"}]})
        assert code == 200 and len(out["sent"]) == 2 and calls == ["https://paper-api.alpaca.markets/v2/orders"] * 2
    finally:
        srv.shutdown()


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
