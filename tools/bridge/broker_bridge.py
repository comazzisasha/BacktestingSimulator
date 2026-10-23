"""QuantLab broker bridge: an optional, separate server that receives orders from the
Paper tab ("Send to Bridge") and forwards them to a broker. QuantLab itself never talks
to a broker; this file is the only part that does, and only when you run it.

    python tools/bridge/broker_bridge.py                     # dry run on http://localhost:8787
    python tools/bridge/broker_bridge.py --alpaca            # Alpaca *paper* account
    python tools/bridge/broker_bridge.py --alpaca --live     # Alpaca live account (real money)

Alpaca keys come from the environment: APCA_API_KEY_ID and APCA_API_SECRET_KEY.
Every request must carry the shared secret in the X-QuantLab-Secret header; one is
generated and printed if you don't pass --secret. Each request and the broker's answers
are appended to bridge-log.jsonl. Standard library only.
"""
import argparse
import datetime as dt
import json
import os
import secrets
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALPACA = {"paper": "https://paper-api.alpaca.markets", "live": "https://api.alpaca.markets"}


def to_alpaca(order):
    """One QuantLab order -> an Alpaca order request (market, day; at the open/close
    auction when the strategy fills at the next open/close)."""
    qty = abs(float(order["qty"]))
    if qty <= 0:
        raise ValueError("order for %s has no quantity" % order.get("ticker"))
    side = order.get("side") or ("buy" if float(order["qty"]) > 0 else "sell")
    if side not in ("buy", "sell"):
        raise ValueError("side must be buy or sell, got %r" % side)
    tif = {"open": "opg", "close": "cls"}.get(order.get("at"), "day")
    if qty != int(qty) and tif != "day":
        tif = "day"                       # Alpaca takes fractional quantities only for day orders
    return {"symbol": str(order["ticker"]).upper(), "qty": ("%.6f" % qty).rstrip("0").rstrip("."),
            "side": side, "type": "market", "time_in_force": tif}


def validate(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("orders"), list):
        raise ValueError("expected {orders: [...]}")
    if len(payload["orders"]) > 500:
        raise ValueError("too many orders in one request (500 at most)")
    return [to_alpaca(o) for o in payload["orders"]]


def send_alpaca(orders, account="paper", opener=None, env=os.environ):
    """POST each order to Alpaca. Returns [{order, status, body}]. `opener` is for tests."""
    key, sec = env.get("APCA_API_KEY_ID"), env.get("APCA_API_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
    opener = opener or urllib.request.urlopen
    out = []
    for o in orders:
        req = urllib.request.Request(ALPACA[account] + "/v2/orders", data=json.dumps(o).encode(), method="POST",
                                     headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec, "Content-Type": "application/json"})
        try:
            with opener(req, timeout=20) as r:
                out.append({"order": o, "status": r.status, "body": json.loads(r.read() or b"{}")})
        except urllib.error.HTTPError as e:
            out.append({"order": o, "status": e.code, "body": e.read().decode("utf-8", "replace")[:500]})
    return out


def make_handler(secret, mode, origin="*", log_path="bridge-log.jsonl", opener=None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-QuantLab-Secret")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):          # the browser's CORS preflight
            self._send(204, {})

        def do_POST(self):
            if self.path.rstrip("/") != "/orders":
                return self._send(404, {"error": "POST to /orders"})
            if not secrets.compare_digest(self.headers.get("X-QuantLab-Secret", ""), secret):
                return self._send(401, {"error": "wrong or missing X-QuantLab-Secret"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(min(n, 2_000_000)) or b"{}")
                orders = validate(payload)
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": str(e)})
            entry = {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "mode": mode, "account": payload.get("account"),
                     "decided": payload.get("decided"), "code_sha256": payload.get("code_sha256"), "orders": orders}
            try:
                if mode == "dry-run":
                    result = {"dry_run": True, "would_send": orders}
                else:
                    result = {"sent": send_alpaca(orders, mode, opener)}
            except RuntimeError as e:
                return self._send(500, {"error": str(e)})
            entry["result"] = result
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            self._send(200, result)

        def log_message(self, fmt, *args):
            sys.stderr.write("bridge: " + fmt % args + "\n")
    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--secret", default=None, help="shared secret (default: a random one, printed)")
    ap.add_argument("--origin", default="*", help="allowed browser origin, e.g. http://localhost:8000")
    ap.add_argument("--alpaca", action="store_true", help="forward to Alpaca (paper account unless --live)")
    ap.add_argument("--live", action="store_true", help="with --alpaca: the live account (real money)")
    ap.add_argument("--log", default="bridge-log.jsonl")
    a = ap.parse_args(argv)
    if a.live and not a.alpaca:
        ap.error("--live needs --alpaca")
    secret = a.secret or secrets.token_urlsafe(18)
    mode = "dry-run" if not a.alpaca else ("live" if a.live else "paper")
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(secret, mode, a.origin, a.log))
    print("QuantLab bridge (%s) on http://%s:%d/orders  secret: %s" % (mode, a.host, a.port, secret))
    if mode == "live":
        print("LIVE: orders go to your real Alpaca account.")
    srv.serve_forever()


if __name__ == "__main__":
    main()
