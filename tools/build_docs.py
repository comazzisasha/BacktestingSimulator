"""Builds docs.html from docs_body.html plus tables generated from engine.py
(statistics) and templates.py (built-in strategies), so the docs always match
the code.  Run from the project folder:  python tools/build_docs.py
"""
import html
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import engine  # noqa: E402


def esc(s):
    return html.escape(str(s), quote=True)


def metrics_table():
    out, group = [], None
    shown = {"core": "Always", "std": "By default", "extra": "Optional"}
    out.append('<div class="scroll"><table><thead><tr><th>Statistic</th><th>Key</th><th>Shown</th><th>Meaning</th></tr></thead><tbody>')
    for key, label, grp, _fmt, tier, desc in engine.METRICS:
        if grp != group:
            group = grp
            out.append('<tr class="group"><td colspan="4">%s</td></tr>' % esc(grp))
        out.append('<tr><td>%s</td><td><code>%s</code></td><td>%s</td><td>%s</td></tr>'
                   % (esc(label), esc(key), shown[tier], esc(desc)))
    out.append('</tbody></table></div>')
    return "\n".join(out)


def strategies_table():
    text = open(os.path.join(ROOT, "templates.py"), encoding="utf-8").read()
    rows, group = [], None
    rows.append('<div class="scroll"><table><thead><tr><th>Strategy</th><th>Idea</th><th>PARAMS</th></tr></thead><tbody>')
    for part in re.split(r"(?m)^#%% id: ", text)[1:]:
        meta = dict(re.findall(r"(?m)^#%% (\w+): (.*)$", part))
        params = re.search(r"(?m)^PARAMS = (\{.*\})$", part)
        if meta.get("group") != group:
            group = meta.get("group")
            rows.append('<tr class="group"><td colspan="3">%s</td></tr>' % esc(group))
        rows.append('<tr><td>%s</td><td>%s</td><td><code>%s</code></td></tr>' % (
            esc(meta.get("name", "")), esc(meta.get("about", "")), esc(params.group(1) if params else "")))
    rows.append('</tbody></table></div>')
    return "\n".join(rows)


def main():
    body = open(os.path.join(ROOT, "tools", "docs_body.html"), encoding="utf-8").read()
    body = body.replace("<!--METRICS-->", metrics_table()).replace("<!--STRATEGIES-->", strategies_table())
    with open(os.path.join(ROOT, "docs.html"), "w", encoding="utf-8") as f:
        f.write(body)
    print("wrote docs.html")


if __name__ == "__main__":
    main()
