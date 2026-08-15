"""Self-contained HTML dashboard for the live run.

No CDN, no build step, no external requests — it renders from the CSV logs and
opens anywhere, including from a phone straight off the repo.
"""

from __future__ import annotations

import os

import pandas as pd

from . import config as C
from .live import load_state

TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Paper Trading Bot</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root {{
    --bg:#0d1117; --panel:#161b22; --line:#30363d; --text:#e6edf3;
    --dim:#8b949e; --up:#3fb950; --down:#f85149; --accent:#58a6ff;
  }}
  * {{ box-sizing:border-box }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--text);
    font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace }}
  h1 {{ font-size:18px; margin:0 0 4px }}
  .sub {{ color:var(--dim); margin-bottom:20px; font-size:12px }}
  .grid {{ display:grid; gap:12px;
    grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); margin-bottom:24px }}
  .card {{ background:var(--panel); border:1px solid var(--line);
    border-radius:8px; padding:14px }}
  .card h2 {{ font-size:12px; margin:0 0 10px; color:var(--dim);
    text-transform:uppercase; letter-spacing:.06em; font-weight:600 }}
  .big {{ font-size:26px; font-weight:600; letter-spacing:-.02em }}
  .aud {{ color:var(--dim); font-size:12px; margin-top:2px }}
  .up {{ color:var(--up) }} .down {{ color:var(--down) }}
  .meta {{ margin-top:10px; font-size:11px; color:var(--dim); line-height:1.7 }}
  .meta b {{ color:var(--text); font-weight:600 }}
  .dead {{ border-color:var(--down) }}
  .badge {{ display:inline-block; background:var(--down); color:#fff;
    border-radius:3px; padding:0 5px; font-size:10px; margin-left:6px }}
  table {{ width:100%; border-collapse:collapse; font-size:12px }}
  th,td {{ text-align:right; padding:5px 8px; border-bottom:1px solid var(--line);
    white-space:nowrap }}
  th:first-child,td:first-child {{ text-align:left }}
  th {{ color:var(--dim); font-weight:600; position:sticky; top:0;
    background:var(--panel) }}
  .wrap {{ background:var(--panel); border:1px solid var(--line);
    border-radius:8px; padding:14px; margin-bottom:24px; overflow-x:auto }}
  svg {{ display:block; width:100%; height:260px }}
  .legend span {{ margin-right:14px; font-size:11px }}
  .sw {{ display:inline-block; width:9px; height:9px; border-radius:2px;
    margin-right:4px; vertical-align:middle }}
</style>
<h1>Crypto perp paper bot &mdash; five books, one signal engine</h1>
<div class="sub">{updated} &middot; tick #{ticks} &middot; universe: {universe}
 &middot; started {created} at A${start_aud:.2f} (US${start_usd:.2f})</div>

<div class="grid">{cards}</div>

<div class="wrap">
  <h2 style="font-size:12px;margin:0 0 10px;color:var(--dim);text-transform:uppercase;
    letter-spacing:.06em">Equity, normalised to starting balance</h2>
  <div class="legend">{legend}</div>
  {chart}
</div>

<div class="wrap">
  <h2 style="font-size:12px;margin:0 0 10px;color:var(--dim);text-transform:uppercase;
    letter-spacing:.06em">Last 40 closed trades</h2>
  {trades}
</div>
"""


def _cards(summary: list[dict], fx: float) -> str:
    out = []
    for row in summary:
        cls = "up" if row["return_pct"] >= 0 else "down"
        dead = ' <span class="badge">DEAD</span>' if row["dead"] else ""
        out.append(f"""<div class="card{' dead' if row['dead'] else ''}">
  <h2>{row['book']}{dead}</h2>
  <div class="big {cls}">US${row['equity']:.2f}</div>
  <div class="aud">A${row['equity'] / fx:.2f} &middot;
    <span class="{cls}">{row['return_pct']:+.1f}%</span></div>
  <div class="meta">
    trades <b>{row['trades']}</b> &middot; win <b>{row['win_rate']:.0f}%</b>
    &middot; open <b>{row['open']}</b><br>
    liquidations <b>{row['liquidations']}</b> &middot; max DD
    <b>{row['max_dd_pct']:.0f}%</b><br>
    fees <b>US${row['fees']:.2f}</b> &middot; funding <b>US${row['funding']:.2f}</b>
  </div></div>""")
    return "\n".join(out)


# One distinct colour per book — modulo wrap would make two books share a line.
COLOURS = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff"]


def _chart(eq: pd.DataFrame, books: list[str], start_usd: float) -> tuple[str, str]:
    if eq.empty or len(eq) < 2:
        return "<div style='color:#8b949e'>no equity history yet</div>", ""

    w, h, pad = 1000, 260, 34
    norm = {b: (eq[b] / start_usd).tolist() for b in books if b in eq.columns}
    flat = [v for vals in norm.values() for v in vals if pd.notna(v)]
    lo, hi = min(min(flat), 0.9), max(max(flat), 1.1)
    span = hi - lo or 1.0
    n = len(eq)

    def x(i): return pad + (w - 2 * pad) * i / max(n - 1, 1)
    def y(v): return h - pad - (h - 2 * pad) * (v - lo) / span

    parts = [f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">']
    for level in (1.0, 2.0):
        if lo <= level <= hi:
            colour = "#8b949e" if level == 1.0 else "#3fb950"
            parts.append(f'<line x1="{pad}" x2="{w - pad}" y1="{y(level):.1f}" '
                         f'y2="{y(level):.1f}" stroke="{colour}" stroke-width="1" '
                         f'stroke-dasharray="4 4" opacity=".5"/>')
            parts.append(f'<text x="{w - pad + 3}" y="{y(level) + 3:.1f}" '
                         f'fill="{colour}" font-size="10">{level:.0f}x</text>')

    legend = []
    for k, (book, vals) in enumerate(norm.items()):
        colour = COLOURS[k % len(COLOURS)]
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}"
                       for i, v in enumerate(vals) if pd.notna(v))
        parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.8" '
                     f'points="{pts}"/>')
        legend.append(f'<span><i class="sw" style="background:{colour}"></i>'
                      f'{book}</span>')

    parts.append("</svg>")
    return "\n".join(parts), " ".join(legend)


def _trades(path: str) -> str:
    if not os.path.exists(path):
        return "<div style='color:#8b949e'>no trades yet</div>"
    df = pd.read_csv(path)
    closed = df[df["pnl"].astype(str).str.len() > 0]
    closed = closed[closed["reason"].astype(str).str.startswith("OPEN") == False]
    if closed.empty:
        return "<div style='color:#8b949e'>no closed trades yet</div>"

    closed = closed.tail(40).iloc[::-1]
    head = ("<tr><th>time</th><th>book</th><th>symbol</th><th>side</th>"
            "<th>module</th><th>lev</th><th>pnl $</th><th>exit</th></tr>")
    rows = []
    for _, r in closed.iterrows():
        pnl = float(r["pnl"])
        cls = "up" if pnl > 0 else "down"
        rows.append(
            f"<tr><td>{str(r['time'])[:16]}</td><td>{r['book']}</td>"
            f"<td>{r['symbol']}</td><td>{r['side']}</td><td>{r['module']}</td>"
            f"<td>{r['leverage']}x</td><td class='{cls}'>{pnl:+.3f}</td>"
            f"<td>{r['reason']}</td></tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def build_status_md(out_path: str = "STATUS.md") -> str:
    """A plain-markdown scoreboard GitHub renders on its own.

    The HTML dashboard is richer, but GitHub serves .html as source text, so on
    a phone it is unreadable. This file is the one that answers "how are they
    going" in one glance, straight from the repo front page.
    """
    state = load_state()
    if not state:
        raise RuntimeError("no live state yet - run `python run.py once` first")

    fx = state.get("aud_usd", C.FALLBACK_AUD_USD)
    start_usd = state["start_usd"]
    lines = [
        "# Live scoreboard",
        "",
        f"**Updated:** {str(state.get('updated_at', ''))[:19]} UTC &middot; "
        f"tick #{state.get('ticks', 0)} &middot; "
        f"started {str(state.get('created_at', ''))[:10]} "
        f"at A${C.START_EQUITY_AUD:.2f} (US${start_usd:.2f})",
        "",
        f"**Trading:** {', '.join(state.get('universe', []))}",
        "",
        "| Book | USD | AUD | Return | Trades | Win% | Open | Liq | Fees | Funding |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in state.get("summary", []):
        name = row["book"] + (" **DEAD**" if row["dead"] else "")
        lines.append(
            f"| {name} | ${row['equity']:.2f} | A${row['equity'] / fx:.2f} "
            f"| {row['return_pct']:+.1f}% | {row['trades']} "
            f"| {row['win_rate']:.0f}% | {row['open']} | {row['liquidations']} "
            f"| ${row['fees']:.2f} | ${row['funding']:.2f} |"
        )

    lines += ["", "## Recent closed trades", ""]
    if not os.path.exists(C.TRADE_LOG):
        lines.append("_No trades yet._")
    else:
        df = pd.read_csv(C.TRADE_LOG)
        closed = df[(df["pnl"].astype(str).str.len() > 0)
                    & (~df["reason"].astype(str).str.startswith("OPEN"))]
        if closed.empty:
            lines.append("_No closed trades yet._")
        else:
            lines += [
                "| Time | Book | Symbol | Side | Module | Lev | PnL $ | Exit |",
                "|---|---|---|---|---|---:|---:|---|",
            ]
            for _, r in closed.tail(25).iloc[::-1].iterrows():
                lines.append(
                    f"| {str(r['time'])[:16]} | {r['book']} | {r['symbol']} "
                    f"| {r['side']} | {r['module']} | {r['leverage']}x "
                    f"| {float(r['pnl']):+.3f} | {r['reason']} |"
                )

    lines += [
        "",
        "---",
        "",
        "Full trade log: [`state/trades.csv`](state/trades.csv) &middot; "
        "equity history: [`state/equity.csv`](state/equity.csv)",
        "",
        "_Paper trading only. No exchange keys, no orders, no money._",
        "",
    ]

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return out_path


def build(out_path: str = "reports/index.html") -> str:
    state = load_state()
    if not state:
        raise RuntimeError("no live state yet — run `python run.py once` first")

    eq = pd.DataFrame()
    if os.path.exists(C.EQUITY_LOG):
        eq = pd.read_csv(C.EQUITY_LOG)

    books = [b.name for b in C.BOOKS]
    fx = state.get("aud_usd", C.FALLBACK_AUD_USD)
    chart, legend = _chart(eq, books, state["start_usd"])

    html = TEMPLATE.format(
        updated=str(state.get("updated_at", ""))[:19],
        ticks=state.get("ticks", 0),
        universe=", ".join(state.get("universe", [])),
        created=str(state.get("created_at", ""))[:10],
        start_aud=C.START_EQUITY_AUD,
        start_usd=state["start_usd"],
        cards=_cards(state.get("summary", []), fx),
        chart=chart,
        legend=legend,
        trades=_trades(C.TRADE_LOG),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path
