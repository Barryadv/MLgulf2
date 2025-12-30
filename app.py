"""
FastAPI dashboard for pair_trade_1120_KSA.py
===========================================

This is intentionally kept as a *second file* so that:
- `pair_trade_1120_KSA.py` remains a standalone research/backtest script (CLI).
- This file focuses on serving results via FastAPI without mixing web concerns
  into the research script.

Run (example):
  python app.py

Then open:
  http://127.0.0.1:8000

Dependencies:
  fastapi, uvicorn
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
import traceback
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse

import pair_trade_1120_KSA as strat


Period = Literal["train", "test", "full"]


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    """Compute drawdown series from an equity curve."""
    cum = equity.copy()
    running_max = cum.cummax()
    dd = (cum / running_max) - 1.0
    return dd


def _month_pivot(returns: pd.Series) -> pd.DataFrame:
    """Monthly return pivot (year x month) using the same monthly sum convention as the script."""
    monthly_ret = returns.resample("M").sum()
    monthly_df = pd.DataFrame(
        {"year": monthly_ret.index.year, "month": monthly_ret.index.month, "return": monthly_ret.values}
    )
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    # Ensure 1..12 columns exist
    pivot = pivot.reindex(columns=list(range(1, 13)))
    pivot.columns = strat.MONTH_NAMES
    return pivot


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-serializable records."""
    # Use ISO for timestamps
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (pd.Timestamp, datetime)):
                rec[k] = v.isoformat()
            elif isinstance(v, (np.floating, np.integer)):
                rec[k] = float(v)
            else:
                rec[k] = v
        out.append(rec)
    return out


def compute_state() -> dict[str, Any]:
    """
    Run the same pipeline as `pair_trade_1120_KSA.py` and return results for the API.

    Note: This calls into the strategy module functions. Those functions print to stdout
    as part of the research script; we intentionally do not suppress prints.
    """
    prices = strat.fetch_prices(strat.TICKERS, strat.DATA_START, strat.DATA_END)
    prices = strat.prepare_data(prices)

    train_mask = (prices.index >= strat.TRAIN_START) & (prices.index <= strat.TRAIN_END)
    test_mask = (prices.index >= strat.TEST_START) & (prices.index <= strat.TEST_END)

    prices_train = prices[train_mask].copy()
    prices_test = prices[test_mask].copy()  # kept for parity with the script (may be unused later)

    beta = strat.fit_hedge_ratio(prices_train)

    full_data, feature_cols = strat.build_features_and_labels(prices, beta)
    train_data = full_data[(full_data.index >= strat.TRAIN_START) & (full_data.index <= strat.TRAIN_END)].copy()
    test_data = full_data[(full_data.index >= strat.TEST_START) & (full_data.index <= strat.TEST_END)].copy()

    X_train = train_data[feature_cols]
    y_train = train_data["label"]
    scaler, model = strat.fit_model(X_train, y_train)

    train_bt = strat.backtest(train_data, feature_cols, scaler, model, beta)
    test_bt = strat.backtest(test_data, feature_cols, scaler, model, beta)

    metrics_train = strat.performance_report(train_bt, label="TRAIN (in-sample)")
    metrics_test = strat.performance_report(test_bt, label="TEST (out-of-sample)")

    all_bt = pd.concat([train_bt, test_bt])
    # Recompute trade_id on full history (same logic as script helper)
    all_bt = all_bt.copy()
    all_bt["trade_id"] = (all_bt["position"] != all_bt["position"].shift()).cumsum()
    all_bt.loc[all_bt["position"] == 0, "trade_id"] = 0
    trades = strat.extract_trades(all_bt)

    # Current position snapshot (last row of full history)
    last_row = all_bt.iloc[-1]
    pos_map = {
        1: "LONG SPREAD (long 1120.SR, short KSA)",
        -1: "SHORT SPREAD (short 1120.SR, long KSA)",
        0: "FLAT (no position)",
    }
    current_position = {
        "as_of": all_bt.index[-1].date().isoformat(),
        "position": int(last_row["position"]),
        "position_text": pos_map[int(last_row["position"])],
        "hold_days": int(last_row.get("hold_days", 0)) if float(last_row.get("position", 0)) != 0 else 0,
        "zscore": float(last_row.get("zscore", np.nan)),
        "prob": float(last_row.get("prob", np.nan)),
    }

    # Precompute chart payloads (test period emphasis)
    test_equity = test_bt["equity"].copy()
    test_drawdown = _compute_drawdown(test_equity)
    test_monthly = _month_pivot(test_bt["strat_ret"].dropna())

    state: dict[str, Any] = {
        "computed_at": datetime.utcnow().isoformat() + "Z",
        "config": {
            "tickers": strat.TICKERS,
            "data_start": strat.DATA_START,
            "data_end": strat.DATA_END,
            "train_start": strat.TRAIN_START,
            "train_end": strat.TRAIN_END,
            "test_start": strat.TEST_START,
            "test_end": strat.TEST_END,
            "cost_bps_per_leg": strat.COST_BPS_PER_LEG,
            "rolling_window": strat.ROLLING_WINDOW,
            "zscore_entry": strat.ZSCORE_ENTRY,
            "zscore_exit": strat.ZSCORE_EXIT,
            "prob_threshold": strat.PROB_THRESHOLD,
            "max_hold_days": strat.MAX_HOLD_DAYS,
        },
        "model": {
            "description": "LogisticRegression with StandardScaler",
            "features": feature_cols,
            "beta": float(beta),
        },
        "metrics": {"train": metrics_train, "test": metrics_test},
        "current_position": current_position,
        # Keep raw objects for endpoints to slice
        "_frames": {
            "train_bt": train_bt,
            "test_bt": test_bt,
            "all_bt": all_bt,
            "trades": trades,
            "test_monthly": test_monthly,
            "test_drawdown": test_drawdown,
        },
    }
    return state


STATE_LOCK = threading.Lock()
STATE: dict[str, Any] | None = None


def get_state() -> dict[str, Any]:
    global STATE
    with STATE_LOCK:
        if STATE is None:
            STATE = compute_state()
        return STATE


def refresh_state() -> dict[str, Any]:
    global STATE
    with STATE_LOCK:
        STATE = compute_state()
        return STATE


app = FastAPI(title="Pairs Trading Dashboard: 1120.SR vs KSA", version="0.1.0")

def _json_500(context: str, exc: Exception) -> JSONResponse:
    """
    Return a JSON 500 response with details for debugging deploy/runtime issues.

    This is intentionally verbose to help diagnose production-only failures
    (e.g., missing deps, yfinance issues, empty data).
    """
    tb = traceback.format_exc()
    payload = {
        "error": str(exc),
        "context": context,
        "traceback": tb,
    }
    return JSONResponse(status_code=500, content=payload)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # Gulf landing hub (directory-style). Served at https://quant.stocks-x.ai/gulf/
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Gulf Strategies</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 18px; color: #111; }
    .muted { color: #666; font-size: 12px; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 12px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 14px; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
    .title { font-weight: 800; font-size: 16px; margin-bottom: 6px; }
    .desc { color: #333; font-size: 13px; margin-bottom: 10px; }
    a.btn { display: inline-block; padding: 8px 10px; border: 1px solid #ddd; border-radius: 8px; background: #f7f7f7; text-decoration: none; color: #111; }
    a.btn:hover { background: #efefef; }
    .disabled { opacity: 0.55; }
  </style>
</head>
<body>
  <h2>Gulf Strategies</h2>
  <div class="muted" style="margin-bottom:12px;">
    Hub for Gulf strategy dashboards under <code>/gulf/</code>.
  </div>

  <div class="grid">
    <div class="card">
      <div class="title">AlRajhi Pair Strategy</div>
      <div class="desc">Pairs trading dashboard (1120.SR vs KSA) with ML filter and analytics.</div>
      <a class="btn" href="./alrajhi-pair/">Open</a>
    </div>

    <div class="card">
      <div class="title">Aramco Pair Strategy</div>
      <div class="desc">Placeholder for an Aramco-related pairs strategy dashboard.</div>
      <a class="btn" href="./aramco-pair/">Open</a>
    </div>

    <div class="card disabled">
      <div class="title">#3 Strategy</div>
      <div class="desc">Placeholder for your next Gulf strategy.</div>
      <a class="btn" href="javascript:void(0)">Coming soon</a>
    </div>

    <div class="card disabled">
      <div class="title">#4 Strategy</div>
      <div class="desc">Placeholder for another Gulf strategy.</div>
      <a class="btn" href="javascript:void(0)">Coming soon</a>
    </div>
  </div>

  <div class="muted" style="margin-top:14px;">
    Tip: If proxied behind <code>https://quant.stocks-x.ai/gulf/</code>, keep links relative (as above).
  </div>
</body>
</html>"""


@app.get("/alrajhi-pair/", response_class=HTMLResponse)
def alrajhi_pair() -> str:
    # Single-file HTML dashboard (no extra static files).
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Pairs Dashboard: 1120.SR vs KSA</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 18px; color: #111; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
    .kpi { font-size: 24px; font-weight: 700; }
    .muted { color: #666; font-size: 12px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; font-size: 13px; }
    th { background: #fafafa; }
    .btn { padding: 8px 10px; border: 1px solid #ddd; border-radius: 8px; background: #f7f7f7; cursor: pointer; }
    .btn:hover { background: #efefef; }
    canvas { max-height: 320px; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 900px) { .grid2 { grid-template-columns: 1fr; } }
    .heatmap { display: grid; grid-template-columns: 60px repeat(12, 1fr); gap: 2px; }
    .hm-cell { padding: 6px 4px; font-size: 12px; text-align: center; border-radius: 4px; }
    .hm-head { font-weight: 700; background: #fafafa; }
  </style>
</head>
<body>
  <div class="muted" style="margin-bottom:8px;"><a href="../">← Back to Gulf hub</a></div>
  <h2>AlRajhi Pair Strategy: 1120.SR vs KSA</h2>
  <div class="row" style="align-items:center;">
    <div class="muted" id="computedAt">Loading...</div>
    <button class="btn" onclick="refresh()">Refresh</button>
  </div>

  <div class="card" id="errorBox" style="display:none; border-color:#e74c3c; background:#fff5f5;">
    <div style="font-weight:700; color:#c0392b; margin-bottom:6px;">Dashboard error</div>
    <div class="muted" id="errorText" style="color:#c0392b; white-space:pre-wrap;"></div>
    <div class="muted" style="margin-top:8px;">
      Tip: confirm the backend is running and `http://127.0.0.1:8000/api/summary` returns JSON.
    </div>
  </div>

  <div class="row">
    <div class="card" style="min-width:260px;">
      <div class="muted">OOS Sharpe</div>
      <div class="kpi" id="kpiSharpe">-</div>
    </div>
    <div class="card" style="min-width:260px;">
      <div class="muted">OOS Max DD</div>
      <div class="kpi" id="kpiMaxDD">-</div>
    </div>
    <div class="card" style="min-width:260px;">
      <div class="muted">Beta (train OLS)</div>
      <div class="kpi" id="kpiBeta">-</div>
    </div>
    <div class="card" style="min-width:260px;">
      <div class="muted">Current Position</div>
      <div class="kpi" id="kpiPos">-</div>
      <div class="muted" id="kpiPosMeta"></div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h3 style="margin:0 0 8px 0;">Performance (Test OOS)</h3>
      <table>
        <tbody id="metricsTable"></tbody>
      </table>
    </div>
    <div class="card">
      <h3 style="margin:0 0 8px 0;">Recent Trades</h3>
      <table>
        <thead>
          <tr><th>Entry</th><th>Exit</th><th>Dir</th><th>Hold</th><th>Gross</th><th>Costs</th><th>Net</th></tr>
        </thead>
        <tbody id="tradesTable"></tbody>
      </table>
    </div>
  </div>

  <div class="grid2" style="margin-top:12px;">
    <div class="card">
      <h3 style="margin:0 0 8px 0;">Equity (Test OOS)</h3>
      <canvas id="equityChart"></canvas>
    </div>
    <div class="card">
      <h3 style="margin:0 0 8px 0;">Drawdown (Test OOS)</h3>
      <canvas id="ddChart"></canvas>
    </div>
  </div>

  <div class="card" style="margin-top:12px;">
    <h3 style="margin:0 0 8px 0;">Position Curve (Test OOS)</h3>
    <div class="muted" style="margin-bottom:6px;">
      Markers: green ▲ = LONG, red ▼ = SHORT, brown ● = FLAT
    </div>
    <canvas id="posChart"></canvas>
  </div>

  <div class="card" style="margin-top:12px;">
    <h3 style="margin:0 0 8px 0;">Monthly Returns Heatmap (Test OOS)</h3>
    <div class="heatmap" id="heatmap"></div>
    <div class="muted" style="margin-top:6px;">Returns shown as monthly sum of daily strategy returns (same convention as the script).</div>
  </div>

<script>
let equityChart = null;
let ddChart = null;
let posChart = null;

function showError(msg) {
  const box = document.getElementById("errorBox");
  const text = document.getElementById("errorText");
  if (!box || !text) return;
  text.innerText = String(msg || "Unknown error");
  box.style.display = "block";
}

function hideError() {
  const box = document.getElementById("errorBox");
  if (!box) return;
  box.style.display = "none";
}

window.addEventListener("error", (e) => {
  showError(e.message || e.error || "Unhandled error");
});
window.addEventListener("unhandledrejection", (e) => {
  showError(e.reason || "Unhandled promise rejection");
});

async function safeFetchJson(url) {
  let resp;
  try {
    resp = await fetch(url, { cache: "no-store" });
  } catch (err) {
    throw new Error(`Network error fetching ${url}: ${err}`);
  }
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status} fetching ${url}: ${body.slice(0, 500)}`);
  }
  return await resp.json();
}

function pct(x, digits=2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "-";
  return (x * 100).toFixed(digits) + "%";
}

function num(x, digits=3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "-";
  return Number(x).toFixed(digits);
}

function colorFor(v) {
  // simple red/green scale centered at 0
  if (v === null || v === undefined || Number.isNaN(v)) return "#f4f4f4";
  const x = Math.max(-0.12, Math.min(0.12, v));
  if (x >= 0) {
    const t = x / 0.12;
    return `rgba(46, 204, 113, ${0.15 + 0.55*t})`;
  } else {
    const t = (-x) / 0.12;
    return `rgba(231, 76, 60, ${0.15 + 0.55*t})`;
  }
}

async function loadAll() {
  try {
    hideError();
    // NOTE: use ../api so this works from /alrajhi-pair/ and when mounted under /gulf/alrajhi-pair/
    const API = "../api";
    const summary = await safeFetchJson(`${API}/summary`);
  document.getElementById("computedAt").innerText = "Computed at (UTC): " + summary.computed_at;
  document.getElementById("kpiSharpe").innerText = num(summary.metrics.test.sharpe, 3);
  document.getElementById("kpiMaxDD").innerText = pct(summary.metrics.test.max_drawdown, 2);
  document.getElementById("kpiBeta").innerText = num(summary.model.beta, 4);

  const posInfo = summary.current_position;
  document.getElementById("kpiPos").innerText = posInfo.position_text;
  document.getElementById("kpiPosMeta").innerText =
    `As of ${posInfo.as_of} | z=${num(posInfo.zscore, 3)} | p=${num(posInfo.prob, 3)} | hold=${posInfo.hold_days}d`;

  // metrics table
  const m = summary.metrics.test;
  const rows = [
    ["Date range", `${m.start_date} to ${m.end_date}`],
    ["Total return", pct(m.total_return)],
    ["Annualized return", pct(m.ann_return)],
    ["Annualized vol", pct(m.ann_vol)],
    ["Sharpe", num(m.sharpe, 3)],
    ["Max drawdown", pct(m.max_drawdown)],
    ["# trades", m.n_trades],
    ["Win rate", pct(m.win_rate)],
    ["Avg hold days", Number(m.avg_hold_days).toFixed(1)],
    ["Total costs", pct(m.total_costs)],
  ];
  document.getElementById("metricsTable").innerHTML =
    rows.map(r => `<tr><th>${r[0]}</th><td>${r[1]}</td></tr>`).join("");

  // trades
  const trades = await safeFetchJson(`${API}/trades?limit=10`);
  document.getElementById("tradesTable").innerHTML = trades.trades.map(t =>
    `<tr>
      <td>${t.entry_date.slice(0,10)}</td>
      <td>${t.exit_date.slice(0,10)}</td>
      <td>${t.direction}</td>
      <td>${t.hold_days}</td>
      <td>${pct(t.gross_pnl, 2)}</td>
      <td>${pct(t.costs, 2)}</td>
      <td>${pct(t.net_pnl, 2)}</td>
    </tr>`
  ).join("");

  // equity chart
  const eq = await safeFetchJson(`${API}/equity?period=test`);
  const eqLabels = eq.points.map(p => p.date.slice(0,10));
  const eqVals = eq.points.map(p => p.equity);
  if (typeof Chart !== "undefined") {
    if (equityChart) equityChart.destroy();
    equityChart = new Chart(document.getElementById("equityChart"), {
      type: "line",
      data: { labels: eqLabels, datasets: [{ label: "Equity", data: eqVals, borderWidth: 1, pointRadius: 0 }]},
      options: { responsive: true, scales: { x: { display: true }, y: { display: true } } }
    });
  }

  // drawdown chart
  const dd = await safeFetchJson(`${API}/drawdown?period=test`);
  const ddLabels = dd.points.map(p => p.date.slice(0,10));
  const ddVals = dd.points.map(p => p.drawdown);
  if (typeof Chart !== "undefined") {
    if (ddChart) ddChart.destroy();
    ddChart = new Chart(document.getElementById("ddChart"), {
      type: "line",
      data: { labels: ddLabels, datasets: [{ label: "Drawdown", data: ddVals, borderWidth: 1, pointRadius: 0 }]},
      options: { responsive: true, scales: { y: { ticks: { callback: (v) => (v*100).toFixed(0) + "%" } } } }
    });
  }

  // position chart
  const posSeries = await safeFetchJson(`${API}/position?period=test`);
  const posLabels = posSeries.points.map(p => p.date.slice(0,10));
  const posVals = posSeries.points.map(p => p.position);
  if (typeof Chart !== "undefined") {
    if (posChart) posChart.destroy();
    posChart = new Chart(document.getElementById("posChart"), {
      type: "line",
      data: {
        labels: posLabels,
        datasets: [{
          label: "Position",
          data: posVals,
          borderWidth: 1,
          stepped: true,
          pointRadius: 2,
          pointHoverRadius: 4,
          pointStyle: (ctx) => {
            const y = ctx.raw;
            if (y === 1) return "triangle";
            if (y === -1) return "triangle";
            return "circle";
          },
          pointRotation: (ctx) => {
            const y = ctx.raw;
            if (y === -1) return 180;
            return 0;
          },
          pointBackgroundColor: (ctx) => {
            const y = ctx.raw;
            if (y === 0) return "saddlebrown";
            if (y === -1) return "red";
            return "green";
          },
          pointBorderColor: (ctx) => {
            const y = ctx.raw;
            if (y === 0) return "saddlebrown";
            if (y === -1) return "red";
            return "green";
          },
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            min: -1.25,
            max: 1.25,
            ticks: {
              callback: (v) => {
                if (v === 1) return "LONG";
                if (v === 0) return "FLAT";
                if (v === -1) return "SHORT";
                return "";
              }
            }
          }
        }
      }
    });
  }

  // heatmap
  const hm = await safeFetchJson(`${API}/monthly?period=test`);
  const container = document.getElementById("heatmap");
  const months = hm.columns;
  const years = hm.index;
  container.innerHTML = "";
  container.innerHTML += `<div class="hm-cell hm-head"></div>` + months.map(m => `<div class="hm-cell hm-head">${m}</div>`).join("");
  years.forEach((y, i) => {
    container.innerHTML += `<div class="hm-cell hm-head">${y}</div>`;
    months.forEach((m, j) => {
      const v = hm.values[i][j];
      const bg = colorFor(v);
      const txt = (v === null || v === undefined) ? "" : (v*100).toFixed(1);
      container.innerHTML += `<div class="hm-cell" style="background:${bg};">${txt}</div>`;
    });
  });
  } catch (err) {
    showError(err);
  }
}

async function refresh() {
  const API = "../api";
  await fetch(`${API}/refresh`, { method: "POST" });
  await loadAll();
}

loadAll();
</script>
</body>
</html>"""


@app.get("/aramco-pair/", response_class=HTMLResponse)
def aramco_pair() -> str:
    # Placeholder page. Strategy not implemented here yet.
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Aramco Pair Strategy (Coming Soon)</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 18px; color: #111; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 14px; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.05); max-width: 760px; }
    .muted { color: #666; font-size: 12px; }
    a { color: #111; }
  </style>
</head>
<body>
  <div class="muted" style="margin-bottom:8px;"><a href="../">← Back to Gulf hub</a></div>
  <h2>Aramco Pair Strategy</h2>
  <div class="card">
    <div style="font-weight:800; margin-bottom:6px;">Coming soon</div>
    <div class="muted">Placeholder page for an Aramco-related pairs strategy dashboard.</div>
  </div>
</body>
</html>"""


@app.get("/api/summary")
def api_summary() -> JSONResponse:
    try:
        st = get_state()
        payload = {
            "computed_at": st["computed_at"],
            "config": st["config"],
            "model": st["model"],
            "metrics": st["metrics"],
            "current_position": st["current_position"],
        }
        # Ensure dates (and other non-JSON-native types) are encoded safely
        return JSONResponse(jsonable_encoder(payload))
    except Exception as e:
        return _json_500("api_summary", e)


@app.post("/api/refresh")
def api_refresh() -> JSONResponse:
    st = refresh_state()
    return JSONResponse({"ok": True, "computed_at": st["computed_at"]})


@app.get("/api/trades")
def api_trades(limit: int = 10) -> JSONResponse:
    st = get_state()
    trades: pd.DataFrame = st["_frames"]["trades"]
    if len(trades) == 0:
        return JSONResponse({"trades": []})
    out = trades.tail(int(limit)).iloc[::-1].copy()
    out["entry_date"] = out["entry_date"].astype(str)
    out["exit_date"] = out["exit_date"].astype(str)
    return JSONResponse({"trades": _df_to_records(out)})


@app.get("/api/equity")
def api_equity(period: Period = "test") -> JSONResponse:
    st = get_state()
    frames = st["_frames"]
    if period == "train":
        df = frames["train_bt"]
    elif period == "test":
        df = frames["test_bt"]
    elif period == "full":
        df = frames["all_bt"]
    else:
        raise HTTPException(status_code=400, detail="invalid period")
    # Starlette's JSONResponse rejects NaN/inf by default ("Out of range float values...").
    # Equity can have an initial NaN due to diff()/cumsum() behavior in the strategy code.
    points: list[dict[str, Any]] = []
    for idx, val in df["equity"].items():
        fv = float(val)
        points.append({"date": idx.isoformat(), "equity": fv if np.isfinite(fv) else None})
    return JSONResponse({"period": period, "points": points})

@app.get("/api/position")
def api_position(period: Period = "test") -> JSONResponse:
    """
    Return the daily position series for charting.

    Position encoding:
      +1 = long spread
       0 = flat
      -1 = short spread
    """
    st = get_state()
    frames = st["_frames"]
    if period == "train":
        df = frames["train_bt"]
    elif period == "test":
        df = frames["test_bt"]
    elif period == "full":
        df = frames["all_bt"]
    else:
        raise HTTPException(status_code=400, detail="invalid period")
    points = [{"date": idx.isoformat(), "position": int(val)} for idx, val in df["position"].items()]
    return JSONResponse({"period": period, "points": points})


@app.get("/api/drawdown")
def api_drawdown(period: Period = "test") -> JSONResponse:
    st = get_state()
    frames = st["_frames"]
    if period == "train":
        df = frames["train_bt"]
        dd = _compute_drawdown(df["equity"])
    elif period == "test":
        dd: pd.Series = frames["test_drawdown"]
    elif period == "full":
        df = frames["all_bt"]
        dd = _compute_drawdown(df["equity"])
    else:
        raise HTTPException(status_code=400, detail="invalid period")
    points: list[dict[str, Any]] = []
    for idx, val in dd.items():
        fv = float(val)
        points.append({"date": idx.isoformat(), "drawdown": fv if np.isfinite(fv) else None})
    return JSONResponse({"period": period, "points": points})


@app.get("/api/monthly")
def api_monthly(period: Period = "test") -> JSONResponse:
    st = get_state()
    frames = st["_frames"]
    if period == "test":
        pivot: pd.DataFrame = frames["test_monthly"]
    else:
        # compute on-demand for other periods
        if period == "train":
            df = frames["train_bt"]
        elif period == "full":
            df = frames["all_bt"]
        else:
            raise HTTPException(status_code=400, detail="invalid period")
        pivot = _month_pivot(df["strat_ret"].dropna())
    # JSON friendly
    values: list[list[float | None]] = []
    for _, row in pivot.iterrows():
        values.append([None if pd.isna(v) else float(v) for v in row.values])
    payload = {
        "period": period,
        "index": [int(y) for y in pivot.index.tolist()],
        "columns": pivot.columns.tolist(),
        "values": values,
    }
    return JSONResponse(payload)


if __name__ == "__main__":
    # Keep run behavior simple: `python app.py`
    try:
        import uvicorn  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: uvicorn. Install with `pip install uvicorn`.\n"
            "Then run: python app.py"
        ) from e

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


