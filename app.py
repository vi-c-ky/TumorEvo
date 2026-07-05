"""
app.py — Flask web server for TumorEvo

Routes:
  GET  /              → serves static/index.html
  POST /api/simulate  → runs all 4 schedules, returns JSON
  GET  /api/health    → health check
  GET  /api/csv       → download simulation as CSV
"""

import csv
import io
import json

from flask import Flask, jsonify, request, send_from_directory, Response

from solver import SCHEDULES, run_simulation, compute_metrics
from trial import trial_bp

app = Flask(__name__, static_folder="static")
app.register_blueprint(trial_bp)


# ── Static files ──────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ── Simulation endpoint ───────────────────────────────────────────

@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Accepts JSON body with model parameters, runs the ODE solver
    for all four dosing schedules in sequence, and returns results
    plus per-schedule metrics.

    Request body:
      r           float   proliferation rate (0.1–0.8)
      mu          float   mutation rate (1e-8 to 1e-4)
      kill_eff    float   drug kill efficiency (0.5–1.0)
      treat_start int     treatment start day (0–120)

    Response:
      results   { schedule: { t, S, RA, RB, RAB, N, T } }
      metrics   { schedule: { peak_N, final_N, max_tox, relapse_day, outcome, ti } }
    """
    data = request.get_json(force=True) or {}

    try:
        params = {
            "r":           _clamp(float(data.get("r",           0.3)),  0.05, 1.0),
            "mu":          _clamp(float(data.get("mu",          1e-6)), 1e-10, 1e-3),
            "kill_eff":    _clamp(float(data.get("kill_eff",    0.8)),   0.0, 1.0),
            "treat_start": _clamp(float(data.get("treat_start", 0)),     0.0, 150.0),
        }
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid parameters: {exc}"}), 400

    results = {}
    metrics = {}

    for sched in SCHEDULES:
        p = {**params, "schedule": sched}
        res = run_simulation(p)
        met = compute_metrics(res, p)
        results[sched] = res
        metrics[sched] = met

    return jsonify({"results": results, "metrics": metrics})


# ── CSV export endpoint ───────────────────────────────────────────

@app.route("/api/csv", methods=["POST"])
def export_csv():
    """
    Same parameters as /api/simulate but returns a CSV file for
    the requested schedule only.
    """
    data = request.get_json(force=True) or {}

    try:
        params = {
            "r":           float(data.get("r",           0.3)),
            "mu":          float(data.get("mu",          1e-6)),
            "kill_eff":    float(data.get("kill_eff",    0.8)),
            "treat_start": float(data.get("treat_start", 0)),
            "schedule":    str(data.get("schedule",      "continuous")),
        }
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    res = run_simulation(params)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["day", "S", "RA", "RB", "RAB", "N_total", "toxicity"])
    for i, t in enumerate(res["t"]):
        writer.writerow([
            f"{t:.1f}",
            f"{res['S'][i]:.4e}",
            f"{res['RA'][i]:.4e}",
            f"{res['RB'][i]:.4e}",
            f"{res['RAB'][i]:.4e}",
            f"{res['N'][i]:.4e}",
            f"{res['T'][i]:.3f}",
        ])

    filename = (
        f"tumorevo_{params['schedule']}"
        f"_r{params['r']:.2f}"
        f"_mu{params['mu']:.0e}.csv"
    )

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Health check ──────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "solver": "RK4", "schedules": list(SCHEDULES)})


# ── Helpers ───────────────────────────────────────────────────────

def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("TumorEvo running at http://localhost:5000")
    app.run(debug=True, port=5000)
