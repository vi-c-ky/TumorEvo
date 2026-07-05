"""
trial.py — Virtual Trial blueprint for TumorEvo

Route: POST /run_trial
"""

import math
import numpy as np
from flask import Blueprint, request, jsonify
from solver import odes, IC, K, T_END

trial_bp = Blueprint("trial", __name__)

_TRIAL_DT      = 1.0   # coarser step for speed; PFS detection accurate to ±1 day
_BASE_R        = 0.3
_BASE_DEATH    = 0.05
_BASE_KILL_EFF = 0.8
_BASE_LOG_MU   = -6.0
_PFS_FACTOR    = 1.5
_BASELINE_N    = float(IC[0])  # 1e6

_VALID_ARMS = {"continuous", "pulsed", "metronomic", "escalating", "control"}


# ── Patient simulation ────────────────────────────────────────────────

def _rk4_run(p: dict) -> tuple:
    """
    Integrate the ODE using solver.odes with a coarser 1-day step for speed.
    Returns (pfs, event, complete_response, resistant).
    """
    dt = _TRIAL_DT
    steps = int(T_END / dt)
    y = IC.copy()
    threshold = _PFS_FACTOR * _BASELINE_N

    pfs = float(T_END)
    event = 0

    for i in range(steps + 1):
        t = float(i * dt)
        y = np.maximum(y, 0.0)
        y[4] = min(y[4], 100.0)
        N = float(y[0] + y[1] + y[2] + y[3])

        if N > 1000.0 * K:
            y[:4] *= (1000.0 * K) / N
            N = 1000.0 * K

        if event == 0 and N > threshold and i > 0:
            pfs = t
            event = 1

        if i < steps:
            k1 = odes(t,        y,              p)
            k2 = odes(t + dt/2, y + dt/2 * k1, p)
            k3 = odes(t + dt/2, y + dt/2 * k2, p)
            k4 = odes(t + dt,   y + dt   * k3, p)
            y  = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    final_n   = max(0.0, float(y[0] + y[1] + y[2] + y[3]))
    rab_final = max(0.0, float(y[3]))
    complete_response = final_n < 0.05 * _BASELINE_N
    resistant = (rab_final > 0.5 * final_n) if final_n > 1.0 else False

    return pfs, event, complete_response, resistant


def _sample_patients(n: int, heterogeneity: float, rng) -> list:
    """Return a list of parameter dicts, one per patient."""
    h = heterogeneity
    growth = np.clip(rng.normal(_BASE_R,        h * _BASE_R,              n), 0.01, None)
    death  = np.clip(rng.normal(_BASE_DEATH,    h * 0.7 * _BASE_DEATH,    n), 0.00, None)
    eff_r  = np.maximum(growth - death, 0.01)
    drug   = np.clip(rng.normal(_BASE_KILL_EFF, h * 1.2 * _BASE_KILL_EFF, n), 0.00, 1.00)
    log_mu = rng.normal(_BASE_LOG_MU, h * 1.5, n)
    mu     = np.clip(10.0 ** log_mu, 1e-10, 1e-3)
    return [
        {"r": float(eff_r[i]), "mu": float(mu[i]),
         "kill_eff": float(drug[i]), "treat_start": 0.0}
        for i in range(n)
    ]


def _arm_params(patient: dict, arm_name: str) -> dict:
    p = dict(patient)
    if arm_name == "control":
        p["kill_eff"] = 0.0
        p["schedule"] = "continuous"
    else:
        p["schedule"] = arm_name
    return p


# ── Kaplan-Meier ──────────────────────────────────────────────────────

def _km_curve(times: list, events: list) -> tuple:
    """KM estimator with Greenwood 95% CI. Returns (km_t, km_s, km_u, km_l)."""
    event_times = sorted({t for t, e in zip(times, events) if e == 1})
    if not event_times:
        return [0.0, float(T_END)], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]

    km_t: list = [0.0]
    km_s: list = [1.0]
    km_u: list = [1.0]
    km_l: list = [1.0]
    S, gw = 1.0, 0.0

    for t in event_times:
        n_risk  = sum(1 for tt in times if tt >= t)
        n_event = sum(1 for tt, e in zip(times, events) if tt == t and e == 1)
        if n_risk == 0:
            continue
        S *= 1.0 - n_event / n_risk
        denom = n_risk * (n_risk - n_event)
        if denom > 0:
            gw += n_event / denom
        se = S * math.sqrt(gw) if S > 0 else 0.0
        km_t.append(float(t))
        km_s.append(float(S))
        km_u.append(float(min(1.0, S + 1.96 * se)))
        km_l.append(float(max(0.0, S - 1.96 * se)))

    return km_t, km_s, km_u, km_l


def _median_pfs(km_t: list, km_s: list) -> float:
    for t, s in zip(km_t, km_s):
        if s <= 0.5:
            return float(t)
    return float(T_END)


# ── Log-rank test ─────────────────────────────────────────────────────

def _logrank_chi2(t_a: list, e_a: list, t_b: list, e_b: list) -> float:
    """Manual chi-squared log-rank test (Mantel 1966). Returns two-sided p-value."""
    from scipy.stats import chi2

    event_times = sorted({t for t, e in zip(t_a + t_b, e_a + e_b) if e == 1})
    if not event_times:
        return 1.0

    O1 = E1 = V = 0.0
    for t in event_times:
        n1 = sum(1 for tt in t_a if tt >= t)
        n2 = sum(1 for tt in t_b if tt >= t)
        d1 = sum(1 for tt, e in zip(t_a, e_a) if tt == t and e == 1)
        d2 = sum(1 for tt, e in zip(t_b, e_b) if tt == t and e == 1)
        n, d = n1 + n2, d1 + d2
        if n < 2:
            continue
        E1 += d * n1 / n
        O1 += d1
        if n > 1:
            V += d * n1 * n2 * (n - d) / (n ** 2 * (n - 1))

    if V <= 0:
        return 1.0
    return float(chi2.sf((O1 - E1) ** 2 / V, df=1))


def _make_pval_fn(arm_outcomes: dict):
    """Returns a (p_value_fn, available) pair, preferring lifelines then scipy."""
    try:
        from lifelines.statistics import logrank_test

        def pval(a, b):
            ta = [r[0] for r in arm_outcomes[a]]
            ea = [r[1] for r in arm_outcomes[a]]
            tb = [r[0] for r in arm_outcomes[b]]
            eb = [r[1] for r in arm_outcomes[b]]
            return float(logrank_test(ta, tb, event_observed_A=ea, event_observed_B=eb).p_value)

        return pval, True
    except Exception:
        pass

    try:
        import scipy  # noqa: F401

        def pval(a, b):
            ta = [r[0] for r in arm_outcomes[a]]
            ea = [r[1] for r in arm_outcomes[a]]
            tb = [r[0] for r in arm_outcomes[b]]
            eb = [r[1] for r in arm_outcomes[b]]
            return _logrank_chi2(ta, ea, tb, eb)

        return pval, True
    except Exception:
        return None, False


# ── Route ─────────────────────────────────────────────────────────────

@trial_bp.route("/run_trial", methods=["POST"])
def run_trial():
    data = request.get_json(force=True) or {}

    try:
        n_patients    = max(50, min(500, int(data.get("n_patients", 200))))
        arms          = [a for a in data.get("arms", ["continuous", "pulsed", "control"])
                         if a in _VALID_ARMS]
        heterogeneity = max(0.05, min(0.50, float(data.get("heterogeneity", 0.2))))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid parameters: {exc}"}), 400

    if not arms:
        arms = ["continuous"]

    rng = np.random.default_rng()
    patients = _sample_patients(n_patients, heterogeneity, rng)

    # Run all patient × arm simulations
    arm_outcomes: dict = {}  # arm -> list of (pfs, event, cr, resistant)
    for arm in arms:
        arm_outcomes[arm] = [_rk4_run(_arm_params(pt, arm)) for pt in patients]

    # Per-arm summaries
    arms_out: dict = {}
    for arm, rows in arm_outcomes.items():
        times  = [r[0] for r in rows]
        events = [r[1] for r in rows]
        crs    = [r[2] for r in rows]
        resis  = [r[3] for r in rows]

        km_t, km_s, km_u, km_l = _km_curve(times, events)
        arms_out[arm] = {
            "km_times":               km_t,
            "km_survival":            km_s,
            "km_ci_upper":            km_u,
            "km_ci_lower":            km_l,
            "median_pfs":             round(_median_pfs(km_t, km_s), 1),
            "pct_complete_response":  round(100.0 * sum(crs)  / len(crs),  1),
            "pct_resistant":          round(100.0 * sum(resis) / len(resis), 1),
            "n":                      len(rows),
        }

    # Pairwise log-rank
    pairwise: dict = {}
    pval_fn, has_pval = _make_pval_fn(arm_outcomes)
    if has_pval:
        arm_list = list(arms_out.keys())
        for i in range(len(arm_list)):
            for j in range(i + 1, len(arm_list)):
                a, b = arm_list[i], arm_list[j]
                try:
                    p = pval_fn(a, b)
                    pairwise[f"{a}_vs_{b}"] = {
                        "p_value":     round(p, 4),
                        "significant": p < 0.05,
                    }
                except Exception:
                    pass

    return jsonify({"arms": arms_out, "pairwise_logrank": pairwise})
