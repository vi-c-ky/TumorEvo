"""
solver.py — Goldie-Coldman two-drug resistance ODE system

State vector: [S, RA, RB, RAB, T]
  S   = sensitive cells
  RA  = resistant to Drug A only
  RB  = resistant to Drug B only
  RAB = resistant to both drugs (no kill term)
  T   = cumulative toxicity index [0, 100]

Logistic carrying capacity shared across all subpopulations — this
encodes competitive suppression between sensitive and resistant cells
(Gatenby & Brown 2020).
"""

import numpy as np
from typing import Literal

SCHEDULES = ("continuous", "pulsed", "metronomic", "escalating")
Schedule = Literal["continuous", "pulsed", "metronomic", "escalating"]

K = 1e10          # Carrying capacity (cells) — fixed
T_END = 200.0     # Simulation duration (days)
DT = 0.1          # Integration step (days)
IC = np.array([1e6, 0.0, 0.0, 0.0, 0.0])  # Initial conditions


# ── Drug kill schedule ────────────────────────────────────────────

def compute_kill(t: float, p: dict) -> tuple[float, float, float]:
    """
    Returns (kA, kB, dose_fraction) at time t.

    Max kill rate per drug = 0.8 * kill_efficiency / day.
    At 100% efficiency and full dose, combined kill (kA+kB=1.6/day)
    comfortably exceeds any proliferation rate in the slider range,
    allowing eradication under the right schedule.
    """
    if t < p["treat_start"]:
        return 0.0, 0.0, 0.0

    elapsed = t - p["treat_start"]
    max_kill = 0.8 * p["kill_eff"]
    sched = p["schedule"]

    if sched == "continuous":
        # Constant low-dose infusion — 30% of maximum
        dose = 0.30

    elif sched == "pulsed":
        # Standard chemotherapy: 1 day on, 6 days rest (7-day cycle)
        dose = 1.0 if (elapsed % 7) < 1.0 else 0.0

    elif sched == "metronomic":
        # Frequent sub-therapeutic dose: 0.5-day on per 2.3-day cycle
        dose = 0.40 if (elapsed % 2.3) < 0.5 else 0.0

    elif sched == "escalating":
        # Linear escalation 50% → 150% over first 100 treatment days
        dose = min(1.5, 0.5 + elapsed / 100.0)

    else:
        dose = 0.0

    k = dose * max_kill
    return k, k, dose


# ── ODE right-hand side ───────────────────────────────────────────

def odes(t: float, y: np.ndarray, p: dict) -> np.ndarray:
    """
    Goldie-Coldman model with logistic growth, fitness costs, and
    time-dependent drug kill rates.

    Fitness costs:
      RA, RB  → grow at 95% of base rate  (5% metabolic burden)
      RAB     → grows at 90% of base rate (10% burden for dual resistance)
    """
    y = np.maximum(y, 0.0)
    S, RA, RB, RAB, T = y
    T = min(T, 100.0)

    r  = p["r"]
    mu = p["mu"]    # mutation rate per division per locus

    N  = S + RA + RB + RAB
    lf = max(0.0, 1.0 - N / K)  # logistic factor

    kA, kB, dose_frac = compute_kill(t, p)

    # Sensitive cells: lost to mutation at both loci, killed by both drugs
    dS = (r * S * lf
          - mu * r * S          # → RA (gain A-resistance)
          - mu * r * S          # → RB (gain B-resistance)
          - kA * S
          - kB * S)

    # RA cells: resistant to A, still killed by B; gain B-resistance → RAB
    dRA = (0.95 * r * RA * lf
           + mu * r * S         # influx from S
           - mu * r * RA        # RA → RAB (gain B-resistance)
           - kB * RA)

    # RB cells: resistant to B, still killed by A; gain A-resistance → RAB
    dRB = (0.95 * r * RB * lf
           + mu * r * S         # influx from S
           - mu * r * RB        # RB → RAB (gain A-resistance)
           - kA * RB)

    # RAB cells: fully resistant, no drug kill; 10% fitness cost
    dRAB = (0.90 * r * RAB * lf
            + mu * r * RA       # RA gains B-resistance
            + mu * r * RB)      # RB gains A-resistance

    # Toxicity: accumulates with dose, decays with half-life ~23 days
    # Steady state at full dose → T = 3.0 / 0.03 = 100 (natural ceiling)
    dT = dose_frac * 3.0 - 0.03 * T

    return np.array([dS, dRA, dRB, dRAB, dT])


# ── 4th-order Runge-Kutta integrator ─────────────────────────────

def rk4_step(t: float, y: np.ndarray, dt: float, p: dict) -> np.ndarray:
    k1 = odes(t,        y,              p)
    k2 = odes(t + dt/2, y + dt/2 * k1, p)
    k3 = odes(t + dt/2, y + dt/2 * k2, p)
    k4 = odes(t + dt,   y + dt   * k3, p)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ── Full simulation ───────────────────────────────────────────────

def run_simulation(params: dict) -> dict:
    """
    Integrates the ODE system from t=0 to T_END with step DT.

    Returns a dict of plain Python lists suitable for JSON serialisation:
      t, S, RA, RB, RAB, N, T
    """
    p = {
        "r":          float(params.get("r",           0.3)),
        "mu":         float(params.get("mu",          1e-6)),
        "kill_eff":   float(params.get("kill_eff",    0.8)),
        "treat_start":float(params.get("treat_start", 0)),
        "schedule":   str(params.get("schedule",      "continuous")),
    }

    steps = int(T_END / DT)
    y = IC.copy()

    t_arr, S_arr, RA_arr, RB_arr, RAB_arr, N_arr, T_arr = (
        [], [], [], [], [], [], []
    )

    for i in range(steps + 1):
        t = round(i * DT, 6)

        # Clamp state
        y = np.maximum(y, 0.0)
        y[4] = min(y[4], 100.0)

        S, RA, RB, RAB, Tv = y
        N = S + RA + RB + RAB

        # Stability guard: cap at 1000×K
        if N > 1000 * K and N > 0:
            y[:4] *= (1000 * K) / N
            S, RA, RB, RAB = y[:4]
            N = S + RA + RB + RAB

        t_arr.append(round(t, 1))
        S_arr.append(float(S))
        RA_arr.append(float(RA))
        RB_arr.append(float(RB))
        RAB_arr.append(float(RAB))
        N_arr.append(float(N))
        T_arr.append(float(Tv))

        y = rk4_step(t, y, DT, p)

    return {
        "t":   t_arr,
        "S":   S_arr,
        "RA":  RA_arr,
        "RB":  RB_arr,
        "RAB": RAB_arr,
        "N":   N_arr,
        "T":   T_arr,
    }


# ── Post-simulation metrics ───────────────────────────────────────

def compute_metrics(results: dict, params: dict) -> dict:
    """
    Derives clinical summary statistics from a completed simulation.
    """
    N   = results["N"]
    T   = results["T"]
    t   = results["t"]
    ts  = float(params.get("treat_start", 0))

    peak_N  = max(N)
    final_N = N[-1]
    max_tox = max(T)

    # Nadir index
    nadir_idx = N.index(min(N))

    # Time to relapse: first crossing of 1e7 after nadir
    relapse_day = None
    for i in range(nadir_idx + 1, len(N)):
        if N[i] > 1e7 and N[i - 1] <= 1e7:
            relapse_day = t[i]
            break

    # Outcome classification
    if final_N < 1.0:
        outcome = "eradication"
    elif final_N > 1e7 or relapse_day is not None:
        outcome = "relapse"
    else:
        outcome = "controlled"

    # Therapeutic index over the treatment window
    treat_idx = next((i for i, ti in enumerate(t) if ti >= ts), 0)
    treat_N = N[treat_idx:] or N
    mean_N = sum(treat_N) / len(treat_N)
    norm_mean_N = mean_N / K
    ti_score = (1.0 / (norm_mean_N + 1e-9)) / (max_tox + 1.0)

    return {
        "peak_N":      peak_N,
        "final_N":     final_N,
        "max_tox":     round(max_tox, 2),
        "relapse_day": relapse_day,
        "outcome":     outcome,
        "ti":          round(ti_score, 4),
    }
