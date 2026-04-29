"""
spatial_solver.py — 2D stochastic cellular automaton for tumour visualisation.

Grid: 36 × 36 cells.
Cell states: 0=empty, 1=S (sensitive), 2=RA, 3=RB, 4=RAB (fully resistant)

Mutation rates are scaled for visual clarity on the grid scale — at μ=1e-6
the ODE model predicts virtually zero resistant clones on a 1296-cell grid
over 200 days (expected: ~0.07 mutations total). The spatial model scales μ
so that ~10–50 resistant cells emerge by mid-simulation. The ODE results
remain the authoritative quantitative model; this is a schematic representation.
"""

import numpy as np
from solver import compute_kill

GRID    = 36
EMPTY   = 0;  S = 1;  RA = 2;  RB = 3;  RAB = 4
FITNESS = np.array([0.0, 1.0, 0.95, 0.95, 0.90], dtype=np.float32)


def _scale_mu(mu: float, r: float) -> float:
    """
    Scale μ so resistant clones are visible on the 36×36 grid.
    Target: ~20 mutation events over 200 days at carrying capacity.
    K_sp = 36² = 1296; μ_vis = 20 / (r × K_sp × 200)
    We use 5× that target for clearly visible heterogeneity.
    """
    K_sp = float(GRID * GRID)
    mu_target = 5.0 * 20.0 / (r * K_sp * 200.0)
    return max(mu, mu_target)


def run_spatial(params: dict, n_snapshots: int = 41) -> list:
    """
    Simulates tumour growth on a 36×36 grid for 200 days.

    Returns a list of snapshot dicts:
      { day: int, grid: List[List[int]], counts: {1:S, 2:RA, 3:RB, 4:RAB} }
    """
    rng = np.random.default_rng(42)
    G   = GRID

    # ── Initial conditions: circular cluster of S cells at centre ─────
    grid = np.zeros((G, G), dtype=np.int8)
    cx = cy = G // 2
    for x in range(G):
        for y in range(G):
            if (x - cx) ** 2 + (y - cy) ** 2 <= 9 and rng.random() < 0.85:
                grid[x, y] = S

    p = {
        'r':           float(params.get('r',           0.3)),
        'mu':          float(params.get('mu',          1e-6)),
        'kill_eff':    float(params.get('kill_eff',    0.8)),
        'treat_start': float(params.get('treat_start', 0)),
        'schedule':    str(params.get('schedule',      'continuous')),
    }

    mu_vis = _scale_mu(p['mu'], p['r'])
    K_sp   = float(G * G)
    dt     = 1.0
    t_end  = 200

    snap_days = set(int(round(d)) for d in np.linspace(0, t_end, n_snapshots))
    snapshots = []

    for day in range(t_end + 1):

        # ── Snapshot ───────────────────────────────────────────────────
        if day in snap_days:
            counts = {
                str(S):   int(np.sum(grid == S)),
                str(RA):  int(np.sum(grid == RA)),
                str(RB):  int(np.sum(grid == RB)),
                str(RAB): int(np.sum(grid == RAB)),
            }
            snapshots.append({
                'day':    day,
                'grid':   grid.tolist(),
                'counts': counts,
            })

        n_total = int(np.sum(grid > 0))
        if n_total == 0:
            break

        lf  = max(0.0, 1.0 - n_total / K_sp)
        kA, kB, _ = compute_kill(float(day), p)

        # ── Vectorised kill ────────────────────────────────────────────
        kill_p = np.zeros((G, G), dtype=np.float32)
        kill_p[grid == S]  = float(np.clip((kA + kB) * dt, 0.0, 1.0))
        kill_p[grid == RA] = float(np.clip(kB * dt,        0.0, 1.0))
        kill_p[grid == RB] = float(np.clip(kA * dt,        0.0, 1.0))
        # RAB: kill_p stays 0

        die_mask = (rng.random((G, G), dtype=np.float32) < kill_p) & (grid > 0)
        grid[die_mask] = EMPTY

        if lf <= 0.0:
            continue

        # ── Division (only iterate cells that will actually divide) ────
        occ = np.argwhere(grid > 0)
        if len(occ) == 0:
            continue

        # Pre-draw all division probability checks at once
        ctypes       = grid[occ[:, 0], occ[:, 1]]
        div_probs    = FITNESS[ctypes] * p['r'] * lf * dt
        will_divide  = occ[rng.random(len(occ), dtype=np.float32) < div_probs]

        for xi, yi in will_divide:
            ctype = int(grid[xi, yi])
            if ctype == EMPTY:
                continue   # cell may have been killed this tick

            # Collect empty 8-connected neighbours
            nb = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = xi + dx, yi + dy
                    if 0 <= nx < G and 0 <= ny < G and grid[nx, ny] == EMPTY:
                        nb.append((nx, ny))

            if not nb:
                continue

            nx, ny = nb[int(rng.integers(len(nb)))]

            # Daughter cell type — apply scaled mutation probability
            daughter = ctype
            rv = float(rng.random())
            if ctype == S:
                if   rv < mu_vis:       daughter = RA
                elif rv < 2.0 * mu_vis: daughter = RB
            elif ctype == RA:
                if rv < mu_vis:         daughter = RAB
            elif ctype == RB:
                if rv < mu_vis:         daughter = RAB

            grid[nx, ny] = daughter

    return snapshots
