# README
# TumorEvo — Drug Resistance Simulator

## Setup

```bash
pip install flask numpy
python app.py
```

Open http://localhost:5000

## File structure

```
tumor_evo/
├── app.py          — Flask routes and CSV export
├── solver.py       — ODE system, RK4 integrator, metrics
├── requirements.txt
└── static/
    ├── index.html  — HTML structure
    ├── style.css   — Styles (warm paper, hard grid)
    └── app.js      — Plotly charts, fetch calls, UI logic
```

## API

POST /api/simulate   { r, mu, kill_eff, treat_start }
POST /api/csv        { r, mu, kill_eff, treat_start, schedule }
GET  /api/health

## Parameter ranges that hit all three outcomes

- Eradication: r=0.10, mu=1e-9, kill_eff=100%, continuous
- Controlled:  r=0.30, mu=1e-6, kill_eff=80%, continuous
- Relapse:     r=0.50, mu=1e-4, kill_eff=60%, pulsed
