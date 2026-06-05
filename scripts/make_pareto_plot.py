"""make_pareto_plot.py — Pareto plot across Run 1 + Run 2."""
import csv, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

R1  = 'data/outputs/search_c001_20260604_164925/results.csv'
R2  = 'data/outputs/search_c001_run2_20260604_184226/results.csv'
OUT = 'data/outputs/pareto_run1_run2_combined.png'

def flt(x):
    try: v=float(x); return v if math.isfinite(v) else None
    except: return None

def load(path, run_label):
    pts = []
    for r in csv.DictReader(open(path)):
        if r['status'] != 'ok': continue
        chf = flt(r['final_chamfer'])
        vs2 = flt(r.get('void_stage2_est'))
        if chf is None or vs2 is None: continue
        pts.append({
            'chamfer':     chf,
            'void_s2':     vs2,
            'run':         run_label,
            'cov':         float(r.get('coverage_w', 1.0)),
            'hidden_dim':  int(r['hidden_dim']),
            'num_layers':  int(r['num_layers']),
            'inner_depth': int(r.get('inner_depth', 2)),
            'force':       float(r['force_value']),
            'trial':       int(r['trial_idx']),
            'vc_w':        float(r['void_closure']),
            'cd_w':        float(r['closure_delta']),
        })
    return pts

def pareto_front(pts):
    xs = [p['chamfer'] for p in pts]
    ys = [p['void_s2'] for p in pts]
    n  = len(pts)
    dominated = [False]*n
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if xs[j]<=xs[i] and ys[j]<=ys[i] and (xs[j]<xs[i] or ys[j]<ys[i]):
                dominated[i] = True; break
    return sorted([pts[i] for i in range(n) if not dominated[i]],
                  key=lambda p: p['chamfer'])

r1 = load(R1, 'Run 1')
r2 = load(R2, 'Run 2')
all_pts = r1 + r2
pf_all  = pareto_front(all_pts)
pf_r2   = pareto_front(r2)

fig, ax = plt.subplots(figsize=(13, 8))

# ── scatter ───────────────────────────────────────────────────────────────────
ax.scatter([p['chamfer'] for p in r1], [p['void_s2'] for p in r1],
           c='#90CAF9', s=16, alpha=0.35, linewidths=0, label='Run 1', zorder=2)
ax.scatter([p['chamfer'] for p in r2 if p['cov']==1.0],
           [p['void_s2'] for p in r2 if p['cov']==1.0],
           c='#FFCC80', s=18, alpha=0.5, linewidths=0,
           label='Run 2  coverage=1.0', zorder=3)
ax.scatter([p['chamfer'] for p in r2 if p['cov']==0.5],
           [p['void_s2'] for p in r2 if p['cov']==0.5],
           c='#E65100', s=18, alpha=0.55, linewidths=0,
           label='Run 2  coverage=0.5', zorder=3)

# ── Pareto fronts ─────────────────────────────────────────────────────────────
pf1 = pareto_front(r1)
pf1_x = [p['chamfer'] for p in pf1]
pf1_y = [p['void_s2'] for p in pf1]
ax.step(pf1_x, pf1_y, where='post', color='#1565C0', lw=1.4,
        alpha=0.7, linestyle='--', label='Pareto (Run 1)', zorder=4)

pf2_x = [p['chamfer'] for p in pf_r2]
pf2_y = [p['void_s2'] for p in pf_r2]
ax.step(pf2_x, pf2_y, where='post', color='#B71C1C', lw=2.0,
        alpha=0.95, label='Pareto (Run 2)', zorder=5)
ax.scatter(pf2_x, pf2_y, c='#B71C1C', s=60, zorder=6, linewidths=0)

# ── annotate Run 2 Pareto points ──────────────────────────────────────────────
offsets = [(8, 10), (8, -22), (-90, 10), (-90, -22), (8, 28), (8, -35)]
for k, p in enumerate(pf_r2):
    ox, oy = offsets[k % len(offsets)]
    cov_s = f"cov={p['cov']:.1f}"
    lbl = (f"t{p['trial']}  {cov_s}\n"
           f"h{p['hidden_dim']} L{p['num_layers']}  F={p['force']:.0f}\n"
           f"vc={p['vc_w']:.0f}  cd={p['cd_w']:.0f}\n"
           f"chf={p['chamfer']:.4f}  vs2={p['void_s2']:.3f}")
    ax.annotate(lbl, xy=(p['chamfer'], p['void_s2']),
                xytext=(ox, oy), textcoords='offset points',
                fontsize=5.5, color='#111',
                arrowprops=dict(arrowstyle='->', color='#888', lw=0.7),
                bbox=dict(boxstyle='round,pad=0.2', fc='#FFF9C4', alpha=0.9, lw=0.4),
                zorder=7)

# ── reference line at void_s2 = 0 ────────────────────────────────────────────
ax.axhline(0, color='#43A047', lw=1.2, linestyle=':', alpha=0.8, label='void_s2 = 0  (fully closed)')

ax.set_xlabel('Chamfer distance  (↓ better shape match)', fontsize=12)
ax.set_ylabel('Remaining void area after loading  (↓ more closed)', fontsize=12)
ax.set_title(
    'Pareto — Chamfer vs. Void Closure  (c001, Run 1 + Run 2 combined, 439 trials)',
    fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, framealpha=0.93)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_xlim(0, 0.07)
ax.set_ylim(-0.1, 3.5)

n_r1  = len(r1);  n_r2 = len(r2)
n_fc  = sum(1 for p in r2 if p['void_s2'] == 0.0)
fig.text(0.5, 0.005,
    f"Run 1: {n_r1} pts (all strategies)  |  Run 2: {n_r2} pts (strategy=both, inner_depth=1 fixed)  "
    f"|  Run 2 fully-closed (vs2=0): {n_fc}/{n_r2}",
    ha='center', fontsize=8, color='#555')

plt.tight_layout(rect=[0, 0.035, 1, 1])
plt.savefig(OUT, dpi=160, bbox_inches='tight')
print(f"Saved: {OUT}")

print(f"\nRun 2 Pareto front ({len(pf_r2)} points):")
for p in pf_r2:
    print(f"  t{p['trial']:>3}  vs2={p['void_s2']:.3f}  chf={p['chamfer']:.4f}  "
          f"cov={p['cov']}  h{p['hidden_dim']} L{p['num_layers']}  "
          f"F={p['force']:.0f}  vc={p['vc_w']:.0f}  cd={p['cd_w']:.0f}")
