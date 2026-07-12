"""Robust analysis + visual of the real CalculiX 1 cm paper-hinge opening.

Derives the fold angle geometrically per .frd frame (mean rotation of the outer +x handle
nodes about the pivot), so it doesn't depend on the .dat/.frd frame-count alignment. Colours
each deformed node by its own max principal (tensile) strain.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))
d = np.load("paper_hinge_result.npz", allow_pickle=True)
xyz = np.asarray(d["xyz"], float)
# saved as object arrays -> cast each frame to float (object dtype breaks numpy ufuncs)
disp = [np.asarray(x, float) for x in d["disp"]]
tostrain = [np.asarray(x, float) for x in d["tostrain"]]
EPS_TEAR = float(d["eps_tear"]); W_LIG = float(d["w_lig"]); THK = float(d["thk"])
PIVOT = np.array([0.0, -W_LIG])

# outer +x handle (rigid_B) reference nodes -> measure their rotation about the pivot
rxy = np.hypot(xyz[:, 0], xyz[:, 1])
handleB = (rxy > 0.95 * 24.0) & (xyz[:, 0] > 0.5)

def fold_angle(u):
    r0 = xyz[handleB, :2] - PIVOT
    r1 = (xyz[handleB, :2] + u[handleB, :2]) - PIVOT
    ang = np.arctan2(r0[:, 0] * r1[:, 1] - r0[:, 1] * r1[:, 0],
                     r0[:, 0] * r1[:, 0] + r0[:, 1] * r1[:, 1])
    return np.degrees(np.abs(ang).mean())

def principal_nodal(E):
    if E is None or len(E) == 0:
        return np.zeros(len(xyz))
    out = np.zeros(len(E))
    for i, (xx, yy, zz, xy, yz, zx) in enumerate(E):
        T = np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])
        out[i] = np.linalg.eigvalsh(T)[-1]
    return out

# tip stress-concentration zone vs ligament body: exclude a 3 mm disk around the cut tip
TIP = np.array([0.0, -W_LIG])
near_tip = np.hypot(xyz[:, 0] - TIP[0], xyz[:, 1] - TIP[1]) < 3.0
body = ~near_tip

theta, p99_full, p99_body = [], [], []
node_eps_by_frame, disp3 = [], []
for u, E in zip(disp, tostrain):
    e = principal_nodal(E)
    if not len(e) or np.percentile(e, 99) < 1e-6:      # skip degenerate/empty frames
        continue
    theta.append(fold_angle(u))
    node_eps_by_frame.append(e)
    disp3.append(np.asarray(u, float)[:, :3])           # some frames parsed 4 cols; keep ux,uy,uz
    p99_full.append(np.percentile(e, 99))
    p99_body.append(np.percentile(e[body], 99))
theta = np.array(theta); p99_full = np.array(p99_full); p99_body = np.array(p99_body)
order = np.argsort(theta)
theta, p99_full, p99_body = theta[order], p99_full[order], p99_body[order]
node_eps_by_frame = [node_eps_by_frame[i] for i in order]
disp3 = [disp3[i] for i in order]
D = p99_full / EPS_TEAR
D_body = p99_body / EPS_TEAR

print(f"nodes={len(xyz)}  usable frames={len(theta)}  max fold={theta.max():.1f} deg")
print(f"{'theta':>6} {'p99_full':>9} {'p99_body':>9}  (strain %, full incl. tip vs ligament body)")
for q in (10, 20, 30, 37):
    i = int(np.argmin(np.abs(theta - q)))
    print(f"{theta[i]:6.1f} {p99_full[i]*100:9.2f} {p99_body[i]*100:9.2f}")
cf = np.polyfit(theta[theta > 8], p99_full[theta > 8], 1)
cb = np.polyfit(theta[theta > 8], p99_body[theta > 8], 1)
print("--- extrapolated to the observed fold ---")
for q in (55, 60):
    print(f"  {q} deg:  full(tip) ~ {np.polyval(cf, q)*100:4.1f}%   body ~ {np.polyval(cb, q)*100:4.1f}%")

# ---------------- figure ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig = plt.figure(figsize=(13, 5.4))
ax = fig.add_subplot(1, 2, 1, projection="3d")
axs = fig.add_subplot(1, 2, 2)
cmap = plt.cm.RdYlGn_r
norm = plt.Normalize(0.0, EPS_TEAR)

pick = np.unique(np.linspace(0, len(theta) - 1, 5).astype(int))
dx = 1.5 * 24.0
for j, fi in enumerate(pick):
    P = xyz + disp3[fi]
    e = node_eps_by_frame[fi]
    sh = j * dx
    ax.scatter(P[:, 0] + sh, P[:, 1], P[:, 2], c=e, cmap=cmap, norm=norm,
               s=1.6, alpha=0.75, depthshade=False, marker=".")
    ax.text(sh, xyz[:, 1].min() - 6, 0, f"{theta[fi]:.0f}°", color="#333",
            fontsize=11, ha="center", fontweight="bold")
ax.set_box_aspect((4.2, 1.0, 0.5)); ax.view_init(elev=24, azim=-74); ax.set_axis_off()
ax.set_title("1 cm paper hinge opening — real FEA deformed shape\n"
             "(colour = tensile strain: green safe → red at eps_tear)", fontsize=10.5)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.018, pad=-0.02)
cb.set_label("principal strain", fontsize=9); cb.ax.tick_params(labelsize=8)

axs.plot(theta, D, color="#D62828", lw=2.6, zorder=3, label="p99 incl. cut tip (concentration)")
axs.plot(theta, D_body, color="#F58025", lw=2.6, zorder=3, label="p99 ligament body (material)")
axs.axhline(1.0, color="#6C757D", ls="--", lw=1.5)
axs.annotate("D = 1  (eps_tear = 1.5%)", (theta.min(), 1.05), color="#6C757D", fontsize=9, va="bottom")
axs.axvspan(50, 60, color="#2A9D8F", alpha=0.13)
axs.annotate("you observe\n50–60° intact", (55, 0.4), color="#1D6F63", fontsize=9.5, ha="center")
axs.set_xlabel("fold angle  [deg]"); axs.set_ylabel("tear margin  D = p99(strain) / eps_tear")
axs.set_title("does a 1 cm paper hinge tear before 60°?", fontsize=11, pad=8)
axs.spines[["top", "right"]].set_visible(False); axs.grid(alpha=0.15)
axs.legend(frameon=False, fontsize=8.5, loc="upper left")
axs.set_xlim(0, 62); axs.set_ylim(0, max(2.0, np.nanmax(D) * 1.1))
fig.tight_layout()
fig.savefig("paper_hinge_open.png", dpi=150, bbox_inches="tight")
print("saved paper_hinge_open.png")
