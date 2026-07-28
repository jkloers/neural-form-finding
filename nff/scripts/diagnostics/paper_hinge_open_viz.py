"""Compare the three tear criteria on the real 60 deg paper-hinge solve.

surface-uniaxial (old)  ->  surface + triaxiality  ->  membrane + triaxiality (shipped).
Left: deformed opening coloured by membrane principal strain. Right: the three D(theta) curves.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))
import sys
sys.path.insert(0, "/Users/julienkloers/Documents/Code2/princeton/neural-form-finding")
from nff.rve.damage import max_principal_strain
# the paper tear criterion lives with the (parked, elastic) paper material -- nff.rve.damage
# carries only the one plastic-dissipation measure, which is undefined for an elastic material
from nff.rve.materials.paper import membrane_tear_from_frame

d = np.load("paper_hinge_result60.npz", allow_pickle=True)
xyz = np.asarray(d["xyz"], float)
disp = [np.asarray(x, float)[:, :3] for x in d["disp"]]
tostrain = [np.asarray(x, float) for x in d["tostrain"]]
stress = [np.asarray(x, float) for x in d["stress"]]
W_LIG = float(d["w_lig"]); EPS0 = 0.03; K = 1.5
PIVOT = np.array([0.0, -W_LIG])

rxy = np.hypot(xyz[:, 0], xyz[:, 1])
handleB = (rxy > 0.95 * 24.0) & (xyz[:, 0] > 0.5)
def fold_angle(u):
    r0 = xyz[handleB, :2] - PIVOT; r1 = (xyz[handleB, :2] + u[handleB, :2]) - PIVOT
    return np.degrees(np.abs(np.arctan2(r0[:, 0]*r1[:, 1]-r0[:, 1]*r1[:, 0],
                                        r0[:, 0]*r1[:, 0]+r0[:, 1]*r1[:, 1])).mean())

theta, D_surf_uni, D_surf_tri, D_mem, memstrain = [], [], [], [], []
for u, E, S in zip(disp, tostrain, stress):
    if not len(E) or np.percentile(max_principal_strain(E), 99) < 1e-6:
        continue
    fr = {"TOSTRAIN": E, "STRESS": S if len(S) == len(E) else None}
    theta.append(fold_angle(u))
    D_surf_uni.append(np.percentile(max_principal_strain(E), 99) / EPS0)
    D_surf_tri.append(membrane_tear_from_frame(fr, None, eps_tear0=EPS0, k=K))  # coords=None -> surface
    D_mem.append(membrane_tear_from_frame(fr, xyz, eps_tear0=EPS0, k=K))
    memstrain.append(None)  # placeholder; nodal membrane strain drawn below per picked frame
theta = np.array(theta)
o = np.argsort(theta)
theta = theta[o]
D_surf_uni = np.array(D_surf_uni)[o]; D_surf_tri = np.array(D_surf_tri)[o]; D_mem = np.array(D_mem)[o]
disp = [disp[i] for i in o]; tostrain = [tostrain[i] for i in o]
print(f"frames={len(theta)}  max fold={theta.max():.1f} deg")
for q in (20, 40, 55, 60):
    if theta.max() >= q - 3:
        i = int(np.argmin(np.abs(theta - q)))
        print(f"  theta~{q:2d} ({theta[i]:5.1f}): D_surf_uni={D_surf_uni[i]:.2f}  "
              f"D_surf_tri={D_surf_tri[i]:.2f}  D_membrane={D_mem[i]:.2f}")

# ---- figure ----
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig = plt.figure(figsize=(13, 5.4))
ax = fig.add_subplot(1, 2, 1, projection="3d")
axs = fig.add_subplot(1, 2, 2)
cmap = plt.cm.RdYlGn_r; norm = plt.Normalize(0.0, EPS0)

def membrane_nodal(E):
    key = np.round(xyz[:, :2] / 0.05).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True); n = int(inv.max())+1
    cnt = np.bincount(inv, minlength=n)
    Ecol = np.stack([np.bincount(inv, weights=E[:, c], minlength=n)/cnt for c in range(6)], axis=1)
    return max_principal_strain(Ecol)[inv]     # broadcast column value back to nodes

pick = np.unique(np.linspace(0, len(theta)-1, 5).astype(int))
for j, fi in enumerate(pick):
    P = xyz + disp[fi]; e = membrane_nodal(tostrain[fi]); sh = j*1.5*24.0
    ax.scatter(P[:, 0]+sh, P[:, 1], P[:, 2], c=e, cmap=cmap, norm=norm, s=1.6,
               alpha=0.75, depthshade=False, marker=".")
    ax.text(sh, xyz[:, 1].min()-6, 0, f"{theta[fi]:.0f}°", color="#333", fontsize=11,
            ha="center", fontweight="bold")
ax.set_box_aspect((4.2, 1.0, 0.5)); ax.view_init(elev=24, azim=-74); ax.set_axis_off()
ax.set_title("1 cm paper hinge opening — real FEA\n(colour = membrane strain)", fontsize=10.5)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.018, pad=-0.02)
cb.set_label("membrane principal strain", fontsize=9); cb.ax.tick_params(labelsize=8)

axs.plot(theta, D_surf_uni, color="#D62828", lw=2.3, label="surface, uniaxial (old)")
axs.plot(theta, D_surf_tri, color="#F58025", lw=2.3, label="surface + triaxiality")
axs.plot(theta, D_mem, color="#2A9D8F", lw=2.8, label="membrane + triaxiality (shipped)")
axs.axhline(1.0, color="#6C757D", ls="--", lw=1.5)
axs.annotate("D = 1 (tear, eps_tear0 = 3%)", (theta.min(), 1.05), color="#6C757D", fontsize=9, va="bottom")
axs.axvspan(50, 60, color="#2A9D8F", alpha=0.10)
axs.annotate("observed\n50–60° intact", (55, 0.4), color="#1D6F63", fontsize=9, ha="center")
axs.set_xlabel("fold angle  [deg]"); axs.set_ylabel("tear margin  D")
axs.set_title("three tear criteria vs fold angle", fontsize=11, pad=8)
axs.spines[["top", "right"]].set_visible(False); axs.grid(alpha=0.15)
axs.legend(frameon=False, fontsize=8.5, loc="upper left")
axs.set_xlim(0, max(62, theta.max()+2)); axs.set_ylim(0, max(2.5, np.nanmax(D_surf_uni)*1.05))
fig.tight_layout()
fig.savefig("paper_hinge_60.png", dpi=150, bbox_inches="tight")
print("saved paper_hinge_60.png")
