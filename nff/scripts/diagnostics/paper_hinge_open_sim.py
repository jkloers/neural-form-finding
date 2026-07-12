"""Real CalculiX simulation of a 1 cm paper hinge opening to ~60 deg.

Single-hinge RVE: w_lig = 10 mm (1 cm ligament), paper thickness 0.10 mm, PaperOrthotropic
(80 gsm copy paper, orthotropic-elastic + tensile-tear, native no-plastic default). We deploy
the pure-rotation spine (a = s = 0) to 60 deg and read energy, moment, out-of-plane buckle, and
the tensile-tear margin D = p99(max principal strain)/eps_tear along the fold.
"""
import os
import numpy as np

os.chdir(os.path.dirname(__file__))
from nff.rve.ccx_solver import deploy
from nff.rve.geometry import RVEParams
from nff.rve.materials import PaperOrthotropic, PAPER_80GSM, Hypotheses

W_LIG = 10.0      # 1 cm ligament
THK = 0.10        # 80 gsm paper thickness [mm]
EPS_TEAR = PAPER_80GSM["eps_tear0"]

p = RVEParams(w_lig=W_LIG, w_c=0.2, alpha_deg=90.0, rho=0.16 * W_LIG,
              thickness=THK, r_win=24.0)

print("Running CalculiX: 1 cm paper hinge, t=0.10 mm, opening to 60 deg ...", flush=True)
# push to 60 deg: bigger pre-crest (~2.5x thickness) to smooth the buckle bifurcation into a
# monotonic bend, 3 elems through thickness, and a much smaller min increment so the solver can
# walk through the deepening fold instead of hitting the 1e-3 floor.
res = deploy(p, material=PaperOrthotropic(), hyp=Hypotheses(),
             angle_deg=60.0, n_steps=30, n_through=3, field_every=2,
             lc_min=0.5, lc_max=4.0, imp_amp=0.25, min_inc=5e-5,
             ncpus=4, eps_f=None, timeout=3000,
             workdir=os.path.join(os.getcwd(), "ccx_paper_job60"))

theta = np.asarray(res["theta_deg"], float)
W = np.asarray(res["W"], float)
M = np.asarray(res["M_theta"], float)
uz = np.asarray(res["uz_max"], float)
strain = np.asarray(res["strain_max"], float)          # max principal strain (raw max)
Draw = np.asarray(res["damage_p99"], float)            # = p99(principal strain)/eps_tear for paper

print(f"ok={res['ok']}  n_frames(dat)={len(theta)}  n_elems={res['n_elems']}  n_nodes={res['n_nodes']}")
print(f"frames(.frd)={len(res['frames'])}  max theta reached = {theta.max() if len(theta) else 0:.1f} deg")
print("theta[deg]:", np.round(theta, 1))
print("W[N.mm]   :", np.round(W, 4))
print("M[N.mm]   :", np.round(M, 3))
print("uz_max[mm]:", np.round(uz, 3))
print("eps_max[%]:", np.round(strain * 100, 3))
print("D=p99/eps_tear:", np.round(Draw, 3))
for d in (50, 55, 60):
    i = int(np.argmin(np.abs(theta - d))) if len(theta) else -1
    if i >= 0:
        print(f"  ~{d} deg (theta={theta[i]:.1f}): eps_max={strain[i]*100:.2f}%  D={Draw[i]:.2f}")

# save everything the viz needs (deformed frames = ref xyz + DISP + strain + stress)
frames = res["frames"]
np.savez_compressed(
    "paper_hinge_result60.npz",
    theta=theta, W=W, M=M, uz=uz, strain=strain, D=Draw,
    xyz=res["xyz"], w_lig=W_LIG, thk=THK,
    disp=np.array([np.asarray(f["DISP"], float)[:, :3] for f in frames], dtype=object),
    tostrain=np.array([f.get("TOSTRAIN", np.zeros((0, 6))) for f in frames], dtype=object),
    stress=np.array([f.get("STRESS", np.zeros((0, 6))) for f in frames], dtype=object),
)
print("saved paper_hinge_result.npz  (stdout tail below)")
print(res["stdout"][-600:])
