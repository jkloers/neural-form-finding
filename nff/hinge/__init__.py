"""nff.hinge — the hinge physics of the closed sheet, swappable by one config key.

How the swap works:

    config  hinge_model.type: rom | surrogate
       │
       ▼  build_hinge_model(config, static_features, state, params)       nff/hinge/model.py
       │     rom        -> nff/hinge/rom.py        no learned energy (+ optional kinematic probe)
       │     surrogate  -> nff/hinge/surrogate.py  CalculiX-trained net (nff/models/hinge_surrogate.py)
       ▼
    HingeModel(bond_energy_fn, probe_fn, geometry_fn, damage_fn, w_lig_logit0)
       │
       ▼  forward_pipeline(..., bond_energy_fn=, hinge_geometry=geometry_fn(params))
    build_potential_energy(bond_energy_fn)                                 nff/stages/physics/energy.py
          bond_energy_fn is None  ->  ligament_energy_linearized   (springs k_stretch, k_shear, k_rot)
          otherwise               ->  the learned per-hinge energy W(a, s, theta; w_lig, alpha)

Both models read hinge kinematics in the same per-hinge cut frame (``nff/hinge/frame.py``).
"""

from nff.hinge.model import HingeModel, build_hinge_model

__all__ = ["HingeModel", "build_hinge_model"]
