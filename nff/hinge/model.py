"""The hinge-energy switch: one config key picks the hinge physics Stage 2 integrates.

    hinge_model:
      type: rom | surrogate

``build_hinge_model`` returns the same five-field ``HingeModel`` either way, and every caller
threads it identically, so swapping the hinge physics never touches the pipeline.
"""

from typing import Any, Callable, NamedTuple, Optional


class HingeModel(NamedTuple):
    """Everything the pipeline needs from a hinge energy. Unpacks as a 5-tuple.

    Attributes:
        bond_energy_fn: per-hinge energy injected into ``build_potential_energy``. ``None`` means
            the linear-spring ROM (``ligament_energy_linearized``).
        probe_fn: per-step per-hinge readings for the design loss (``eta_a``, ``eta_s``, ``theta``;
            plus ``D`` and ``ood`` for the surrogate). ``None`` when no loss term reads one.
        geometry_fn: ``map_params -> HingeGeometry(w_lig, alpha, sec_dir)``, traced in the design.
        damage_fn: per-hinge damage reader (surrogate only; diagnostics and visuals).
        w_lig_logit0: initial logit when ``hinge_model.learn_w_lig`` makes ligament width a design
            variable, else ``None``.
    """
    bond_energy_fn: Optional[Callable]
    probe_fn: Optional[Callable]
    geometry_fn: Optional[Callable]
    damage_fn: Optional[Callable]
    w_lig_logit0: Optional[Any]

    @property
    def kind(self) -> str:
        """``'surrogate'`` when a learned bond energy is injected, else ``'rom'``."""
        return 'rom' if self.bond_energy_fn is None else 'surrogate'


def build_hinge_model(config, static_features, state, init_map_params) -> HingeModel:
    """Build the hinge model the config asks for (``hinge_model.type``, default ``rom``).

    Args:
        config: parsed ExperimentConfig.
        static_features: closed-chart static structure from ``init_closed_les_params``.
        state: the flat initial CentroidalState.
        init_map_params: the initial design (``{'z', 'bnd_logits'}``).

    Returns:
        HingeModel.
    """
    hm = getattr(config, 'hinge_model', None)
    if hm is not None and getattr(hm, 'type', 'rom') == 'surrogate':
        from nff.hinge.surrogate import build_surrogate_hinge
        return build_surrogate_hinge(config, static_features, state, init_map_params)
    from nff.hinge.rom import build_rom_hinge
    return build_rom_hinge(config, static_features, state)
