"""Pluggable RVE materials + modeling hypotheses.

``coerce_material`` is the boundary adapter: legacy call sites pass ``material`` as a
plain params dict (``STEEL``), which is wrapped into a :class:`SteelJ2`. A
:class:`Material` instance passes through unchanged.
"""

from __future__ import annotations

from nff.rve.materials.base import Hypotheses, Material
from nff.rve.materials.paper import PAPER_80GSM, PaperOrthotropic
from nff.rve.materials.pet import PET, PETIsotropic
from nff.rve.materials.steel import STEEL, SteelJ2

_REGISTRY: dict[str, type[Material]] = {
    "steel": SteelJ2,
    "paper": PaperOrthotropic,
    "pet": PETIsotropic,
}


def get_material(name: str, **kwargs) -> Material:
    """Instantiate a registered material by name (e.g. ``get_material("steel")``)."""
    try:
        cls = _REGISTRY[name.lower()]
    except KeyError:
        raise KeyError(f"unknown material {name!r}; known: {sorted(_REGISTRY)}") from None
    return cls(**kwargs)


def coerce_material(material) -> Material:
    """Accept a :class:`Material`, a params dict (legacy steel), or a name -> Material."""
    if isinstance(material, Material):
        return material
    if isinstance(material, dict):
        return SteelJ2.from_dict(material)
    if isinstance(material, str):
        return get_material(material)
    raise TypeError(f"cannot coerce {type(material).__name__} to a Material")


__all__ = ["Hypotheses", "Material", "SteelJ2", "STEEL", "PaperOrthotropic", "PAPER_80GSM",
           "PETIsotropic", "PET", "get_material", "coerce_material"]
