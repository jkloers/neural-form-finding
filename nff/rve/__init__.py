"""Single-hinge RVE (representative volume element) for the condensation surrogate.

Offline data-generation side: from a hinge descriptor, build the standalone RVE
geometry (``geometry.py``), mesh it, and (later) run the elasto-plastic FEM. This
package is independent of the JAX pipeline in ``nff/stages`` — it only consumes the
descriptor parameters.
"""
