"""Single-hinge RVE (representative volume element) for the condensation surrogate.

Offline data-generation side: from a hinge descriptor, build the standalone RVE
geometry (``geometry.py``) and deploy it with CalculiX (``ccx_solver.py``) to read the
stored energy, out-of-plane displacement, and strain. The elasto-plastic + buckling
physics is entirely CalculiX's — we write no constitutive models. This package is
independent of the JAX pipeline in ``nff/stages``; it only consumes descriptor parameters.
"""
