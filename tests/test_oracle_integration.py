"""Live round-trip against a running oracle (skipped if :8000 is down).

Verifies the SDK apply/jacobian return native, finite values for all 9 knobs.
Start the oracle first:
    docker run -d -p 8000:8000 -e TESSERACT_RUNTIME_SERVE_HOST=0.0.0.0 \\
        --name nff_oracle_test nff-sofa-oracle:latest serve
"""
import pathlib
import numpy as np
import pytest
import yaml
from nff.sofa import oracle_payload as op
from nff.sofa.hinge_optimizer import _initial_params, _phys

REPO = pathlib.Path(__file__).resolve().parents[1]
CFG = REPO / "data/configs/sofa/hinge_opt_2face.yaml"
URL = "http://localhost:8000"


def _oracle_up() -> bool:
    try:
        import requests
        return requests.get(URL + "/health", timeout=2).ok
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _oracle_up(),
                                reason="oracle not serving on :8000")


def _payload():
    cfg = yaml.safe_load(open(CFG))
    cs = op.build_physical_cs(cfg)
    p = op.build_payload(cs, _phys(_initial_params(cfg, cs)), cfg, [0], [1])
    p.update(n_steps=20, skip_secondary_modes=True, mesh_refine=1.0)
    return p


def test_apply_returns_finite_native_scalars():
    from tesseract_core import Tesseract
    fwd = Tesseract.from_url(URL).apply(_payload())
    assert np.isfinite(float(fwd["smooth_principal_strain"]))
    assert float(fwd["max_von_mises_stress"]) > 0.0


def test_jacobian_returns_nine_finite_gradients():
    from tesseract_core import Tesseract
    jac = Tesseract.from_url(URL).jacobian(
        _payload(), jac_inputs=op.PARAM_NAMES,
        jac_outputs=["smooth_principal_strain"])
    g = [float(jac["smooth_principal_strain"][k]) for k in op.PARAM_NAMES]
    assert len(g) == 9 and all(np.isfinite(g))
