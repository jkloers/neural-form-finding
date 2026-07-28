"""Physical tensile-calibration analysis (Instron 34SC-5 + Bluehill + video extensometer).

Turns raw experiments into material-model inputs for the CalculiX RVE / PET material
class. Standalone from the JAX pipeline — only needs numpy (+ ffmpeg for ``ladder``).
See ``docs/physical_calibration_series1_tensile_protocol.md`` and
``docs/physical_calibration_video_extensometer_plan.md``.

Modules:
    bluehill_io          parse Bluehill results+raw CSV -> per-specimen curves
    summary_io           parse the operator's manual summary sheet (source of truth)
    stress_strain        stress-strain curves, E / yield / draw extraction, true-stress
    compliance           load-train compliance C [mm/N]: crosshead travel that is not specimen
    ladder               ink-ladder video strain + grip-based video/Bluehill time sync
"""
from nff.calibration import (
    bluehill_io,
    compliance,
    ladder,
    stress_strain,
    summary_io,
)

__all__ = ["bluehill_io", "compliance", "ladder", "stress_strain", "summary_io"]
