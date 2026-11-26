"""CLI for the training film: the cut pattern morphing beside the full loss.

Reads a finished (or partial) closed run and renders a 1080p H.264 clip. Decoupled from training on
purpose -- the run persists its design trajectory, so restyling the film costs a minute rather than
the hours the run itself took.

Run:
    conda run -n kgnn_mac python nff/scripts/figures/animate_training_film.py \
        --run-dir data/outputs/runs/run_<ts>_pet45_big_cP10_dP --duration 30 --fps 60
"""

import os
import argparse

# Before jax is imported anywhere below. The film replays the same collinearity solve the run used,
# and the Metal backend rejects the zero-sized tensors that solve produces; CPU is also what every
# closed run trains on, so replaying there is what reproduces the run's own geometry.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from nff.utils.figstyle import use_agg, apply_charter, paper_rc

use_agg()   # before the first pyplot import, or the backend is already fixed

from nff.closed.training_film import render_training_film, FILM_SCALE   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run folder holding config.yaml, design_traj.npz and loss_history.json")
    ap.add_argument("--out", default=None, help="output mp4 (default: <run-dir>/training_film.mp4)")
    ap.add_argument("--duration", type=float, default=30.0, help="clip length in seconds")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--dpi", type=int, default=150, help="150 gives exactly 1920x1080")
    ap.add_argument("--ease", default="smoothstep", choices=["smoothstep", "linear"],
                    help="smoothstep eases the sweep at both ends so the clip settles")
    ap.add_argument("--hold-start", type=int, default=None,
                    help="frames held on the untrained design (default: fps/3)")
    ap.add_argument("--hold-end", type=int, default=None,
                    help="frames held on the final design (default: fps)")
    ap.add_argument("--keep-frames", action="store_true",
                    help="keep the PNG sequence for \\animategraphics in the deck")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality; lower is better")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    apply_charter(**paper_rc(scale=FILM_SCALE))

    out = args.out or os.path.join(args.run_dir, "training_film.mp4")
    render_training_film(args.run_dir, out, duration=args.duration, fps=args.fps, dpi=args.dpi,
                         ease=args.ease, hold_start=args.hold_start, hold_end=args.hold_end,
                         keep_frames=args.keep_frames, crf=args.crf, title=args.title)


if __name__ == "__main__":
    main()
