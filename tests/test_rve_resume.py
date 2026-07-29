"""A 16-hour campaign must survive being stopped and relaunched."""
import json

import numpy as np

from nff.rve.dataset import resume_state


def _write(tmp_path, n_jobs, n_per):
    out = str(tmp_path / "ds")
    np.savez_compressed(out + ".npz", a=np.arange(n_jobs * n_per, dtype=float),
                        job_id=np.repeat(np.arange(n_jobs), n_per).astype(float))
    json.dump({"jobs": [{"job_id": i, "n_samples": n_per} for i in range(n_jobs)]},
              open(out + ".json", "w"))
    return out


def test_a_fresh_output_resumes_to_nothing(tmp_path):
    acc, meta = resume_state(str(tmp_path / "does_not_exist"))
    assert acc == {} and meta == []


def test_resume_reads_back_every_finished_job(tmp_path):
    out = _write(tmp_path, n_jobs=7, n_per=3)
    acc, meta = resume_state(out)
    assert len(meta) == 7
    assert [m["job_id"] for m in meta] == list(range(7))
    assert np.concatenate(acc["a"]).shape == (21,)


def test_resume_needs_both_files(tmp_path):
    """A half-written checkpoint must not be mistaken for a complete one."""
    out = _write(tmp_path, n_jobs=4, n_per=2)
    (tmp_path / "ds.json").unlink()
    assert resume_state(out) == ({}, [])
