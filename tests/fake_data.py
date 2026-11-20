"""
tests/fake_data.py -- write small synthetic SEG-Y shot gathers with the file names
expected by config.py, for tests that must not depend on the real data.

Each file holds n_shots shot gathers of linear and hyperbolic events plus
noise, recorded on the streamer of the corresponding line (25 m spacing,
offsets from the near offset to the streamer length), 2 ms, 4.5 s.
"""
from pathlib import Path

import numpy as np


def write_fake_line(path, streamer_m, n_shots=20, near=150.0, dx=25.0,
                    dt_ms=2.0, t_s=4.5, seed=0):
    import segyio
    rng = np.random.default_rng(seed)
    offsets = np.arange(near, streamer_m + near + 1e-6, dx)
    n_tr, n_s = len(offsets), int(round(t_s * 1000 / dt_ms))
    t = np.arange(n_s) * dt_ms / 1000
    spec = segyio.spec()
    spec.format = 5
    spec.samples = list(range(n_s))
    spec.tracecount = n_shots * n_tr
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = int(dt_ms * 1000)
        f.bin[segyio.BinField.Samples] = n_s
        k = 0
        for s in range(n_shots):
            t0 = rng.uniform(0.3, 2.0, size=4)
            v = rng.uniform(1500, 3500, size=4)
            for x in offsets:
                tr = np.zeros(n_s, np.float32)
                for a, b in zip(t0, v):
                    tt = np.sqrt(a ** 2 + (x / b) ** 2)
                    tr += np.exp(-((t - tt) / 0.01) ** 2).astype(np.float32)
                tr += 0.05 * rng.standard_normal(n_s).astype(np.float32)
                f.trace[k] = tr
                f.header[k] = {segyio.TraceField.FieldRecord: 100 + s,
                               segyio.TraceField.offset: -int(x),
                               segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(dt_ms * 1000)}
                k += 1


def write_fake_training_pool(folder, n_shots=20):
    """Six lines with the names and streamer lengths of config.py."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import get_config
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    dc = get_config().data
    for i, (p, L) in enumerate(zip(dc.dataset_paths, dc.streamer_lengths)):
        write_fake_line(folder / Path(p).name, L, n_shots=n_shots, seed=i)
    return folder


if __name__ == "__main__":
    import sys
    print(write_fake_training_pool(sys.argv[1] if len(sys.argv) > 1 else "fake_raw"))
