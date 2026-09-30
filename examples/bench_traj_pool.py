"""Throughput of pooled EMRI trajectories vs a serial loop.

usage: python examples/bench_traj_pool.py [N_TRAJ] [WORKERS ...]
env:   T_YR (trajectory span, default 1.0), ERR (default 1e-11)

Draws N_TRAJ plausible KerrEccEqFlux sources (m1 5e5-2e6, q 1e-5-1e-4, a 0-0.95,
p0 9-14, e0 0-0.5), times a serial loop on one module, then the same batch on a
TrajectoryPool for each worker count (pool start-up timed separately: it is paid once
per run). Prints trajectories/s and the speedup over serial.
"""
import os
import sys
import time

import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.pool import TrajectoryPool, call_key, reset_stepper


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    workers = [int(w) for w in sys.argv[2:]] or [2]
    T = float(os.environ.get("T_YR", "1.0"))
    err = float(os.environ.get("ERR", "1e-11"))
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(n):
        m1 = 10 ** rng.uniform(np.log10(5e5), np.log10(2e6))
        m2 = m1 * 10 ** rng.uniform(-5, -4)
        rows.append((m1, m2, rng.uniform(0, 0.95), rng.uniform(9, 14), rng.uniform(0, 0.5), 1.0))
    kw = dict(T=T, dt=10.0, err=err)

    insp = EMRIInspiral(func="KerrEccEqFlux")
    reset_stepper(insp)
    insp(*rows[0], **kw)                                 # warm numba caches
    t0 = time.perf_counter()
    knots = 0
    for r in rows:
        reset_stepper(insp)
        knots += insp(*r, **kw)[0].size
    serial = time.perf_counter() - t0
    print(f"serial: {n} trajectories ({T} yr, {knots / n:.0f} knots avg) in {serial:.2f} s "
          f"= {n / serial:.2f} traj/s", flush=True)

    calls = [(call_key(r, kw), r, kw) for r in rows]
    for w in workers:
        t0 = time.perf_counter()
        pool = TrajectoryPool(n_workers=w)
        pool.run(calls[:w])                              # spawn + build + warm every worker
        start = time.perf_counter() - t0
        t0 = time.perf_counter()
        snaps = pool.run(calls)
        wall = time.perf_counter() - t0
        pool.close()
        nbytes = sum(sum(a.nbytes for a in s.output) + sum(v.nbytes for v in s.integrator_state.values()
                                                           if isinstance(v, np.ndarray)) for s in snaps)
        print(f"pool {w}: start-up {start:.1f} s, batch {wall:.2f} s = {n / wall:.2f} traj/s, "
              f"speedup {serial / wall:.2f}x, shipped {nbytes / n / 1e3:.0f} kB/traj", flush=True)


if __name__ == "__main__":
    main()
