"""Trajectory fan-out: pooled trajectories are served bit-identically to a waveform.

* a snapshot installed into ANOTHER inspiral module reproduces the output, the stored
  trajectory and the dense-output evaluators bit for bit;
* a pooled (spawned worker) trajectory equals a freshly built module's, bit for bit;
* a waveform-shaped caller served from the cache gives the uncached result, bit for bit,
  and every trajectory is a cache hit;
* capture survives ``except Exception`` in the caller; misses fall through; worker
  errors are recorded and the real call raises.
"""
import unittest

import numpy as np

YR = 0.25       # trajectory span [yr]: short, keeps the suite cheap
DT = 10.0
ROWS = [(1e6, 10.0, 0.9, 10.0, 0.3, 1.0), (1e6, 20.0, 0.5, 11.0, 0.2, 1.0),
        (5e5, 10.0, 0.7, 12.0, 0.4, 1.0), (1e6, 10.0, -0.6, 10.5, 0.1, 1.0)]


def _inspiral():
    from few.trajectory.inspiral import EMRIInspiral

    return EMRIInspiral(func="KerrEccEqFlux")


class FakeWave:
    """Waveform-shaped caller: transforms its inputs (like sanity_check_init), calls the
    inspiral with call-time kwargs, then reads dense-output state from the module."""

    def __init__(self):
        self.inspiral_generator = _inspiral()
        self.inspiral_kwargs = {"func": "KerrEccEqFlux", "err": 1e-11}

    def __call__(self, m1, m2, a, p0, e0, x0):
        if a < 0:                                   # FEW's retrograde convention
            a, x0 = -a, -x0
        out = self.inspiral_generator(m1, m2, a, p0, e0, x0, T=YR, dt=DT, **self.inspiral_kwargs)
        integ = self.inspiral_generator.inspiral_generator
        tq = np.linspace(0.0, float(out[0][-1]), 50)
        return (out, integ.eval_integrator_spline(tq), integ.eval_integrator_derivative_spline(tq, order=1),
                np.array(self.inspiral_generator.integrator_spline_phase_coeff),
                np.array(self.inspiral_generator.integrator_spline_t))


def _fresh(fn, *row):
    """Uncached reference: reset the stepper so the result has no trajectory history."""
    from few.trajectory.pool import reset_stepper

    reset_stepper(fn.inspiral_generator)
    return fn(*row)


def _assert_same(test, a, b):
    if isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            _assert_same(test, x, y)
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class SnapshotTest(unittest.TestCase):
    def test_install_into_another_module_is_bitwise(self):
        from few.trajectory.pool import install_snapshot, reset_stepper, snapshot_inspiral

        src, dst = _inspiral(), _inspiral()
        reset_stepper(src)
        out = src(1e6, 10.0, 0.9, 10.0, 0.3, 1.0, T=YR, dt=DT, err=1e-11)
        snap = snapshot_inspiral(src, out)
        dst(1e6, 30.0, 0.2, 14.0, 0.1, 1.0, T=YR, dt=DT, err=1e-11)      # different prior state
        got = install_snapshot(dst, snap)
        _assert_same(self, got, out)
        tq = np.linspace(0.0, float(out[0][-1]), 40)
        a, b = src.inspiral_generator, dst.inspiral_generator
        _assert_same(self, b.eval_integrator_spline(tq), a.eval_integrator_spline(tq))
        for order in (1, 2):
            _assert_same(self, b.eval_integrator_derivative_spline(tq, order=order),
                         a.eval_integrator_derivative_spline(tq, order=order))
        _assert_same(self, b.trajectory, a.trajectory)
        _assert_same(self, dst.integrator_spline_phase_coeff, src.integrator_spline_phase_coeff)


class WorkerResetTest(unittest.TestCase):
    def test_worker_clears_a_carried_rejected_step(self):
        """DOPR853 keeps previousRejectTemp across trajectories; a trajectory that ended on
        a rejected step changes the NEXT one's knots. The worker must reset it."""
        from few.trajectory import pool as P

        init = {"func": "KerrEccEqFlux"}
        insp = _inspiral()
        call = (ROWS[0], dict(T=YR, dt=DT, err=1e-11))
        P.reset_stepper(insp)
        want = insp(*call[0], **call[1])
        insp.inspiral_generator.dopr.previousRejectTemp = np.array([True])   # poisoned history
        insp.inspiral_generator.dopr.errOldTemp = np.array([1e-4])
        P._WORKER_INSPIRALS[P.call_key((), init)] = insp
        try:
            snap = P._worker_run(init, [(P.call_key(*call), call[0], call[1])])[0]
        finally:
            P._WORKER_INSPIRALS.clear()
        _assert_same(self, snap.output, want)


class PoolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from few.trajectory.pool import TrajectoryPool

        cls.pool = TrajectoryPool(n_workers=2, inspiral_init_kwargs={"func": "KerrEccEqFlux"})

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()

    def test_pooled_trajectory_equals_fresh_module(self):
        from few.trajectory.pool import call_key, reset_stepper

        calls = [((m1, m2, a, p0, e0, x0), dict(T=YR, dt=DT, err=1e-11)) for m1, m2, a, p0, e0, x0 in ROWS[:3]]
        snaps = self.pool.run([(call_key(*c), c[0], c[1]) for c in calls])
        ref = _inspiral()
        for c, s in zip(calls, snaps):
            reset_stepper(ref)
            want = ref(*c[0], **c[1])
            _assert_same(self, s.output, want)
            _assert_same(self, s.integrator_state["dopr_spline_output"],
                         ref.inspiral_generator.integrator_spline_coeff)

    def test_waveform_served_from_cache_is_bitwise_and_all_hits(self):
        from few.trajectory.pool import TrajectoryCache

        wave = FakeWave()
        want = [_fresh(wave, *r) for r in ROWS]
        cache = TrajectoryCache.install(wave)
        info = cache.precompute(wave, ROWS, self.pool)
        self.assertEqual(info, dict(rows=4, captured=4, computed=4, errors=0))
        got = [wave(*r) for r in ROWS[::-1]][::-1]              # any order
        self.assertEqual((cache.hits, cache.misses), (4, 0))
        for g, w in zip(got, want):
            _assert_same(self, g, w)
        again = cache.precompute(wave, ROWS, self.pool)          # already stored: nothing new
        self.assertEqual(again["computed"], 0)
        TrajectoryCache.uninstall(wave)
        self.assertNotIsInstance(wave.inspiral_generator, TrajectoryCache)

    def test_worker_error_is_recorded_and_real_call_raises(self):
        from few.trajectory.pool import TrajectoryCache

        wave = FakeWave()
        cache = TrajectoryCache.install(wave)
        bad = (1e6, 10.0, 0.9, 1.5, 0.3, 1.0)                    # inside the separatrix
        info = cache.precompute(wave, [bad], self.pool)
        # the waveform-level validity check may fire at capture (no call) or in the worker
        self.assertEqual(info["computed"], info["errors"])
        with self.assertRaises(Exception):
            wave(*bad)


class CaptureTest(unittest.TestCase):
    def test_capture_survives_except_exception_and_miss_falls_through(self):
        from few.trajectory.pool import TrajectoryCache

        wave = FakeWave()
        cache = TrajectoryCache.install(wave)

        continued = []

        def swallowing(*row):
            try:
                wave(*row)
            except Exception:
                pass
            continued.append(True)                               # must never run in capture

        calls = cache.capture(swallowing, [ROWS[3]])
        self.assertEqual(continued, [])
        self.assertIsNotNone(calls[0])
        args, kwargs = calls[0]
        self.assertEqual(args[2], 0.6)                           # the TRANSFORMED spin was captured
        self.assertEqual(args[5], -1.0)
        self.assertEqual(kwargs["T"], YR)
        want = _fresh(FakeWave(), *ROWS[3])
        cache.inspiral_generator.dopr.__dict__.pop("errOldTemp", None)
        cache.inspiral_generator.dopr.__dict__.pop("previousRejectTemp", None)
        got = wave(*ROWS[3])                                     # nothing stored: a miss
        self.assertEqual((cache.hits, cache.misses), (0, 1))
        _assert_same(self, got, want)


if __name__ == "__main__":
    unittest.main()
