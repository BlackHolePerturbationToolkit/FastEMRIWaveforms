"""Batched (numSys) trajectory pieces on CPU: the reference for the GPU port (Task B1).

* ``brentq_array``: vectorised mask-driven Brent == scipy.optimize.brentq per element.
* ``tricubic_eval_batch``: numba tricubic from multispline's own coefficients ==
  ``TricubicSpline.__call__`` (the CPU mirror of the CUDA evaluator of Task B2).
* ``KerrEccEqFlux.evaluate_rhs_batch``: (6, S) states, per-system spin -> (6, S) rates +
  per-system status, == the scalar ``ODEBase.__call__`` column by column; out-of-bounds
  systems get a status code and NaN rates instead of an exception.
"""
import unittest

import numpy as np


class BrentqArrayTest(unittest.TestCase):
    def test_matches_scipy_on_a_batch(self):
        from scipy.optimize import brentq

        from few.utils.utility import brentq_array

        c = np.linspace(0.5, 3.0, 25)
        got = brentq_array(lambda x, args: x ** 2 - args, np.zeros_like(c), np.full_like(c, 4.0), c, 1e-14)
        want = np.array([brentq(lambda x: x ** 2 - ci, 0.0, 4.0, xtol=1e-14) for ci in c])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    def test_per_element_args_and_brackets(self):
        from few.utils.utility import brentq_array

        a = np.array([0.0, -2.0, 1.0])
        b = np.array([2.0, 0.0, 5.0])
        k = np.array([1.0, -1.5, 3.3])
        got = brentq_array(lambda x, args: np.sin(x - args), a, b, k, 1e-14)
        np.testing.assert_allclose(got, k, atol=1e-12)

    def test_raises_without_sign_change(self):
        from few.utils.utility import brentq_array

        with self.assertRaises(ValueError):
            brentq_array(lambda x, a: x + 1, np.zeros(2), np.ones(2), None, 1e-12)


class TricubicBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from few.trajectory.ode import KerrEccEqFlux

        cls.ode = KerrEccEqFlux(force_backend="cpu")

    def test_matches_multispline_all_four(self):
        from few.trajectory.ode.flux import tricubic_eval_batch, tricubic_grid

        rng = np.random.default_rng(3)
        pts = rng.uniform(0.0, 1.0, (3, 2000))
        for name in ("pdot_interp_A", "edot_interp_A", "pdot_interp_B", "edot_interp_B"):
            sp = getattr(self.ode, name)
            got = tricubic_eval_batch(tricubic_grid(sp), pts[0], pts[1], pts[2])
            want = np.array([sp(*p) for p in pts.T])
            np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13 * np.abs(want).max(), err_msg=name)


class BatchedRhsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from few.trajectory.ode import KerrEccEqFlux

        cls.ode = KerrEccEqFlux(force_backend="cpu")

    def _scalar(self, y, a):
        self.ode.add_fixed_parameters(1e6, 1e2, a)
        return np.asarray(self.ode(np.asarray(y, float)), float)

    def test_batched_matches_scalar_columns_mixed_spins(self):
        from few.utils.geodesic import get_separatrix

        rng = np.random.default_rng(1)
        S = 24
        a = rng.uniform(0.05, 0.95, S)
        e = rng.uniform(0.0, 0.6, S)
        p = np.array([get_separatrix(ai, ei, 1.0) for ai, ei in zip(a, e)]) + rng.uniform(0.3, 12.0, S)
        y = np.stack([p, e, np.ones(S), rng.uniform(0, 6, S), np.zeros(S), rng.uniform(0, 6, S)])
        ydot, status = self.ode.evaluate_rhs_batch(y, a)
        self.assertTrue(np.all(status == 0), status)
        for j in range(S):
            np.testing.assert_allclose(ydot[:, j], self._scalar(y[:, j], a[j]), rtol=1e-12, atol=0,
                                       err_msg=f"system {j}")

    def test_out_of_bounds_systems_flagged_others_exact(self):
        a = np.array([0.9, 0.9, 0.9])
        y = np.array([[10.0, 2.0, 10.0], [0.2, 0.2, -0.1], [1.0, 1.0, 1.0],
                      [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ydot, status = self.ode.evaluate_rhs_batch(y, a)
        self.assertEqual(list(status), [0, 2, 1])            # ok / inside separatrix / e < 0
        np.testing.assert_allclose(ydot[:, 0], self._scalar(y[:, 0], 0.9), rtol=1e-12)
        self.assertTrue(np.all(np.isnan(ydot[:, 1])) and np.all(np.isnan(ydot[:, 2])))

    def test_off_grid_system_flagged(self):
        a = np.array([0.9])
        y = np.array([[500.0], [0.2], [1.0], [0.0], [0.0], [0.0]])   # far beyond the flux grid
        ydot, status = self.ode.evaluate_rhs_batch(y, a)
        self.assertEqual(int(status[0]), 3)
        self.assertTrue(np.all(np.isnan(ydot[:, 0])))


class BatchedIntegrateTest(unittest.TestCase):
    """BatchedIntegrate: many systems per DOPR853 step, each following the scalar
    integrator's logic (controller, retries, separatrix-buffer stop, final point placed
    by the scalar finishing functions)."""

    @classmethod
    def setUpClass(cls):
        from few.trajectory.inspiral import EMRIInspiral
        from few.trajectory.integrate import BatchedIntegrate

        cls.scalar = EMRIInspiral(func="KerrEccEqFlux", force_backend="cpu")
        cls.batched = BatchedIntegrate(func="KerrEccEqFlux")

    def _scalar(self, m1, m2, a, p0, e0, T):
        # a FRESH scalar integrator: DOPR853 keeps its controller state (errOldTemp,
        # previousRejectTemp) across trajectories, so a reused one depends on its history
        from few.trajectory.inspiral import EMRIInspiral

        sc = EMRIInspiral(func="KerrEccEqFlux", force_backend="cpu")
        return np.array(sc(m1, m2, a, p0, e0, 1.0, T=T, dt=10.0))   # (7, n): t p e x Pp Pt Pr

    # Knot-by-knot equality with the scalar path is NOT a valid requirement: DOPR853's error
    # estimate sits at roundoff (err ~1e-5 of abstol 1e-11, i.e. ~1e-16 absolute), so last-bit
    # differences between the batched and scalar RHS move the accepted step sizes by tens of
    # percent. Both integrate the same solution: compare the batched knots against the scalar
    # DENSE output evaluated at the same times, and the end time.
    def _scalar_dense(self, m1, m2, a, p0, e0, T, t_new, err):
        from few.trajectory.inspiral import EMRIInspiral

        sc = EMRIInspiral(func="KerrEccEqFlux", force_backend="cpu")
        full = np.array(sc(m1, m2, a, p0, e0, 1.0, T=T, dt=10.0, err=err))
        sc = EMRIInspiral(func="KerrEccEqFlux", force_backend="cpu")
        dense = np.array(sc(m1, m2, a, p0, e0, 1.0, T=T, dt=10.0, err=err, new_t=t_new,
                            upsample=True, fix_t=True))
        return full, dense

    def _check(self, m1, m2, a, p0, e0, T, want_status):
        """Batched vs scalar (both err=1e-11) must agree within 10x the scalar path's own
        deviation from a tighter scalar reference (err=1e-12): the integrator's accuracy
        scale, not an arbitrary threshold."""
        out = self.batched.run(np.array([m1]), np.array([m2]), np.array([a]),
                               np.array([[p0], [e0], [1.0], [0.0], [0.0], [0.0]]), T=T, dt=10.0)
        traj = out.trajectories[0]
        self.assertEqual(int(out.status[0]), want_status)
        t_in = traj[:-1, 0]
        full, dense = self._scalar_dense(m1, m2, a, p0, e0, T, t_in, 1e-11)
        full_ref, dense_ref = self._scalar_dense(m1, m2, a, p0, e0, T, t_in, 1e-12)
        n = min(dense.shape[1], dense_ref.shape[1], traj.shape[0] - 1)
        for rows, name in ((slice(1, 3), "p,e"), (slice(4, 7), "phases")):
            spread = np.max(np.abs(dense[rows, :n] - dense_ref[rows, :n]))
            dev = np.max(np.abs(traj[:n, rows].T - dense[rows, :n]))
            print(f"\n[{name}] batched-vs-scalar {dev:.2e}, scalar(1e-11)-vs-scalar(1e-12) {spread:.2e}")
            self.assertLessEqual(dev, 10 * spread + 1e-13, name)
        t_spread = abs(full[0, -1] - full_ref[0, -1])
        self.assertLessEqual(abs(traj[-1, 0] - full[0, -1]), 10 * t_spread + 1e-6 * full[0, -1] * 1e-6, "end time")
        return traj

    def test_one_system_matches_scalar_solution(self):
        self._check(1e6, 1e2, 0.9, 12.0, 0.4, 0.3, want_status=-2)                  # reaches tmax

    def test_plunging_system_matches_scalar_solution(self):
        self._check(1e6, 1e2, 0.9, 7.0, 0.4, 0.2, want_status=-1)                   # separatrix buffer

    def test_systems_are_independent(self):
        y0 = np.array([[12.0, 7.0, 10.0], [0.4, 0.4, 0.2], [1.0, 1.0, 1.0],
                       [0.0, 0.0, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
        m1, m2, a = np.array([1e6, 1e6, 5e5]), np.array([1e2, 1e2, 30.0]), np.array([0.9, 0.9, 0.5])
        both = self.batched.run(m1, m2, a, y0, T=0.2, dt=10.0)
        for j in range(3):
            solo = self.batched.run(m1[j:j + 1], m2[j:j + 1], a[j:j + 1], y0[:, j:j + 1], T=0.2, dt=10.0)
            np.testing.assert_array_equal(both.trajectories[j], solo.trajectories[0])
            self.assertEqual(int(both.status[j]), int(solo.status[0]))


if __name__ == "__main__":
    unittest.main()
