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


if __name__ == "__main__":
    unittest.main()
