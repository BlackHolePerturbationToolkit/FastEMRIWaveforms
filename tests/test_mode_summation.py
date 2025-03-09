import unittest
import numpy as np

from few.summation.interpolatedmodesum import CubicSplineInterpolant, InterpolatedModeSum
from scipy.interpolate import CubicSpline
from few.summation.directmodesum import DirectModeSum
from few.utils.globals import get_logger, get_first_backend

few_logger = get_logger()

best_backend = get_first_backend(InterpolatedModeSum.supported_backends())
few_logger.warning(
    "Mode-summation test is running with backend '{}'".format(best_backend.name)
)

class CubicSplineTest(unittest.TestCase):
    def test_cubic_spline(self):
        # check that the cubic spline correctly interpolates test functions
        t_test = np.linspace(0, 100, 1000)
        f_test = np.vstack((np.cos(t_test), np.sin(t_test)))
        few_spl = CubicSplineInterpolant(t_test, f_test, force_backend=best_backend)
        scipy_spl = CubicSpline(t_test, f_test, axis=-1)

        # first check the knots are returned
        few_eval = few_spl(t_test)
        if best_backend.uses_gpu:
            few_eval = few_eval.get()
        np.testing.assert_allclose(few_eval, f_test, rtol=1e-10)
        
        # check that derivatives agree with scipy on the knots
        for deriv_order in range(1,4):
            few_eval = few_spl(t_test, deriv_order=deriv_order)
            scipy_eval = scipy_spl(t_test, nu=deriv_order)
            if best_backend.uses_gpu:
                few_eval = few_eval.get()
            np.testing.assert_allclose(few_eval, scipy_eval, rtol=1e-10)

        # now, we check on a denser grid that we match scipy's spline (as they use the same BCs)
        # check derivatives as well
        t_eval = np.linspace(0, 100, 1852)
        for deriv_order in range(4):
            few_eval = few_spl(t_eval, deriv_order=deriv_order)
            scipy_eval = scipy_spl(t_eval, nu=deriv_order)
            if best_backend.uses_gpu:
                few_eval = few_eval.get()
            np.testing.assert_allclose(few_eval, scipy_eval, rtol=1e-10)


class SummationTest(unittest.TestCase):
    def test_interpolated_mode_sum(self):
        # check that the interpolatedmodesum accurately computes a dummy case
        summation = InterpolatedModeSum(force_backend=best_backend)

    def test_direct_mode_sum(self):
        # check that the directmodesum accurately computes a dummy case
        summation = DirectModeSum(force_backend=best_backend)

        t_eval = np.linspace(0, 100, 10001)

        num_modes = 10
        # dummy amplitudes: linearly ramping in real and imaginary part
        amp_temp = np.linspace(np.random.uniform(-1, 1, (2, num_modes)), np.random.uniform(-1, 1, (2,num_modes)), t_eval.size, axis=-1)
        amplitude = (amp_temp[0] + 1j*amp_temp[1]).T

        # dummy phases: linear ramp
        Phi_phi_temp = np.pi/3 + 1e-4 * t_eval
        Phi_r_temp = np.pi/4 + 3e-5 * t_eval

        phases_in = np.asarray([Phi_phi_temp, Phi_phi_temp.copy(), Phi_r_temp])

        # dummy mode indices
        l_arr = np.ones(num_modes) * 2  # not used in this fictitious summation
        m_arr = np.random.randint(2, 5, num_modes)
        n_arr = np.random.randint(-3, 3, num_modes)

        # dummy 'spherical harmonics'
        ylms = np.ones(2 * num_modes)
        ylms[num_modes:] = 0   # disable the mode symmetry

        # perform the summation manually
        mode_phase_values = Phi_phi_temp[:,None] * m_arr[None,:] + Phi_r_temp[:,None] * n_arr[None,:]
        phasors = amplitude * np.exp(-1j * mode_phase_values)
        manual_sum = phasors.sum(-1)

        few_sum = summation(t_eval, amplitude, ylms, t_eval, phases_in, l_arr, m_arr, n_arr)

        if best_backend.uses_gpu:
            few_sum = few_sum.get()

        np.testing.assert_allclose(manual_sum, few_sum, rtol=1e-10)
        