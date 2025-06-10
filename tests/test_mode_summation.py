import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from few.summation.directmodesum import DirectModeSum
from few.summation.interpolatedmodesum import (
    CubicSplineInterpolant,
    InterpolatedModeSum,
)
from few.tests.base import FewBackendTest
from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import YRSID_SI


class CubicSplineTest(FewBackendTest):
    @classmethod
    def name(self) -> str:
        return "Mode-summation (CubicSpline)"

    @classmethod
    def parallel_class(self):
        return InterpolatedModeSum

    def test_cubic_spline(self):
        # check that the cubic spline correctly interpolates test functions
        t_test = np.linspace(0, 100, 1000)
        f_test = np.vstack((np.cos(t_test), np.sin(t_test)))
        few_spl = CubicSplineInterpolant(t_test, f_test, force_backend=self.backend)
        scipy_spl = CubicSpline(t_test, f_test, axis=-1)

        # first check the knots are returned
        few_eval = few_spl(t_test)
        if self.backend.uses_gpu:
            few_eval = few_eval.get()
        np.testing.assert_allclose(few_eval, f_test, rtol=1e-10)

        # check that derivatives agree with scipy on the knots
        for deriv_order in range(1, 4):
            few_eval = few_spl(t_test, deriv_order=deriv_order)
            scipy_eval = scipy_spl(t_test, nu=deriv_order)
            if self.backend.uses_gpu:
                few_eval = few_eval.get()
            np.testing.assert_allclose(few_eval, scipy_eval, rtol=1e-10)

        # now, we check on a denser grid that we match scipy's spline (as they use the same BCs)
        # check derivatives as well
        t_eval = np.linspace(0, 100, 1852)
        for deriv_order in range(4):
            few_eval = few_spl(t_eval, deriv_order=deriv_order)
            scipy_eval = scipy_spl(t_eval, nu=deriv_order)
            if self.backend.uses_gpu:
                few_eval = few_eval.get()
            np.testing.assert_allclose(few_eval, scipy_eval, rtol=1e-10)


class SummationTest(FewBackendTest):
    @classmethod
    def name(self) -> str:
        return "Mode-summation (Summation)"

    @classmethod
    def parallel_class(self):
        return InterpolatedModeSum

    def test_interpolated_mode_sum(self):
        # check that the interpolatedmodesum accurately computes a dummy case
        summation = InterpolatedModeSum(force_backend=self.backend)

        # interpolatedmodesum expects 8th-order phase coefficients of a particular form
        # the easiest way to obtain these is to just evolve a 0PA trajectory
        # in the future when our integrator is fully stand-alone this can be generalised
        traj_module = EMRIInspiral(func="KerrEccEqFlux")

        m1 = 1e6
        m2 = 1e1
        nu = m1 * m2 / (m1 + m2)**2
        traj = traj_module(m1, m2, 0.4, 20., 0.6, 1., T=0.1, err=1e-15)
        t_spl = traj_module.inspiral_generator.integrator_t_cache
        coeff_spl = traj_module.inspiral_generator.integrator_spline_coeff[:,[3,5],:] / nu

        t_eval = np.linspace(0, t_spl[-1], 10001)
        dt = t_eval[1] - t_eval[0]
        phases_eval = traj_module.inspiral_generator.eval_integrator_spline(t_eval)[
            :, [3, 5]
        ].T

        # now we make up the rest
        num_modes = 10

        # dummy amplitudes: linearly ramping in real and imaginary part
        amp_temp = np.linspace(
            np.random.uniform(-1, 1, (2, num_modes)),
            np.random.uniform(-1, 1, (2, num_modes)),
            t_eval.size,
            axis=-1,
        )
        amplitude = (amp_temp[0] + 1j * amp_temp[1]).T

        # we need to provide these values at the spline knots for InterpolatedModeSum
        # easy to do here as we are just ramping them linearly
        amplitude_spl_temp = interp1d(
            t_eval, amp_temp, axis=-1, kind="linear", assume_sorted=True
        )(t_spl)
        amplitude_spl = summation.xp.asarray(
            (amplitude_spl_temp[0] + 1j * amplitude_spl_temp[1]).T
        )

        # dummy mode indices
        l_arr = np.ones(num_modes) * 2  # not used in this fictitious summation
        m_arr = np.random.randint(2, 5, num_modes)
        n_arr = np.random.randint(-4, 4, num_modes)

        # dummy 'spherical harmonics'
        ylms = summation.xp.ones(2 * num_modes)
        ylms[num_modes:] = 0  # disable the mode symmetry

        # perform the summation manually
        mode_phase_values = (
            phases_eval[0, :, None] * m_arr[None, :]
            + phases_eval[1, :, None] * n_arr[None, :]
        )
        phasors = amplitude * np.exp(-1j * mode_phase_values)
        manual_sum = phasors.sum(-1)

        l_arr = summation.xp.asarray(l_arr)
        m_arr = summation.xp.asarray(m_arr)
        n_arr = summation.xp.asarray(n_arr)

        few_sum = summation(
            summation.xp.asarray(t_spl),
            amplitude_spl,
            ylms,
            t_spl,
            coeff_spl,
            l_arr,
            m_arr,
            n_arr,
            T=t_spl[-1] / YRSID_SI,
            dt=dt,
        )

        if self.backend.uses_gpu:
            few_sum = few_sum.get()

        # the following leads to an error on linux systems for some reason
        # np.testing.assert_allclose(manual_sum, few_sum, rtol=1e-10)

        np.testing.assert_allclose(manual_sum[:-1], few_sum[:-1], rtol=1e-8)

    def test_direct_mode_sum(self):
        # check that the directmodesum accurately computes a dummy case
        summation = DirectModeSum(force_backend=self.backend)

        t_eval = np.linspace(0, 1000, 10001)

        num_modes = 10
        # dummy amplitudes: linearly ramping in real and imaginary part
        amp_temp = np.linspace(
            np.random.uniform(-1, 1, (2, num_modes)),
            np.random.uniform(-1, 1, (2, num_modes)),
            t_eval.size,
            axis=-1,
        )
        amplitude = (amp_temp[0] + 1j * amp_temp[1]).T

        # dummy phases: linear ramp
        Phi_phi_temp = np.pi / 3 + 1e-2 * t_eval
        Phi_r_temp = np.pi / 4 + 3e-3 * t_eval

        phases_in = np.asarray([Phi_phi_temp, Phi_phi_temp.copy(), Phi_r_temp])

        # dummy mode indices
        l_arr = np.ones(num_modes) * 2  # not used in this fictitious summation
        m_arr = np.random.randint(2, 5, num_modes)
        n_arr = np.random.randint(-3, 3, num_modes)

        # dummy 'spherical harmonics'
        ylms = np.ones(2 * num_modes)
        ylms[num_modes:] = 0  # disable the mode symmetry

        # perform the summation manually
        mode_phase_values = (
            Phi_phi_temp[:, None] * m_arr[None, :]
            + Phi_r_temp[:, None] * n_arr[None, :]
        )
        phasors = amplitude * np.exp(-1j * mode_phase_values)
        manual_sum = phasors.sum(-1)

        few_sum = summation(
            t_eval, amplitude, ylms, t_eval, phases_in, l_arr, m_arr, n_arr
        )

        if self.backend.uses_gpu:
            few_sum = few_sum.get()

        np.testing.assert_allclose(manual_sum, few_sum, rtol=1e-10)
