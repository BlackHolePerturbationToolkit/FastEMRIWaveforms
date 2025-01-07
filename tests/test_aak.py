import unittest
import numpy as np
import warnings

from few.waveform import Pn5AAKWaveform,GenerateEMRIWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import PN5
from few.utils.utility import get_overlap, get_mismatch

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False


class AAKWaveformTest(unittest.TestCase):
    def test_aak(self):

        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "buffer_length": int(
                1e3
            )  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {"pad_output": False}

        # set initial parameters
        M = 1e6
        mu = 1e1
        a = 0.2
        p0 = 14.0
        e0 = 0.2
        iota0 = 0.1
        Y0 = np.cos(iota0)

        qS = 0.2
        phiS = 0.2
        qK = 0.8
        phiK = 0.8
        dist = 1.0
        mich = False
        dt = 10.0
        T = 0.001

        wave_cpu = Pn5AAKWaveform(inspiral_kwargs, sum_kwargs, use_gpu=False)

        waveform_cpu = wave_cpu(
            M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
        )

        self.assertTrue(
            np.all(np.abs(waveform_cpu) > 0.0)
            and np.all(np.isnan(waveform_cpu) == False)
        )

        if gpu_available:
            wave_gpu = Pn5AAKWaveform(inspiral_kwargs, sum_kwargs, use_gpu=True)

            waveform = wave_gpu(
                M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
            )

            mm = get_mismatch(waveform, waveform_cpu, use_gpu=gpu_available)
            self.assertLess(mm, 1e-10)

        few_gen = GenerateEMRIWaveform("Pn5AAKWaveform",inspiral_kwargs = inspiral_kwargs, use_gpu=gpu_available)

        Phi_phi0 = np.pi/2
        Phi_theta0 = 0.0
        Phi_r0 = np.pi/2
        h_p_c_phase = few_gen(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=2.0, dt=dt)
        h_p_c_phase2 = few_gen(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, Phi_phi0+0.5, Phi_theta0, Phi_r0, T=2.0, dt=dt)

        # print(get_overlap(h_p_c_phase.get(), h_p_c_phase2.get()))

    def test_aak_backwards(self):

        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs_forward = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "buffer_length": int(
                1e3
            ),  # all of the trajectories will be well under len = 1000
            "integrate_backwards":False
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {"pad_output": False}

        traj_module = EMRIInspiral(func = PN5)
        # set initial parameters
        M = 1e6
        mu = 1e1
        a = 0.2
        p0 = 14.0
        e0 = 0.2
        iota0 = 0.1
        Y0 = np.cos(iota0)

        qS = 0.2
        phiS = 0.2
        qK = 0.8
        phiK = 0.8
        dist = 1.0
        mich = False
        dt = 10.0
        T = 0.001

        _, p_forward, e_forward, Y_forward, _, _, _ = traj_module(M,mu,a,p0,e0,Y0, T=T, dt = dt, **inspiral_kwargs_forward)

        p_f = p_forward[-1]
        e_f = e_forward[-1]
        Y_f = Y_forward[-1]

        wave_cpu_forward = Pn5AAKWaveform(inspiral_kwargs = inspiral_kwargs_forward, use_gpu=False)

        inspiral_kwargs_backward = inspiral_kwargs_forward.copy()
        inspiral_kwargs_backward['integrate_backwards'] = True
        wave_cpu_backward = Pn5AAKWaveform(inspiral_kwargs = inspiral_kwargs_backward, use_gpu=False)

        waveform_cpu_forward = wave_cpu_forward(
            M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
        )

        waveform_cpu_backward = wave_cpu_backward(
            M, mu, a, p_f, e_f, Y_f, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
        )

        mm = get_mismatch(waveform_cpu_forward, waveform_cpu_backward[::-1], use_gpu=gpu_available)
        self.assertLess(mm, 1e-3)
