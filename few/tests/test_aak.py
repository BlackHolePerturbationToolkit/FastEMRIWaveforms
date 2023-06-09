import unittest
import numpy as np
import warnings

from few.waveform import Pn5AAKWaveform
from few.utils.utility import get_overlap, get_mismatch
from few.trajectory.inspiral import EMRIInspiral

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
    def test_trajectory_AAK(self):

        # initialize trajectory class
        traj_forwards = EMRIInspiral(func="pn5")
        traj_back = EMRIInspiral(func="pn5",integrate_backwards=True)

        # set initial parameters
        M = 1e6
        mu = 1e1
        a = 0.9
        p0 = 10.0
        e0 = 0.7
        iota0 = 0.5
        Y0 = np.cos(iota0)
        # run trajectory
        t_forward, *out_forward = traj_forwards(M, mu, 0.9, p0, e0, Y0)
        p_f = out_forward[0][-1]
        e_f = out_forward[1][-1]
        Y_f = out_forward[2][-1]

        t_back, *out_back = traj_back(M, mu, a, p_f, e_f, Y_f)
        
        self.assertAlmostEqual(p0, out_back[0][-1])
    
    def test_aak(self):

        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "max_init_len": int(
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

    def test_aak_backwards(self):

        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs_backward = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "max_init_len": int(
                1e3
            ),  # all of the trajectories will be well under len = 1000
            "integrate_backwards": True
        }
        
        inspiral_kwargs_forward = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "max_init_len": int(
                1e3
            ),  # all of the trajectories will be well under len = 1000
            "integrate_backwards": False
        }
        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {"pad_output": False}

        # set initial parameters
        M = 1e6
        mu = 1e1
        a = 0.2
        p_f = 10.0
        e_f = 0.2
        iota_f = 0.1
        Y_f = np.cos(iota_f)

        qS = 0.2
        phiS = 0.2
        qK = 0.8
        phiK = 0.8
        dist = 1.0
        mich = False
        dt = 10.0
        T = 0.01

        wave_cpu_back = Pn5AAKWaveform(inspiral_kwargs_backward, sum_kwargs, use_gpu=False)

        waveform_cpu_back = wave_cpu_back(
            M, mu, a, p_f, e_f, Y_f, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
        )

        self.assertTrue(
            np.all(np.abs(waveform_cpu_back) > 0.0)
            and np.all(np.isnan(waveform_cpu_back) == False)
        )

        traj_back = EMRIInspiral(func="pn5",integrate_backwards=True)
        _ , *out_back = traj_back(M, mu, a, p_f, e_f, Y_f, T = T)
        p0 = out_back[0][-1]
        e0 = out_back[1][-1] 
        Y0 = out_back[2][-1]

        wave_cpu_forward = Pn5AAKWaveform(inspiral_kwargs_forward, sum_kwargs, use_gpu=False)

        waveform_cpu_forward = wave_cpu_forward(
            M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
        )
 
        mm = get_mismatch(waveform_cpu_back,waveform_cpu_forward, use_gpu = False)
        self.assertLess(mm, 1e-7)


    def test_aak_fixed_timestep(self):

        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs_adaptive = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "max_init_len": int(
                1e3
            )  # all of the trajectories will be well under len = 1000
        }
        inspiral_kwargs_fixed = {
            "DENSE_STEPPING": 1,
            "max_init_len": int(1e3),
            "dT": 200
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

        wave_gen_adaptive = Pn5AAKWaveform(inspiral_kwargs_adaptive, sum_kwargs, use_gpu=False)
        wave_gen_fixed =  Pn5AAKWaveform(inspiral_kwargs_fixed, sum_kwargs, use_gpu=False)

        waveform_adapt_step = wave_gen_adaptive(
            M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
        )
        wave_fixed_step = wave_gen_fixed(
            M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
        )

        mm = get_mismatch(waveform_adapt_step, wave_fixed_step, use_gpu=gpu_available)
        self.assertLess(mm, 1e-5)