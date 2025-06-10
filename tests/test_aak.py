import numpy as np

from few.tests.base import FewBackendTest, tagged_test
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import PN5
from few.utils.utility import get_mismatch, get_overlap
from few.waveform import GenerateEMRIWaveform, Pn5AAKWaveform


class AAKWaveformTest(FewBackendTest):
    @classmethod
    def name(self) -> str:
        return "AAKWaveformTest"

    @classmethod
    def parallel_class(self):
        return Pn5AAKWaveform

    @tagged_test(slow=True)
    def test_aak(self):
        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "buffer_length": int(
                1e3
            ),  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {"pad_output": False}

        # set initial parameters
        m1 = 1e6
        m2 = 1e1
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

        wave_cpu = Pn5AAKWaveform(inspiral_kwargs, sum_kwargs, force_backend="cpu")

        waveform_cpu = wave_cpu(
            m1, m2, a, p0, e0, Y0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
        )

        self.assertTrue(
            np.all(np.abs(waveform_cpu) > 0.0) and not np.any(np.isnan(waveform_cpu))
        )

        if self.backend.uses_gpu:
            wave_gpu = Pn5AAKWaveform(inspiral_kwargs, sum_kwargs)

            waveform = wave_gpu(
                m1, m2, a, p0, e0, Y0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
            )

            mm = get_mismatch(waveform, waveform_cpu, use_gpu=self.backend.uses_gpu)
            self.assertLess(mm, 1e-10)

        few_gen = GenerateEMRIWaveform(
            "Pn5AAKWaveform", inspiral_kwargs=inspiral_kwargs
        )

        Phi_phi0 = np.pi / 2
        Phi_theta0 = 0.0
        Phi_r0 = np.pi / 2
        h_p_c_phase = few_gen(
            m1,
            m2,
            a,
            p0,
            e0,
            Y0,
            dist,
            qS,
            phiS,
            qK,
            phiK,
            Phi_phi0,
            Phi_theta0,
            Phi_r0,
            T=2.0,
            dt=dt,
        )
        h_p_c_phase2 = few_gen(
            m1,
            m2,
            a,
            p0,
            e0,
            Y0,
            dist,
            qS,
            phiS,
            qK,
            phiK,
            Phi_phi0 + 0.5,
            Phi_theta0,
            Phi_r0,
            T=2.0,
            dt=dt,
        )

        self.logger.info(
            "  Overlap is: {}".format(
                get_overlap(h_p_c_phase, h_p_c_phase2, use_gpu=self.backend.uses_gpu)
            )
        )

    def test_aak_backwards(self):
        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs_forward = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "buffer_length": int(
                1e3
            ),  # all of the trajectories will be well under len = 1000
            "integrate_backwards": False,
        }

        traj_module = EMRIInspiral(func=PN5)
        # set initial parameters
        m1 = 1e6
        m2 = 1e1
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

        _, p_forward, e_forward, Y_forward, _, _, _ = traj_module(
            m1, m2, a, p0, e0, Y0, T=T, dt=dt, **inspiral_kwargs_forward
        )

        p_f = p_forward[-1]
        e_f = e_forward[-1]
        Y_f = Y_forward[-1]

        wave_cpu_forward = Pn5AAKWaveform(
            inspiral_kwargs=inspiral_kwargs_forward, force_backend="cpu"
        )

        inspiral_kwargs_backward = inspiral_kwargs_forward.copy()
        inspiral_kwargs_backward["integrate_backwards"] = True
        wave_cpu_backward = Pn5AAKWaveform(
            inspiral_kwargs=inspiral_kwargs_backward, force_backend="cpu"
        )

        waveform_cpu_forward = wave_cpu_forward(
            m1, m2, a, p0, e0, Y0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
        )

        waveform_cpu_backward = wave_cpu_backward(
            m1, m2, a, p_f, e_f, Y_f, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
        )

        mm = get_mismatch(
            waveform_cpu_forward, waveform_cpu_backward[::-1], use_gpu=False
        )
        self.assertLess(mm, 1e-3)
