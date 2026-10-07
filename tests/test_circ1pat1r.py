from few.tests.base import FewBackendTest
from few.utils.utility import get_mismatch
from few.waveform import Circ1PAT1R, GenerateEMRIWaveform

# parameters (m1, m2, chi1, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, chi2)
m1 = 1e6
m2 = 1e1
chi1 = 1e-5
p0 = 10.0
e0 = 0.0
xI0 = 1.0
chi2 = 0.1

dist = 1.0
qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8

Phi_phi0 = 1.0
Phi_theta0 = 2.0
Phi_r0 = 3.0

params = [
    m1,
    m2,
    chi1,
    p0,
    e0,
    xI0,
    dist,
    qS,
    phiS,
    qK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
    chi2,
]

dt = 10.0
T = 0.1


class Circ1PAT1RWaveformTest(FewBackendTest):
    @classmethod
    def name(self) -> str:
        return "Circ1PAT1R"

    @classmethod
    def parallel_class(self):
        return Circ1PAT1R

    def test_waveform_runs(self):
        # Test that the default Circ1PAT1R waveform generates without error.
        wave_generator = GenerateEMRIWaveform("Circ1PAT1R", force_backend=self.backend)

        wave = wave_generator(*params, T=T, dt=dt)

        self.assertGreater(len(wave), 0)
        xp = self.backend.xp
        self.assertTrue(bool(xp.all(xp.isfinite(wave))))

    def test_fd_waveform_runs(self):
        # Test that the Circ1PAT1R waveform generates in the frequency domain.
        wave_generator = GenerateEMRIWaveform(
            "Circ1PAT1R",
            sum_kwargs={"pad_output": True, "output_type": "fd"},
            force_backend=self.backend,
        )

        wave = wave_generator(*params, T=T, dt=dt)

        self.assertGreater(len(wave), 0)
        xp = self.backend.xp
        self.assertTrue(bool(xp.all(xp.isfinite(wave))))

    def test_no_evolve_primary_zero_PA_amps(self):
        # Test that turning off primary evolution and 1PA amplitude corrections
        # gives a waveform with a small (but nonzero) mismatch from the default.
        self.logger.info("Testing evolve_primary=False and zero_PA_amps_only=True")

        wave_generator_default = GenerateEMRIWaveform(
            "Circ1PAT1R", force_backend=self.backend
        )
        wave_generator_off = GenerateEMRIWaveform(
            "Circ1PAT1R",
            inspiral_kwargs={"evolve_primary": False},
            amplitude_kwargs={"zero_PA_amps_only": True},
            force_backend=self.backend,
        )

        # (m2, min mismatch, max mismatch): the effect grows with mass ratio
        cases = [
            (1e1, 0.0, 1e-8),
            (1e5, 1e-5, 1e-2),
        ]

        for m2_case, mm_min, mm_max in cases:
            with self.subTest(m2=m2_case):
                params_case = list(params)
                params_case[1] = m2_case
                # scale primary spin with the mass ratio
                params_case[2] = m2_case / m1

                wave_default = wave_generator_default(*params_case, T=T, dt=dt)
                wave_off = wave_generator_off(*params_case, T=T, dt=dt)

                n = min(len(wave_default), len(wave_off))
                mm = get_mismatch(
                    wave_default[:n], wave_off[:n], use_gpu=self.backend.uses_gpu
                )

                self.assertGreater(mm, mm_min)
                self.assertLess(mm, mm_max)

    def test_no_evolve_primary(self):
        # Test that turning off primary evolution alone gives a small (but
        # nonzero) mismatch from the default. The dephasing from the horizon
        # fluxes builds up slowly, so use a long inspiral starting at large p0.
        self.logger.info("Testing evolve_primary=False")

        wave_generator_default = GenerateEMRIWaveform(
            "Circ1PAT1R", force_backend=self.backend
        )
        wave_generator_off = GenerateEMRIWaveform(
            "Circ1PAT1R",
            inspiral_kwargs={"evolve_primary": False},
            force_backend=self.backend,
        )

        m2_case = 1e4
        params_case = list(params)
        params_case[1] = m2_case
        # scale primary spin with the mass ratio
        params_case[2] = m2_case / m1
        params_case[3] = 20.0  # p0
        T_case = 1.0

        wave_default = wave_generator_default(*params_case, T=T_case, dt=dt)
        wave_off = wave_generator_off(*params_case, T=T_case, dt=dt)

        n = min(len(wave_default), len(wave_off))
        mm = get_mismatch(
            wave_default[:n], wave_off[:n], use_gpu=self.backend.uses_gpu
        )

        self.assertGreater(mm, 1e-10)
        self.assertLess(mm, 1e-6)
