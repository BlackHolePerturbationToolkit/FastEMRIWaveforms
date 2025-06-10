import pickle

from few.tests.base import FewBackendTest, tagged_test
from few.waveform import FastSchwarzschildEccentricFlux, GenerateEMRIWaveform


class WaveformTest(FewBackendTest):
    @classmethod
    def name(self) -> str:
        return "DetectorWave"

    @classmethod
    def parallel_class(self):
        return FastSchwarzschildEccentricFlux

    def run_detector_frame_test(self, test_pickle=False):
        # specific waveform kwargs
        waveform_kwargs = dict(
            force_backend=self.backend,
        )

        gen_wave = GenerateEMRIWaveform(
            "FastSchwarzschildEccentricFlux", **waveform_kwargs
        )

        gen_wave_aak = GenerateEMRIWaveform("Pn5AAKWaveform", **waveform_kwargs)

        # parameters
        T = 0.001  # years
        dt = 15.0  # seconds
        m1 = 1e6
        a = 0.1  # will be ignored in Schwarz waveform
        m2 = 1e1
        p0 = 12.0
        e0 = 0.2
        Y0 = 1.0
        qK = 0.2  # polar spin angle
        phiK = 0.2  # azimuthal viewing angle
        qS = 0.3  # polar sky angle
        phiS = 0.3  # azimuthal viewing angle
        dist = 1.0  # distance
        Phi_phi0 = 1.0
        Phi_theta0 = 2.0
        Phi_r0 = 3.0

        if test_pickle:
            check_pickle_gen_wave = pickle.dumps(gen_wave)
            extracted_gen_wave = pickle.loads(check_pickle_gen_wave)

            check_pickle_gen_wave_aak = pickle.dumps(gen_wave_aak)
            extracted_gen_wave_aak = pickle.loads(check_pickle_gen_wave_aak)

        else:
            extracted_gen_wave = gen_wave
            extracted_gen_wave_aak = gen_wave_aak

        _ = extracted_gen_wave(
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
            T=T,
            dt=dt,
        )

        _ = extracted_gen_wave_aak(
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
            T=T,
            dt=dt,
        )

    @tagged_test(slow=True)
    def test_pickle(self):
        self.run_detector_frame_test(test_pickle=True)

    @tagged_test(slow=True)
    def test_detector_frame(self):
        self.run_detector_frame_test(test_pickle=False)
