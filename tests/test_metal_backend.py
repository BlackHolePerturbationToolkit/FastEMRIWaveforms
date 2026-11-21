"""Check explicit selection and the native summation against the CPU kernel."""

import unittest
from importlib.util import find_spec

import numpy as np

from few import get_backend, has_backend
from few.cutils import BackendAccessException, CUDA_BACKENDS
from few.summation.interpolatedmodesum import InterpolatedModeSum


class MetalSelectionTest(unittest.TestCase):
    def test_metal_is_explicit(self):
        self.assertEqual(has_backend("gpu"), has_backend("cuda"))
        self.assertNotIn("metal", CUDA_BACKENDS)
        self.assertNotEqual(InterpolatedModeSum().backend_name, "metal")


    @unittest.skipIf(find_spec("few_backend_metal"), "Metal extension is installed")
    def test_metal_unavailable_in_default_build(self):
        self.assertFalse(has_backend("metal"))
        with self.assertRaises(BackendAccessException):
            get_backend("metal")
        self.assertEqual(get_backend("cpu").name, "cpu")


@unittest.skipUnless(find_spec("few_backend_metal"), "Metal extension was not built")
class MetalSummationTest(unittest.TestCase):
    def test_features(self):
        backend = get_backend("metal")
        self.assertIs(backend.xp, np)
        self.assertTrue(backend.uses_metal and backend.uses_gpu)
        self.assertFalse(backend.uses_cupy or backend.uses_cuda)

    def test_accumulation_knots_and_padding(self):
        # Include m=0, its missing negative-m partner, signed k/n, non-zero
        # phases, and a last sample exactly on an interior trajectory knot.
        rng = np.random.default_rng(42)
        times = np.array([0.0, 3.5, 8.0, 11.5])
        modes = 3
        interpolant = rng.normal(size=(4, len(times), 2 * modes))
        phase_coeff = rng.normal(size=(8, len(times) - 1, 3))
        phase_coeff[0] += 1e4
        indices = [np.array(x, dtype=np.int32) for x in
                   ([0, 1, 2], [0, -2, 1], [-3, 0, 1])]
        harmonics = rng.normal(size=2 * modes) + 1j * rng.normal(size=2 * modes)
        seed = np.full(19, 2 - 3j)
        expected, actual = seed.copy(), seed.copy()
        for backend, wave in [("cpu", expected), ("metal", actual)]:
            get_backend(backend).get_waveform_wrap(
                wave, interpolant, times, phase_coeff, *indices,
                len(times), 17, modes, harmonics, 0.5, times, 0
            )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(actual[17:], seed[17:])

    @staticmethod
    def constant_mode(wave):
        """One unit-amplitude positive-m mode, four samples, no negative partner."""
        knots = np.arange(4, dtype=np.float64)
        spline = np.zeros((4, 4, 2))
        spline[0, :, 0] = 1
        phases = np.zeros((8, 3, 3))
        one = np.ones(1, dtype=np.int32)
        zero = np.zeros_like(one)
        weights = np.array([1, 0], dtype=np.complex128)
        return [wave, spline, knots, phases, one, zero, zero, 4, 4, 1, weights, 1., knots, 0]

    def test_strided_inputs_outputs_and_padding(self):
        for step in (2, -2):
            with self.subTest(step=step):
                base = np.arange(12, dtype=np.complex128) - 2j
                expected = base.copy()
                expected[::step][:4] += 1
                args = self.constant_mode(base[::step])
                # Logical input values are separated by deliberately different values.
                for i in (1, 2, 3, 4, 5, 6, 10, 12):
                    backing = np.full((*args[i].shape, 2), 9, dtype=args[i].dtype)
                    backing[..., 0] = args[i]
                    args[i] = backing[..., 0]
                get_backend("metal").get_waveform_wrap(*args)
                np.testing.assert_array_equal(base, expected)

    def test_unaligned_output(self):
        storage = bytearray(1 + 6 * np.dtype(np.complex128).itemsize)
        wave = np.ndarray(6, dtype=np.complex128, buffer=storage, offset=1)
        wave[:] = 2 - 3j
        self.assertFalse(wave.flags.aligned)
        get_backend("metal").get_waveform_wrap(*self.constant_mode(wave))
        np.testing.assert_array_equal(wave[:4], 3 - 3j)
        np.testing.assert_array_equal(wave[4:], 2 - 3j)

    def test_invalid_buffers_are_rejected_before_accumulating(self):
        wave = np.zeros(4, dtype=np.complex128)
        for position, value, exception in (
            (1, np.zeros(31), ValueError),
            (4, np.zeros(1, dtype=np.int64), TypeError),
            (8, 5, ValueError),
            (11, float("nan"), ValueError),
            (12, np.array([0., 1., 1., 3.]), RuntimeError),
            (12, np.array([0., 1., 2., 2.5]), RuntimeError),
        ):
            with self.subTest(position=position):
                args = self.constant_mode(wave)
                args[position] = value
                with self.assertRaises(exception):
                    get_backend("metal").get_waveform_wrap(*args)
                np.testing.assert_array_equal(wave, 0)
        wave.flags.writeable = False
        with self.assertRaisesRegex(ValueError, "writable"):
            get_backend("metal").get_waveform_wrap(*self.constant_mode(wave))

    def test_small_evolution_on_large_phase(self):
        # Sample inside one long segment. At 2**55, 1/8 rad is below one FP64 ULP.
        times = np.array([0., 4., 8., 12.])
        for offset in (2.**33, 2.**55, -2.**55):
            with self.subTest(offset=offset):
                wave = np.zeros(4, dtype=np.complex128)
                args = self.constant_mode(wave)
                args[2] = args[12] = times
                args[3][0, 0, 0] = offset
                args[3][1, 0, 0] = .5
                expected = np.exp(-1j * offset) * np.exp(-1j * np.arange(4) / 8)
                get_backend("metal").get_waveform_wrap(*args)
                np.testing.assert_allclose(wave, expected, atol=1e-12, rtol=0)

    def test_interior_knot_neighbors(self):
        # Give successive segments different values to expose which one was used.
        for knot, value in ((np.nextafter(1., 0.), 2), (1., 2), (np.nextafter(1., 2.), 1)):
            with self.subTest(knot=knot):
                knots = np.array([0., knot, 2.375, 4.])
                expected = np.array([1, 1, value, 2, 2, 3, 3])
                for backend in ("cpu", "metal"):
                    wave = np.zeros(7, dtype=np.complex128)
                    args = self.constant_mode(wave)
                    args[1][0, :, 0] = [1, 2, 3, 4]
                    args[2] = args[12] = knots
                    args[8], args[11] = len(wave), .5
                    get_backend(backend).get_waveform_wrap(*args)
                    np.testing.assert_array_equal(wave, expected)

    def test_nonlinear_mixed_modes(self):
        # A(t) is a complex cubic in physical time. Store its Taylor coefficients
        # at irregular knots and compute the reference from the global polynomial.
        knots = np.array([0., 1.25, 2.75, 5.])
        t = np.arange(39) / 8
        amplitude = np.polynomial.Polynomial([1 + .5j, -.5 + .25j, .125j, 1 / 32])
        derivatives = (amplitude, amplitude.deriv(), amplitude.deriv(2) / 2,
                       amplitude.deriv(3) / 6)
        spline = np.empty((4, len(knots), 4))
        for order, derivative in enumerate(derivatives):
            values = derivative(knots)
            spline[order, :, :2] = values.real[:, None]
            spline[order, :, 2:] = values.imag[:, None]

        # Each fundamental phase is offset + rate*(j+s) + bend*(s^7-s),
        # where j is the segment and s is its fractional coordinate. Expanding
        # s^7-s in the integrator's alternating s, (1-s) basis gives these six
        # coefficients after the additive and linear terms.
        rate = np.array([.25, -.5, .125])
        bend = np.array([.125, .25, -.375])
        phases = np.empty((8, len(knots) - 1, 3))
        phases[1] = rate
        phases[2:] = np.array([-1, -5, 4, 6, -3, -1])[:, None, None] * bend
        indices = np.array([[1, 2], [-2, 1], [55, -55]], dtype=np.int32)
        weights = np.array([1, -1 + 2.**-27, .25j, -.25j])
        segment = np.searchsorted(knots, t, side="right") - 1
        s = (t - knots[segment]) / np.diff(knots)[segment]
        local_phase = (segment + s)[:, None] * rate + (s**7 - s)[:, None] * bend
        for backend, offset in (("cpu", 0.), ("metal", 0.), ("metal", 2.**33)):
            with self.subTest(backend=backend, offset=offset):
                initial = offset * np.array([1., -.5, .25])
                phases[0] = initial + np.arange(len(knots) - 1)[:, None] * rate
                # Factoring out the exactly representable constant angle keeps
                # the small nonlinear evolution observable at large offsets.
                modes = (amplitude(t)[:, None] * np.exp(-1j * (initial @ indices))
                         * np.exp(-1j * (local_phase @ indices)))
                expected = modes @ weights[:2] + modes.conj() @ weights[2:]
                wave = np.zeros(len(t), dtype=np.complex128)
                get_backend(backend).get_waveform_wrap(
                    wave, spline, knots, phases, *indices, len(knots),
                    len(t), 2, weights, .125, knots, 0,
                )
                # Opposite weights nearly cancel at t=0 for zero offset.
                # Absolute error uses the envelope of the individual modes.
                envelope = max(abs(amplitude(t))) * sum(abs(weights))
                np.testing.assert_allclose(wave, expected, atol=1e-12 * envelope, rtol=0)
