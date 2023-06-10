import unittest
import numpy as np
import warnings

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux
from few.utils.utility import get_overlap, get_mismatch, get_p_at_t, get_mu_at_t
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False


class WaveformTest(unittest.TestCase):
    def test_fast_and_slow(self):
        """
        Test the FastSchwarzschildEccentricFlux and SlowSchwarzschildEccentricFlux waveforms.

        This test compares the waveforms generated by the fast and slow inspiral generators
        and checks the similarity between them. It verifies the accuracy of the waveform
        generation process.

        Steps:
        - Set up the keyword arguments for the inspiral, amplitude, Ylm, and summation generators.
        - Create instances of the fast and slow inspiral generators with the specified arguments.
        - Set up the parameters for the waveforms (primary mass, secondary mass, initial p and e, viewing angles, and distance).
        - Generate the slow and fast waveforms using the specified parameters.
        - Calculate the mismatch between the slow and fast waveforms.
        - Assert that the mismatch is less than 1e-4.

        Raises:
            AssertionError: If the mismatch between the slow and fast waveforms is greater than 1e-4.
        """

        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "max_init_len": int(
                1e3
            ),  # all of the trajectories will be well under len = 1000
            "integrate_backwards": False
        }

        # keyword arguments for inspiral generator (RomanAmplitude)
        amplitude_kwargs = {
            "max_init_len": int(
                1e3
            )  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for Ylm generator (GetYlms)
        Ylm_kwargs = {
            "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {}

        fast = FastSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=gpu_available,
        )

        # setup slow

        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 1,  # we want a sparsely sampled trajectory
            "max_init_len": int(1e7),  # dense stepping trajectories
            "integrate_backwards": False
        }

        # keyword arguments for inspiral generator (RomanAmplitude)
        amplitude_kwargs = {"max_init_len": int(1e4)}  # this must be >= batch_size

        # keyword arguments for Ylm generator (GetYlms)
        Ylm_kwargs = {
            "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {"use_gpu": False}  # GPU is availabel for this type of summation
        mode_selector_kwargs = {}
        slow = SlowSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            mode_selector_kwargs=mode_selector_kwargs,
            use_gpu=False,
        )

        # parameters
        T = 0.001  # years
        dt = 15.0  # seconds
        M = 1e6
        mu = 1e1
        p0 = 8.0
        e0 = 0.2
        theta = np.pi / 3  # polar viewing angle
        phi = np.pi / 4  # azimuthal viewing angle
        dist = 1.0  # distance
        batch_size = int(1e4)

        slow_wave = slow(
            M, mu, p0, e0, theta, phi, dist, T=T, dt=dt, batch_size=batch_size
        )

        fast_wave = fast(M, mu, p0, e0, theta, phi, dist, T=T, dt=dt)

        mm = get_mismatch(slow_wave, fast_wave, use_gpu=gpu_available)

        self.assertLess(mm, 1e-4)

        # test_rk4
        fast.inspiral_kwargs["use_rk4"] = True
        fast_wave = fast(M, mu, p0, e0, theta, phi, dist, T=T, dt=dt)


    def test_few_backwards(self):
        """
        Test the integration of the waveform generation in the backward direction.

        This test verifies the ability of the waveform generator to integrate the waveform
        backwards by comparing the waveforms generated with forward and backward integration.
        It checks if the two waveforms are similar.

        Steps:
        - Set up the keyword arguments for the inspiral, amplitude, and Ylm generators.
        - Create instances of the fast inspiral generator for both forward and backward integration.
        - Set up the initial parameters for the trajectory and waveform generation.
        - Generate the forward and backward waveforms using the specified parameters.
        - Calculate the mismatch between the forward and backward waveforms.
        - Assert that the mismatch is less than 1e-4.

        Raises:
            AssertionError: If the mismatch between the forward and backward waveforms is greater than 1e-4.
        """

        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
            "integrate_backwards": False  # Choosing not to integrate backwards
        }

        # keyword arguments for inspiral generator (RomanAmplitude)
        amplitude_kwargs = {
            "max_init_len": int(1e3)  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for Ylm generator (GetYlms)
        Ylm_kwargs = {
            "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
        }
        
        sum_kwargs = {"use_gpu": False}  # GPU is availabel for this type of summation

        # ================ Trajectory ====================== 
        traj_forwards = EMRIInspiral(func="SchwarzEccFlux")
        traj_back = EMRIInspiral(func="SchwarzEccFlux",integrate_backwards=True) # Integrating backwards

        # set initial parameters at t = 0
        M = 1e6
        mu = 1e1
        p0 = 10.0
        e0 = 0.7
        
        T = 0.1  # years
        dt = 15.0  # seconds
        # run trajectory
        _, *out_forward = traj_forwards(M, mu, 0.0, p0, e0, 1.0, dt = dt, T = T)

        p_f = out_forward[0][-1] # Defining p(t = T)
        e_f = out_forward[1][-1] # Defining e(t = T)

        # ==================== Waveform ===================
        few_forwards = FastSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=gpu_available,
        )

        # extra waveform parameters
        theta = np.pi / 3  # polar viewing angle
        phi = np.pi / 4  # azimuthal viewing angle
        dist = 1.0  # distance

        gen_wave_forwards = few_forwards(M, mu, p0, e0, theta, phi, dist, T=T, dt=dt)
        
        # Now integrating backwards 
        inspiral_kwargs['integrate_backwards'] = True

        few_backwards= FastSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=gpu_available,
        )

        gen_wave_backwards = few_backwards(M, mu, p_f, e_f, theta, phi, dist, T=T, dt=dt)

        # Compute mismatch between forwards and backwards evolution.
        mm = get_mismatch(gen_wave_backwards,gen_wave_forwards)
        self.assertLess(mm, 1e-4)

def amplitude_test(amp_class): 
    # initialize ROMAN class
    amp = RomanAmplitude(max_init_len=5000)  # max_init_len creates memory buffers

    p = np.linspace(10.0, 14.0, 10)
    e = np.linspace(0.1, 0.7, 10)

    p_all, e_all = np.asarray([temp.ravel() for temp in np.meshgrid(p, e)])

    teuk_modes = amp_class(p_all, e_all)

    # (2, 2, 0) and (7, -3, 1) modes
    specific_modes = [(2, 2, 0), (7, -3, 1)]

    # notice this returns a dictionary with keys as the mode tuple and values as the mode values at all trajectory points
    specific_teuk_modes = amp_class(p_all, e_all, specific_modes=specific_modes)

    # we can find the index to these modes to check
    inds = np.array([amp.special_index_map[lmn] for lmn in specific_modes])

    first_check = np.allclose(specific_teuk_modes[(2, 2, 0)], teuk_modes[:, inds[0]])
    second_check = np.allclose(
        specific_teuk_modes[(7, -3, 1)], np.conj(teuk_modes[:, inds[1]])
    )
    return first_check, second_check
    
class ModuleTest(unittest.TestCase):
    def test_amplitudes(self):

        amp = RomanAmplitude()

        first_check, second_check = amplitude_test(amp)

        # make sure they are the same
        self.assertTrue(first_check)

        # to check -m modes we need to take the conjugate
        self.assertTrue(second_check)

    def test_amplitudes_bicubic(self):
        # initialize class
        amp2 = Interp2DAmplitude()

        first_check, second_check = amplitude_test(amp2)

        # make sure they are the same
        self.assertTrue(first_check)

        # to check -m modes we need to take the conjugate
        self.assertTrue(second_check)


    def test_trajectory_few(self):
        """
        Test the trajectory integration for forward and backward evolution.

        This test verifies the integration of the trajectory for both forward and backward evolution
        using the EMRIInspiral class. It checks if the final values of the parameters obtained
        from the forward trajectory match the initial values of the parameters in the backward trajectory.

        Steps:
        - Initialize instances of the EMRIInspiral class for forward and backward integration.
        - Set up the initial parameters for the trajectory (primary mass, secondary reduced mass, initial p and e).
        - Run the forward trajectory integration.
        - Get the final values of the parameters from the forward trajectory.
        - Run the backward trajectory integration using the final parameters from the forward trajectory.
        - Assert that the final values of p and e in the backward trajectory match the initial values.

        Raises:
            AssertionError: If the final values of p and e in the backward trajectory do not match the initial values.
        """        
        traj_forwards = EMRIInspiral(func="SchwarzEccFlux")
        traj_back = EMRIInspiral(func="SchwarzEccFlux",integrate_backwards=True)

        # set initial parameters
        M = 1e6
        mu = 1e1
        p0 = 10.0
        e0 = 0.7

        # run trajectory
        _, *out_forward = traj_forwards(M, mu, 0.0, p0, e0, 1.0)
        p_f = out_forward[0][-1]
        e_f = out_forward[1][-1]

        _, *out_back = traj_back(M, mu, 0.0, p_f, e_f, 1.0)
        
        self.assertAlmostEqual(p0, out_back[0][-1])
        self.assertAlmostEqual(e0, out_back[1][-1])

    def test_mismatch(self):

        dt = 1.0
        t = np.arange(10000) * dt
        x0 = np.sin(t) + 1j * np.sin(t)

        # check 1
        x1 = np.sin(t) + 1j * np.sin(t)
        self.assertAlmostEqual(get_overlap(x0, x1), 1.0)
        self.assertAlmostEqual(1.0 - get_overlap(x0, x1), get_mismatch(x0, x1))

        # check 1
        x2 = np.sin(t) + 1j * np.cos(t)
        self.assertAlmostEqual(get_overlap(x0, x2), 0.499981442642142)
        self.assertAlmostEqual(1.0 - get_overlap(x0, x1), get_mismatch(x0, x1))

    def test_get_p_at_t(self):
        """
        Test the calculation of the semilatus rectum (p) at a specific time during the trajectory.

        This test case verifies the accuracy of the `get_p_at_t` function by calculating the value of `p`
        corresponding to a specific time during the trajectory and comparing it with the expected result.
        It uses the provided trajectory module and sets the initial parameters for the trajectory.
        The trajectory arguments and the desired time `t_out` are specified.
        The `get_p_at_t` function is then called with the necessary parameters to obtain the value of `p`.
        The obtained value of `p` is used to run the trajectory backwards from `t_out` and the final time
        of the backward trajectory is compared with the expected time.

        Steps:
        - Initialize an instance of the trajectory module (EMRIInspiral).
        - Set the initial parameters for the trajectory (M, mu, e0).
        - Set the trajectory arguments (traj_args) including the initial parameters.
        - Set the time during the trajectory to calculate `p` for (t_out).
        - Calculate the value of `p` corresponding to `t_out` using the `get_p_at_t` function.
        - Run the trajectory backwards from `t_out` using the obtained value of `p`.
        - Compare the final time of the backward trajectory with the expected time.

        Raises:
            AssertionError: If the final time of the backward trajectory does not match the expected time.
        """

        traj_module = EMRIInspiral(func="SchwarzEccFlux")

        # set initial parameters
        M = 1e6
        mu = 1e1
        e0 = 0.7

        # Set trajectory arguments
        traj_args = [M, mu, 0.0, e0, 1.0]

        # Set time of trajectory
        t_out = 1  

        # Calculate value of p corresponding to trajectory length t_out 
        p_new = get_p_at_t(
                traj_module,
                t_out,
                traj_args,
                index_of_p=3,
                xtol=2e-12,
                rtol=8.881784197001252e-16,
                bounds=None,
                        )

        t_back, *out = traj_module(M, mu, 0.0, p_new, e0, 1.0, T = t_out)
        one_year = 365 * 24 * 60 * 60
        
        self.assertAlmostEqual(t_back[-1]/(one_year), t_out, places = 2)

    def test_get_mu_at_t(self):
        """
        Test the calculation of the reduced mass (mu) at a specific time during the trajectory.

        This test case verifies the accuracy of the `get_mu_at_t` function by calculating the value of `mu`
        corresponding to a specific time during the trajectory and comparing it with the expected result.
        It uses the provided trajectory module and sets the initial parameters for the trajectory.
        The trajectory arguments and the desired time `t_out` are specified.
        The `get_mu_at_t` function is then called with the necessary parameters to obtain the value of `mu`.
        The obtained value of `mu` is used to run the trajectory backwards from `t_out` and the final time
        of the backward trajectory is compared with the expected time.

        Steps:
        - Initialize an instance of the trajectory module (EMRIInspiral).
        - Set the initial parameters for the trajectory (M, p0, e0).
        - Set the trajectory arguments (traj_args) including the initial parameters.
        - Set the time during the trajectory to calculate `mu` for (t_out).
        - Specify the index of `mu` in the trajectory arguments (index_of_mu).
        - Calculate the value of `mu` corresponding to `t_out` using the `get_mu_at_t` function.
        - Run the trajectory backwards from `t_out` using the obtained value of `mu`.
        - Compare the final time of the backward trajectory with the expected time.

        Raises:
            AssertionError: If the final time of the backward trajectory does not match the expected time.
        """
        traj_module = EMRIInspiral(func="SchwarzEccFlux")

        # set initial parameters
        M = 1e6
        p0 = 20.0
        e0 = 0.7

        # Set trajectory arguments
        traj_args = [M, 0.0, p0, e0, 1.0]

        # Set time of trajectory
        t_out = 1  

        index_of_mu = 1
        
        # Calculate value of p corresponding to trajectory length t_out 
        mu_new = get_mu_at_t(
            traj_module,
            t_out,
            traj_args,
            index_of_mu=index_of_mu,
            xtol=2e-12,
            rtol=8.881784197001252e-16,
            bounds=None,
        )

        t_back, *out = traj_module(M, mu_new, 0.0, p0, e0, 1.0, T = t_out)
        one_year = 365 * 24 * 60 * 60
        
        self.assertAlmostEqual(t_back[-1]/(one_year), t_out, places = 2)