import sys
import os
from abc import ABC

import numpy as np
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline

# check if cupy is available / GPU is available
try:
    import cupy as cp
    import numpy as xp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

from ..utils.baseclasses import (
    Pn5AAK,
    ParallelModuleBase,
    SphericalHarmonic
)

from ..utils.utility import (
    p_to_y,
)
from ..utils.ylm import GetYlms
from ..utils.constants import *
from ..utils.citations import *

from typing import Union, Optional


class SphericalHarmonicWaveformBase(ParallelModuleBase, ABC):
    """Base class for waveforms built with amplitudes expressed in a spherical harmonic basis.

    This class contains the methods required to build the core waveform for Kerr equatorial eccentric
    (to be upgraded to Kerr generic once that is available). Stock waveform classes constructed in
    this basis can subclass this class and implement their own "__call__" method to fill in the
    relevant data.
    """

    def __init__(
        self,
        inspiral_module,
        amplitude_module,
        sum_module,
        mode_selector_module,
        inspiral_kwargs={},
        amplitude_kwargs={},
        sum_kwargs={},
        Ylm_kwargs={},
        mode_selector_kwargs={},
        use_gpu=False,
        num_threads=None,
        normalize_amps=False,
    ):
        ParallelModuleBase.__init__(self, use_gpu=use_gpu, num_threads=num_threads)

        (
            amplitude_kwargs,
            sum_kwargs,
            Ylm_kwargs,
            mode_selector_kwargs,
        ) = self.adjust_gpu_usage(
            use_gpu, [amplitude_kwargs, sum_kwargs, Ylm_kwargs, mode_selector_kwargs]
        )

        # normalize amplitudes to flux at each step from trajectory
        self.normalize_amps = normalize_amps

        # kwargs that are passed to the inspiral call function
        self.inspiral_kwargs = inspiral_kwargs

        # function for generating the inpsiral
        self.inspiral_generator = inspiral_module(**inspiral_kwargs)

        # function for generating the amplitude
        self.amplitude_generator = amplitude_module(**amplitude_kwargs)

        # summation generator
        self.create_waveform = sum_module(**sum_kwargs)

        # angular harmonics generation
        self.ylm_gen = GetYlms(**Ylm_kwargs)

        # selecting modes that contribute at threshold to the waveform
        self.mode_selector = mode_selector_module(
            self.l_arr_no_mask,
            self.m_arr_no_mask,
            self.n_arr_no_mask,
            **mode_selector_kwargs,
        )

    def _generate_waveform(
        self,
        M: float,
        mu: float,
        a: float,
        p0: float,
        e0: float,
        xI0: float,
        theta: float,
        phi: float,
        *args: Optional[tuple],
        dist: Optional[float]=None,
        Phi_phi0: float=0.0,
        Phi_r0: float=0.0,
        dt: float=10.0,
        T: float=1.0,
        eps: float=1e-5,
        show_progress: bool=False,
        batch_size: int=-1,
        mode_selection: Optional[Union[str, list]]=None,
        include_minus_m: bool=True,
        **kwargs: Optional[dict],
    ):
        """Call function for waveform models built in the spherical harmonic basis.

        This function will take input parameters and produce waveforms. It will use all of the modules preloaded to
        compute desired outputs.

        args:
            M (double): Mass of larger black hole in solar masses.
            mu (double): Mass of compact object in solar masses.
            a (double): Dimensionless spin parameter of larger black hole.
            p0 (double): Initial (osculating) semilatus rectum of inspiral trajectory.
            e0 (double): Initial (osculating) eccentricity of inspiral trajectory.
            theta (double): Polar viewing angle in radians (:math:`-\pi/2\leq\Theta\leq\pi/2`).
            phi (double): Azimuthal viewing angle in radians.
            *args (list): extra args for trajectory model.
            dist (double, optional): Luminosity distance in Gpc. Default is None. If None,
                will return source frame.
            Phi_phi0 (double, optional): Initial phase for :math:`\Phi_\phi`.
                Default is 0.0.
            Phi_r0 (double, optional): Initial phase for :math:`\Phi_r`.
                Default is 0.0.
            dt (double, optional): Time between samples in seconds (inverse of
                sampling frequency). Default is 10.0.
            T (double, optional): Total observation time in years.
                Default is 1.0.
            eps (double, optional): Controls the fractional accuracy during mode
                filtering. Raising this parameter will remove modes. Lowering
                this parameter will add modes. Default that gives a good overalp
                is 1e-5.
            show_progress (bool, optional): If True, show progress through
                amplitude/waveform batches using
                `tqdm <https://tqdm.github.io/>`_. Default is False.
            batch_size (int, optional): If less than 0, create the waveform
                without batching. If greater than zero, create the waveform
                batching in sizes of batch_size. Default is -1.
            mode_selection (str or list or None): Determines the type of mode
                filtering to perform. If None, perform our base mode filtering
                with eps as the fractional accuracy on the total power.
                If 'all', it will run all modes without filtering. If a list of
                tuples (or lists) of mode indices
                (e.g. [(:math:`l_1,m_1,n_1`), (:math:`l_2,m_2,n_2`)]) is
                provided, it will return those modes combined into a
                single waveform.
            include_minus_m (bool, optional): If True, then include -m modes when
                computing a mode with m. This only effects modes if :code:`mode_selection`
                is a list of specific modes. Default is True.

        Returns:
            1D complex128 self.xp.ndarray: The output waveform.

        Raises:
            ValueError: user selections are not allowed.

        """

        # makes sure viewing angles are allowable
        theta, phi = self.sanity_check_viewing_angles(theta, phi)

        self.sanity_check_init(M, mu, a, p0, e0, xI0)

        # get trajectory
        (t, p, e, xI, Phi_phi, Phi_theta, Phi_r) = self.inspiral_generator(
            M,
            mu,
            a,
            p0,
            e0,
            xI0,
            *args,
            Phi_phi0=Phi_phi0,
            Phi_theta0=0.0,
            Phi_r0=Phi_r0,
            T=T,
            dt=dt,
            **self.inspiral_kwargs,
        )
        # makes sure p and e are generally within the model
        self.sanity_check_traj(a, p, e, xI)

        if self.normalize_amps:
            # get the vector norm
            amp_norm = self.amplitude_generator.amp_norm_spline.ev(
                p_to_y(p, e), e
            )  # TODO: handle this grid parameter change, fix to Schwarzschild for now
            amp_norm = self.xp.asarray(amp_norm)

        self.end_time = t[-1]

        # convert for gpu
        t = self.xp.asarray(t)
        p = self.xp.asarray(p)
        e = self.xp.asarray(e)
        xI = self.xp.asarray(xI)
        Phi_phi = self.xp.asarray(Phi_phi)
        Phi_r = self.xp.asarray(Phi_r)

        # get ylms only for unique (l,m) pairs
        # then expand to all (lmn with self.inverse_lm)
        ylms = self.ylm_gen(self.unique_l, self.unique_m, theta, phi).copy()[
            self.inverse_lm
        ]
        # if mode selector is predictive, run now to avoid generating amplitudes that are not required
        if self.mode_selector.is_predictive:
            # overwrites mode_selection so it's now a list of modes to keep, ready to feed into amplitudes
            mode_selection = self.mode_selector(
                M, mu, a * xI0, p0, e0, 1.0, theta, phi, T, eps
            )  # TODO: update this if more arguments are required

        # split into batches

        if batch_size == -1 or self.allow_batching is False:
            inds_split_all = [self.xp.arange(len(t))]
        else:
            split_inds = []
            i = 0
            while i < len(t):
                i += batch_size
                if i >= len(t):
                    break
                split_inds.append(i)

            inds_split_all = self.xp.split(self.xp.arange(len(t)), split_inds)

        # select tqdm if user wants to see progress
        iterator = enumerate(inds_split_all)
        iterator = tqdm(iterator, desc="time batch") if show_progress else iterator

        if show_progress:
            print("total:", len(inds_split_all))

        for i, inds_in in iterator:
            # get subsections of the arrays for each batch
            t_temp = t[inds_in]
            p_temp = p[inds_in]
            e_temp = e[inds_in]
            Phi_phi_temp = Phi_phi[inds_in]
            Phi_theta_temp = Phi_theta[inds_in]
            Phi_r_temp = Phi_r[inds_in]

            if self.normalize_amps:
                amp_norm_temp = amp_norm[inds_in]

            # if we aren't requesting a subset of modes, compute them all now
            if not isinstance(mode_selection, list) and not isinstance(
                mode_selection, self.xp.ndarray
            ):
                # amplitudes
                teuk_modes = self.xp.asarray(
                    self.amplitude_generator(a, p_temp, e_temp, xI0)
                )

                # normalize by flux produced in trajectory
                if self.normalize_amps:
                    amp_for_norm = self.xp.sum(
                        self.xp.abs(
                            self.xp.concatenate(
                                [teuk_modes, self.xp.conj(teuk_modes[:, self.m0mask])],
                                axis=1,
                            )
                        )
                        ** 2,
                        axis=1,
                    ) ** (1 / 2)

                    # normalize
                    factor = amp_norm_temp / amp_for_norm
                    teuk_modes = teuk_modes * factor[:, np.newaxis]

            # different types of mode selection
            # sets up ylm and teuk_modes properly for summation
            if isinstance(mode_selection, str):
                # use all modes
                if mode_selection == "all":
                    self.ls = self.l_arr[: teuk_modes.shape[1]]
                    self.ms = self.m_arr[: teuk_modes.shape[1]]
                    self.ns = self.n_arr[: teuk_modes.shape[1]]

                    keep_modes = self.xp.arange(teuk_modes.shape[1])
                    temp2 = keep_modes * (keep_modes < self.num_m0) + (
                        keep_modes + self.num_m_1_up
                    ) * (keep_modes >= self.num_m0)

                    ylmkeep = self.xp.concatenate([keep_modes, temp2])
                    ylms_in = ylms[ylmkeep]
                    teuk_modes_in = teuk_modes

                else:
                    raise ValueError("If mode selection is a string, must be `all`.")

            # get a specific subset of modes
            elif isinstance(mode_selection, list) or isinstance(
                mode_selection, self.xp.ndarray
            ):
                if len(mode_selection) == 0:
                    raise ValueError("If mode selection is a list, cannot be empty.")

                if self.normalize_amps:
                    assert isinstance(mode_selection, list)

                    # compute all amplitudes
                    teuk_modes = self.xp.asarray(
                        self.amplitude_generator(a, p_temp, e_temp, xI0)
                    )

                    amp_for_norm = self.xp.sum(
                        self.xp.abs(
                            self.xp.concatenate(
                                [teuk_modes, self.xp.conj(teuk_modes[:, self.m0mask])],
                                axis=1,
                            )
                        )
                        ** 2,
                        axis=1,
                    ) ** (1 / 2)

                    keep_inds = self.xp.asarray([self.amplitude_generator.special_index_map[md] for md in mode_selection])

                    # filter modes and normalize
                    factor = amp_norm_temp / amp_for_norm
                    teuk_modes = teuk_modes[:, keep_inds] * factor[:, np.newaxis]


                else:
                    # generate only the required modes with the amplitude module
                    teuk_modes = self.amplitude_generator(
                        a, p_temp, e_temp, xI0, specific_modes=mode_selection
                    )

                # unpack the dictionary
                if isinstance(teuk_modes, dict):
                    teuk_modes_in = self.xp.asarray(
                        [teuk_modes[lmn] for lmn in mode_selection]
                    ).T
                else:
                    teuk_modes_in = teuk_modes

                # for removing opposite m modes
                fix_include_ms = self.xp.full(2 * len(mode_selection), False)
                if isinstance(mode_selection, list):
                    keep_modes = self.xp.zeros(len(mode_selection), dtype=self.xp.int32)
                    for jj, lmn in enumerate(mode_selection):
                        l, m, n = tuple(lmn)

                        # keep modes only works with m>=0
                        lmn_in = (l, abs(m), n)
                        keep_modes[jj] = self.xp.int32(self.lmn_indices[lmn_in])

                        if not include_minus_m:
                            if m > 0:
                                # minus m modes blocked
                                fix_include_ms[len(mode_selection) + jj] = True
                            elif m < 0:
                                # positive m modes blocked
                                fix_include_ms[jj] = True
                else:
                    keep_modes = mode_selection
                    m_temp = abs(self.m_arr[mode_selection])
                    for jj, m_here in enumerate(m_temp):
                        if not include_minus_m:
                            if m_here > 0:
                                # minus m modes blocked
                                fix_include_ms[len(mode_selection) + jj] = True
                            elif m_here < 0:
                                # positive m modes blocked
                                fix_include_ms[jj] = True

                self.ls = self.l_arr[keep_modes]
                self.ms = self.m_arr[keep_modes]
                self.ns = self.n_arr[keep_modes]

                temp2 = keep_modes * (keep_modes < self.num_m0) + (
                    keep_modes + self.num_m_1_up
                ) * (keep_modes >= self.num_m0)

                ylmkeep = self.xp.concatenate([keep_modes, temp2])
                ylms_in = ylms[ylmkeep]

                # remove modes if include_minus_m is False
                ylms_in[fix_include_ms] = 0.0 + 1j * 0.0

            # mode selection based on input module
            else:
                fund_freq_args = (
                    M,
                    0.0,
                    p_temp,
                    e_temp,
                    self.xp.zeros_like(e_temp),
                )
                modeinds = [self.l_arr, self.m_arr, self.n_arr]
                (
                    teuk_modes_in,
                    ylms_in,
                    self.ls,
                    self.ms,
                    self.ns,
                ) = self.mode_selector(
                    teuk_modes,
                    ylms,
                    modeinds,
                    fund_freq_args=fund_freq_args,
                    eps=eps,
                )

            # store number of modes for external information
            self.num_modes_kept = teuk_modes_in.shape[1]
            

            # prepare phases for summation modules
            if not self.inspiral_generator.inspiral_generator.dopr.fix_step:
                # prepare phase spline coefficients
                phase_spline_coeff = self.inspiral_generator.inspiral_generator.integrator_spline_coeff  # TODO make these accessible from EMRIInspiral
                
                # scale coefficients here by the mass ratio
                phase_information_in = phase_spline_coeff[:,[3,5],:] / (mu / M)

                if self.inspiral_generator.inspiral_generator.integrate_backwards:
                    phase_information_in[:,:,0] += self.xp.array([Phi_phi[-1] + Phi_phi[0], Phi_r[-1] + Phi_r[0]])

                phase_t_in = self.inspiral_generator.inspiral_generator.integrator_t_cache
            else:
                phase_information_in = [Phi_phi_temp, Phi_theta_temp, Phi_r_temp]
                if self.inspiral_generator.inspiral_generator.integrate_backwards:
                    phase_information_in[0] += self.xp.array([Phi_phi[-1] + Phi_phi[0]])
                    phase_information_in[1] += self.xp.array([Phi_theta[-1] + Phi_theta[0]])
                    phase_information_in[2] += self.xp.array([Phi_r[-1] + Phi_r[0]])

                phase_t_in = None
                
            # create waveform
            waveform_temp = self.create_waveform(
                t_temp,
                teuk_modes_in,
                ylms_in,
                phase_t_in, 
                phase_information_in,
                self.ms,
                self.ns,
                M,
                a,
                p,
                e,
                xI,
                dt=dt,
                T=T,
                include_minus_m=include_minus_m,
                integrate_backwards=self.inspiral_generator.inspiral_generator.integrate_backwards,
                **kwargs,
            )

            # if batching, need to add the waveform
            if i > 0:
                waveform = self.xp.concatenate([waveform, waveform_temp])

            # return entire waveform
            else:
                waveform = waveform_temp

        if dist is not None:
            dist_dimensionless = (dist * Gpc) / (mu * MRSUN_SI)

        else:
            dist_dimensionless = 1.0

        return waveform / dist_dimensionless



class AAKWaveformBase(Pn5AAK, ParallelModuleBase, ABC):
    """Waveform generation class for AAK with arbitrary trajectory.

    This class generates waveforms based on the Augmented Analytic Kludge
    given in the
    `EMRI Kludge Suite <https://github.com/alvincjk/EMRI_Kludge_Suite/>`_.
    The trajectory is chosen by user or by default in child classes.

    The trajectory is calculated until the orbit reaches
    within 0.1 of the separatrix, determined from
    `arXiv:1912.07609 <https://arxiv.org/abs/1912.07609/>`_. The
    fundamental frequencies along the trajectory at each point are then
    calculated from the orbital parameters and the spin value given by (`Schmidt 2002 <https://arxiv.org/abs/gr-qc/0202090>`_).

    These frequencies along the trajectory are then used to map to the
    frequency basis of the `Analytic Kludge <https://arxiv.org/abs/gr-qc/0310125>`_. This mapping
    takes the form of time evolving large mass and spin parameters, as
    well as the use of phases and frequencies in
    :math:`(alpha, \Phi, \gamma)`:

    .. math:: \Phi = \Phi_\phi,

    .. math:: \gamma = \Phi_\phi + \Phi_\Theta,

    .. math:: alpha = \Phi_\phi + \Phi_\Theta + \Phi_r.

    The frequencies in that basis are found by taking the time derivatives
    of each equation above.

    This class has GPU capabilities and works from the sparse trajectory
    methodoligy with cubic spine interpolation of the smoothly varying
    waveform quantities.

    **Please note:** the AAK waveform takes the parameter
    :math:`Y\equiv\cos{\iota}=L/\sqrt{L^2 + Q}` rather than :math:`x_I` as is accepted
    for relativistic waveforms and in the generic waveform interface discussed above.
    The generic waveform interface directly converts :math:`x_I` to :math:`Y`.

    args:
        inspiral_module (obj): Class object representing the module
            for creating the inspiral. This returns the phases and orbital
            parameters. See :ref:`trajectory-label`.
        sum_module (obj): Class object representing the module for summing the
            final waveform from the amplitude and phase information. See
            :ref:`summation-label`.
        inspiral_kwargs (dict, optional): Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}. This is stored as an attribute.
        sum_kwargs (dict, optional): Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        use_gpu (bool, optional): If True, use GPU resources. Default is False.
        num_threads (int, optional): Number of parallel threads to use in OpenMP.
            If :code:`None`, will not set the global variable :code:`OMP_NUM_THREADS`.
            Default is None.

    """

    def __init__(
        self,
        inspiral_module: object,
        sum_module: object,
        inspiral_kwargs: Optional[dict]=None,
        sum_kwargs: Optional[dict]=None,
        use_gpu: bool=False,
        num_threads: Optional[int]=None,
    ):
        if inspiral_kwargs is None:
            inspiral_kwargs = {}
        if sum_kwargs is None:
            sum_kwargs = {}

        ParallelModuleBase.__init__(self, use_gpu=use_gpu, num_threads=num_threads)
        Pn5AAK.__init__(self)

        sum_kwargs = self.adjust_gpu_usage(use_gpu, sum_kwargs)

        # kwargs that are passed to the inspiral call function
        self.inspiral_kwargs = inspiral_kwargs

        # function for generating the inpsiral
        self.inspiral_generator = inspiral_module(**inspiral_kwargs)

        # summation generator
        self.create_waveform = sum_module(**sum_kwargs)

    def attributes_AAKWaveform(self):
        """
        attributes:
            inspiral_generator (obj): instantiated trajectory module.
            create_waveform (obj): instantiated summation module.
            inspiral_kwargs (dict): Kwargs related to the inspiral.
            xp (obj): numpy or cupy based on gpu usage.
            num_modes_kept/nmodes (int): Number of modes for final waveform.
                For this model, it is solely determined from the
                eccentricity.


        """
        pass

    @property
    def citation(self):
        """Return citations related to this module"""
        return (
            larger_few_citation
            + few_citation
            + few_software_citation
            + fd_citation
            + AAK_citation_1
            + AAK_citation_2
            + AK_citation
            + kerr_separatrix_citation
        )

    @property
    def gpu_capability(self):
        return True

    @property
    def is_source_frame(self):
        return False

    @property
    def allow_batching(self):
        return False

    def __call__(
        self,
        M: float,
        mu: float,
        a: float,
        p0: float,
        e0: float,
        Y0: float,
        dist: float,
        qS: float,
        phiS: float,
        qK: float,
        phiK: float,
        *args: Optional[tuple],
        Phi_phi0: float=0.0,
        Phi_theta0: float=0.0,
        Phi_r0: float=0.0,
        mich: bool=False,
        dt: float=10.0,
        T: float=1.0,
        nmodes: Optional[int]=None,
    ) -> xp.ndarray:
        """Call function for AAK + 5PN model.

        This function will take input parameters and produce AAK waveforms with 5PN trajectories in generic Kerr.

        args:
            M (double): Mass of larger black hole in solar masses.
            mu (double): Mass of compact object in solar masses.
            a (double): Dimensionless spin of massive black hole.
            p0 (double): Initial semilatus rectum (Must be greater than
                the separatrix at the the given e0 and Y0).
                See documentation for more information.
            e0 (double): Initial eccentricity.
            Y0 (double): Initial cosine of :math:`\iota`. :math:`Y=\cos{\iota}\equiv L_z/\sqrt{L_z^2 + Q}`
                in the semi-relativistic formulation.
            dist (double): Luminosity distance in Gpc.
            qS (double): Sky location polar angle in ecliptic
                coordinates.
            phiS (double): Sky location azimuthal angle in
                ecliptic coordinates.
            qK (double): Initial BH spin polar angle in ecliptic
                coordinates.
            phiK (double): Initial BH spin azimuthal angle in
                ecliptic coordinates.
            *args (tuple, optional): Any additional arguments required for the
                trajectory.
            Phi_phi0 (double, optional): Initial phase for :math:`\Phi_\phi`.
                Default is 0.0.
            Phi_theta0 (double, optional): Initial phase for :math:`\Phi_\Theta`.
                Default is 0.0.
            Phi_r0 (double, optional): Initial phase for :math:`\Phi_r`.
                Default is 0.0.
            mich (bool, optional): If True, produce waveform with
                long-wavelength response approximation (hI, hII). Please
                note this is not TDI. If False, return hplus and hcross.
                Default is False.
            dt (double, optional): Time between samples in seconds
                (inverse of sampling frequency). Default is 10.0.
            T (double, optional): Total observation time in years.
                Default is 1.0.

        Returns:
            1D complex128 self.xp.ndarray: The output waveform.

        Raises:
            ValueError: user selections are not allowed.

        """

        # makes sure angular extrinsic parameters are allowable
        qS, phiS, qK, phiK = self.sanity_check_angles(qS, phiS, qK, phiK)

        self.sanity_check_init(M, mu, a, p0, e0, Y0)

        # get trajectory
        t, p, e, Y, Phi_phi, Phi_theta, Phi_r = self.inspiral_generator(
            M,
            mu,
            a,
            p0,
            e0,
            Y0,
            *args,
            Phi_phi0=Phi_phi0,
            Phi_theta0=Phi_theta0,
            Phi_r0=Phi_r0,
            T=T,
            dt=dt,
            **self.inspiral_kwargs,
        )

        if nmodes == None:
            if (p[0] - p[1]) < 0:  # Integrating backwards
                # Need to keep the number of modes equivalent
                initial_e = e[-1]
                self.num_modes_kept = self.nmodes = int(30 * initial_e)
            else:
                # number of modes to use (from original AAK model)
                self.num_modes_kept = self.nmodes = int(30 * e0)

            if self.num_modes_kept < 4:
                self.num_modes_kept = self.nmodes = 4
        else:
            self.num_modes_kept = self.nmodes = nmodes

        # makes sure p, Y, and e are generally within the model
        self.sanity_check_traj(p, e, Y)

        self.end_time = t[-1]

        # prepare phase spline coefficients
        traj_spline_coeff = self.inspiral_generator.inspiral_generator.integrator_spline_coeff  # TODO make these accessible from EMRIInspiral
        
        # scale coefficients here by the mass ratio
        traj_spline_coeff_in = traj_spline_coeff.copy()
        traj_spline_coeff_in[:,3:,:] /= (mu / M)

        if self.inspiral_generator.inspiral_generator.integrate_backwards:
            traj_spline_coeff_in[:,3:,0] += self.xp.array([Phi_phi[-1] + Phi_phi[0], Phi_theta[-1] + Phi_theta[0], Phi_r[-1] + Phi_r[0]])

        waveform = self.create_waveform(
            t,
            M,
            a,
            dist,
            mu,
            qS,
            phiS,
            qK,
            phiK,
            self.nmodes,
            self.inspiral_generator.inspiral_generator.integrator_t_cache,
            traj_spline_coeff_in,
            mich=mich,
            dt=dt,
            T=T,
            integrate_backwards=self.inspiral_generator.inspiral_generator.integrate_backwards,
        )

        return waveform
