# Main waveform class location

# Copyright (C) 2020 Michael L. Katz, Alvin J.K. Chua, Niels Warburton, Scott A. Hughes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
import os
from abc import ABC

import numpy as np
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline

# check if cupy is available / GPU is available
try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

from few.utils.baseclasses import SchwarzschildEccentric, Pn5AAK, ParallelModuleBase
from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.utils.utility import get_mismatch, xI_to_Y, p_to_y, check_for_file_download
from few.amplitude.romannet import RomanAmplitude
from few.utils.modeselector import ModeSelector
from few.utils.ylm import GetYlms
from few.summation.directmodesum import DirectModeSum
from few.summation.aakwave import AAKSummation
from few.utils.constants import *
from few.utils.citations import *
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.fdinterp import FDInterpolatedModeSum


class GenerateEMRIWaveform:
    """Generic waveform generator for data analysis

    This class allows the user interface to be the exact same between any
    waveform in the FEW package. For waveforms built in the source frame,
    like :class:`few.waveform.FastSchwarzschildEccentricFlux`, the waveforms
    are transformed to the detector frame. Waveforms like
    :class:`few.waveform.Pn5AAKWaveform`that are built in the detector frame
    are left alone effectively.

    For waveforms that are less than Kerr generic (i.e. certain parameters are
    unnecessary), this interface automatically removes the waveforms dependence
    on those parameters.

    Args:
        waveform_class (str or obj): String with the name of the waveform class to use.
            See the `pre-built waveform models
            <https://bhptoolkit.org/FastEMRIWaveforms/html/user/main.html#prebuilt-waveform-models>`_.
            If an object is provided, must be a waveform class.
        *args (list or tuple, optional): Arguments for the instantiation of
            the waveform generation class.
        frame (str, optional): Which frame to produce the output waveform in.
            Default is "detector." Right now, the source frame is not implemented
            for waveforms that are built in the detector frame.
        return_list (bool, optional): If True, return :math:`h_p` and
            :math:`h_x` as a list. If False, return :math:`hp - ihx`. Default
            is False.
        **kwargs (dict, optional): Dictionary with kwargs for the instantiation of
            the waveform generator.

    """

    def __init__(
        self, waveform_class, *args, frame="detector", return_list=False, **kwargs
    ):
        # instantiate the class
        if isinstance(waveform_class, str):
            try:
                waveform = globals()[waveform_class]
                self.waveform_generator = waveform(*args, **kwargs)
            except KeyError:
                raise ValueError(
                    "{} waveform class is not available.".format(waveform_class)
                )
        else:
            self.waveform_generator = waveform_class(*args, **kwargs)

        self.frame = frame

        self.return_list = return_list

        # setup arguments to remove based on the specific waveform
        # also get proper phases
        self.args_remove = []
        if self.waveform_generator.descriptor == "eccentric":
            self.args_remove.append(5)

            self.phases_needed = {"Phi_phi0": 11, "Phi_r0": 13}

        else:
            self.phases_needed = {"Phi_phi0": 11, "Phi_theta0": 12, "Phi_r0": 13}

        if self.waveform_generator.background == "Schwarzschild":
            # remove spin
            self.args_remove.append(2)

        # remove sky and orientation parameters
        if self.waveform_generator.frame == "source":
            for i in range(6, 11):
                self.args_remove.append(i)

        # these are the arguments that go in to the generator
        self.args_keep = np.delete(np.arange(11), self.args_remove)

    @property
    def stock_waveform_options(self):
        print(
            """
            FastSchwarzschildEccentricFlux
            SlowSchwarzschildEccentricFlux
            Pn5AAKWaveform
            """
        )

    def _get_viewing_angles(self, qS, phiS, qK, phiK):
        """Transform from the detector frame to the source frame"""

        cqS = np.cos(qS)
        sqS = np.sin(qS)

        cphiS = np.cos(phiS)
        sphiS = np.sin(phiS)

        cqK = np.cos(qK)
        sqK = np.sin(qK)

        cphiK = np.cos(phiK)
        sphiK = np.sin(phiK)

        # sky location vector
        R = np.array([sqS * cphiS, sqS * sphiS, cqS])

        # spin vector
        S = np.array([sqK * cphiK, sqK * sphiK, cqK])

        # get viewing angles
        phi = -np.pi / 2.0  # by definition of source frame

        theta = np.arccos(-np.dot(R, S))  # normalized vector

        return (theta, phi)

    def _to_SSB_frame(self, hp, hc, qS, phiS, qK, phiK):
        """Transform to SSB frame"""

        cqS = np.cos(qS)
        sqS = np.sin(qS)

        cphiS = np.cos(phiS)
        sphiS = np.sin(phiS)

        cqK = np.cos(qK)
        sqK = np.sin(qK)

        cphiK = np.cos(phiK)
        sphiK = np.sin(phiK)

        # get polarization angle

        up_ldc = cqS * sqK * np.cos(phiS - phiK) - cqK * sqS
        dw_ldc = sqK * np.sin(phiS - phiK)

        if dw_ldc != 0.0:
            psi_ldc = -np.arctan2(up_ldc, dw_ldc)

        else:
            psi_ldc = 0.5 * np.pi

        c2psi_ldc = np.cos(2.0 * psi_ldc)
        s2psi_ldc = np.sin(2.0 * psi_ldc)

        # rotate
        FplusI = c2psi_ldc
        FcrosI = -s2psi_ldc
        FplusII = s2psi_ldc
        FcrosII = c2psi_ldc

        hp_new = FplusI * hp + FcrosI * hc
        hc_new = FplusII * hp + FcrosII * hc

        return hp_new, hc_new

    def __call__(
        self,
        M,
        mu,
        a,
        p0,
        e0,
        x0,
        dist,
        qS,
        phiS,
        qK,
        phiK,
        Phi_phi0,
        Phi_theta0,
        Phi_r0,
        *add_args,
        **kwargs,
    ):
        """Generate the waveform with the given parameters.

        Args:
            M (double): Mass of larger black hole in solar masses.
            mu (double): Mass of compact object in solar masses.
            a (double): Dimensionless spin of massive black hole.
            p0 (double): Initial semilatus rectum (Must be greater than
                the separatrix at the the given e0 and x0).
                See documentation for more information on :math:`p_0<10`.
            e0 (double): Initial eccentricity.
            x0 (double): Initial cosine of the inclination angle.
                (:math:`x_I=\cos{I}`). This differs from :math:`Y=\cos{\iota}\equiv L_z/\sqrt{L_z^2 + Q}`
                used in the semi-relativistic formulation. When running kludge waveforms,
                :math:`x_{I,0}` will be converted to :math:`Y_0`.
            dist (double): Luminosity distance in Gpc.
            qS (double): Sky location polar angle in ecliptic
                coordinates.
            phiS (double): Sky location azimuthal angle in
                ecliptic coordinates.
            qK (double): Initial BH spin polar angle in ecliptic
                coordinates.
            phiK (double): Initial BH spin azimuthal angle in
                ecliptic coordinates.
            Phi_phi0 (double, optional): Initial phase for :math:`\Phi_\phi`.
                Default is 0.0.
            Phi_theta0 (double, optional): Initial phase for :math:`\Phi_\Theta`.
                Default is 0.0.
            Phi_r0 (double, optional): Initial phase for :math:`\Phi_r`.
                Default is 0.0.
            *args (tuple, optional): Tuple of any extra parameters that go into the model.
            **kwargs (dict, optional): Dictionary with kwargs for online waveform
                generation.

        """

        args_all = (
            M,
            mu,
            a,
            p0,
            e0,
            x0,
            dist,
            qS,
            phiS,
            qK,
            phiK,
            Phi_phi0,
            Phi_theta0,
            Phi_r0,
        )

        # if Y is needed rather than x (inclination definition)
        if (
            hasattr(self.waveform_generator, "needs_Y")
            and self.waveform_generator.background.lower() == "kerr"
            and self.waveform_generator.needs_Y
        ):
            x0 = xI_to_Y(a, p0, e0, x0)

        # remove the arguments that are not used in this waveform
        args = tuple([args_all[i] for i in self.args_keep])

        # pick out the phases to be used
        initial_phases = {key: args_all[i] for key, i in self.phases_needed.items()}

        # generate waveform in source frame
        if self.waveform_generator.frame == "source":
            # distance factor
            dist_dimensionless = (dist * Gpc) / (mu * MRSUN_SI)

            # get viewing angles in the source frame
            (
                theta_source,
                phi_source,
            ) = self._get_viewing_angles(qS, phiS, qK, phiK)

            args += (theta_source, phi_source)

        else:
            dist_dimensionless = 1.0

        # if output is to be in the source frame, need to properly scale with distance
        if self.frame == "source":
            if self.waveform_generator.frame == "detector":
                dist_dimensionless = 1.0 / ((dist * Gpc) / (mu * MRSUN_SI))
            else:
                dist_dimensionless = 1.0

        # add additional arguments to waveform interface
        args += add_args

        # get waveform
        h = (
            self.waveform_generator(*args, **{**initial_phases, **kwargs})
            / dist_dimensionless
        )

        # by definition of the source frame, need to rotate by pi
        if self.waveform_generator.frame == "source":
            h *= -1

        # transform to SSB frame if desired
        if self.waveform_generator.create_waveform.output_type == "td":
            if self.frame == "detector":
                hp, hc = self._to_SSB_frame(h.real, -h.imag, qS, phiS, qK, phiK)
            elif self.frame == "source":
                hp, hc = h.real, -h.imag

        # if FD, h is of length 2 rather than h+ - ihx
        if self.waveform_generator.create_waveform.output_type == "fd":
            if self.frame == "detector":
                hp, hc = self._to_SSB_frame(h[0], h[1], qS, phiS, qK, phiK)
            elif self.frame == "source":
                hp, hc = h[0], h[1]

        if self.return_list is False:
            return hp - 1j * hc
        else:
            return [hp, hc]


# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))


class SchwarzschildEccentricWaveformBase(
    SchwarzschildEccentric, ParallelModuleBase, ABC
):
    """Base class for the actual Schwarzschild eccentric waveforms.

    This class carries information and methods that are common to any
    implementation of Schwarzschild eccentric waveforms. These include
    initialization and the actual base code for building a waveform. This base
    code calls the various modules chosen by the user or according to the
    predefined waveform classes available. See
    :class:`few.utils.baseclasses.SchwarzschildEccentric` for information
    high level information on these waveform models.

    args:
        inspiral_module (obj): Class object representing the module
            for creating the inspiral. This returns the phases and orbital
            parameters. See :ref:`trajectory-label`.
        amplitude_module (obj): Class object representing the module for
            generating amplitudes. See :ref:`amplitude-label` for more
            information.
        sum_module (obj): Class object representing the module for summing the
            final waveform from the amplitude and phase information. See
            :ref:`summation-label`.
        inspiral_kwargs (dict, optional): Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}. This is stored as an attribute.
        amplitude_kwargs (dict, optional): Optional kwargs to pass to the
            amplitude generator during instantiation. Default is {}.
        sum_kwargs (dict, optional): Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        Ylm_kwargs (dict, optional): Optional kwargs to pass to the
            Ylm generator during instantiation. Default is {}.
        mode_selector_kwargs (dict, optional): Optional kwargs to pass to the
            mode selector during instantiation. Default is {}.
        use_gpu (bool, optional): If True, use GPU resources. Default is False.
        num_threads (int, optional): Number of parallel threads to use in OpenMP.
            If :code:`None`, will not set the global variable :code:`OMP_NUM_THREADS`.
            Default is None.
        normalize_amps (bool, optional): If True, it will normalize amplitudes
            to flux information output from the trajectory modules. Default
            is True. This is stored as an attribute.

    """

    def attributes_SchwarzschildEccentricWaveformBase(self):
        """
        attributes:
            inspiral_generator (obj): instantiated trajectory module.
            amplitude_generator (obj): instantiated amplitude module.
            ylm_gen (obj): instantiated ylm module.
            create_waveform (obj): instantiated summation module.
            ylm_gen (obj): instantiated Ylm module.
            mode_selector (obj): instantiated mode selection module.
            num_teuk_modes (int): number of Teukolsky modes in the model.
            ls, ms, ns (1D int xp.ndarray): Arrays of mode indices :math:`(l,m,n)`
                after filtering operation. If no filtering, these are equivalent
                to l_arr, m_arr, n_arr.
            xp (obj): numpy or cupy based on gpu usage.
            num_modes_kept (int): Number of modes for final waveform after mode
                selection.

        """
        pass

    def __init__(
        self,
        inspiral_module,
        amplitude_module,
        sum_module,
        inspiral_kwargs={},
        amplitude_kwargs={},
        sum_kwargs={},
        Ylm_kwargs={},
        mode_selector_kwargs={},
        use_gpu=False,
        num_threads=None,
        normalize_amps=True,
    ):
        ParallelModuleBase.__init__(self, use_gpu=use_gpu, num_threads=num_threads)
        SchwarzschildEccentric.__init__(self, use_gpu=use_gpu)

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
        self.mode_selector = ModeSelector(self.m0mask, **mode_selector_kwargs)

        # setup amplitude normalization
        fp = "AmplitudeVectorNorm.dat"
        few_dir = dir_path + "/../"
        check_for_file_download(fp, few_dir)

        y_in, e_in, norm = np.genfromtxt(
            few_dir + "/few/files/AmplitudeVectorNorm.dat"
        ).T

        num_y = len(np.unique(y_in))
        num_e = len(np.unique(e_in))

        self.amp_norm_spline = RectBivariateSpline(
            np.unique(y_in), np.unique(e_in), norm.reshape(num_e, num_y).T
        )

    @property
    def citation(self):
        """Return citations related to this module"""
        return (
            larger_few_citation
            + few_citation
            + few_software_citation
            + fd_citation
            + romannet_citation
            + FD_citation
        )

    def __call__(
        self,
        M,
        mu,
        p0,
        e0,
        theta,
        phi,
        *args,
        dist=None,
        Phi_phi0=0.0,
        Phi_r0=0.0,
        dt=10.0,
        T=1.0,
        eps=1e-5,
        show_progress=False,
        batch_size=-1,
        mode_selection=None,
        include_minus_m=True,
        **kwargs,
    ):
        """Call function for SchwarzschildEccentric models.

        This function will take input parameters and produce Schwarzschild
        eccentric waveforms. It will use all of the modules preloaded to
        compute desired outputs.

        args:
            M (double): Mass of larger black hole in solar masses.
            mu (double): Mass of compact object in solar masses.
            p0 (double): Initial semilatus rectum (:math:`10\leq p_0\leq16 + e_0`).
                See documentation for more information on :math:`p_0<10`.
            e0 (double): Initial eccentricity (:math:`0.0\leq e_0\leq0.7`).
            theta (double): Polar viewing angle (:math:`-\pi/2\leq\Theta\leq\pi/2`).
            phi (double): Azimuthal viewing angle.
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
            1D complex128 xp.ndarray: The output waveform.

        Raises:
            ValueError: user selections are not allowed.

        """

        if self.use_gpu:
            xp = cp
        else:
            xp = np

        # makes sure viewing angles are allowable
        theta, phi = self.sanity_check_viewing_angles(theta, phi)
        self.sanity_check_init(M, mu, p0, e0)

        # get trajectory
        (t, p, e, x, Phi_phi, Phi_theta, Phi_r) = self.inspiral_generator(
            M,
            mu,
            0.0,
            p0,
            e0,
            1.0,
            *args,
            Phi_phi0=Phi_phi0,
            Phi_theta0=0.0,
            Phi_r0=Phi_r0,
            T=T,
            dt=dt,
            **self.inspiral_kwargs,
        )

        # makes sure p and e are generally within the model
        self.sanity_check_traj(p, e)

        # get the vector norm
        amp_norm = self.amp_norm_spline.ev(p_to_y(p, e), e)

        self.end_time = t[-1]
        # convert for gpu
        t = xp.asarray(t)
        p = xp.asarray(p)
        e = xp.asarray(e)
        Phi_phi = xp.asarray(Phi_phi)
        Phi_r = xp.asarray(Phi_r)
        amp_norm = xp.asarray(amp_norm)

        # get ylms only for unique (l,m) pairs
        # then expand to all (lmn with self.inverse_lm)
        ylms = self.ylm_gen(self.unique_l, self.unique_m, theta, phi).copy()[
            self.inverse_lm
        ]

        # split into batches

        if batch_size == -1 or self.allow_batching is False:
            inds_split_all = [xp.arange(len(t))]
        else:
            split_inds = []
            i = 0
            while i < len(t):
                i += batch_size
                if i >= len(t):
                    break
                split_inds.append(i)

            inds_split_all = xp.split(xp.arange(len(t)), split_inds)

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
            Phi_r_temp = Phi_r[inds_in]
            amp_norm_temp = amp_norm[inds_in]

            # amplitudes
            teuk_modes = self.amplitude_generator(p_temp, e_temp)

            # normalize by flux produced in trajectory
            if self.normalize_amps:
                amp_for_norm = xp.sum(
                    xp.abs(
                        xp.concatenate(
                            [teuk_modes, xp.conj(teuk_modes[:, self.m0mask])],
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

                    keep_modes = xp.arange(teuk_modes.shape[1])
                    temp2 = keep_modes * (keep_modes < self.num_m0) + (
                        keep_modes + self.num_m_1_up
                    ) * (keep_modes >= self.num_m0)

                    ylmkeep = xp.concatenate([keep_modes, temp2])
                    ylms_in = ylms[ylmkeep]
                    teuk_modes_in = teuk_modes

                else:
                    raise ValueError("If mode selection is a string, must be `all`.")

            # get a specific subset of modes
            elif isinstance(mode_selection, list):
                if mode_selection == []:
                    raise ValueError("If mode selection is a list, cannot be empty.")

                keep_modes = xp.zeros(len(mode_selection), dtype=xp.int32)

                # for removing opposite m modes
                fix_include_ms = xp.full(2 * len(mode_selection), False)
                for jj, lmn in enumerate(mode_selection):
                    l, m, n = tuple(lmn)

                    # keep modes only works with m>=0
                    lmn_in = (l, abs(m), n)
                    keep_modes[jj] = xp.int32(self.lmn_indices[lmn_in])

                    if not include_minus_m:
                        if m > 0:
                            # minus m modes blocked
                            fix_include_ms[len(mode_selection) + jj] = True
                        elif m < 0:
                            # positive m modes blocked
                            fix_include_ms[jj] = True

                self.ls = self.l_arr[keep_modes]
                self.ms = self.m_arr[keep_modes]
                self.ns = self.n_arr[keep_modes]

                temp2 = keep_modes * (keep_modes < self.num_m0) + (
                    keep_modes + self.num_m_1_up
                ) * (keep_modes >= self.num_m0)

                ylmkeep = xp.concatenate([keep_modes, temp2])
                ylms_in = ylms[ylmkeep]

                # remove modes if include_minus_m is False
                ylms_in[fix_include_ms] = 0.0 + 1j * 0.0

                teuk_modes_in = teuk_modes[:, keep_modes]

            # mode selection based on input module
            else:
                fund_freq_args = (
                    M,
                    0.0,
                    p_temp,
                    e_temp,
                    xp.zeros_like(e_temp),
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

            # create waveform
            waveform_temp = self.create_waveform(
                t_temp,
                teuk_modes_in,
                ylms_in,
                Phi_phi_temp,
                Phi_r_temp,
                self.ms,
                self.ns,
                M,
                p,
                e,
                dt=dt,
                T=T,
                include_minus_m=include_minus_m,
                **kwargs,
            )

            # if batching, need to add the waveform
            if i > 0:
                waveform = xp.concatenate([waveform, waveform_temp])

            # return entire waveform
            else:
                waveform = waveform_temp

        if dist is not None:
            dist_dimensionless = (dist * Gpc) / (mu * MRSUN_SI)

        else:
            dist_dimensionless = 1.0

        return waveform / dist_dimensionless


class FastSchwarzschildEccentricFlux(SchwarzschildEccentricWaveformBase):
    """Prebuilt model for fast Schwarzschild eccentric flux-based waveforms.

    This model combines the most efficient modules to produce the fastest
    accurate EMRI waveforms. It leverages GPU hardware for maximal acceleration,
    but is also available on for CPUs. Please see
    :class:`few.utils.baseclasses.SchwarzschildEccentric` for general
    information on this class of models.

    The trajectory module used here is :class:`few.trajectory.flux` for a
    flux-based, sparse trajectory. This returns approximately 100 points.

    The amplitudes are then determined with
    :class:`few.amplitude.romannet.RomanAmplitude` along these sparse
    trajectories. This gives complex amplitudes for all modes in this model at
    each point in the trajectory. These are then filtered with
    :class:`few.utils.modeselector.ModeSelector`.

    The modes that make it through the filter are then summed by
    :class:`few.summation.interpolatedmodesum.InterpolatedModeSum`.

    See :class:`few.waveform.SchwarzschildEccentricWaveformBase` for information
    on inputs. See examples as well.

    args:
        inspiral_kwargs (dict, optional): Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}.
        amplitude_kwargs (dict, optional): Optional kwargs to pass to the
            amplitude generator during instantiation. Default is {}.
        sum_kwargs (dict, optional): Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        Ylm_kwargs (dict, optional): Optional kwargs to pass to the
            Ylm generator during instantiation. Default is {}.
        use_gpu (bool, optional): If True, use GPU resources. Default is False.
        *args (list, placeholder): args for waveform model.
        **kwargs (dict, placeholder): kwargs for waveform model.

    """

    def __init__(
        self,
        inspiral_kwargs={},
        amplitude_kwargs={},
        sum_kwargs={},
        Ylm_kwargs={},
        use_gpu=False,
        *args,
        **kwargs,
    ):
        inspiral_kwargs["func"] = "SchwarzEccFlux"

        if "output_type" in sum_kwargs:
            if sum_kwargs["output_type"] == "fd":
                mode_summation_module = FDInterpolatedModeSum

            else:
                mode_summation_module = InterpolatedModeSum

        else:
            mode_summation_module = InterpolatedModeSum

        SchwarzschildEccentricWaveformBase.__init__(
            self,
            EMRIInspiral,
            RomanAmplitude,
            mode_summation_module,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            use_gpu=use_gpu,
            *args,
            **kwargs,
        )

    def attributes_FastSchwarzschildEccentricFlux(self):
        """
        Attributes:
            gpu_capability (bool): If True, this wavefrom can leverage gpu
                resources. For this class it is True.
            allow_batching (bool): If True, this waveform can use the batch_size
                kwarg. For this class it is False.

        """
        pass

    @property
    def gpu_capability(self):
        return True

    @property
    def allow_batching(self):
        return False


class SlowSchwarzschildEccentricFlux(SchwarzschildEccentricWaveformBase):
    """Prebuilt model for slow Schwarzschild eccentric flux-based waveforms.

    This model combines the various modules to produce the a reference waveform
    against which we test our fast models. Please see
    :class:`few.utils.baseclasses.SchwarzschildEccentric` for general
    information on this class of models.

    The trajectory module used here is :class:`few.trajectory.flux` for a
    flux-based trajectory. For this slow waveform, the DENSE_SAMPLING parameter
    from :class:`few.utils.baseclasses.TrajectoryBase` is fixed to 1 to create
    a densely sampled trajectory.

    The amplitudes are then determined with
    :class:`few.amplitude.interp2dcubicspline.Interp2DAmplitude`
    along a densely sampled trajectory. This gives complex amplitudes
    for all modes in this model at each point in the trajectory. These, can be
    chosent to be filtered, but for reference waveforms, they should not be.

    The modes that make it through the filter are then summed by
    :class:`few.summation.directmodesum.DirectModeSum`.

    See :class:`few.waveform.SchwarzschildEccentricWaveformBase` for information
    on inputs. See examples as well.

    args:
        inspiral_kwargs (dict, optional): Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}.
        amplitude_kwargs (dict, optional): Optional kwargs to pass to the
            amplitude generator during instantiation. Default is {}.
        sum_kwargs (dict, optional): Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        Ylm_kwargs (dict, optional): Optional kwargs to pass to the
            Ylm generator during instantiation. Default is {}.
        use_gpu (bool, optional): If True, use GPU resources. Default is False.
        *args (list, placeholder): args for waveform model.
        **kwargs (dict, placeholder): kwargs for waveform model.

    """

    @property
    def gpu_capability(self):
        return False

    @property
    def allow_batching(self):
        return True

    def attributes_SlowSchwarzschildEccentricFlux(self):
        """
        attributes:
            gpu_capability (bool): If True, this wavefrom can leverage gpu
                resources. For this class it is False.
            allow_batching (bool): If True, this waveform can use the batch_size
                kwarg. For this class it is True.
        """
        pass

    def __init__(
        self,
        inspiral_kwargs={},
        amplitude_kwargs={},
        sum_kwargs={},
        Ylm_kwargs={},
        use_gpu=False,
        *args,
        **kwargs,
    ):
        # declare specific properties
        inspiral_kwargs["DENSE_STEPPING"] = 1
        inspiral_kwargs["func"] = "SchwarzEccFlux"

        SchwarzschildEccentricWaveformBase.__init__(
            self,
            EMRIInspiral,
            Interp2DAmplitude,
            DirectModeSum,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            use_gpu=use_gpu,
            *args,
            **kwargs,
        )


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
        inspiral_module,
        sum_module,
        inspiral_kwargs={},
        sum_kwargs={},
        use_gpu=False,
        num_threads=None,
    ):
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
        M,
        mu,
        a,
        p0,
        e0,
        Y0,
        dist,
        qS,
        phiS,
        qK,
        phiK,
        *args,
        Phi_phi0=0.0,
        Phi_theta0=0.0,
        Phi_r0=0.0,
        mich=False,
        dt=10.0,
        T=1.0,
    ):
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
            1D complex128 xp.ndarray: The output waveform.

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

        # makes sure p, Y, and e are generally within the model
        self.sanity_check_traj(p, e, Y)

        self.end_time = t[-1]

        # number of modes to use (from original AAK model)
        self.num_modes_kept = self.nmodes = int(30 * e0)
        if self.num_modes_kept < 4:
            self.num_modes_kept = self.nmodes = 4

        waveform = self.create_waveform(
            t,
            M,
            a,
            p,
            e,
            Y,
            dist,
            Phi_phi,
            Phi_theta,
            Phi_r,
            mu,
            qS,
            phiS,
            qK,
            phiK,
            self.nmodes,
            mich=mich,
            dt=dt,
            T=T,
        )

        return waveform


class Pn5AAKWaveform(AAKWaveformBase, Pn5AAK, ParallelModuleBase, ABC):
    """Waveform generation class for AAK with 5PN trajectory.

    This class generates waveforms based on the Augmented Analytic Kludge
    given in the
    `EMRI Kludge Suite <https://github.com/alvincjk/EMRI_Kludge_Suite/>`_.
    However, here the trajectory is vastly improved by employing the 5PN
    fluxes for generic Kerr orbits from
    `Fujita & Shibata 2020 <https://arxiv.org/abs/2008.13554>`_.

    The 5PN trajectory produces orbital and phase trajectories.
    The trajectory is calculated until the orbit reaches
    within 0.2 of the separatrix, determined from
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
    waveform quantities. This waveform does not have the freedom in terms
    of user-chosen quantitites that
    :class:`few.waveform.SchwarzschildEccentricWaveformBase` contains.
    This is mainly due to the specific waveform constructions particular
    to the AAK/AK.

    **Please note:** the 5PN trajectory and AAK waveform take the parameter
    :math:`Y\equiv\cos{\iota}=L/\sqrt{L^2 + Q}` rather than :math:`x_I` as is accepted
    for relativistic waveforms and in the generic waveform interface discussed above.
    The generic waveform interface directly converts :math:`x_I` to :math:`Y`.

    args:
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
        self, inspiral_kwargs={}, sum_kwargs={}, use_gpu=False, num_threads=None
    ):
        inspiral_kwargs["func"] = "pn5"

        AAKWaveformBase.__init__(
            self,
            EMRIInspiral,
            AAKSummation,
            inspiral_kwargs=inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=use_gpu,
            num_threads=num_threads,
        )

    def attributes_Pn5AAKWaveform(self):
        """
        attributes:
            inspiral_generator (obj): instantiated trajectory module.
            create_waveform (obj): instantiated summation module.
            inspiral_kwargs (dict): Kwargs related to the inspiral class:
                :class:`few.trajectory.pn5.RunKerrGenericPn5Inspiral`.
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
            + Pn5_citation
            + kerr_separatrix_citation
        )
