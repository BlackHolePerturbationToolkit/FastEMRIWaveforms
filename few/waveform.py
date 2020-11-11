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

# check if cupy is available / GPU is available
try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

from few.utils.baseclasses import SchwarzschildEccentric, Pn5AAK
from few.trajectory.pn5 import RunKerrGenericPn5Inspiral
from few.trajectory.flux import RunSchwarzEccFluxInspiral
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.utils.utility import get_mismatch
from few.amplitude.romannet import RomanAmplitude
from few.utils.modeselector import ModeSelector
from few.utils.ylm import GetYlms
from few.summation.directmodesum import DirectModeSum
from few.summation.aakwave import AAKSummation
from few.utils.constants import *
from few.utils.citations import *
from few.summation.interpolatedmodesum import InterpolatedModeSum


class SchwarzschildEccentricWaveformBase(SchwarzschildEccentric, ABC):
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
        use_gpu (bool, optional): If True, use GPU resources. Default is False.
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
        use_gpu=False,
        normalize_amps=True,
    ):

        SchwarzschildEccentric.__init__(self, use_gpu)

        amplitude_kwargs, sum_kwargs = self.adjust_gpu_usage(
            use_gpu, [amplitude_kwargs, sum_kwargs]
        )

        # normalize amplitudes to flux at each step from trajectory
        self.normalize_amps = normalize_amps

        # kwargs that are passed to the inspiral call function
        self.inspiral_kwargs = inspiral_kwargs

        # function for generating the inpsiral
        self.inspiral_generator = inspiral_module()

        # function for generating the amplitude
        self.amplitude_generator = amplitude_module(**amplitude_kwargs)

        # summation generator
        self.create_waveform = sum_module(**sum_kwargs)

        # angular harmonics generation
        self.ylm_gen = GetYlms(use_gpu=use_gpu, **Ylm_kwargs)

        # selecting modes that contribute at threshold to the waveform
        self.mode_selector = ModeSelector(self.m0mask, use_gpu=use_gpu)

    @property
    def citation(self):
        """Return citations related to this module"""
        return few_citation + few_software_citation + romannet_citation

    def __call__(
        self,
        M,
        mu,
        p0,
        e0,
        theta,
        phi,
        dist,
        dt=10.0,
        T=1.0,
        eps=1e-5,
        show_progress=False,
        batch_size=-1,
        mode_selection=None,
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
            dist (double): Luminosity distance in Gpc.
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

        Returns:
            1D complex128 xp.ndarray: The output waveform.

        Raises:
            ValueError: user selections are not allowed.

        """

        # makes sure viewing angles are allowable
        theta, phi = self.sanity_check_viewing_angles(theta, phi)
        self.sanity_check_init(M, mu, p0, e0)

        # get trajectory
        (t, p, e, Phi_phi, Phi_r, amp_norm) = self.inspiral_generator(
            M, mu, p0, e0, T=T, dt=dt, **self.inspiral_kwargs
        )

        # makes sure p and e are generally within the model
        self.sanity_check_traj(p, e)

        self.end_time = t[-1]
        # convert for gpu
        t = self.xp.asarray(t)
        p = self.xp.asarray(p)
        e = self.xp.asarray(e)
        Phi_phi = self.xp.asarray(Phi_phi)
        Phi_r = self.xp.asarray(Phi_r)
        amp_norm = self.xp.asarray(amp_norm)

        # get ylms only for unique (l,m) pairs
        # then expand to all (lmn with self.inverse_lm)
        ylms = self.ylm_gen(self.unique_l, self.unique_m, theta, phi).copy()[
            self.inverse_lm
        ]

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
            Phi_r_temp = Phi_r[inds_in]
            amp_norm_temp = amp_norm[inds_in]

            # amplitudes
            teuk_modes = self.amplitude_generator(p_temp, e_temp)

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
            elif isinstance(mode_selection, list):
                if mode_selection == []:
                    raise ValueError("If mode selection is a list, cannot be empty.")

                keep_modes = self.xp.zeros(len(mode_selection), dtype=self.xp.int32)
                for jj, lmn in enumerate(mode_selection):
                    keep_modes[jj] = self.xp.int32(self.lmn_indices[tuple(lmn)])

                self.ls = self.l_arr[keep_modes]
                self.ms = self.m_arr[keep_modes]
                self.ns = self.n_arr[keep_modes]

                temp2 = keep_modes * (keep_modes < self.num_m0) + (
                    keep_modes + self.num_m_1_up
                ) * (keep_modes >= self.num_m0)

                ylmkeep = self.xp.concatenate([keep_modes, temp2])
                ylms_in = ylms[ylmkeep]
                teuk_modes_in = teuk_modes[:, keep_modes]

            # mode selection based on input module
            else:
                modeinds = [self.l_arr, self.m_arr, self.n_arr]
                (
                    teuk_modes_in,
                    ylms_in,
                    self.ls,
                    self.ms,
                    self.ns,
                ) = self.mode_selector(teuk_modes, ylms, modeinds, eps=eps)

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
                dt=dt,
                T=T,
            )

            # if batching, need to add the waveform
            if i > 0:
                waveform = self.xp.concatenate([waveform, waveform_temp])

            # return entire waveform
            else:
                waveform = waveform_temp

        dist_dimensionless = (dist * Gpc) / (mu * MRSUN_SI)
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
        **kwargs
    ):

        SchwarzschildEccentricWaveformBase.__init__(
            self,
            RunSchwarzEccFluxInspiral,
            RomanAmplitude,
            InterpolatedModeSum,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            use_gpu=use_gpu,
            *args,
            **kwargs
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
        **kwargs
    ):

        # declare specific properties
        inspiral_kwargs["DENSE_STEPPING"] = 1

        SchwarzschildEccentricWaveformBase.__init__(
            self,
            RunSchwarzEccFluxInspiral,
            Interp2DAmplitude,
            DirectModeSum,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            use_gpu=use_gpu,
            *args,
            **kwargs
        )


class Pn5AAKWaveform(Pn5AAK, ABC):
    """Waveform generation class for AAK with 5PN trajectory.

    This class generates waveforms based on the Augmented Analytic Kludge
    given in the
    `EMRI Kludge Suite <https://github.com/alvincjk/EMRI_Kludge_Suite/>`_.
    However, here the trajectory is vastly improved by employing the 5PN
    fluxes for generic Kerr orbits from
    `Fujita & Shibata 2020<https://arxiv.org/abs/2008.13554>`_.

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


    args:
        inspiral_kwargs (dict, optional): Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}. This is stored as an attribute.
        sum_kwargs (dict, optional): Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        use_gpu (bool, optional): If True, use GPU resources. Default is False.

    """

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

    def __init__(self, inspiral_kwargs={}, sum_kwargs={}, use_gpu=False):

        Pn5AAK.__init__(self, use_gpu)

        sum_kwargs = self.adjust_gpu_usage(use_gpu, sum_kwargs)

        # kwargs that are passed to the inspiral call function
        self.inspiral_kwargs = inspiral_kwargs

        # function for generating the inpsiral
        self.inspiral_generator = RunKerrGenericPn5Inspiral(**inspiral_kwargs)

        # summation generator
        self.create_waveform = AAKSummation(**sum_kwargs)

    @property
    def citation(self):
        """Return citations related to this module"""
        return (
            few_citation
            + few_software_citation
            + AAK_citation_1
            + AAK_citation_2
            + AK_citation
            + Pn5_citation
            + kerr_separatrix_citation
        )

    @property
    def gpu_capability(self):
        return True

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
        qS,
        phiS,
        qK,
        phiK,
        dist,
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
            p0 (double): Initial semilatus rectum (Must be greater than
                the separatrix at the the given e0 and Y0).
                See documentation for more information on :math:`p_0<10`.
            e0 (double): Initial eccentricity.
            Y0 (double): Initial cosine of the inclination angle
                (:math:`\cos{\iota}`).
            qS (double): Sky location polar angle in ecliptic
                coordinates.
            phiS (double): Sky location azimuthal angle in
                ecliptic coordinates.
            qK (double): Initial BH spin polar angle in ecliptic
                coordinates.
            phiK (double): Initial BH spin azimuthal angle in
                ecliptic coordinates.
            dist (double): Luminosity distance in Gpc.
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
            Phi_phi0=Phi_phi0,
            Phi_theta0=Phi_theta0,
            Phi_r0=Phi_r0,
            T=T,
            dt=dt,
            **self.inspiral_kwargs
        )

        # makes sure p, Y, and e are generally within the model
        self.sanity_check_traj(p, e, Y)

        self.end_time = t[-1]

        # number of modes to use (from original AAK model)
        self.num_modes_kept = self.nmodes = int(30 * e0)

        waveform = self.create_waveform(
            t,
            M,
            a,
            p,
            e,
            Y,
            Phi_phi,
            Phi_theta,
            Phi_r,
            mu,
            qS,
            phiS,
            qK,
            phiK,
            dist,
            self.nmodes,
            mich=mich,
            dt=dt,
            T=T,
        )

        return waveform
