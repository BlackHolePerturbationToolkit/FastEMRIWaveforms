from abc import ABC
import warnings

try:
    import cupy as xp
except:
    import numpy as xp

import numpy as np

# TODO: get bounds on p


class SchwarzschildEccentric(ABC):
    """Base class for Schwarzschild eccentric waveforms.

    This class creates shared traits between different implementations of the
    same model. Particularly, this class includes descriptive traits as well as
    the sanity check class method that should be used in all implementations of
    this model. This method can be overwritten if necessary. Here we describe
    the overall qualities of this base class.

    The user inputs orbital parameter trajectories and is returned the complex
    amplitudes of each harmonic mode, :math:`A_{lmn}`, given by,

    .. math:: A_{lmn}=-2Z_{lmn}/\omega_{mn}^2,

    where :math:`Z_{lmn}` and :math:`\omega_{mn}` are functions of the
    orbital paramters. :math:`l` ranges from 2 to 10; :math:`m` from :math:`-l` to :math:`l`;
    and :math:`n` from -30 to 30. This is for Schwarzschild eccentric.
    The model validity ranges from (TODO: add limits).

    args:
        use_gpu (bool, optional): If True, will allocate arrays on the GPU.
            Default is False.

    attributes:
        xp (module): numpy or cupy based on hardware chosen.
        background (str): Spacetime background for this model.
        descriptor (str): Short description for model validity.
        num_modes, num_teuk_modes (int): Total number of Tuekolsky modes
            in the model.
        lmax, nmax (int): Maximum :math:`l`, :math:`n`  values

    """

    def __init__(self, use_gpu=False, **kwargs):

        if use_gpu is True:
            self.xp = xp
        else:
            self.xp = np
        self.background = "Schwarzschild"
        self.descriptor = "eccentric"

        self.num_teuk_modes = 3843
        self.num_modes = 3843

        self.lmax = 10
        self.nmax = 30

    def sanity_check_traj(self, p, e):
        """Sanity check on parameters output from thte trajectory module.

        Make sure parameters are within allowable ranges.

        args:
            p (1D np.ndarray): Array of semi-latus rectum values produced by
                the trajectory module.
            e (1D np.ndarray): Array of eccentricity values produced by
                the trajectory module.

        Raises:
            ValueError: If any of the trajectory points are not allowed.
            UserWarning: If any points in the trajectory are allowable,
                but outside calibration region.

        """
        if e[-1] > 0.5:
            warnings.UserWarning(
                "Plunge (or final) eccentricity value above 0.5 is outside of calibration for this model."
            )

        if self.xp.any(e < 0.0):
            raise ValueError("Members of e array are less than zero.")

        if self.xp.any(p < 0.0):
            raise ValueError("Members of p array are less than zero.")

    def sanity_check_init(self, p0, e0, M, mu, theta, phi):
        """Sanity check initial parameters.

        Make sure parameters are within allowable ranges.

        args:
            p0 (double): Initial semilatus rectum in units of M. TODO: Fix this. :math:`(\leq e0\leq0.7)`
            e0 (double): Initial eccentricity :math:`(0\leq e0\leq0.7)`
            M (double): Massive black hole mass in solar masses.
            mu (double): compact object mass in solar masses.
            theta (double): Polar viewing angle.
            phi (double): Azimuthal viewing angle.

        Raises:
            ValueError: If any of the parameters are not allowed.

        """

        # TODO: add stuff
        if e0 > 0.7:
            raise ValueError(
                "Initial eccentricity above 0.7 not allowed. (e0={})".format(e0)
            )

        if e0 < 0.0:
            raise ValueError(
                "Initial eccentricity below 0.0 not physical. (e0={})".format(e0)
            )

        if mu / M > 1e-4:
            warnings.UserWarning(
                "Mass ratio is outside of generally accepted range for an extreme mass ratio (1e-4). (q={})".format(
                    mu / M
                )
            )
