from numba import jit
from multispline.spline import CubicSpline
import numpy as np
import h5py

from typing import Union, Optional

from ...utils.globals import get_file_manager, get_logger
from ...utils.mappings.common import chi_to_chit, chit1_nu_to_chi1
from ...utils.geodesic import get_separatrix
from .base import ODEBase

@jit
def dEdOmega0PA(p):
    """Derivative of the specific binding energy with respect to the orbital
    frequency at adiabatic order for a circular orbit in a Schwarzschild background.

    Args:
        p (float): Dimensionless semi-latus rectum p/M, with M the total mass.

    Returns:
        float: dE_0PA/dOmega_phi.
    """
    return p**(1./2.)*(-1.+6./p)/(3.*(1.-3./p)**(3./2.))

@jit
def dEdOmega_1PA_chit1(p):
    """Linear-in-chit1 1PA correction to dE/dOmega_phi.

    Coefficient of chit1 in the 1PA expansion
    of dE/dOmega_phi for a circular orbit in a Schwarzschild background, at
    fixed symmetric mass ratio and total mass.

    Args:
        p (float): Dimensionless semi-latus rectum p/M, with M the total mass.

    Returns:
        float: (dE/dOmega_phi)^(1, delta_chi1), Ref. 2510.16113.
    """
    return -2./p*(10.-33./p+36./p**2.)/(9.*(1.-3./p)**(5./2.))

@jit
def dEdOmega_1PA_chit2(p):
    """Linear-in-chit2 1PA correction to dE/dOmega_phi.

    Coefficient of chit2 in the 1PA expansion
    of dE/dOmega_phi for a circular orbit in a Schwarzschild background, at
    fixed symmetric mass ratio and total mass.

    Args:
        p (float): Dimensionless semi-latus rectum p/M, with M the total mass.

    Returns:
        float: (dE/dOmega_phi)^(1, delta_chi2), Ref. 2510.16113.
    """
    return (12./p-5.)/(3.*p*(1.-3./p)**(3./2.))

@jit
def dOmegadp(p):
    """Derivative of the (geodesically-related) orbital frequency with respect to the semi-latus rectum.

    For a Schwarzschild circular orbit Omega_phi = p^(-3/2).

    Args:
        p (float): Dimensionless semi-latus rectum p/M, with M the total mass.

    Returns:
        float: dOmega_phi/dp = -3/2 * p^(-5/2).
    """
    return -3./2. * p**(-5./2.)

@jit
def dEdeltaMdOmega(p):
    """1PA correction to dE/dOmega_phi from the evolution of the primary's mass.

    Coefficient of deltaM in the 1PA expansion of dE/dOmega_phi,
    Ref. 2510.16113.

    Args:
        p (float): Dimensionless semi-latus rectum p/M, with M the total mass.

    Returns:
        float: (dE/dOmega_phi)^(1, deltaM), Ref. 2510.16113.
    """
    return p**(1./2.)*(-2.+21./p-18./p**2.)/(9.*(1.-3./p)**(5./2.))

@jit
def EdeltaM(p):
    """Coefficient multiplying deltaM in Eq. (87) of Ref. 2510.16113.

    Args:
        p (float): Dimensionless semi-latus rectum p/M, with M the total mass.

    Returns:
        float: E_(deltaM)(p).
    """
    return -(1-6./p)/(3.*p*(1.-3./p)**(3./2.))

class TrajectoryCirc1PAT1R(ODEBase):
    """Trajectory of the first-post-adiabatic, slowly-spinning circular inspiral model 1PAT1R of Ref. 2510.16113.

    Circular equatorial inspiral of an arbitrary spinning secondary (chit2) BH
    around a slowly-spinning primary (chit1), at first post-adiabatic order.
    The model also includes 1PA corrections due to the evolution of the
    primary's mass and spin.

    The state vector has 8 components:

        y = [p, e, xI, Phi_phi, Phi_theta, Phi_r, deltaM, delta_chit1]

    where e = 0 and xI = 1 (circular, equatorial motion).  The extra slots
    deltaM and delta_chit1 are the deviations of the total mass and reduced
    primary spin from their initial values.

    The flux interpolants are read from the HDF5 data file
    ``Trajectory1PAT1R.h5``.

    Args:
        *args
        downsample (dict, optional): Dictionary of integer downsampling factors
            for each flux grid, keyed by grid name.  Useful for testing
            convergence.  Defaults to ``None`` (no downsampling). Get the
            default dictionary of downsampling factors with class attribute
            ``default_downsample``.
        evolve_primary (bool, optional): Whether to evolve the primary's mass and spin.
            Defaults to ``True``. If ``False``, the primary's mass and spin are held
            fixed at their initial values.
        **kwargs:


    Raises:
        ValueError: If any downsampling value is not an integer.
        ValueError: If a downsampling key is not in the set of grid names.
        ValueError: If the loaded flux data contains NaN values.
    """

    def __init__(
        self,
        *args,
        downsample=None,
        evolve_primary=True,
        **kwargs
    ):
        super().__init__(*args, downsample=downsample, **kwargs)

        fp = "Trajectory1PAT1R.h5"
        fm = get_file_manager()
        file_path = fm.get_file(fp)

        self.interpolation_keys = ["Flux/Energy/0PA/Infinity", "Flux/Energy/0PA/Horizon", "Flux/AngularMomentum/0PA/Horizon", "Flux/Energy/1PA/Infinity", "Flux/Energy/1PAchi1/Infinity", "Flux/Energy/1PAchi1/Horizon", "Flux/Energy/1PAchi2/Infinity", "Flux/Energy/1PAchi2/Horizon", "Energy/1PA/dEdOmega"]
        self.default_downsample = {key: 1 for key in self.interpolation_keys}
        self.interpolant = {}
        
        if downsample is None:
            downsample = self.default_downsample
        else:
            downsample = {**self.default_downsample, **downsample}

        if not all(isinstance(v, int) for v in downsample.values()):
            raise ValueError("(TrajectoryCirc1PAT1R) All downsample values must be integers.")
        if not all(k in self.default_downsample for k in downsample.keys()):
            raise ValueError(f"(TrajectoryCirc1PAT1R) Downsample keys must be in the default_downsample keys {self.default_downsample.keys()}.")

        min_p_grid = 0
        max_p_grid = np.inf
        with h5py.File(file_path, "r") as trajectoryData:
            for key in self.interpolation_keys:
                grid = trajectoryData[key.rsplit('/', 1)[0] + '/Grid'][::downsample[key]]
                value = trajectoryData[key][()][::downsample[key]]

                # check whether there are no nans in the grids and values
                if (
                    np.isnan(grid).any()
                    or np.isnan(value).any()
                ):
                    raise ValueError(f"(TrajectoryCirc1PAT1R) Interpolation: nans in grids or values of interpolating functions of {key}")
                min_p_grid = max(min_p_grid, np.min(grid))
                max_p_grid = min(max_p_grid, np.max(grid))
                self.interpolant[key] = CubicSpline(grid, value)
                del grid, value # free memory

            # Add derivative of 0PA flux manually
            grid = trajectoryData['Flux/Energy/0PA/Grid'][::max(downsample["Flux/Energy/0PA/Infinity"], downsample["Flux/Energy/0PA/Horizon"])]
            value = trajectoryData['Flux/Energy/0PA/Infinity'][()][::max(downsample["Flux/Energy/0PA/Infinity"], downsample["Flux/Energy/0PA/Horizon"])] + trajectoryData['Flux/Energy/0PA/Horizon'][()][::max(downsample["Flux/Energy/0PA/Infinity"], downsample["Flux/Energy/0PA/Horizon"])]
            self.interpolant["Flux/Energy/0PA/Deriv"] = CubicSpline(grid, value).deriv
            min_p_grid = max(min_p_grid, np.min(grid))
            max_p_grid = min(max_p_grid, np.max(grid))
            del grid, value # free memory

        self.evolve_primary = evolve_primary
        self._min_nu = 0.0
        self._max_nu = 0.25
        self._min_p = min_p_grid
        self._max_p = max_p_grid
        self._p_sep_schw = 6.0 # Schwarzschild separatrix for circular orbits
        self._min_chi1 = -1.0
        self._max_chi1 = 1.0
        self._min_chi2 = -1.0
        self._max_chi2 = 1.0

    @property
    def equatorial(self):
        """bool: Always ``True``; this model is restricted to equatorial orbits."""
        return True

    @property
    def circular(self):
        """bool: Always ``True``; this model is restricted to circular orbits (e = 0)."""
        return True

    @property
    def background(self):
        """
        str: Spacetime background for this trajectory.

        Returns 'Kerr' so that the primary spin (chi1) passed as 'a' is not
        zeroed out by the inspiral integrator.
        """
        return "Kerr"

    @property
    def nparams(self):
        """
        An integer describing the number of parameters this ODE will integrate.
        Defaults to 6 (three orbital elements, three orbital phases).
        """
        return 8

    @property
    def required_add_args(self):
        """
        The secondary spin ``chi2`` must always be supplied as an additional
        argument.
        """
        return ["chi2"]

    @property
    def separatrix_buffer_dist_grid(self):
        """
        The distance from the separatrix for the minimum p value of the grid.
        """
        return self._min_p - self._p_sep_schw

    @property
    def separatrix_buffer_dist(self):
        """
        The distance from the separatrix to truncate ODE integration.
        """
        return self._min_p - self._p_sep_schw + 0.01
    
    
    def isvalid_x(self, x, **kwargs):
        if np.any(x != 1):
            raise ValueError("Interpolation: x out of bounds. Must be 1.")

    def min_e(
        self,
        p: Union[float, np.ndarray],
        x: Union[float, np.ndarray] = 1.0,
        a: Optional[Union[float, np.ndarray]] = 0.0,
        **kwargs
    ) -> Union[float, np.ndarray]:
        """
        Computes the minimum value of the eccentricity e for a given semilatus rectum and inclination for this model.
        Trajectory models implementing their own interpolants should override this function to return the minimum value
        corresponding to the precomputed grid boundaries.

        By default, this function assumes minimal eccentricity corresponds to circular orbits and returns 0.
        """
        if isinstance(p, float):
            return 0
        else:
            return np.zeros_like(p)
        
    def max_e(
        self,
        p: Union[float, np.ndarray],
        x: Union[float, np.ndarray] = 1,
        a: Optional[Union[float, np.ndarray]] = 0,
        **kwargs
    ) -> Union[float, np.ndarray]:
        """
        Computes the maximum value of the eccentricity e for a given semilatus rectum and inclination for this model.
        Trajectory models implementing their own interpolants should override this function to return the minimum value
        corresponding to the precomputed grid boundaries.

        By default, this function assumes no orbital bounds on eccentricity and returns np.inf.
        """
        if isinstance(p, float):
            return 0
        else:
            return np.zeros_like(p)
    
    def isvalid_e(self, e, e_buffer=[0, 0],
        **kwargs):
        if np.any(e != 0):
            raise ValueError("Interpolation: e out of bounds. Must be 0.")
    
    def _isvalidnu(self, nu,
        **kwargs):
        """Raise ``ValueError`` if the symmetric mass ratio is outside the valid range.

        Args:
            nu (float): Symmetric mass ratio nu = m1*m2/(m1+m2)^2.

        Raises:
            ValueError: If ``nu`` is outside ``[_min_nu, _max_nu]``.
        """
        if nu > self._max_nu or nu < self._min_nu:
            raise ValueError(f"(TrajectoryCirc1PAT1R) Interpolation: nu = {nu} out of bounds. Must be between {self._min_nu} and {self._max_nu}.")

    def _isvalidchi1(self, chi1,
        **kwargs):
        """Raise ``ValueError`` if the primary spin is outside the valid range.

        Args:
            chi1 (float): Primary dimensionless spin.

        Raises:
            ValueError: If ``chi1`` is outside ``[_min_chi1, _max_chi1]``.
        """
        if chi1 > self._max_chi1 or chi1 < self._min_chi1:
            raise ValueError(f"(TrajectoryCirc1PAT1R) Domain validity: chi1 = {chi1} out of bounds. Must be between {self._min_chi1} and {self._max_chi1}.")
    
    def isvalid_a(self, a,
        **kwargs):
        """Raise ``ValueError`` if the primary spin is outside the valid range.

        Args:
            a (float): Primary dimensionless spin.

        Raises:
            ValueError: If ``a`` is outside ``[_min_chi1, _max_chi1]``.
        """
        if a > self._max_chi1 or a < self._min_chi1:
            raise ValueError(f"(TrajectoryCirc1PAT1R) Domain validity: a = {a} out of bounds. Must be between {self._min_chi1} and {self._max_chi1}.")
     

    def _isvalidchi2(self, chi2,
        **kwargs):
        """Raise ``ValueError`` if the secondary spin is outside the valid range.

        Args:
            chi2 (float): Secondary dimensionless spin.

        Raises:
            ValueError: If ``chi2`` is outside ``[_min_chi2, _max_chi2]``.
        """
        if chi2 > self._max_chi2 or chi2 < self._min_chi2:
            raise ValueError(f"(TrajectoryCirc1PAT1R) Domain validity: chi2 = {chi2} out of bounds. Must be between {self._min_chi2} and {self._max_chi2}.")

    def min_p(self, e=0.0, x = 1.0, a=0.0, separatrix_buffer=None,
        **kwargs):
        """Return the minumum valid p for this model (data grid upper boundary).

        Args:
            e (float): Eccentricity (ignored since this model is circular).
            x (float): Cosine of the inclination angle (ignored since this model is equatorial
            a (float): Primary spin"""
        if separatrix_buffer is None:
            separatrix_buffer = self.separatrix_buffer_dist_grid
        self.isvalid_e(e, **kwargs)
        self.isvalid_x(x, **kwargs)
        self._isvalidchi1(a, **kwargs)

        # for retrograde spin we decide to truncate the valid p near the Kerr separatrix when
        # it is larger than the minimum p of the grid. We subtract the separatrix buffer distance
        # so that when the integration is performed, the ODE will stop some small distance from 
        # the separatrix.
        psep_of_a_minus_buffer = get_separatrix(a, e, x) - self.separatrix_buffer_dist_grid
        return np.max([self._p_sep_schw, psep_of_a_minus_buffer]) + separatrix_buffer

    def max_p(self,e=0.0, x = 1.0, a=0.0,
        **kwargs):
        """Return the maximum valid p for this model (data grid upper boundary).

        Args:
            e (float): Eccentricity (ignored since this model is circular).
            x (float): Cosine of the inclination angle (ignored since this model is equatorial
            a (float): Primary spin"""
        
        self.isvalid_e(e, **kwargs)
        self.isvalid_x(x, **kwargs)
        self._isvalidchi1(a, **kwargs)

        return self._max_p
    
    
    def bounds_a(self, a_buffer=[0, 0], **kwargs):
        """Return the valid a range for this model with optional buffer.

        Args:
            a_buffer (list[float, float], optional): Two-element list of non-negative floats specifying how much to shrink the valid a range from below and above, respectively. Defaults to [0.0, 0.0] (no buffer)."""
      
        return [self._min_chi1 + a_buffer[0], self._max_chi1 - a_buffer[1]]

    def bounds_p(self, e=0.0, x=1.0, a=0.0, p_buffer=[0.0, 0.0], separatrix_buffer=None, **kwargs):
        """Return the valid p range for this model, accounting for the separatrix and optional buffer.

        Args:
            e (float): Eccentricity (ignored since this model is circular).
            x (float): Cosine of the inclination angle (ignored since this model is equatorial).
            a (float): Primary spin.
            p_buffer (list[float, float], optional): Two-element list of non-negative floats specifying how much to shrink the valid p range from below and above, respectively. Defaults to [0.0, 0.0] (no buffer)."""
        self._isvalidchi1(a, **kwargs)
        self.isvalid_e(e, **kwargs)
        self.isvalid_x(x, **kwargs)

        pmin = self.min_p(e, x, a=a, separatrix_buffer=separatrix_buffer, **kwargs) + p_buffer[0]
        pmax = self.max_p(e, x, a=a, **kwargs) - p_buffer[1]
        if pmin > pmax:
            raise ValueError(f"(TrajectoryCirc1PAT1R) Invalid p_buffer: lower buffer {p_buffer[0]} and upper buffer {p_buffer[1]} together exclude the entire valid p range [{self.min_p(e, x, a=a, **kwargs)}, {self.max_p(e, x, a=a, **kwargs)}].")
        return pmin, pmax

    def isvalid_p(self, p, e=0.0, x=1.0, a=0.0, separatrix_buffer=None, **kwargs):
        """Raise ``ValueError`` if p is outside the flux-data grid.

        Args:
            p (float): Dimensionless semi-latus rectum p/M.
            e (float): Eccentricity (ignored since this model is circular).
            x (float): Cosine of the inclination angle (ignored since this model is equatorial).
            a (float): Primary spin."""
        
        self._isvalidchi1(a, **kwargs)
        self.isvalid_e(e, **kwargs)
        self.isvalid_x(x, **kwargs)

        pmin, pmax = self.bounds_p(e, x, a, separatrix_buffer=separatrix_buffer, **kwargs)
        if p > pmax or p < pmin:
            raise ValueError(f"(TrajectoryCirc1PAT1R) Interpolation: p = {p} out of bounds. Must be between {pmin} and {pmax}.")
        
    # Note: This does not seem to be used. Consider deleting
    def _isvalidp(self, p, chi1):
        """Raise ``ValueError`` if p is outside the flux-data grid.

        Args:
            p (float): Dimensionless semi-latus rectum p/M.

        Raises:
            ValueError: If ``p`` is outside ``[_min_p, _max_p]``.
        """
        pmax = self.max_p(e=0.,x=1.,a=chi1)
        pmin = self.min_p(e=0.,x=1,a=chi1)

        if p > pmax or p < pmin:
            raise ValueError(f"(TrajectoryCirc1PAT1R) Interpolation: p = {p} out of bounds. Must be between {pmin} and {pmax}.")
        
    
    
    def isvalid_pex(self, p=20, e=0, x=1, a=0, p_buffer=[0, 0], e_buffer=[0, 0], separatrix_buffer=None, **kwargs):
        self.isvalid_x(x, **kwargs)
        self.isvalid_e(e, e_buffer=e_buffer, **kwargs)
        self._isvalidchi1(a, **kwargs)
        pmin, pmax = self.bounds_p(e, x, a, p_buffer=p_buffer, separatrix_buffer=separatrix_buffer, **kwargs)
        assert p >= pmin and p <= pmax, (
            f"(TrajectoryCirc1PAT1R) Interpolation: p = {p} out of bounds. Must be between {pmin} and {pmax}."
        )

    def add_fixed_parameters(self, m1, m2, chi1, additional_args):
        """Store the symmetric mass ratio nu, the spin parameters, and the reduced spins chit1, chit2.

        The ``additional_args`` array must contain, in order:

        1. ``chi2`` — dimensionless spin of the secondary compact object.

        Args:
            m1 (float): Mass of the primary in solar masses.
            m2 (float): Mass of the secondary in solar masses.
            chi1 (float): Dimensionless spin of the primary.
            additional_args (array-like): Extra arguments; ``additional_args[0]``
                is the secondary spin ``chi2``.

        Raises:
            ValueError: If nu, chi1, or chi2 fall outside their respective
                valid ranges.
        """
        # call base to set self.a, self.massratio, self.additional_args, self.num_add_args
        super().add_fixed_parameters(m1, m2, chi1, additional_args)
        # chi2 (additional_args[0]) is guaranteed to be present here: get_inspiral
        # enforces required_add_args before reaching this point.
        self.num_add_args = len(additional_args)
        if self.num_add_args != 1:
            get_logger().warning(f"WARNING (TrajectoryCirc1PAT1R.add_fixed_parameters): Only 1 additional argument (chi2) expected in TrajectoryCirc1PAT1R but {self.num_add_args} received. Excess arguments will be ignored.")

        self.args['nu'] = m1 * m2 / (m1 + m2)**2
        if self.args['nu'] < 1e-8:
            get_logger().warning(f"WARNING (TrajectoryCirc1PAT1R.add_fixed_parameteters): Mass ratio nu = {self.args['nu']:.2e} < 1e-8. Phase error may exceed 1 radian.  ")

        self.args['chi1'] = chi1
        self.args['chi2'] = float(additional_args[0])
        self.args['deltaM'] = 0.0

        self.args['chit1'], self.args['chit2'] = chi_to_chit(self.args['chi1'], self.args['chi2'], self.args['nu'])

        self._isvalidnu(self.args['nu'])
        self._isvalidchi1(self.args['chi1'])
        self._isvalidchi2(self.args['chi2'])

    def Flux(self, nu, p, chit1, chit2, deltaM):
        """Total 1PA numerator of the flux-balance law, Eq. (83) of Ref. 2510.16113.

        Returns the numerator of the flux-balance law at 1PA order.

        Args:
            nu (float): Symmetric mass ratio nu = m1*m2/(m1+m2)^2.
            p (float): Dimensionless semi-latus rectum p/M.
            chit1 (float): Primary reduced spin.
            chit2 (float): Secondary reduced spin.
            deltaM (float): Deviation of the total mass from its initial value.

        Returns:
            float: Numerator of the flux-balance law.
        """
        return (self.interpolant["Flux/Energy/0PA/Infinity"](p) + self.interpolant["Flux/Energy/0PA/Horizon"](p)) + nu * (self.interpolant["Flux/Energy/1PA/Infinity"](p) + chit1/nu * (self.interpolant["Flux/Energy/1PAchi1/Infinity"](p)+self.interpolant["Flux/Energy/1PAchi1/Horizon"](p)) + chit2/nu * (self.interpolant["Flux/Energy/1PAchi2/Infinity"](p)+ self.interpolant["Flux/Energy/1PAchi2/Horizon"](p)) - deltaM * (2./3.*p) * self.interpolant["Flux/Energy/0PA/Deriv"](p) + EdeltaM(p) * self.interpolant["Flux/Energy/0PA/Horizon"](p) - 2.*(-3.+2.*p)/(3.*(-3.+p)**(3./2.)*p**2.) * self.interpolant["Flux/AngularMomentum/0PA/Horizon"](p))

    def dEdOmega(self, nu, p, chit1, chit2, deltaM):
        """1PA derivative of the binding energy with respect to the orbital frequency.

        Args:
            nu (float): Symmetric mass ratio nu = m1*m2/(m1+m2)^2.
            p (float): Dimensionless semi-latus rectum p/M.
            chit1 (float): Primary reduced spin.
            chit2 (float): Secondary reduced spin.
            deltaM (float): Deviation of the total mass from its initial value.

        Returns:
            float: dE/dOmega_phi at 1PA order.
        """
        return (dEdOmega0PA(p) + chit1 * dEdOmega_1PA_chit1(p) + chit2 * dEdOmega_1PA_chit2(p) + nu *  (deltaM * dEdeltaMdOmega(p) + self.interpolant["Energy/1PA/dEdOmega"](p)))

    def evaluate_rhs(self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        """Evaluate the right-hand side of the 1PAT1R ODE.

        Computes the time derivatives of all state-vector components at the
        current state y.  The state vector layout is:

            y = [p, e, xI, nu*Phi_phi, nu*Phi_theta, nu*Phi_r, deltaM, delta_chit1]

        The evolution equations are derived from the 1PA flux-balance law,
        Eq. (83) of Ref. 2510.16113.

        Args:
            y (array-like): Current state vector of length 8:
                [p, e, xI, nu*Phi_phi, nu*Phi_theta, nu*Phi_r, deltaM, deltaChit1].

        Returns:
            list[float]: Derivatives
                [pdot, 0, 0, Omega_phi, 0, 0, deltaMdot, chit1dot].

        Raises:
            ValueError: If p is outside the grid bounds.
        """
        nu = self.args['nu']
        chit2 = self.args['chit2']

        assert len(y) == 8, f"State vector y must have length 8, but got length {len(y)}."

        p = y[0]
        e = y[1]
        xI = y[2]

        chit1 = self.args['chit1'] + y[-1]
        deltaM = y[-2]
        


        Omega_phi = 1/p**(3./2.)
        pdot = -self.Flux(nu, p, chit1, chit2, deltaM)/self.dEdOmega(nu, p, chit1, chit2, deltaM)/dOmegadp(p)
        chit1dot = nu * self.interpolant["Flux/AngularMomentum/0PA/Horizon"](p) if self.evolve_primary else 0.0
        deltaMdot = self.interpolant["Flux/Energy/0PA/Horizon"](p) if self.evolve_primary else 0.0

        return [pdot, 0.0, 0.0, Omega_phi, 0.0, 0.0, deltaMdot, chit1dot]
    
    def __call__(
        self,
        y: Union[list, np.ndarray],
        out: Optional[np.ndarray] = None,
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        """Evaluate the ODE and return the derivative array.

        Overrides :meth:`~few.trajectory.ode.base.ODEBase.__call__` to
        pre-allocate an 8-element output buffer (instead of the base-class
        default of 6) and to skip the ELQ Jacobian transform, which is not
        used in this model.

        Args:
            y (array-like): Current state vector of length 8.
            out (np.ndarray, optional): Pre-allocated output array of length 8.
                If ``None``, a new array is allocated.
            **kwargs

        Returns:
            np.ndarray: Derivative array of length 8.
        """
        in_bounds = self.cache_values_and_check_bounds(y)

        if out is None:
            out = np.zeros(8)

        if in_bounds:
            out[:] = self.evaluate_rhs(y, **kwargs)
        else:
            out *= np.nan

        self.modify_rhs(out, y, **kwargs)

        if self.integrate_backwards:
            out *= -1.0

        return out
