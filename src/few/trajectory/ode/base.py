"""
Contains the ODEBase baseclass that handles evaluating the ODE
"""

from typing import Optional, Type, Union
import numpy as np
import os
from ...utils.utility import ELQ_to_pex, get_separatrix
from ...utils.mappings import ELdot_to_PEdot_Jacobian
from ...utils.pn_map import Y_to_xI

dir_path = os.path.dirname(os.path.realpath(__file__))

# def get_ode_function_options():
#     return _STOCK_TRAJECTORY_OPTIONS


class ODEBase:
    """
    A baseclass for handling the evaluation of ODE derivatives in the trajectory module.

    To define a new trajectory function, subclass this function and define `evaluate_rhs`.
    Make sure to update the relevant class attributes as well.
    See the documentation for examples on how to do this.

    """

    def __init__(self, *args, use_ELQ=False, **kwargs):
        if use_ELQ:
            assert self.supports_ELQ, "This ODE does not support ELQ evaluation."
        self.use_ELQ = use_ELQ
        """
        bool: If True, the ODE will take as input (and output derivatives of) the integrals of motion (E, L, Q). Defaults to False.
        """
        self.num_add_args = 0
        """int: Number of additional arguments being passed to the ODE function."""

        self.apply_Jacobian_bool = (
            self.flux_output_convention == "ELQ" and not self.use_ELQ
        )

    @property
    def convert_Y(self):
        """
        If True, the inclination coordinate is assumed to be Y and is converted accordingly.
        Defaults to False.
        """
        return False

    @property
    def equatorial(self):
        """
        If True, the inclination coordinate is assumed to be +/- 1.
        Defaults to False.
        """
        return False

    @property
    def circular(self):
        """
        If True, the eccentricity coordinate is assumed to be 0.
        Defaults to False.
        """
        return False

    @property
    def supports_ELQ(self):
        """
        If True, this ODE can take as input (and output derivatives of)
        the integrals of motion (E, L, Q) if initialised with `use_ELQ=True`.
        Defaults to False.
        """
        return False

    @property
    def background(self):
        """
        A string describing the background spacetime. Either "Kerr" or "Schwarzschild".
        Defaults to "Kerr".
        """
        return "Kerr"

    @property
    def separatrix_buffer_dist(self):
        """
        A float describing the value of "p" at which the trajectory should terminate at,
        with respect to the separatrix.
        A value of 0 would mean that the trajectory terminates at the separatrix.
        Defaults to 0.05
        """
        return 0.05

    @property
    def nparams(self):
        """
        An integer describing the number of parameters this ODE will integrate.
        Defaults to 6 (three orbital elements, three orbital phases).
        """
        return 6

    @property
    def flux_output_convention(self):
        """
        A string describing the coordinate convention of the fluxes for this model, as output by `evaluate_rhs`.
        These are either "pex" or "ELQ". If "ELQ", a Jacobian transformation is performed if
        the output derivatives of the model are expected to be in the "pex" convention.
        Defaults to "ELQ".

        For models that do not perform interpolation to generate fluxes (e.g., PN5), this property is not accessed.
        """
        return "ELQ"

    def add_fixed_parameters(self, M: float, mu: float, a: float, additional_args=None):
        self.epsilon = mu / M
        self.a = a
        self.additional_args = additional_args

        if additional_args is None:
            self.num_add_args = 0
        else:
            self.num_add_args = len(additional_args)

    def evaluate_rhs(self, y: np.ndarray, **kwargs) -> NotImplementedError:
        """
        This function evaluates the right-hand side of the ODE at the point y.
        An ODE model can be defined by subclassing the ODEBase class and implementing this method.
        """
        raise NotImplementedError

    def modify_rhs_before_Jacobian(
        self, ydot: np.ndarray, y: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        This function allows the user to modify the right-hand side of the ODE before the Jacobian transform is applied.
        This is particularly useful if the user wishes to apply modifications to fluxes of the integrals of motion "ELQ",
        but integrate a trajectory in terms of "pex".

        By default, this function returns the input right-hand side unchanged.
        """
        return ydot

    def modify_rhs(self, ydot: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        This function allows the user to modify the right-hand side of the ODE after any required Jacobian transforms
        have been applied.

        By default, this function returns the input right-hand side unchanged.
        """
        return ydot

    def interpolate_flux_grids(self, *args) -> NotImplementedError:
        """
        This function handles the interpolation of the fluxes from the precomputed grids, including parameter transformations.
        Each stock model implements this function to handle the specifics of their own interpolants.
        To easily incorporate interpolated fluxes into their own models, users can subclass the stock models; the
        interpolated fluxes can then be accessed from `evaluate_rhs` by calling this function.

        This method should also handle checking that the input parameters are within the bounds of the precomputed grids.
        Failure to do so may result in erroneous behaviour during trajectory evaluation in which fluxes are extrapolated.
        """
        raise NotImplementedError

    def distance_to_outer_boundary(self, y: np.ndarray) -> float:
        """
        This function returns the distance to the outer boundary of the interpolation grid. This is necessary for
        backwards integration, which performs root-finding to ensure that the trajectory ends on this outer boundary.
        Root-finding is initiated when this function returns a negative value.

        Each stock model implements this function to handle the specifics of their own interpolants. For models that
        do not use interpolation, this function returns a positive constant (such that root-finding never occurs)
        and does not need to be implemented.
        """
        return 1e10

    def get_pex(self, y: np.ndarray) -> tuple[float]:
        """
        This function converts the integrals of motion (E, L, Q) to the orbital elements (p, e, x), if required.
        """
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)
        else:
            p, e, x = y[:3]
        return p, e, x

    def cache_values_and_check_bounds(self, y: np.ndarray) -> bool:
        """
        This function checks the input points to ensure they are within the physical bounds of the inspiral parameter space.
        These checks include ensuring that the separatrix has not been crossed, and that the eccentricity is within bounds.

        Returns a boolean indicating whether the input was in bounds.
        """

        p, e, x_or_Y = self.get_pex(y)

        # first: check the eccentricity
        in_bounds = e >= 0
        if in_bounds:
            # second: check the separatrix
            if self.convert_Y:
                x = Y_to_xI(self.a, p, e, x_or_Y)
                self.xI_cache = x
            else:
                x = x_or_Y

            self.p_sep_cache = get_separatrix(self.a, e, x)
            in_bounds = p > self.p_sep_cache

        return in_bounds

    def __call__(
        self,
        y: Union[list, np.ndarray],
        out: Optional[np.ndarray] = None,
        scale_by_eps=False,
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        in_bounds = self.cache_values_and_check_bounds(y)

        if in_bounds:
            derivs = self.evaluate_rhs(y, **kwargs)
        else:
            derivs = np.zeros(6) * np.nan

        if out is None:
            out = np.asarray(derivs)
        else:
            out[:] = derivs

        self.modify_rhs_before_Jacobian(out, y, **kwargs)

        if self.apply_Jacobian_bool:  # implicitly this means that y contains (p, e, x)
            out[:2] = ELdot_to_PEdot_Jacobian(self.a, *y[:3], *out[:2])

        self.modify_rhs(out, y, **kwargs)

        if scale_by_eps:
            out[:3] *= self.epsilon

        return out

    def __reduce__(self):
        #  to ensure pickleability of the trajectory & waveform modules
        #  TODO: re-examine this in future, this is a band-aid fix that breaks
        #  if the user adds their own args/kwargs to their class
        #  Or optionally, we can ask the user to define this as well (not ideal)
        return self.__class__, (self.use_ELQ,)


def _properties(cls: type) -> list[str]:
    return [key for key, value in cls.__dict__.items() if isinstance(value, property)]


def get_ode_properties(inst_cls: Type[ODEBase]):
    cls = inst_cls.__class__

    # first get all the properties of ODEBase
    parent = cls.__bases__[0]
    parentprops = _properties(parent)
    props = {pkey: getattr(parent, pkey).fget(parent) for pkey in parentprops}

    # now update with what is changed by this subclass
    childprops = _properties(cls)
    props.update({ckey: getattr(cls, ckey).fget(cls) for ckey in childprops})
    return props
