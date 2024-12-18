"""
Contains the ODEBase baseclass that handles evaluating the ODE
"""
from typing import Optional, Type
import numpy as np

def get_ode_function_options():
    pass


class ODEBase:
    """
    A baseclass for handling the evaluation of ODE derivatives in the trajectory module. 

    To define a new trajectory function, subclass this function and define `evaluate_rhs`.
    Make sure to update the relevant class attributes as well.
    See the documentation for examples on how to do this.

    """
    def __init__(self, *args, **kwargs):
        self.file_dir = None
        self.num_add_args = 0

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
    def integrate_constants_of_motion(self):
        """
        If True, the orbital element coordinates are assumed to be (E, L, Q).
        If False, they are assumed to be (p, e, xI).
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

    def add_fixed_parameters(self, M: float, mu: float, a: float, additional_args=None):
        self.epsilon = mu / M
        self.a = a
        self.additional_args = additional_args

        if additional_args is None:
            self.num_add_args = 0
        else:
            self.num_add_args = len(additional_args)

    def evaluate_rhs_arr(self, y, **kwargs):
        return self.evaluate_rhs(*y, **kwargs)

    def evaluate_rhs(self, *args, **kwargs) -> NotImplementedError:
        raise NotImplementedError

    def __call__(self, *args: list[float], out: Optional[np.ndarray] = None, scale_by_eps=False, **kwargs: Optional[dict]) -> np.ndarray:
        derivs = self.evaluate_rhs_arr(*args, **kwargs)
        if out is None:
            out = np.asarray(derivs)
        else:
            out[:] = derivs

        if scale_by_eps:
            out[:3] *= self.epsilon

        return out

    def __reduce__(self):
        #  to ensure pickleability of the trajectory & waveform modules
        #  TODO: re-examine this in future, this is a band-aid fix that breaks
        #  if the user adds their own args/kwargs to their class
        #  Or optionally, we can ask the user to define this as well (not ideal)
        return (self.__class__, (self.file_dir, ))


def _properties(cls: type) -> list[str]:
    return [
        key
        for key, value in cls.__dict__.items()
        if isinstance(value, property)
    ]

def get_ode_properties(inst_cls: Type[ODEBase]):
    cls = inst_cls.__class__

    # first get all the properties of ODEBase
    parent = cls.__bases__[0]
    parentprops = _properties(parent)
    props = {pkey : getattr(parent, pkey).fget(parent) for pkey in parentprops}

    # now update with what is changed by this subclass
    childprops = _properties(cls)
    props.update({ckey : getattr(cls, ckey).fget(cls) for ckey in childprops})
    return props
