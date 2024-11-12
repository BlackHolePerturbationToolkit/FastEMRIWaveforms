from abc import ABC
from few.utils.citations import *

class AmplitudeBase(ABC):
    """Base class used for amplitude modules.

    This class provides a common flexible interface to various amplitude
    implementations. Specific arguments to each amplitude module can be found
    with each associated module discussed below.

    args:
        pad_output (bool, optional): Add zero padding to the waveform for time
            between plunge and observation time. Default is False.

    """

    def __init__(self, **kwargs):
        pass

    @classmethod
    def get_amplitudes(self, *args, **kwargs):
        """Amplitude Generator

        @classmethod that requires a child class to have a get_amplitudes method.

        raises:
            NotImplementedError: The child class does not have this method.

        """
        raise NotImplementedError

    @property
    def citation(self):
        """Return citation for this class"""
        return larger_few_citation + few_citation + few_software_citation

    def __call__(self, *args, specific_modes=None, **kwargs):
        """Common call for Teukolsky amplitudes

        This function takes the inputs the trajectory in :math:`(p,e)` as arrays
        and returns the complex amplitude of all modes to adiabatic order at
        each step of the trajectory.

        args:
            *args (tuple, placeholder): Added to create future flexibility when calling different
                amplitude modules. Transfers directly into get_amplitudes function.
            specific_modes (list, optional): List of tuples for (l, m, n) values
                desired modes. Default is None. This is not available for all waveforms.
            **kwargs (dict, placeholder): Added to create flexibility when calling different
                amplitude modules. It is not used.

        returns:
            2D array (double): If specific_modes is None, Teukolsky modes in shape (number of trajectory points, number of modes)
            dict: Dictionary with requested modes.


        """

        return self.get_amplitudes(*args, specific_modes=specific_modes, **kwargs)
