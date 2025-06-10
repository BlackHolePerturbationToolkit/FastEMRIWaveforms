from typing import Optional, Union

import numpy as np

from ..utils.citations import Citable


class AmplitudeBase(Citable):
    """Base class used for amplitude modules.

    This class provides a common flexible interface to various amplitude
    implementations. Specific arguments to each amplitude module can be found
    with each associated module discussed below.

    args:
        pad_output (bool, optional): Add zero padding to the waveform for time
            between plunge and observation time. Default is False.

    """

    @classmethod
    def get_amplitudes(self, *args, **kwargs):
        """Amplitude Generator

        @classmethod that requires a child class to have a get_amplitudes method.

        raises:
            NotImplementedError: The child class does not have this method.

        """
        raise NotImplementedError

    def __call__(
        self, *args, specific_modes: Optional[Union[list, np.ndarray]] = None, **kwargs
    ) -> Union[dict, np.ndarray]:
        """Common call for Teukolsky amplitudes

        This function takes the inputs the trajectory in :math:`(p,e)` as arrays
        and returns the complex amplitude of all modes to adiabatic order at
        each step of the trajectory.

        Args:
            *args: Added to create future flexibility when calling different
                amplitude modules. Transfers directly into get_amplitudes function.
            specific_modes: Either indices or mode index tuples of modes to be generated (optional; defaults to all modes). This is not available for all waveforms.
            **kwargs (dict, placeholder): Added to create flexibility when calling different
                amplitude modules. It is not used.

        Returns:
            If specific_modes is a list of tuples, returns a dictionary of complex mode amplitudes.
            Else, returns an array of complex mode amplitudes.

        """

        return self.get_amplitudes(*args, specific_modes=specific_modes, **kwargs)
