from abc import ABC
from ..utils.citations import *
from ..utils.constants import *

# try to import cupy
try:
    import cupy as cp

    gpu_available = True

except:
    import numpy as np

    gpu_available = False

class SummationBase(ABC):
    """Base class used for summation modules.

    This class provides a common flexible interface to various summation
    implementations. Specific arguments to each summation module can be found
    with each associated module discussed below.

    args:
        pad_output (bool, optional): Add zero padding to the waveform for time
            between plunge and observation time. Default is False.
        output_type (str, optional): Type of domain in which to calculate the waveform.
            Default is 'td' for time domain. Options are 'td' (time domain) or 'fd' (Fourier domain). In the future we hope to add 'tf'
            (time-frequency) and 'wd' (wavelet domain).
        odd_len (bool, optional): The waveform output will be padded to be an odd number if True.
            If ``output_type == "fd"``, odd_len will be set to ``True``. Default is False.

    """

    def __init__(
        self, *args, output_type="td", pad_output=False, odd_len=False, **kwargs
    ):
        self.pad_output = pad_output
        self.odd_len = odd_len

        if output_type not in ["td", "fd"]:
            raise ValueError(
                "{} waveform domain not available. Choices are 'td' (time domain) or 'fd' (frequency domain).".format(
                    output_type
                )
            )
        self.output_type = output_type
        if self.output_type == "fd":
            self.odd_len = True

    def attributes_SummationBase(self):
        """
        attributes:
            waveform (1D complex128 np.ndarray): Complex waveform given by
                :math:`h_+ + i*h_x`.
        """
        pass

    @property
    def citation(self):
        """Return citation for this class"""
        return larger_few_citation + few_citation + few_software_citation

    @classmethod
    def sum(self, *args, **kwargs):
        """Sum Generator

        @classmethod that requires a child class to have a sum method.

        raises:
            NotImplementedError: The child class does not have this method.

        """
        raise NotImplementedError

    def __call__(self, t, *args, T=1.0, dt=10.0, t_window=None, **kwargs):
        """Common call function for summation modules.

        Provides a common interface for summation modules. It can adjust for
        more dimensions in a model.

        args:
            t (1D double xp.ndarray): Array of t values.
            *args (list): Added for flexibility with summation modules. `args`
                tranfers directly into sum function.
            dt (double, optional): Time spacing between observations in seconds (inverse of sampling
                rate). Default is 10.0.
            T (double, optional): Maximum observing time in years. Default is 1.0.
            **kwargs (dict, placeholder): Added for future flexibility.

        """

        if self.use_gpu:
            xp = cp
        else:
            xp = np

        n_pts = int(T * YRSID_SI / dt)
        T = n_pts * dt
        # determine the output array setup

        # adjust based on if observations time is less than or more than trajectory time array
        # if the user wants zero-padding, add number of zero pad points
        if T < t[-1].item():
            num_pts = int((T - t[0]) / dt) + 1
            num_pts_pad = 0

        else:
            num_pts = int((t[-1] - t[0]) / dt) + 1
            if self.pad_output:
                num_pts_pad = int((T - t[0]) / dt) + 1 - num_pts
            else:
                num_pts_pad = 0

        self.num_pts, self.num_pts_pad = num_pts, num_pts_pad
        self.dt = dt

        # impose to be always odd
        if self.odd_len:
            if (self.num_pts + self.num_pts_pad) % 2 == 0:
                self.num_pts_pad = self.num_pts_pad + 1
                # print("n points",self.num_pts + self.num_pts_pad)

        # make sure that the FD waveform has always an odd number of points
        if self.output_type == "fd":
            if "f_arr" in kwargs:
                frequency = kwargs["f_arr"]
                dt = float(xp.max(frequency) * 2)
                Nf = len(frequency)
                # total
                self.waveform = xp.zeros(Nf, dtype=xp.complex128)
                # print("user defined frequencies Nf=", Nf)
            else:
                self.waveform = xp.zeros(
                    (self.num_pts + self.num_pts_pad,), dtype=xp.complex128
                )
            # if self.num_pts + self.num_pts_pad % 2:
            #     self.num_pts_pad = self.num_pts_pad + 1
            #     print("n points",self.num_pts + self.num_pts_pad)
        else:
            # setup waveform holder for time domain
            self.waveform = xp.zeros(
                (self.num_pts + self.num_pts_pad,), dtype=xp.complex128
            )

        # get the waveform summed in place
        self.sum(t, *args, dt=dt, **kwargs)

        return self.waveform
