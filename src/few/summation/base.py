from ..utils.baseclasses import BackendLike, ParallelModuleBase
from ..utils.constants import YRSID_SI


class SummationBase(ParallelModuleBase):
    """Base class used for summation modules.

    This class provides a common flexible interface to various summation
    implementations. Specific arguments to each summation module can be found
    with each associated module discussed below.

    args:
        pad_output: Add zero padding to the waveform for time
            between plunge and observation time. Default is False.
        output_type: Type of domain in which to calculate the waveform.
            Default is 'td' for time domain. Options are 'td' (time domain) or 'fd' (Fourier domain). In the future we hope to add 'tf'
            (time-frequency) and 'wd' (wavelet domain).
        odd_len: The waveform output will be padded to be an odd number if True.
            If ``output_type == "fd"``, odd_len will be set to ``True``. Default is False.
    """

    def __init__(
        self,
        /,
        output_type: str = "td",
        pad_output: bool = False,
        odd_len: bool = False,
        force_backend: BackendLike = None,
    ):
        ParallelModuleBase.__init__(self, force_backend=force_backend)

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

        self.num_pts = None
        """int: Number of points in the output waveform."""

        self.num_pts_pad = None
        """int: Number of points the output waveform has been padded by."""

        self.waveform = None
        """Complex waveform given by :math:`h_+ + i*h_x`."""

    @classmethod
    def sum(self, *args, **kwargs):
        """Sum Generator

        @classmethod that requires a child class to have a sum method.

        raises:
            NotImplementedError: The child class does not have this method.

        """
        raise NotImplementedError

    def __call__(self, t: float, *args, T: float = 1.0, dt: float = 10.0, **kwargs):
        """Common call function for summation modules.

        Provides a common interface for summation modules. It can adjust for
        more dimensions in a model.

        args:
            t: Array of t values.
            *args: Added for flexibility with summation modules. `args`
                tranfers directly into sum function.
            dt: Time spacing between observations in seconds (inverse of sampling
                rate). Default is 10.0.
            T: Maximum observing time in years. Default is 1.0.
            **kwargs: Added for future flexibility.

        """

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

        self.num_pts = num_pts

        self.num_pts_pad = num_pts_pad

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
                dt = float(self.xp.max(frequency) * 2)
                Nf = len(frequency)
                # total
                waveform = self.xp.zeros(Nf, dtype=self.xp.complex128)
                # print("user defined frequencies Nf=", Nf)
            else:
                waveform = self.xp.zeros(
                    (self.num_pts + self.num_pts_pad,), dtype=self.xp.complex128
                )
            # if self.num_pts + self.num_pts_pad % 2:
            #     self.num_pts_pad = self.num_pts_pad + 1
            #     print("n points",self.num_pts + self.num_pts_pad)
        else:
            # setup waveform holder for time domain
            waveform = self.xp.zeros(
                (self.num_pts + self.num_pts_pad,), dtype=self.xp.complex128
            )

        self.waveform = waveform

        # get the waveform summed in place
        self.sum(t, *args, dt=dt, **kwargs)

        return self.waveform
