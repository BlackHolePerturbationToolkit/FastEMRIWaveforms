"""Common call interface of the FEW summation modules (:class:`SummationBase`)."""

from ..utils.baseclasses import BackendLike, FewBackendConsumer
from ..utils.constants import YRSID_SI


class SummationBase(FewBackendConsumer):
    r"""Base class of the summation modules.

    :meth:`__call__` sets the output length from ``T``, ``dt`` and the trajectory span,
    allocates the zeroed complex output buffer :attr:`waveform` on the backend and calls the
    subclass's :meth:`sum`, which fills it (or replaces it). The summed source-frame
    waveform is :math:`h_+ - i h_\times` (see
    :class:`~few.waveform.waveform.GenerateEMRIWaveform`).

    Args:
        output_type: Output domain: ``"td"`` (time domain, default) or ``"fd"`` (frequency
            domain; forces ``odd_len=True``).
        pad_output: If ``True``, zero-pad the output from the end of the trajectory (plunge)
            to the observation time ``T``. Default ``False``.
        odd_len: If ``True``, add one zero sample when needed so the output length is odd.
            Default ``False``.
        force_backend: Backend name or object (see
            :class:`~few.utils.baseclasses.FewBackendConsumer`). Default ``None``.

    Raises:
        ValueError: If ``output_type`` is not ``"td"`` or ``"fd"``.

    Attributes:
        output_type (str): The output domain.
        pad_output (bool): Whether the output is zero-padded to ``T``.
        odd_len (bool): Whether the output length is forced odd.
        dt (float): Sampling interval of the last call in seconds.
    """

    def __init__(
        self,
        /,
        output_type: str = "td",
        pad_output: bool = False,
        odd_len: bool = False,
        force_backend: BackendLike = None,
    ):
        FewBackendConsumer.__init__(self, force_backend=force_backend)

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
        """Output buffer: the complex waveform :math:`h_+ - i h_x` (time domain)."""

    @classmethod
    def sum(self, *args, **kwargs):
        """Fill :attr:`waveform` (implemented by the subclasses).

        Called by :meth:`__call__` as ``sum(t, *args, dt=dt, **kwargs)`` after the output
        buffer has been allocated.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def __call__(self, t: float, *args, T: float = 1.0, dt: float = 10.0, **kwargs):
        """Set the output length, allocate the output buffer and run :meth:`sum`.

        With ``T`` rounded down to a whole number of samples,
        ``T_s = int(T * YRSID_SI / dt) * dt``: if ``T_s < t[-1]`` the output covers ``[t[0], T_s]``
        (``int((T_s - t[0]) / dt) + 1`` samples); otherwise it covers the trajectory
        (``int((t[-1] - t[0]) / dt) + 1`` samples, :attr:`num_pts`) plus, with
        ``pad_output``, zeros up to ``T_s`` (:attr:`num_pts_pad`). With ``odd_len`` one more
        zero is added when the total is even. For ``output_type="fd"`` with an ``f_arr``
        keyword, the buffer has ``len(f_arr)`` entries instead (and the ``dt`` passed to
        :meth:`sum` becomes ``2 * max(f_arr)``).

        Args:
            t: Trajectory times in seconds, array of shape ``(N,)`` (numpy or cupy).
            *args: Forwarded to :meth:`sum` (amplitudes, harmonics, phases, mode indices, ...).
            T: Maximum duration in years. Default ``1.0``.
            dt: Sampling interval in seconds. Default ``10.0``.
            **kwargs: Forwarded to :meth:`sum` (``f_arr`` is also read here).

        Returns:
            xp.ndarray: :attr:`waveform` after :meth:`sum`, on the backend: complex, shape
            ``(num_pts + num_pts_pad,)`` in the time domain (the subclass may replace it, e.g.
            the ``(2, n_f)`` array of :class:`~few.summation.fdinterp.FDInterpolatedModeSum`).
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
