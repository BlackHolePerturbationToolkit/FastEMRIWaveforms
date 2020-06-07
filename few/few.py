import numpy as np

try:
    import cupy as xp

except ImportError:
    import numpy as xp

from flux import RunFluxInspiral
from amplitude import Amplitude
from interpolated_mode_sum import InterpolatedModeSum


class FEW:
    def __init__(self, inspiral_kwargs={}, amplitude_kwargs={}):
        """
        Carrier class for FEW
        """
        self.inspiral_gen = RunFluxInspiral()
        self.inspiral_kwargs = inspiral_kwargs

        self.amplitude_gen = Amplitude(**amplitude_kwargs)
        self.sum = InterpolatedModeSum()

    def __call__(self, M, mu, p0, e0, theta, phi):

        # get trajectory
        (t, p, e, Phi_phi, Phi_r) = self.inspiral_gen(
            M, mu, p0, e0, **self.inspiral_kwargs
        )

        # convert for gpu
        t = xp.asarray(t)
        p = xp.asarray(p)
        e = xp.asarray(e)
        Phi_phi = xp.asarray(Phi_phi)
        Phi_r = xp.asarray(Phi_r)

        # amplitudes
        teuk_modes = self.amplitude_gen(p, e)

        import pdb

        pdb.set_trace()

        # TODO: implement normalization to flux
        power = xp.abs(teuk_modes) ** 2

        inds_sort = xp.argsort(power, axis=1)[:, ::-1]
        power = xp.sort(power, axis=1)[:, ::-1]
        cumsum = xp.cumsum(power, axis=1)

        eps = 1e-2

        inds_keep = xp.full(cumsum.shape, True)

        inds_keep[:, 1:] = cumsum[:, :-1] < cumsum[:, -1][:, xp.newaxis] * (1 - eps)

        keep_modes = xp.unique(inds_sort[inds_keep])

        self.sum(t, p, e, phi_phi, phi_r, teuk_modes, mode_keep_inds=keep_modes)
        return


if __name__ == "__main__":
    import time

    few = FEW(inspiral_kwargs={}, amplitude_kwargs={"max_input_len": 3000})
    M = 1e5
    mu = 1e1
    p0 = 10.0
    e0 = 0.3
    theta = np.pi / 3.0
    phi = np.pi / 4.0

    check = few(M, mu, p0, e0, theta, phi)
    num = 100

    st = time.perf_counter()
    for _ in range(num):
        check = few(M, mu, p0, e0, theta, phi)
    et = time.perf_counter()

    print((et - st) / num)
