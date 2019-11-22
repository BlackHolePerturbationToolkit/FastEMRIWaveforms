import numpy as np
import time

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from .nn import NN
from .ylm import get_ylms


class GetZlmnk:
    def __init__(
        self, batch_size, transform_file="files/reduced_basis.dat", nn_kwargs={}
    ):
        self.neural_net = NN(**nn_kwargs)

        self.transform_matrix = xp.asarray(
            np.genfromtxt(transform_file, dtype=xp.complex64)
        )

        self.transform_factor = 1000.0

        self.buffer = xp.zeros((batch_size, 2))

    def __call__(self, p, e):
        self.buffer[:, 0] = p
        self.buffer[:, 1] = e
        output = self.neural_net(self.buffer)

        re = output[:, :80]
        im = output[:, 80:]

        temp = re + 1j * im

        Zlmkn = xp.matmul(temp / self.transform_factor, self.transform_matrix)

        return Zlmkn


class CreateWaveform:
    def __init__(self, **kwargs):
        batch_size = kwargs["batch_size"]
        self.get_zlmnk = GetZlmnk(**kwargs)
        self.expiphases = xp.zeros((batch_size, 2214), dtype=xp.complex64)
        self.zlmnk = xp.zeros((batch_size, 2214), dtype=xp.complex64)
        self.ylms = xp.zeros(2214, dtype=xp.complex64)
        self.buffer = xp.zeros(54, dtype=xp.complex64)

    def __call__(self, p, e, Phi_r, Phi_phi, l, m, n, theta, phi):
        st = time.perf_counter()
        for i in range(10):

            self.ylms[:] = xp.tile(
                get_ylms(l[0::41], m[0::41], theta, phi, self.buffer), (41,)
            )
        et = time.perf_counter()
        print((et - st) / 10)
        self.zlmnk[:] = self.get_zlmnk(p, e)
        self.expiphases[:] = xp.exp(
            -1j
            * (
                m[xp.newaxis, :] * Phi_phi[:, xp.newaxis]
                + n[xp.newaxis, :] * Phi_r[:, xp.newaxis]
            )
        )

        return xp.sum(self.zlmnk * self.ylms[xp.newaxis, :] * self.expiphases, axis=1)


if __name__ == "__main__":
    nn_kwargs = dict(input_str="SE_", folder="files/weights/", activation_kwargs={})

    kwargs = dict(transform_file="files/reduced_basis.dat", nn_kwargs=nn_kwargs)

    traj = np.genfromtxt("insp_p12.5_e0.7_tspacing_1M.dat")[0::3][:100000]

    batch_size = kwargs["batch_size"] = len(traj)

    p = xp.asarray(traj[:, 0], dtype=xp.float32)
    e = xp.asarray(traj[:, 1], dtype=xp.float32)
    Phi_phi = xp.asarray(traj[:, 2], dtype=xp.float32)
    Phi_r = xp.asarray(traj[:, 3], dtype=xp.float32)

    l = xp.zeros(2214, dtype=int)
    m = xp.zeros(2214, dtype=int)
    n = xp.zeros(2214, dtype=int)

    ind = 0
    for l_i in range(2, 10 + 1):
        for m_i in range(1, l_i + 1):
            for n_i in range(-20, 20 + 1):
                l[ind] = l_i
                m[ind] = m_i
                n[ind] = n_i
                ind += 1

    cw = CreateWaveform(**kwargs)
    theta = np.pi / 4
    phi = np.pi / 3

    num = 10
    out = cw(p, e, Phi_r, Phi_phi, l, m, n, theta, phi)
    check = []
    for _ in range(num):
        st = time.perf_counter()
        out = cw(p, e, Phi_r, Phi_phi, l, m, n, theta, phi)
        et = time.perf_counter()
        print("Timing:", (et - st))
    import pdb

    pdb.set_trace()
