import numpy as np

from flux import RunFluxInspiral


class FEW:
    def __init__(self, inspiral_kwargs={}):
        """
        Carrier class for FEW
        """
        self.inspiral_gen = RunFluxInspiral()
        self.inspiral_kwargs = inspiral_kwargs

    def __call__(self, M, mu, p0, e0, theta, phi):

        (t, p, e, Phi_phi, Phi_r) = self.inspiral_gen(
            M, mu, p0, e0, **self.inspiral_kwargs
        )

        return


if __name__ == "__main__":
    import time

    few = FEW()
    M = 1e6
    mu = 1e1
    p0 = 10.0
    e0 = 0.3
    theta = np.pi / 3.0
    phi = np.pi / 4.0

    num = 1000

    st = time.perf_counter()
    for _ in range(num):
        check = few(M, mu, p0, e0, theta, phi)
    et = time.perf_counter()

    print((et - st) / num)
