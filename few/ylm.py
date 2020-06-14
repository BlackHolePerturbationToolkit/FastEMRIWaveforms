try:
    import cupy as xp
except ImportError:
    import numpy as xp

import numpy as np


class GetYlms:
    def __init__(self, num_teuk_modes, assume_positive_m=True):
        self.num_teuk_modes = num_teuk_modes
        self.assume_positive_m = assume_positive_m

        self.buffer = xp.zeros((2 * self.num_teuk_modes,), dtype=xp.complex128)

    # These are the spin-weighted spherical harmonics with s=2
    def __call__(self, l_in, m_in, theta, phi):

        if self.assume_positive_m:
            l = xp.zeros(2 * l_in.shape[0], dtype=int)
            m = xp.zeros(2 * l_in.shape[0], dtype=int)

            l[: l_in.shape[0]] = l_in
            l[l_in.shape[0] :] = l_in

            m[: l_in.shape[0]] = m_in
            m[l_in.shape[0] :] = -m_in

        else:
            l = l_in
            m = m_in

        costheta = xp.cos(theta / 2.0)
        sintheta = xp.sin(theta / 2.0)
        ylms_dict = {
            (2, -2): (xp.sqrt(5 / np.pi) * sintheta ** 4)
            / (2.0 * xp.exp(2 * 1j * phi)),
            (2, -1): (xp.sqrt(5 / np.pi) * costheta * sintheta ** 3) / xp.exp(1j * phi),
            (2, 0): xp.sqrt(15 / (2.0 * np.pi)) * costheta ** 2 * sintheta ** 2,
            (2, 1): xp.exp(1j * phi) * xp.sqrt(5 / np.pi) * costheta ** 3 * sintheta,
            (2, 2): (xp.exp(2 * 1j * phi) * xp.sqrt(5 / np.pi) * costheta ** 4) / 2.0,
            (3, -3): (xp.sqrt(21 / (2.0 * np.pi)) * costheta * sintheta ** 5)
            / xp.exp(3 * 1j * phi),
            (3, -2): (
                xp.sqrt(7 / np.pi) * (5 * costheta ** 2 * sintheta ** 4 - sintheta ** 6)
            )
            / (2.0 * xp.exp(2 * 1j * phi)),
            (3, -1): -(
                (
                    xp.sqrt(7 / (10.0 * np.pi))
                    * (
                        -10 * costheta ** 3 * sintheta ** 3
                        + 5 * costheta * sintheta ** 5
                    )
                )
                / xp.exp(1j * phi)
            ),
            (3, 0): (
                xp.sqrt(21 / (10.0 * np.pi))
                * (
                    10 * costheta ** 4 * sintheta ** 2
                    - 10 * costheta ** 2 * sintheta ** 4
                )
            )
            / 2.0,
            (3, 1): -(
                xp.exp(1j * phi)
                * xp.sqrt(7 / (10.0 * np.pi))
                * (-5 * costheta ** 5 * sintheta + 10 * costheta ** 3 * sintheta ** 3)
            ),
            (3, 2): (
                xp.exp(2 * 1j * phi)
                * xp.sqrt(7 / np.pi)
                * (costheta ** 6 - 5 * costheta ** 4 * sintheta ** 2)
            )
            / 2.0,
            (3, 3): -(
                xp.exp(3 * 1j * phi)
                * xp.sqrt(21 / (2.0 * np.pi))
                * costheta ** 5
                * sintheta
            ),
            (4, -4): (3 * xp.sqrt(7 / np.pi) * costheta ** 2 * sintheta ** 6)
            / xp.exp(4 * 1j * phi),
            (4, -3): (
                -3
                * xp.sqrt(7 / (2.0 * np.pi))
                * (-6 * costheta ** 3 * sintheta ** 5 + 2 * costheta * sintheta ** 7)
            )
            / (2.0 * xp.exp(3 * 1j * phi)),
            (4, -2): (
                3
                * (
                    15 * costheta ** 4 * sintheta ** 4
                    - 12 * costheta ** 2 * sintheta ** 6
                    + sintheta ** 8
                )
            )
            / (2.0 * xp.exp(2 * 1j * phi) * xp.sqrt(np.pi)),
            (4, -1): (
                -3
                * (
                    -20 * costheta ** 5 * sintheta ** 3
                    + 30 * costheta ** 3 * sintheta ** 5
                    - 6 * costheta * sintheta ** 7
                )
            )
            / (2.0 * xp.exp(1j * phi) * xp.sqrt(2 * np.pi)),
            (4, 0): (
                3
                * (
                    15 * costheta ** 6 * sintheta ** 2
                    - 40 * costheta ** 4 * sintheta ** 4
                    + 15 * costheta ** 2 * sintheta ** 6
                )
            )
            / xp.sqrt(10 * np.pi),
            (4, 1): (
                -3
                * xp.exp(1j * phi)
                * (
                    -6 * costheta ** 7 * sintheta
                    + 30 * costheta ** 5 * sintheta ** 3
                    - 20 * costheta ** 3 * sintheta ** 5
                )
            )
            / (2.0 * xp.sqrt(2 * np.pi)),
            (4, 2): (
                3
                * xp.exp(2 * 1j * phi)
                * (
                    costheta ** 8
                    - 12 * costheta ** 6 * sintheta ** 2
                    + 15 * costheta ** 4 * sintheta ** 4
                )
            )
            / (2.0 * xp.sqrt(np.pi)),
            (4, 3): (
                -3
                * xp.exp(3 * 1j * phi)
                * xp.sqrt(7 / (2.0 * np.pi))
                * (2 * costheta ** 7 * sintheta - 6 * costheta ** 5 * sintheta ** 3)
            )
            / 2.0,
            (4, 4): 3
            * xp.exp(4 * 1j * phi)
            * xp.sqrt(7 / np.pi)
            * costheta ** 6
            * sintheta ** 2,
            (5, -5): (xp.sqrt(330 / np.pi) * costheta ** 3 * sintheta ** 7)
            / xp.exp(5 * 1j * phi),
            (5, -4): (
                xp.sqrt(33 / np.pi)
                * (
                    7 * costheta ** 4 * sintheta ** 6
                    - 3 * costheta ** 2 * sintheta ** 8
                )
            )
            / xp.exp(4 * 1j * phi),
            (5, -3): -(
                (
                    xp.sqrt(22 / (3.0 * np.pi))
                    * (
                        -21 * costheta ** 5 * sintheta ** 5
                        + 21 * costheta ** 3 * sintheta ** 7
                        - 3 * costheta * sintheta ** 9
                    )
                )
                / xp.exp(3 * 1j * phi)
            ),
            (5, -2): (
                xp.sqrt(11 / np.pi)
                * (
                    35 * costheta ** 6 * sintheta ** 4
                    - 63 * costheta ** 4 * sintheta ** 6
                    + 21 * costheta ** 2 * sintheta ** 8
                    - sintheta ** 10
                )
            )
            / (2.0 * xp.exp(2 * 1j * phi)),
            (5, -1): -(
                (
                    xp.sqrt(11 / (7.0 * np.pi))
                    * (
                        -35 * costheta ** 7 * sintheta ** 3
                        + 105 * costheta ** 5 * sintheta ** 5
                        - 63 * costheta ** 3 * sintheta ** 7
                        + 7 * costheta * sintheta ** 9
                    )
                )
                / xp.exp(1j * phi)
            ),
            (5, 0): xp.sqrt(55 / (42.0 * np.pi))
            * (
                21 * costheta ** 8 * sintheta ** 2
                - 105 * costheta ** 6 * sintheta ** 4
                + 105 * costheta ** 4 * sintheta ** 6
                - 21 * costheta ** 2 * sintheta ** 8
            ),
            (5, 1): -(
                xp.exp(1j * phi)
                * xp.sqrt(11 / (7.0 * np.pi))
                * (
                    -7 * costheta ** 9 * sintheta
                    + 63 * costheta ** 7 * sintheta ** 3
                    - 105 * costheta ** 5 * sintheta ** 5
                    + 35 * costheta ** 3 * sintheta ** 7
                )
            ),
            (5, 2): (
                xp.exp(2 * 1j * phi)
                * xp.sqrt(11 / np.pi)
                * (
                    costheta ** 10
                    - 21 * costheta ** 8 * sintheta ** 2
                    + 63 * costheta ** 6 * sintheta ** 4
                    - 35 * costheta ** 4 * sintheta ** 6
                )
            )
            / 2.0,
            (5, 3): -(
                xp.exp(3 * 1j * phi)
                * xp.sqrt(22 / (3.0 * np.pi))
                * (
                    3 * costheta ** 9 * sintheta
                    - 21 * costheta ** 7 * sintheta ** 3
                    + 21 * costheta ** 5 * sintheta ** 5
                )
            ),
            (5, 4): xp.exp(4 * 1j * phi)
            * xp.sqrt(33 / np.pi)
            * (3 * costheta ** 8 * sintheta ** 2 - 7 * costheta ** 6 * sintheta ** 4),
            (5, 5): -(
                xp.exp(5 * 1j * phi)
                * xp.sqrt(330 / np.pi)
                * costheta ** 7
                * sintheta ** 3
            ),
            (6, -6): (3 * xp.sqrt(715 / np.pi) * costheta ** 4 * sintheta ** 8)
            / (2.0 * xp.exp(6 * 1j * phi)),
            (6, -5): -(
                xp.sqrt(2145 / np.pi)
                * (
                    -8 * costheta ** 5 * sintheta ** 7
                    + 4 * costheta ** 3 * sintheta ** 9
                )
            )
            / (4.0 * xp.exp(5 * 1j * phi)),
            (6, -4): (
                xp.sqrt(195 / (2.0 * np.pi))
                * (
                    28 * costheta ** 6 * sintheta ** 6
                    - 32 * costheta ** 4 * sintheta ** 8
                    + 6 * costheta ** 2 * sintheta ** 10
                )
            )
            / (2.0 * xp.exp(4 * 1j * phi)),
            (6, -3): (
                -3
                * xp.sqrt(13 / np.pi)
                * (
                    -56 * costheta ** 7 * sintheta ** 5
                    + 112 * costheta ** 5 * sintheta ** 7
                    - 48 * costheta ** 3 * sintheta ** 9
                    + 4 * costheta * sintheta ** 11
                )
            )
            / (4.0 * xp.exp(3 * 1j * phi)),
            (6, -2): (
                xp.sqrt(13 / np.pi)
                * (
                    70 * costheta ** 8 * sintheta ** 4
                    - 224 * costheta ** 6 * sintheta ** 6
                    + 168 * costheta ** 4 * sintheta ** 8
                    - 32 * costheta ** 2 * sintheta ** 10
                    + sintheta ** 12
                )
            )
            / (2.0 * xp.exp(2 * 1j * phi)),
            (6, -1): -(
                xp.sqrt(65 / (2.0 * np.pi))
                * (
                    -56 * costheta ** 9 * sintheta ** 3
                    + 280 * costheta ** 7 * sintheta ** 5
                    - 336 * costheta ** 5 * sintheta ** 7
                    + 112 * costheta ** 3 * sintheta ** 9
                    - 8 * costheta * sintheta ** 11
                )
            )
            / (4.0 * xp.exp(1j * phi)),
            (6, 0): (
                xp.sqrt(195 / (7.0 * np.pi))
                * (
                    28 * costheta ** 10 * sintheta ** 2
                    - 224 * costheta ** 8 * sintheta ** 4
                    + 420 * costheta ** 6 * sintheta ** 6
                    - 224 * costheta ** 4 * sintheta ** 8
                    + 28 * costheta ** 2 * sintheta ** 10
                )
            )
            / 4.0,
            (6, 1): -(
                xp.exp(1j * phi)
                * xp.sqrt(65 / (2.0 * np.pi))
                * (
                    -8 * costheta ** 11 * sintheta
                    + 112 * costheta ** 9 * sintheta ** 3
                    - 336 * costheta ** 7 * sintheta ** 5
                    + 280 * costheta ** 5 * sintheta ** 7
                    - 56 * costheta ** 3 * sintheta ** 9
                )
            )
            / 4.0,
            (6, 2): (
                xp.exp(2 * 1j * phi)
                * xp.sqrt(13 / np.pi)
                * (
                    costheta ** 12
                    - 32 * costheta ** 10 * sintheta ** 2
                    + 168 * costheta ** 8 * sintheta ** 4
                    - 224 * costheta ** 6 * sintheta ** 6
                    + 70 * costheta ** 4 * sintheta ** 8
                )
            )
            / 2.0,
            (6, 3): (
                -3
                * xp.exp(3 * 1j * phi)
                * xp.sqrt(13 / np.pi)
                * (
                    4 * costheta ** 11 * sintheta
                    - 48 * costheta ** 9 * sintheta ** 3
                    + 112 * costheta ** 7 * sintheta ** 5
                    - 56 * costheta ** 5 * sintheta ** 7
                )
            )
            / 4.0,
            (6, 4): (
                xp.exp(4 * 1j * phi)
                * xp.sqrt(195 / (2.0 * np.pi))
                * (
                    6 * costheta ** 10 * sintheta ** 2
                    - 32 * costheta ** 8 * sintheta ** 4
                    + 28 * costheta ** 6 * sintheta ** 6
                )
            )
            / 2.0,
            (6, 5): -(
                xp.exp(5 * 1j * phi)
                * xp.sqrt(2145 / np.pi)
                * (
                    4 * costheta ** 9 * sintheta ** 3
                    - 8 * costheta ** 7 * sintheta ** 5
                )
            )
            / 4.0,
            (6, 6): (
                3
                * xp.exp(6 * 1j * phi)
                * xp.sqrt(715 / np.pi)
                * costheta ** 8
                * sintheta ** 4
            )
            / 2.0,
            (7, -7): (xp.sqrt(15015 / (2.0 * np.pi)) * costheta ** 5 * sintheta ** 9)
            / xp.exp(7 * 1j * phi),
            (7, -6): (
                xp.sqrt(2145 / np.pi)
                * (
                    9 * costheta ** 6 * sintheta ** 8
                    - 5 * costheta ** 4 * sintheta ** 10
                )
            )
            / (2.0 * xp.exp(6 * 1j * phi)),
            (7, -5): -(
                (
                    xp.sqrt(165 / (2.0 * np.pi))
                    * (
                        -36 * costheta ** 7 * sintheta ** 7
                        + 45 * costheta ** 5 * sintheta ** 9
                        - 10 * costheta ** 3 * sintheta ** 11
                    )
                )
                / xp.exp(5 * 1j * phi)
            ),
            (7, -4): (
                xp.sqrt(165 / (2.0 * np.pi))
                * (
                    84 * costheta ** 8 * sintheta ** 6
                    - 180 * costheta ** 6 * sintheta ** 8
                    + 90 * costheta ** 4 * sintheta ** 10
                    - 10 * costheta ** 2 * sintheta ** 12
                )
            )
            / (2.0 * xp.exp(4 * 1j * phi)),
            (7, -3): -(
                (
                    xp.sqrt(15 / (2.0 * np.pi))
                    * (
                        -126 * costheta ** 9 * sintheta ** 5
                        + 420 * costheta ** 7 * sintheta ** 7
                        - 360 * costheta ** 5 * sintheta ** 9
                        + 90 * costheta ** 3 * sintheta ** 11
                        - 5 * costheta * sintheta ** 13
                    )
                )
                / xp.exp(3 * 1j * phi)
            ),
            (7, -2): (
                xp.sqrt(15 / np.pi)
                * (
                    126 * costheta ** 10 * sintheta ** 4
                    - 630 * costheta ** 8 * sintheta ** 6
                    + 840 * costheta ** 6 * sintheta ** 8
                    - 360 * costheta ** 4 * sintheta ** 10
                    + 45 * costheta ** 2 * sintheta ** 12
                    - sintheta ** 14
                )
            )
            / (2.0 * xp.exp(2 * 1j * phi)),
            (7, -1): -(
                (
                    xp.sqrt(5 / (2.0 * np.pi))
                    * (
                        -84 * costheta ** 11 * sintheta ** 3
                        + 630 * costheta ** 9 * sintheta ** 5
                        - 1260 * costheta ** 7 * sintheta ** 7
                        + 840 * costheta ** 5 * sintheta ** 9
                        - 180 * costheta ** 3 * sintheta ** 11
                        + 9 * costheta * sintheta ** 13
                    )
                )
                / xp.exp(1j * phi)
            ),
            (7, 0): (
                xp.sqrt(35 / np.pi)
                * (
                    36 * costheta ** 12 * sintheta ** 2
                    - 420 * costheta ** 10 * sintheta ** 4
                    + 1260 * costheta ** 8 * sintheta ** 6
                    - 1260 * costheta ** 6 * sintheta ** 8
                    + 420 * costheta ** 4 * sintheta ** 10
                    - 36 * costheta ** 2 * sintheta ** 12
                )
            )
            / 4.0,
            (7, 1): -(
                xp.exp(1j * phi)
                * xp.sqrt(5 / (2.0 * np.pi))
                * (
                    -9 * costheta ** 13 * sintheta
                    + 180 * costheta ** 11 * sintheta ** 3
                    - 840 * costheta ** 9 * sintheta ** 5
                    + 1260 * costheta ** 7 * sintheta ** 7
                    - 630 * costheta ** 5 * sintheta ** 9
                    + 84 * costheta ** 3 * sintheta ** 11
                )
            ),
            (7, 2): (
                xp.exp(2 * 1j * phi)
                * xp.sqrt(15 / np.pi)
                * (
                    costheta ** 14
                    - 45 * costheta ** 12 * sintheta ** 2
                    + 360 * costheta ** 10 * sintheta ** 4
                    - 840 * costheta ** 8 * sintheta ** 6
                    + 630 * costheta ** 6 * sintheta ** 8
                    - 126 * costheta ** 4 * sintheta ** 10
                )
            )
            / 2.0,
            (7, 3): -(
                xp.exp(3 * 1j * phi)
                * xp.sqrt(15 / (2.0 * np.pi))
                * (
                    5 * costheta ** 13 * sintheta
                    - 90 * costheta ** 11 * sintheta ** 3
                    + 360 * costheta ** 9 * sintheta ** 5
                    - 420 * costheta ** 7 * sintheta ** 7
                    + 126 * costheta ** 5 * sintheta ** 9
                )
            ),
            (7, 4): (
                xp.exp(4 * 1j * phi)
                * xp.sqrt(165 / (2.0 * np.pi))
                * (
                    10 * costheta ** 12 * sintheta ** 2
                    - 90 * costheta ** 10 * sintheta ** 4
                    + 180 * costheta ** 8 * sintheta ** 6
                    - 84 * costheta ** 6 * sintheta ** 8
                )
            )
            / 2.0,
            (7, 5): -(
                xp.exp(5 * 1j * phi)
                * xp.sqrt(165 / (2.0 * np.pi))
                * (
                    10 * costheta ** 11 * sintheta ** 3
                    - 45 * costheta ** 9 * sintheta ** 5
                    + 36 * costheta ** 7 * sintheta ** 7
                )
            ),
            (7, 6): (
                xp.exp(6 * 1j * phi)
                * xp.sqrt(2145 / np.pi)
                * (
                    5 * costheta ** 10 * sintheta ** 4
                    - 9 * costheta ** 8 * sintheta ** 6
                )
            )
            / 2.0,
            (7, 7): -(
                xp.exp(7 * 1j * phi)
                * xp.sqrt(15015 / (2.0 * np.pi))
                * costheta ** 9
                * sintheta ** 5
            ),
            (8, -8): (xp.sqrt(34034 / np.pi) * costheta ** 6 * sintheta ** 10)
            / xp.exp(8 * 1j * phi),
            (8, -7): -(
                xp.sqrt(17017 / (2.0 * np.pi))
                * (
                    -10 * costheta ** 7 * sintheta ** 9
                    + 6 * costheta ** 5 * sintheta ** 11
                )
            )
            / (2.0 * xp.exp(7 * 1j * phi)),
            (8, -6): (
                xp.sqrt(17017 / (15.0 * np.pi))
                * (
                    45 * costheta ** 8 * sintheta ** 8
                    - 60 * costheta ** 6 * sintheta ** 10
                    + 15 * costheta ** 4 * sintheta ** 12
                )
            )
            / (2.0 * xp.exp(6 * 1j * phi)),
            (8, -5): -(
                xp.sqrt(2431 / (10.0 * np.pi))
                * (
                    -120 * costheta ** 9 * sintheta ** 7
                    + 270 * costheta ** 7 * sintheta ** 9
                    - 150 * costheta ** 5 * sintheta ** 11
                    + 20 * costheta ** 3 * sintheta ** 13
                )
            )
            / (2.0 * xp.exp(5 * 1j * phi)),
            (8, -4): (
                xp.sqrt(187 / (10.0 * np.pi))
                * (
                    210 * costheta ** 10 * sintheta ** 6
                    - 720 * costheta ** 8 * sintheta ** 8
                    + 675 * costheta ** 6 * sintheta ** 10
                    - 200 * costheta ** 4 * sintheta ** 12
                    + 15 * costheta ** 2 * sintheta ** 14
                )
            )
            / xp.exp(4 * 1j * phi),
            (8, -3): -(
                xp.sqrt(187 / (6.0 * np.pi))
                * (
                    -252 * costheta ** 11 * sintheta ** 5
                    + 1260 * costheta ** 9 * sintheta ** 7
                    - 1800 * costheta ** 7 * sintheta ** 9
                    + 900 * costheta ** 5 * sintheta ** 11
                    - 150 * costheta ** 3 * sintheta ** 13
                    + 6 * costheta * sintheta ** 15
                )
            )
            / (2.0 * xp.exp(3 * 1j * phi)),
            (8, -2): (
                xp.sqrt(17 / np.pi)
                * (
                    210 * costheta ** 12 * sintheta ** 4
                    - 1512 * costheta ** 10 * sintheta ** 6
                    + 3150 * costheta ** 8 * sintheta ** 8
                    - 2400 * costheta ** 6 * sintheta ** 10
                    + 675 * costheta ** 4 * sintheta ** 12
                    - 60 * costheta ** 2 * sintheta ** 14
                    + sintheta ** 16
                )
            )
            / (2.0 * xp.exp(2 * 1j * phi)),
            (8, -1): -(
                xp.sqrt(119 / (10.0 * np.pi))
                * (
                    -120 * costheta ** 13 * sintheta ** 3
                    + 1260 * costheta ** 11 * sintheta ** 5
                    - 3780 * costheta ** 9 * sintheta ** 7
                    + 4200 * costheta ** 7 * sintheta ** 9
                    - 1800 * costheta ** 5 * sintheta ** 11
                    + 270 * costheta ** 3 * sintheta ** 13
                    - 10 * costheta * sintheta ** 15
                )
            )
            / (2.0 * xp.exp(1j * phi)),
            (8, 0): (
                xp.sqrt(119 / (5.0 * np.pi))
                * (
                    45 * costheta ** 14 * sintheta ** 2
                    - 720 * costheta ** 12 * sintheta ** 4
                    + 3150 * costheta ** 10 * sintheta ** 6
                    - 5040 * costheta ** 8 * sintheta ** 8
                    + 3150 * costheta ** 6 * sintheta ** 10
                    - 720 * costheta ** 4 * sintheta ** 12
                    + 45 * costheta ** 2 * sintheta ** 14
                )
            )
            / 3.0,
            (8, 1): -(
                xp.exp(1j * phi)
                * xp.sqrt(119 / (10.0 * np.pi))
                * (
                    -10 * costheta ** 15 * sintheta
                    + 270 * costheta ** 13 * sintheta ** 3
                    - 1800 * costheta ** 11 * sintheta ** 5
                    + 4200 * costheta ** 9 * sintheta ** 7
                    - 3780 * costheta ** 7 * sintheta ** 9
                    + 1260 * costheta ** 5 * sintheta ** 11
                    - 120 * costheta ** 3 * sintheta ** 13
                )
            )
            / 2.0,
            (8, 2): (
                xp.exp(2 * 1j * phi)
                * xp.sqrt(17 / np.pi)
                * (
                    costheta ** 16
                    - 60 * costheta ** 14 * sintheta ** 2
                    + 675 * costheta ** 12 * sintheta ** 4
                    - 2400 * costheta ** 10 * sintheta ** 6
                    + 3150 * costheta ** 8 * sintheta ** 8
                    - 1512 * costheta ** 6 * sintheta ** 10
                    + 210 * costheta ** 4 * sintheta ** 12
                )
            )
            / 2.0,
            (8, 3): -(
                xp.exp(3 * 1j * phi)
                * xp.sqrt(187 / (6.0 * np.pi))
                * (
                    6 * costheta ** 15 * sintheta
                    - 150 * costheta ** 13 * sintheta ** 3
                    + 900 * costheta ** 11 * sintheta ** 5
                    - 1800 * costheta ** 9 * sintheta ** 7
                    + 1260 * costheta ** 7 * sintheta ** 9
                    - 252 * costheta ** 5 * sintheta ** 11
                )
            )
            / 2.0,
            (8, 4): xp.exp(4 * 1j * phi)
            * xp.sqrt(187 / (10.0 * np.pi))
            * (
                15 * costheta ** 14 * sintheta ** 2
                - 200 * costheta ** 12 * sintheta ** 4
                + 675 * costheta ** 10 * sintheta ** 6
                - 720 * costheta ** 8 * sintheta ** 8
                + 210 * costheta ** 6 * sintheta ** 10
            ),
            (8, 5): -(
                xp.exp(5 * 1j * phi)
                * xp.sqrt(2431 / (10.0 * np.pi))
                * (
                    20 * costheta ** 13 * sintheta ** 3
                    - 150 * costheta ** 11 * sintheta ** 5
                    + 270 * costheta ** 9 * sintheta ** 7
                    - 120 * costheta ** 7 * sintheta ** 9
                )
            )
            / 2.0,
            (8, 6): (
                xp.exp(6 * 1j * phi)
                * xp.sqrt(17017 / (15.0 * np.pi))
                * (
                    15 * costheta ** 12 * sintheta ** 4
                    - 60 * costheta ** 10 * sintheta ** 6
                    + 45 * costheta ** 8 * sintheta ** 8
                )
            )
            / 2.0,
            (8, 7): -(
                xp.exp(7 * 1j * phi)
                * xp.sqrt(17017 / (2.0 * np.pi))
                * (
                    6 * costheta ** 11 * sintheta ** 5
                    - 10 * costheta ** 9 * sintheta ** 7
                )
            )
            / 2.0,
            (8, 8): xp.exp(8 * 1j * phi)
            * xp.sqrt(34034 / np.pi)
            * costheta ** 10
            * sintheta ** 6,
            (9, -9): (6 * xp.sqrt(4199 / np.pi) * costheta ** 7 * sintheta ** 11)
            / xp.exp(9 * 1j * phi),
            (9, -8): (
                xp.sqrt(8398 / np.pi)
                * (
                    11 * costheta ** 8 * sintheta ** 10
                    - 7 * costheta ** 6 * sintheta ** 12
                )
            )
            / xp.exp(8 * 1j * phi),
            (9, -7): (
                -2
                * xp.sqrt(247 / np.pi)
                * (
                    -55 * costheta ** 9 * sintheta ** 9
                    + 77 * costheta ** 7 * sintheta ** 11
                    - 21 * costheta ** 5 * sintheta ** 13
                )
            )
            / xp.exp(7 * 1j * phi),
            (9, -6): (
                xp.sqrt(741 / np.pi)
                * (
                    165 * costheta ** 10 * sintheta ** 8
                    - 385 * costheta ** 8 * sintheta ** 10
                    + 231 * costheta ** 6 * sintheta ** 12
                    - 35 * costheta ** 4 * sintheta ** 14
                )
            )
            / (2.0 * xp.exp(6 * 1j * phi)),
            (9, -5): -(
                (
                    xp.sqrt(247 / (5.0 * np.pi))
                    * (
                        -330 * costheta ** 11 * sintheta ** 7
                        + 1155 * costheta ** 9 * sintheta ** 9
                        - 1155 * costheta ** 7 * sintheta ** 11
                        + 385 * costheta ** 5 * sintheta ** 13
                        - 35 * costheta ** 3 * sintheta ** 15
                    )
                )
                / xp.exp(5 * 1j * phi)
            ),
            (9, -4): (
                xp.sqrt(247 / (14.0 * np.pi))
                * (
                    462 * costheta ** 12 * sintheta ** 6
                    - 2310 * costheta ** 10 * sintheta ** 8
                    + 3465 * costheta ** 8 * sintheta ** 10
                    - 1925 * costheta ** 6 * sintheta ** 12
                    + 385 * costheta ** 4 * sintheta ** 14
                    - 21 * costheta ** 2 * sintheta ** 16
                )
            )
            / xp.exp(4 * 1j * phi),
            (9, -3): -(
                (
                    xp.sqrt(57 / (7.0 * np.pi))
                    * (
                        -462 * costheta ** 13 * sintheta ** 5
                        + 3234 * costheta ** 11 * sintheta ** 7
                        - 6930 * costheta ** 9 * sintheta ** 9
                        + 5775 * costheta ** 7 * sintheta ** 11
                        - 1925 * costheta ** 5 * sintheta ** 13
                        + 231 * costheta ** 3 * sintheta ** 15
                        - 7 * costheta * sintheta ** 17
                    )
                )
                / xp.exp(3 * 1j * phi)
            ),
            (9, -2): (
                xp.sqrt(19 / np.pi)
                * (
                    330 * costheta ** 14 * sintheta ** 4
                    - 3234 * costheta ** 12 * sintheta ** 6
                    + 9702 * costheta ** 10 * sintheta ** 8
                    - 11550 * costheta ** 8 * sintheta ** 10
                    + 5775 * costheta ** 6 * sintheta ** 12
                    - 1155 * costheta ** 4 * sintheta ** 14
                    + 77 * costheta ** 2 * sintheta ** 16
                    - sintheta ** 18
                )
            )
            / (2.0 * xp.exp(2 * 1j * phi)),
            (9, -1): -(
                (
                    xp.sqrt(38 / (11.0 * np.pi))
                    * (
                        -165 * costheta ** 15 * sintheta ** 3
                        + 2310 * costheta ** 13 * sintheta ** 5
                        - 9702 * costheta ** 11 * sintheta ** 7
                        + 16170 * costheta ** 9 * sintheta ** 9
                        - 11550 * costheta ** 7 * sintheta ** 11
                        + 3465 * costheta ** 5 * sintheta ** 13
                        - 385 * costheta ** 3 * sintheta ** 15
                        + 11 * costheta * sintheta ** 17
                    )
                )
                / xp.exp(1j * phi)
            ),
            (9, 0): 3
            * xp.sqrt(19 / (55.0 * np.pi))
            * (
                55 * costheta ** 16 * sintheta ** 2
                - 1155 * costheta ** 14 * sintheta ** 4
                + 6930 * costheta ** 12 * sintheta ** 6
                - 16170 * costheta ** 10 * sintheta ** 8
                + 16170 * costheta ** 8 * sintheta ** 10
                - 6930 * costheta ** 6 * sintheta ** 12
                + 1155 * costheta ** 4 * sintheta ** 14
                - 55 * costheta ** 2 * sintheta ** 16
            ),
            (9, 1): -(
                xp.exp(1j * phi)
                * xp.sqrt(38 / (11.0 * np.pi))
                * (
                    -11 * costheta ** 17 * sintheta
                    + 385 * costheta ** 15 * sintheta ** 3
                    - 3465 * costheta ** 13 * sintheta ** 5
                    + 11550 * costheta ** 11 * sintheta ** 7
                    - 16170 * costheta ** 9 * sintheta ** 9
                    + 9702 * costheta ** 7 * sintheta ** 11
                    - 2310 * costheta ** 5 * sintheta ** 13
                    + 165 * costheta ** 3 * sintheta ** 15
                )
            ),
            (9, 2): (
                xp.exp(2 * 1j * phi)
                * xp.sqrt(19 / np.pi)
                * (
                    costheta ** 18
                    - 77 * costheta ** 16 * sintheta ** 2
                    + 1155 * costheta ** 14 * sintheta ** 4
                    - 5775 * costheta ** 12 * sintheta ** 6
                    + 11550 * costheta ** 10 * sintheta ** 8
                    - 9702 * costheta ** 8 * sintheta ** 10
                    + 3234 * costheta ** 6 * sintheta ** 12
                    - 330 * costheta ** 4 * sintheta ** 14
                )
            )
            / 2.0,
            (9, 3): -(
                xp.exp(3 * 1j * phi)
                * xp.sqrt(57 / (7.0 * np.pi))
                * (
                    7 * costheta ** 17 * sintheta
                    - 231 * costheta ** 15 * sintheta ** 3
                    + 1925 * costheta ** 13 * sintheta ** 5
                    - 5775 * costheta ** 11 * sintheta ** 7
                    + 6930 * costheta ** 9 * sintheta ** 9
                    - 3234 * costheta ** 7 * sintheta ** 11
                    + 462 * costheta ** 5 * sintheta ** 13
                )
            ),
            (9, 4): xp.exp(4 * 1j * phi)
            * xp.sqrt(247 / (14.0 * np.pi))
            * (
                21 * costheta ** 16 * sintheta ** 2
                - 385 * costheta ** 14 * sintheta ** 4
                + 1925 * costheta ** 12 * sintheta ** 6
                - 3465 * costheta ** 10 * sintheta ** 8
                + 2310 * costheta ** 8 * sintheta ** 10
                - 462 * costheta ** 6 * sintheta ** 12
            ),
            (9, 5): -(
                xp.exp(5 * 1j * phi)
                * xp.sqrt(247 / (5.0 * np.pi))
                * (
                    35 * costheta ** 15 * sintheta ** 3
                    - 385 * costheta ** 13 * sintheta ** 5
                    + 1155 * costheta ** 11 * sintheta ** 7
                    - 1155 * costheta ** 9 * sintheta ** 9
                    + 330 * costheta ** 7 * sintheta ** 11
                )
            ),
            (9, 6): (
                xp.exp(6 * 1j * phi)
                * xp.sqrt(741 / np.pi)
                * (
                    35 * costheta ** 14 * sintheta ** 4
                    - 231 * costheta ** 12 * sintheta ** 6
                    + 385 * costheta ** 10 * sintheta ** 8
                    - 165 * costheta ** 8 * sintheta ** 10
                )
            )
            / 2.0,
            (9, 7): -2
            * xp.exp(7 * 1j * phi)
            * xp.sqrt(247 / np.pi)
            * (
                21 * costheta ** 13 * sintheta ** 5
                - 77 * costheta ** 11 * sintheta ** 7
                + 55 * costheta ** 9 * sintheta ** 9
            ),
            (9, 8): xp.exp(8 * 1j * phi)
            * xp.sqrt(8398 / np.pi)
            * (
                7 * costheta ** 12 * sintheta ** 6 - 11 * costheta ** 10 * sintheta ** 8
            ),
            (9, 9): -6
            * xp.exp(9 * 1j * phi)
            * xp.sqrt(4199 / np.pi)
            * costheta ** 11
            * sintheta ** 7,
            (10, -10): (
                3 * xp.sqrt(146965 / (2.0 * np.pi)) * costheta ** 8 * sintheta ** 12
            )
            / xp.exp(10 * 1j * phi),
            (10, -9): (
                -3
                * xp.sqrt(29393 / (2.0 * np.pi))
                * (
                    -12 * costheta ** 9 * sintheta ** 11
                    + 8 * costheta ** 7 * sintheta ** 13
                )
            )
            / (2.0 * xp.exp(9 * 1j * phi)),
            (10, -8): (
                3
                * xp.sqrt(1547 / np.pi)
                * (
                    66 * costheta ** 10 * sintheta ** 10
                    - 96 * costheta ** 8 * sintheta ** 12
                    + 28 * costheta ** 6 * sintheta ** 14
                )
            )
            / (2.0 * xp.exp(8 * 1j * phi)),
            (10, -7): -(
                xp.sqrt(4641 / (2.0 * np.pi))
                * (
                    -220 * costheta ** 11 * sintheta ** 9
                    + 528 * costheta ** 9 * sintheta ** 11
                    - 336 * costheta ** 7 * sintheta ** 13
                    + 56 * costheta ** 5 * sintheta ** 15
                )
            )
            / (2.0 * xp.exp(7 * 1j * phi)),
            (10, -6): (
                xp.sqrt(273 / (2.0 * np.pi))
                * (
                    495 * costheta ** 12 * sintheta ** 8
                    - 1760 * costheta ** 10 * sintheta ** 10
                    + 1848 * costheta ** 8 * sintheta ** 12
                    - 672 * costheta ** 6 * sintheta ** 14
                    + 70 * costheta ** 4 * sintheta ** 16
                )
            )
            / xp.exp(6 * 1j * phi),
            (10, -5): -(
                xp.sqrt(1365 / (2.0 * np.pi))
                * (
                    -792 * costheta ** 13 * sintheta ** 7
                    + 3960 * costheta ** 11 * sintheta ** 9
                    - 6160 * costheta ** 9 * sintheta ** 11
                    + 3696 * costheta ** 7 * sintheta ** 13
                    - 840 * costheta ** 5 * sintheta ** 15
                    + 56 * costheta ** 3 * sintheta ** 17
                )
            )
            / (4.0 * xp.exp(5 * 1j * phi)),
            (10, -4): (
                xp.sqrt(273 / np.pi)
                * (
                    924 * costheta ** 14 * sintheta ** 6
                    - 6336 * costheta ** 12 * sintheta ** 8
                    + 13860 * costheta ** 10 * sintheta ** 10
                    - 12320 * costheta ** 8 * sintheta ** 12
                    + 4620 * costheta ** 6 * sintheta ** 14
                    - 672 * costheta ** 4 * sintheta ** 16
                    + 28 * costheta ** 2 * sintheta ** 18
                )
            )
            / (4.0 * xp.exp(4 * 1j * phi)),
            (10, -3): -(
                xp.sqrt(273 / (2.0 * np.pi))
                * (
                    -792 * costheta ** 15 * sintheta ** 5
                    + 7392 * costheta ** 13 * sintheta ** 7
                    - 22176 * costheta ** 11 * sintheta ** 9
                    + 27720 * costheta ** 9 * sintheta ** 11
                    - 15400 * costheta ** 7 * sintheta ** 13
                    + 3696 * costheta ** 5 * sintheta ** 15
                    - 336 * costheta ** 3 * sintheta ** 17
                    + 8 * costheta * sintheta ** 19
                )
            )
            / (4.0 * xp.exp(3 * 1j * phi)),
            (10, -2): (
                xp.sqrt(21 / np.pi)
                * (
                    495 * costheta ** 16 * sintheta ** 4
                    - 6336 * costheta ** 14 * sintheta ** 6
                    + 25872 * costheta ** 12 * sintheta ** 8
                    - 44352 * costheta ** 10 * sintheta ** 10
                    + 34650 * costheta ** 8 * sintheta ** 12
                    - 12320 * costheta ** 6 * sintheta ** 14
                    + 1848 * costheta ** 4 * sintheta ** 16
                    - 96 * costheta ** 2 * sintheta ** 18
                    + sintheta ** 20
                )
            )
            / (2.0 * xp.exp(2 * 1j * phi)),
            (10, -1): (
                -3
                * xp.sqrt(7 / np.pi)
                * (
                    -220 * costheta ** 17 * sintheta ** 3
                    + 3960 * costheta ** 15 * sintheta ** 5
                    - 22176 * costheta ** 13 * sintheta ** 7
                    + 51744 * costheta ** 11 * sintheta ** 9
                    - 55440 * costheta ** 9 * sintheta ** 11
                    + 27720 * costheta ** 7 * sintheta ** 13
                    - 6160 * costheta ** 5 * sintheta ** 15
                    + 528 * costheta ** 3 * sintheta ** 17
                    - 12 * costheta * sintheta ** 19
                )
            )
            / (4.0 * xp.exp(1j * phi)),
            (10, 0): (
                3
                * xp.sqrt(35 / (22.0 * np.pi))
                * (
                    66 * costheta ** 18 * sintheta ** 2
                    - 1760 * costheta ** 16 * sintheta ** 4
                    + 13860 * costheta ** 14 * sintheta ** 6
                    - 44352 * costheta ** 12 * sintheta ** 8
                    + 64680 * costheta ** 10 * sintheta ** 10
                    - 44352 * costheta ** 8 * sintheta ** 12
                    + 13860 * costheta ** 6 * sintheta ** 14
                    - 1760 * costheta ** 4 * sintheta ** 16
                    + 66 * costheta ** 2 * sintheta ** 18
                )
            )
            / 2.0,
            (10, 1): (
                -3
                * xp.exp(1j * phi)
                * xp.sqrt(7 / np.pi)
                * (
                    -12 * costheta ** 19 * sintheta
                    + 528 * costheta ** 17 * sintheta ** 3
                    - 6160 * costheta ** 15 * sintheta ** 5
                    + 27720 * costheta ** 13 * sintheta ** 7
                    - 55440 * costheta ** 11 * sintheta ** 9
                    + 51744 * costheta ** 9 * sintheta ** 11
                    - 22176 * costheta ** 7 * sintheta ** 13
                    + 3960 * costheta ** 5 * sintheta ** 15
                    - 220 * costheta ** 3 * sintheta ** 17
                )
            )
            / 4.0,
            (10, 2): (
                xp.exp(2 * 1j * phi)
                * xp.sqrt(21 / np.pi)
                * (
                    costheta ** 20
                    - 96 * costheta ** 18 * sintheta ** 2
                    + 1848 * costheta ** 16 * sintheta ** 4
                    - 12320 * costheta ** 14 * sintheta ** 6
                    + 34650 * costheta ** 12 * sintheta ** 8
                    - 44352 * costheta ** 10 * sintheta ** 10
                    + 25872 * costheta ** 8 * sintheta ** 12
                    - 6336 * costheta ** 6 * sintheta ** 14
                    + 495 * costheta ** 4 * sintheta ** 16
                )
            )
            / 2.0,
            (10, 3): -(
                xp.exp(3 * 1j * phi)
                * xp.sqrt(273 / (2.0 * np.pi))
                * (
                    8 * costheta ** 19 * sintheta
                    - 336 * costheta ** 17 * sintheta ** 3
                    + 3696 * costheta ** 15 * sintheta ** 5
                    - 15400 * costheta ** 13 * sintheta ** 7
                    + 27720 * costheta ** 11 * sintheta ** 9
                    - 22176 * costheta ** 9 * sintheta ** 11
                    + 7392 * costheta ** 7 * sintheta ** 13
                    - 792 * costheta ** 5 * sintheta ** 15
                )
            )
            / 4.0,
            (10, 4): (
                xp.exp(4 * 1j * phi)
                * xp.sqrt(273 / np.pi)
                * (
                    28 * costheta ** 18 * sintheta ** 2
                    - 672 * costheta ** 16 * sintheta ** 4
                    + 4620 * costheta ** 14 * sintheta ** 6
                    - 12320 * costheta ** 12 * sintheta ** 8
                    + 13860 * costheta ** 10 * sintheta ** 10
                    - 6336 * costheta ** 8 * sintheta ** 12
                    + 924 * costheta ** 6 * sintheta ** 14
                )
            )
            / 4.0,
            (10, 5): -(
                xp.exp(5 * 1j * phi)
                * xp.sqrt(1365 / (2.0 * np.pi))
                * (
                    56 * costheta ** 17 * sintheta ** 3
                    - 840 * costheta ** 15 * sintheta ** 5
                    + 3696 * costheta ** 13 * sintheta ** 7
                    - 6160 * costheta ** 11 * sintheta ** 9
                    + 3960 * costheta ** 9 * sintheta ** 11
                    - 792 * costheta ** 7 * sintheta ** 13
                )
            )
            / 4.0,
            (10, 6): xp.exp(6 * 1j * phi)
            * xp.sqrt(273 / (2.0 * np.pi))
            * (
                70 * costheta ** 16 * sintheta ** 4
                - 672 * costheta ** 14 * sintheta ** 6
                + 1848 * costheta ** 12 * sintheta ** 8
                - 1760 * costheta ** 10 * sintheta ** 10
                + 495 * costheta ** 8 * sintheta ** 12
            ),
            (10, 7): -(
                xp.exp(7 * 1j * phi)
                * xp.sqrt(4641 / (2.0 * np.pi))
                * (
                    56 * costheta ** 15 * sintheta ** 5
                    - 336 * costheta ** 13 * sintheta ** 7
                    + 528 * costheta ** 11 * sintheta ** 9
                    - 220 * costheta ** 9 * sintheta ** 11
                )
            )
            / 2.0,
            (10, 8): (
                3
                * xp.exp(8 * 1j * phi)
                * xp.sqrt(1547 / np.pi)
                * (
                    28 * costheta ** 14 * sintheta ** 6
                    - 96 * costheta ** 12 * sintheta ** 8
                    + 66 * costheta ** 10 * sintheta ** 10
                )
            )
            / 2.0,
            (10, 9): (
                -3
                * xp.exp(9 * 1j * phi)
                * xp.sqrt(29393 / (2.0 * np.pi))
                * (
                    8 * costheta ** 13 * sintheta ** 7
                    - 12 * costheta ** 11 * sintheta ** 9
                )
            )
            / 2.0,
            (10, 10): 3
            * xp.exp(10 * 1j * phi)
            * xp.sqrt(146965 / (2.0 * np.pi))
            * costheta ** 12
            * sintheta ** 8,
        }
        for i, (l_i, m_i) in enumerate(zip(l, m)):
            temp = ylms_dict[(l_i.item(), m_i.item())]
            self.buffer[i] = temp * (-1) ** l_i if m_i < 0 else temp

        return self.buffer[: len(l)]
