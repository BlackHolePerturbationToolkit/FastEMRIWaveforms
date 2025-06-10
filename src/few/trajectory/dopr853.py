import numpy as np

# try:
#     import cupy as xp
#     from .pydopr853 import dormandPrinceSteps as dormandPrinceSteps_gpu
#     from .pydopr853 import error as error_gpu
#     from .pydopr853 import controllerSuccess as controllerSuccess_gpu
#     gpu_available = True
#     from cupy.cuda.runtime import setDevice
#     #setDevice(4)

# except ModuleNotFoundError:
#     import numpy as xp
#     gpu_available = False

"""from .pydopr853_cpu import dormandPrinceSteps as dormandPrinceSteps_cpu
from .pydopr853_cpu import error as error_cpu
from .pydopr853_cpu import controllerSuccess as controllerSuccess_cpu
"""

np.random.seed(5)

# Tolerances
# rtol = 0#1e-17
# abstol = 1e-12

# Coefficients for using in Dormand Prince Solver
c2 = 0.526001519587677318785587544488e-01
c3 = 0.789002279381515978178381316732e-01
c4 = 0.118350341907227396726757197510e00
c5 = 0.281649658092772603273242802490e00
c6 = 0.333333333333333333333333333333e00
c7 = 0.25e00
c8 = 0.307692307692307692307692307692e00
c9 = 0.651282051282051282051282051282e00
c10 = 0.6e00
c11 = 0.857142857142857142857142857142e00
c14 = 0.1e00
c15 = 0.2e00
c16 = 0.777777777777777777777777777778e00

b1 = 5.42937341165687622380535766363e-2
b6 = 4.45031289275240888144113950566e0
b7 = 1.89151789931450038304281599044e0
b8 = -5.8012039600105847814672114227e0
b9 = 3.1116436695781989440891606237e-1
b10 = -1.52160949662516078556178806805e-1
b11 = 2.01365400804030348374776537501e-1
b12 = 4.47106157277725905176885569043e-2

bhh1 = 0.244094488188976377952755905512e00
bhh2 = 0.733846688281611857341361741547e00
bhh3 = 0.220588235294117647058823529412e-01

er1 = 0.1312004499419488073250102996e-01
er6 = -0.1225156446376204440720569753e01
er7 = -0.4957589496572501915214079952e00
er8 = 0.1664377182454986536961530415e01
er9 = -0.3503288487499736816886487290e00
er10 = 0.3341791187130174790297318841e00
er11 = 0.8192320648511571246570742613e-01
er12 = -0.2235530786388629525884427845e-01

a21 = 5.26001519587677318785587544488e-2

a31 = 1.97250569845378994544595329183e-2
a32 = 5.91751709536136983633785987549e-2

a41 = 2.95875854768068491816892993775e-2
a43 = 8.87627564304205475450678981324e-2

a51 = 2.41365134159266685502369798665e-1
a53 = -8.84549479328286085344864962717e-1
a54 = 9.24834003261792003115737966543e-1

a61 = 3.7037037037037037037037037037e-2
a64 = 1.70828608729473871279604482173e-1
a65 = 1.25467687566822425016691814123e-1

a71 = 3.7109375e-2
a74 = 1.70252211019544039314978060272e-1
a75 = 6.02165389804559606850219397283e-2
a76 = -1.7578125e-2

a81 = 3.70920001185047927108779319836e-2
a84 = 1.70383925712239993810214054705e-1
a85 = 1.07262030446373284651809199168e-1
a86 = -1.53194377486244017527936158236e-2
a87 = 8.27378916381402288758473766002e-3

a91 = 6.24110958716075717114429577812e-1
a94 = -3.36089262944694129406857109825e0
a95 = -8.68219346841726006818189891453e-1
a96 = 2.75920996994467083049415600797e1
a97 = 2.01540675504778934086186788979e1
a98 = -4.34898841810699588477366255144e1

a101 = 4.77662536438264365890433908527e-1
a104 = -2.48811461997166764192642586468e0
a105 = -5.90290826836842996371446475743e-1
a106 = 2.12300514481811942347288949897e1
a107 = 1.52792336328824235832596922938e1
a108 = -3.32882109689848629194453265587e1
a109 = -2.03312017085086261358222928593e-2

a111 = -9.3714243008598732571704021658e-1
a114 = 5.18637242884406370830023853209e0
a115 = 1.09143734899672957818500254654e0
a116 = -8.14978701074692612513997267357e0
a117 = -1.85200656599969598641566180701e1
a118 = 2.27394870993505042818970056734e1
a119 = 2.49360555267965238987089396762e0
a1110 = -3.0467644718982195003823669022e0

a121 = 2.27331014751653820792359768449e0
a124 = -1.05344954667372501984066689879e1
a125 = -2.00087205822486249909675718444e0
a126 = -1.79589318631187989172765950534e1
a127 = 2.79488845294199600508499808837e1
a128 = -2.85899827713502369474065508674e0
a129 = -8.87285693353062954433549289258e0
a1210 = 1.23605671757943030647266201528e1
a1211 = 6.43392746015763530355970484046e-1

a141 = 5.61675022830479523392909219681e-2
a147 = 2.53500210216624811088794765333e-1
a148 = -2.46239037470802489917441475441e-1
a149 = -1.24191423263816360469010140626e-1
a1410 = 1.5329179827876569731206322685e-1
a1411 = 8.20105229563468988491666602057e-3
a1412 = 7.56789766054569976138603589584e-3
a1413 = -8.298e-3

a151 = 3.18346481635021405060768473261e-2
a156 = 2.83009096723667755288322961402e-2
a157 = 5.35419883074385676223797384372e-2
a158 = -5.49237485713909884646569340306e-2
a1511 = -1.08347328697249322858509316994e-4
a1512 = 3.82571090835658412954920192323e-4
a1513 = -3.40465008687404560802977114492e-4
a1514 = 1.41312443674632500278074618366e-1

a161 = -4.28896301583791923408573538692e-1
a166 = -4.69762141536116384314449447206e0
a167 = 7.68342119606259904184240953878e0
a168 = 4.06898981839711007970213554331e0
a169 = 3.56727187455281109270669543021e-1
a1613 = -1.39902416515901462129418009734e-3
a1614 = 2.9475147891527723389556272149e0
a1615 = -9.15095847217987001081870187138e0

d41 = -0.84289382761090128651353491142e01
d46 = 0.56671495351937776962531783590e00
d47 = -0.30689499459498916912797304727e01
d48 = 0.23846676565120698287728149680e01
d49 = 0.21170345824450282767155149946e01
d410 = -0.87139158377797299206789907490e00
d411 = 0.22404374302607882758541771650e01
d412 = 0.63157877876946881815570249290e00
d413 = -0.88990336451333310820698117400e-01
d414 = 0.18148505520854727256656404962e02
d415 = -0.91946323924783554000451984436e01
d416 = -0.44360363875948939664310572000e01

d51 = 0.10427508642579134603413151009e02
d56 = 0.24228349177525818288430175319e03
d57 = 0.16520045171727028198505394887e03
d58 = -0.37454675472269020279518312152e03
d59 = -0.22113666853125306036270938578e02
d510 = 0.77334326684722638389603898808e01
d511 = -0.30674084731089398182061213626e02
d512 = -0.93321305264302278729567221706e01
d513 = 0.15697238121770843886131091075e02
d514 = -0.31139403219565177677282850411e02
d515 = -0.93529243588444783865713862664e01
d516 = 0.35816841486394083752465898540e02

d61 = 0.19985053242002433820987653617e02
d66 = -0.38703730874935176555105901742e03
d67 = -0.18917813819516756882830838328e03
d68 = 0.52780815920542364900561016686e03
d69 = -0.11573902539959630126141871134e02
d610 = 0.68812326946963000169666922661e01
d611 = -0.10006050966910838403183860980e01
d612 = 0.77771377980534432092869265740e00
d613 = -0.27782057523535084065932004339e01
d614 = -0.60196695231264120758267380846e02
d615 = 0.84320405506677161018159903784e02
d616 = 0.11992291136182789328035130030e02

d71 = -0.25693933462703749003312586129e02
d76 = -0.15418974869023643374053993627e03
d77 = -0.23152937917604549567536039109e03
d78 = 0.35763911791061412378285349910e03
d79 = 0.93405324183624310003907691704e02
d710 = -0.37458323136451633156875139351e02
d711 = 0.10409964950896230045147246184e03
d712 = 0.29840293426660503123344363579e02
d713 = -0.43533456590011143754432175058e02
d714 = 0.96324553959188282948394950600e02
d715 = -0.39177261675615439165231486172e02
d716 = -0.14972683625798562581422125276e03

# Some additional constants for the controller
beta = 0.0
alpha = 1.0 / 8.0 - beta * 0.2
safe = 0.9
minscale = 1.0 / 3.0
maxscale = 6.0


def ODE_pendulum(x, arg, k, additionalArgs):
    b = additionalArgs[0]
    c = additionalArgs[1]
    theta, omega = arg[0], arg[1]
    k[0] = omega
    k[1] = -b * omega - c * xp.sin(theta)


class DOPR853:
    def __init__(
        self,
        ode,
        stopping_criterion=None,
        tmax=1e300,
        max_step=int(1e6),
        use_gpu=False,
        read_out_to_cpu=True,
    ):
        self.ode = ode
        self.stopping_criterion = stopping_criterion
        self.tmax, self.max_step = tmax, max_step
        self.use_gpu = use_gpu
        self.read_out_to_cpus = read_out_to_cpu

        # self.dormandPrinceSteps2 = dormandPrinceSteps_gpu
        # self.error2 = error_gpu
        # self.controllerSuccess2 = controllerSuccess_gpu

        # self.dormandPrinceSteps2 = dormandPrinceSteps_cpu
        # self.error2 = error_cpu
        # self.controllerSuccess2 = controllerSuccess_cpu

        self.fix_step = False
        self.abstol = 1e-10

    @property
    def xp(self):
        return xp if self.use_gpu else np

    @property
    def xp_read_out(self):
        return np if self.read_out_to_cpus else xp

    def dormandPrinceSteps(
        self,
        x,
        solOld,
        h,
        additionalArgs,
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
        k8,
        k9,
        k10,
    ):
        # Create temporary index for use in loops

        # Step 1
        arg = solOld.copy()
        xCurrent = x.copy()

        self.ode(xCurrent, arg, k1, additionalArgs)

        # Step 2
        arg[:] = solOld + h * (a21 * k1)
        xCurrent = x + c2 * h

        self.ode(xCurrent, arg, k2, additionalArgs)

        # Step 3
        arg[:] = solOld + h * (a31 * k1 + a32 * k2)
        xCurrent = x + c3 * h

        self.ode(xCurrent, arg, k3, additionalArgs)

        # Step 4
        arg[:] = solOld + h * (a41 * k1 + a43 * k3)
        xCurrent = x + c4 * h

        self.ode(xCurrent, arg, k4, additionalArgs)

        # Step 5
        arg[:] = solOld + h * (a51 * k1 + a53 * k3 + a54 * k4)
        xCurrent = x + c5 * h

        self.ode(xCurrent, arg, k5, additionalArgs)

        # Step 6
        # pragma unroll
        arg[:] = solOld + h * (a61 * k1 + a64 * k4 + a65 * k5)
        xCurrent = x + c6 * h

        self.ode(xCurrent, arg, k6, additionalArgs)

        # Step 7
        arg[:] = solOld + h * (a71 * k1 + a74 * k4 + a75 * k5 + a76 * k6)
        xCurrent = x + c7 * h

        self.ode(xCurrent, arg, k7, additionalArgs)

        # Step 8
        arg[:] = solOld + h * (a81 * k1 + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7)
        xCurrent = x + c8 * h

        self.ode(xCurrent, arg, k8, additionalArgs)  #

        # Step 9
        arg[:] = solOld + h * (
            a91 * k1 + a94 * k4 + a95 * k5 + a96 * k6 + a97 * k7 + a98 * k8
        )  #
        xCurrent = x + c9 * h

        self.ode(xCurrent, arg, k9, additionalArgs)

        # Step 10
        arg[:] = solOld + h * (
            a101 * k1
            + a104 * k4
            + a105 * k5
            + a106 * k6
            + a107 * k7
            + a108 * k8
            + a109 * k9
        )
        xCurrent = x + c10 * h

        self.ode(xCurrent, arg, k10, additionalArgs)

        # Step 11
        arg[:] = solOld + h * (
            a111 * k1
            + a114 * k4
            + a115 * k5
            + a116 * k6
            + a117 * k7
            + a118 * k8
            + a119 * k9
            + a1110 * k10
        )
        xCurrent = x + c11 * h

        self.ode(xCurrent, arg, k2, additionalArgs)

        # Step 12 - Note the use of x + h for this step
        arg[:] = solOld + h * (
            a121 * k1
            + a124 * k4
            + a125 * k5
            + a126 * k6
            + a127 * k7
            + a128 * k8
            + a129 * k9
            + a1210 * k10
            + a1211 * k2
        )
        xCurrent = x + h

        self.ode(xCurrent, arg, k3, additionalArgs)

    # Error calculation
    def error(
        self,
        err,
        solOld,
        solNew,
        h,
        k1,
        k2,
        k3,
        k6,
        k7,
        k8,
        k9,
        k10,
    ):
        # Variables used in system
        # err     = 0.0
        # double err2 = 0.0
        # double sk, denom, temp

        err[:] = 0.0

        n = solOld.shape[0]

        # Number of equations in system
        temp = (
            b1 * k1
            + b6 * k6
            + b7 * k7
            + b8 * k8
            + b9 * k9
            + b10 * k10
            + b11 * k2
            + b12 * k3
        )

        solNew[:] = solOld + h * temp

        # sk = 1.0 / (
        #     abstol + rtol * self.xp.max(self.xp.array([solOld, solNew]), axis=0)
        # )

        sk = 1.0 / (
            self.abstol + 0.0 * self.xp.max(self.xp.array([solOld, solNew]), axis=0)
        )

        err2 = (((temp - bhh1 * k1 - bhh2 * k9 - bhh3 * k3) * sk) ** 2).sum(axis=0)
        err[:] = (
            (
                (
                    er1 * k1
                    + er6 * k6
                    + er7 * k7
                    + er8 * k8
                    + er9 * k9
                    + er10 * k10
                    + er11 * k2
                    + er12 * k3
                )
                * sk
            )
            ** 2
        ).sum(axis=0)

        # Now calculate the denominator and return the error
        denom = err + 0.01 * err2

        denom = 1.0 * (denom <= 0.0) + denom * (denom > 0.0)

        err[:] = self.xp.abs(h) * err * self.xp.sqrt(1.0 / (n * denom))

    # Contoller for setting step size on next iteration
    def controllerSuccess(self, flagSuccess, err, errOld, previousReject, h, x):
        # flagSuccess and previousReject were bool*

        # The error was acceptable
        acceptable = err <= 1.0
        scale = self.xp.zeros_like(err)

        scale[acceptable] = (
            safe * err[acceptable] ** (-alpha) * errOld[acceptable] ** beta
        )
        scale[acceptable] = self.xp.clip(scale[acceptable], minscale, maxscale)
        scale[err == 0.0] = maxscale

        if self.xp.any(h == 0.0):
            breakpoint()

        # accept_and_prev_reject = previousReject & acceptable

        # TODO: do we want this to fix the adjustment if the last was rejected
        # h[accept_and_prev_reject] = h[accept_and_prev_reject] * self.xp.clip(
        #     scale[accept_and_prev_reject], 0.0, 1.0
        # )
        # h[accept_and_prev_reject] = h[accept_and_prev_reject] * min(scale[accept_and_prev_reject] * scale[accept_and_prev_reject], 1.0)

        accept_and_prev_accept = ~previousReject & acceptable
        h[accept_and_prev_accept] = (
            h[accept_and_prev_accept] * scale[accept_and_prev_accept]
        )

        if self.xp.any(h == 0.0):
            raise ValueError("Step-size went to zero during trajectory evolution.")
        errOld[acceptable] = self.xp.clip(err[acceptable], 1e-04, 1e300)
        previousReject[acceptable] = False

        # Return that we have accepted the current step
        flagSuccess[acceptable] = True

        # The error was too big, we need to make the step size smaller
        unacceptable = ~acceptable
        # Reduce the size of the step
        scale[unacceptable] = self.xp.clip(
            safe * (err[unacceptable] ** -alpha), minscale, 1e300
        )
        # htmp = h.copy()

        h[unacceptable] = h[unacceptable] * scale[unacceptable]
        previousReject[unacceptable] = True

        # Return that we have failed to advance
        flagSuccess[unacceptable] = False

    def take_step_single(
        self,
        x,
        h,
        y,
        tmax_dimensionless,
        additional_args,
    ):
        xTemp = np.array([x])
        hTemp = np.array([h])
        solOldTemp = np.array([y.copy()]).T
        additionalArgsTemp = np.array([additional_args])

        flagSuccess = self.take_step(
            xTemp,
            hTemp,
            solOldTemp,
            tmax_dimensionless,
            additionalArgsTemp,
            inds=None,
        )

        # inplace update to y
        y[:] = solOldTemp[:, 0]
        return (flagSuccess[0], xTemp[0], hTemp[0])

    def take_step(
        self,
        xTemp,
        hTemp,
        solOldTemp,
        tmax_dimensionless,
        additionalArgsTemp,
        inds=None,
    ):
        hOldTemp = hTemp.copy()

        if inds is None:
            inds = np.arange(hTemp.shape[0])

        nODE, numSysTemp = solOldTemp.shape

        (
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            k7,
            k8,
            k9,
            k10,
        ) = [self.xp.zeros((nODE, numSysTemp)) for _ in range(10)]

        self.dormandPrinceSteps(
            xTemp,
            solOldTemp,
            hTemp,
            additionalArgsTemp,
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            k7,
            k8,
            k9,
            k10,
        )

        if not self.fix_step:
            err = np.zeros_like(xTemp)
            solNewTemp = np.zeros_like(solOldTemp)

            self.error(
                err,
                solOldTemp,
                solNewTemp,
                hTemp,
                k1,
                k2,
                k3,
                k6,
                k7,
                k8,
                k9,
                k10,
                # numEq,
                # nargs,
            )

            # hOldTemp[:] = hTemp
            # xOldTemp[:] = xTemp

            flagSuccess = np.zeros_like(xTemp, dtype=bool)

            if not hasattr(self, "errOldTemp"):
                self.errOldTemp = np.full_like(err, 1e-4)

            if not hasattr(self, "previousRejectTemp"):
                self.previousRejectTemp = np.zeros_like(err, dtype=bool)

            errOldTemp = self.errOldTemp[inds].copy()
            previousRejectTemp = self.previousRejectTemp[inds].copy()

            self.controllerSuccess(
                flagSuccess,
                err,
                errOldTemp,
                previousRejectTemp,
                hTemp,
                xTemp,
                # numEq,
                # nargs,
            )

            solOldTemp = solOldTemp.reshape(nODE, numSysTemp)
            solOldTemp[:, flagSuccess] = solNewTemp.reshape(nODE, numSysTemp)[
                :, flagSuccess
            ]

            xTemp[flagSuccess] = xTemp[flagSuccess] + hOldTemp[flagSuccess]
            self.errOldTemp[inds[flagSuccess]] = err[flagSuccess]
            self.previousRejectTemp[inds] = ~flagSuccess

            self.k_coefficient_storage = [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10]

        else:  # Not controlling for error so return successes and advance xTemp for all
            flagSuccess = np.ones_like(xTemp, dtype=bool)
            xTemp[flagSuccess] = xTemp + hOldTemp

            temp = (
                b1 * k1
                + b6 * k6
                + b7 * k7
                + b8 * k8
                + b9 * k9
                + b10 * k10
                + b11 * k2
                + b12 * k3
            )

            solOldTemp += hOldTemp * temp
            solOldTemp = solOldTemp.reshape(nODE, numSysTemp)

        return flagSuccess

    def prep_evaluate_single(self, x0, y0, h0, x1, y1, additionalArgs):
        assert not self.fix_step

        x0_tmp = np.array([x0])
        y0_tmp = y0[:, None].copy()
        h0_tmp = np.array([h0])
        x1_tmp = np.array([x1])
        y1_tmp = y1[:, None].copy()

        spline_output = self.prep_evaluate(
            x0_tmp, y0_tmp, h0_tmp, x1_tmp, y1_tmp, additionalArgs
        )

        return spline_output[0]

    def prep_evaluate(self, x0, y0, h0, x1, y1, additionalArgs):
        assert not self.fix_step

        newDer = np.zeros_like(y0)
        (k1, k2, k3, k4, k5, k6, k7, k8, k9, k10) = self.k_coefficient_storage

        # Derivative at new time step
        self.ode(x1, y1, newDer, additionalArgs)

        # Step 1
        rcont1 = y0

        # Step 2
        diff = y1 - y0
        rcont2 = diff

        # Step 3
        bspl = h0 * k1 - diff
        rcont3 = bspl

        # Step 4
        rcont4 = diff - h0 * newDer - bspl

        # Step 5
        rcont5 = (
            d41 * k1
            + d46 * k6
            + d47 * k7
            + d48 * k8
            + d49 * k9
            + d410 * k10
            + d411 * k2
            + d412 * k3
        )

        # Step 6
        rcont6 = (
            d51 * k1
            + d56 * k6
            + d57 * k7
            + d58 * k8
            + d59 * k9
            + d510 * k10
            + d511 * k2
            + d512 * k3
        )

        # Step 7
        rcont7 = (
            d61 * k1
            + d66 * k6
            + d67 * k7
            + d68 * k8
            + d69 * k9
            + d610 * k10
            + d611 * k2
            + d612 * k3
        )

        # Step 8
        rcont8 = (
            d71 * k1
            + d76 * k6
            + d77 * k7
            + d78 * k8
            + d79 * k9
            + d710 * k10
            + d711 * k2
            + d712 * k3
        )

        # Now do the additional derivative steps

        # Step 1
        arg = y0 + h0 * (
            a141 * k1
            + a147 * k7
            + a148 * k8
            + a149 * k9
            + a1410 * k10
            + a1411 * k2
            + a1412 * k3
            + a1413 * newDer
        )

        x1_tmp = x0 + c14 * h0

        self.ode(x1_tmp, arg, k10, additionalArgs)

        # Step 2
        arg = y0 + h0 * (
            a151 * k1
            + a156 * k6
            + a157 * k7
            + a158 * k8
            + a1511 * k2
            + a1512 * k3
            + a1513 * newDer
            + a1514 * k10
        )
        x1_tmp = x0 + c15 * h0

        self.ode(x1_tmp, arg, k2, additionalArgs)

        # Step 3
        arg = y0 + h0 * (
            a161 * k1
            + a166 * k6
            + a167 * k7
            + a168 * k8
            + a169 * k9
            + a1613 * newDer
            + a1614 * k10
            + a1615 * k2
        )

        x1_tmp = x0 + c16 * h0

        self.ode(x1_tmp, arg, k3, additionalArgs)

        # Now complete rcont5-8 with the updated values
        rcont5 = h0 * (rcont5 + d413 * newDer + d414 * k10 + d415 * k2 + d416 * k3)
        rcont6 = h0 * (rcont6 + d513 * newDer + d514 * k10 + d515 * k2 + d516 * k3)
        rcont7 = h0 * (rcont7 + d613 * newDer + d614 * k10 + d615 * k2 + d616 * k3)
        rcont8 = h0 * (rcont8 + d713 * newDer + d714 * k10 + d715 * k2 + d716 * k3)

        return np.array(
            [
                rcont1,
                rcont2,
                rcont3,
                rcont4,
                rcont5,
                rcont6,
                rcont7,
                rcont8,
            ]
        ).T

    def eval(self, t_new: np.ndarray, t_old: np.ndarray, spline_coeffs: np.ndarray):
        assert not self.fix_step

        t_min = t_old.min()
        t_max = t_old.max()

        if not np.all((t_min <= t_new) & (t_new <= t_max)):
            raise ValueError(
                f"All t_new values must be between t_min ({t_min}) and t_max ({t_max})."
            )

        segments = np.searchsorted(t_old, t_new, side="right") - 1
        segments[t_new == t_max] = t_old.shape[0] - 2  # there is 1 less spline segment

        # NOT MEMORY EFFICIENT
        tmp_coeffs = spline_coeffs[segments]
        tmp_t_old = t_old[segments]
        diffs = np.diff(t_old)[segments]

        assert spline_coeffs.ndim == 3 and spline_coeffs.shape[-1] == 8

        rcont1 = tmp_coeffs[:, :, 0]
        rcont2 = tmp_coeffs[:, :, 1]
        rcont3 = tmp_coeffs[:, :, 2]
        rcont4 = tmp_coeffs[:, :, 3]
        rcont5 = tmp_coeffs[:, :, 4]
        rcont6 = tmp_coeffs[:, :, 5]
        rcont7 = tmp_coeffs[:, :, 6]
        rcont8 = tmp_coeffs[:, :, 7]

        s = ((t_new - tmp_t_old) / diffs)[:, None]  # add axes to match rcont shape
        s1 = 1.0 - s

        output = rcont1 + s * (
            rcont2
            + s1
            * (
                rcont3
                + s
                * (rcont4 + s1 * (rcont5 + s * (rcont6 + s1 * (rcont7 + s * rcont8))))
            )
        )
        # output = rcont1 + s*rcont2 + rcont3 * (s - s**2)  + rcont4 * (s**2 - s**3) + rcont5 * (s**4 - s**5) + rcont6 * (s**5 - s**6) + rcont7 * (s**6 - s**7) + rcont8 * (s**7 - s**8)

        return output

    def eval_derivative(
        self,
        t_new: np.ndarray,
        t_old: np.ndarray,
        spline_coeffs: np.ndarray,
        order: int = 1,
    ):
        t_min = t_old.min()
        t_max = t_old.max()
        if not np.all((t_min <= t_new) & (t_new <= t_max)):
            raise ValueError(
                f"All t_new values must be between t_min ({t_min}) and t_max ({t_max})."
            )

        segments = np.searchsorted(t_old, t_new, side="right") - 1
        segments[t_new == t_max] = t_old.shape[0] - 2

        tmp_coeffs = spline_coeffs[segments]
        tmp_t_old = t_old[segments]
        diffs = np.diff(t_old)[segments]

        assert spline_coeffs.ndim == 3 and spline_coeffs.shape[-1] == 8

        # rcont1 = tmp_coeffs[:, :, 0]
        rcont2 = tmp_coeffs[:, :, 1]
        rcont3 = tmp_coeffs[:, :, 2]
        rcont4 = tmp_coeffs[:, :, 3]
        rcont5 = tmp_coeffs[:, :, 4]
        rcont6 = tmp_coeffs[:, :, 5]
        rcont7 = tmp_coeffs[:, :, 6]
        rcont8 = tmp_coeffs[:, :, 7]

        s = ((t_new - tmp_t_old) / diffs)[:, None]  # add axes to match rcont shape
        s2 = s**2
        s3 = s**3
        s4 = s**4
        s5 = s**5
        s6 = s**6

        if order == 0:
            d_by_ds = self.eval(t_new, t_old, spline_coeffs)
        if order == 1:
            d_by_ds = (
                rcont2
                + rcont3 * (1 - 2 * s)
                + rcont4 * (2 * s - 3 * s2)
                + rcont5 * (2 * s - 6 * s2 + 4 * s3)
                + rcont6 * (3 * s2 - 8 * s3 + 5 * s4)
                + rcont7 * (3 * s2 - 12 * s3 + 15 * s4 - 6 * s5)
                + rcont8 * (4 * s3 - 15 * s4 + 18 * s5 - 7 * s6)
            )
        if order == 2:
            d_by_ds = (
                -2 * rcont3
                + rcont4 * (2 - 6 * s)
                + rcont5 * (2 - 12 * s + 12 * s2)
                + rcont6 * (6 * s - 24 * s2 + 20 * s3)
                + rcont7 * (6 * s - 36 * s2 + 60 * s3 - 30 * s4)
                + rcont8 * (12 * s2 - 60 * s3 + 90 * s4 - 42 * s5)
            )
        if order == 3:
            d_by_ds = (
                -6 * rcont4
                + rcont5 * (-12 + 24 * s)
                + rcont6 * (6 - 48 * s + 60 * s2)
                + rcont7 * (6 - 72 * s + 180 * s2 - 120 * s3)
                + rcont8 * (24 * s - 180 * s2 + 360 * s3 - 210 * s4)
            )

        # Scale by dt/ds = diffs to get derivative with respect to t
        derivative = d_by_ds / diffs[:, None] ** order

        return derivative

    #                         denseOutput[indexOut] = rcont1 + s * (rcont2 + s1 * (rcont3 + s * (rcont4 + s1 * (rcont5 + s * (rcont6 + s1 * (rcont7 + s * rcont8))))))

    #                             denseDerivOutput[indexOut] = (rcont2 + rcont3 - 2*rcont3*s + rcont4*(2 - 3*s)*s - (-1 + s)*s*(rcont5*(2 - 4*s) + rcont6*(3 - 5*s)*s + (-1 + s)*s*(rcont7*(-3 + 6*s) + rcont8*s*(-4 + 7*s))))/h

    # Solver
    def integrate(
        self,
        condBound,  # Initial condition stored as a vector
        argsData,
        hInit=None,  # Initial spacing
        step_num=None,
        denseOutput=None,
        denseOutputLoc=None,
        fix_step=False,
    ):
        """
        # Declare all shared memory
        # Vectors for current solution
        CUDA_SHARED double solOldData[BLOCK * NMAX]
        CUDA_SHARED double solNewData[BLOCK * NMAX]

        # Vectors to store current position and spacing
        CUDA_SHARED double x[BLOCK]
        CUDA_SHARED double h[BLOCK]

        # Arguements for use in ODE
        CUDA_SHARED double additionalArgs[BLOCK * MAXARGS]

        # Vectors for use in stepping
        CUDA_SHARED double k1[BLOCK * NMAX]
        CUDA_SHARED double k2[BLOCK * NMAX]
        CUDA_SHARED double k3[BLOCK * NMAX]
        CUDA_SHARED double k4[BLOCK * NMAX]
        CUDA_SHARED double k5[BLOCK * NMAX]
        CUDA_SHARED double k6[BLOCK * NMAX]
        CUDA_SHARED double k7[BLOCK * NMAX]
        CUDA_SHARED double k8[BLOCK * NMAX]
        CUDA_SHARED double k9[BLOCK * NMAX]
        CUDA_SHARED double k10[BLOCK * NMAX]

        # Vectors for recovering dense output
        CUDA_SHARED double rcont1[BLOCK * NMAX]
        CUDA_SHARED double rcont2[BLOCK * NMAX]
        CUDA_SHARED double rcont3[BLOCK * NMAX]
        CUDA_SHARED double rcont4[BLOCK * NMAX]
        CUDA_SHARED double rcont5[BLOCK * NMAX]
        CUDA_SHARED double rcont6[BLOCK * NMAX]
        CUDA_SHARED double rcont7[BLOCK * NMAX]
        CUDA_SHARED double rcont8[BLOCK * NMAX]

        # Vectors used for temp values
        CUDA_SHARED double arg[BLOCK * NMAX]
        CUDA_SHARED double xCurrent[BLOCK]
        CUDA_SHARED double newDer[BLOCK * NMAX]

        # Error tracking
        CUDA_SHARED double err[BLOCK]
        CUDA_SHARED double errOld[BLOCK]

        # Flag to determine if the step made was a success
        CUDA_SHARED bool flagSuccess[BLOCK]

        # Temporary values for x and h, need these for dense storage
        CUDA_SHARED double xOld[BLOCK], hOld[BLOCK]

        # Check if the previous step was rejected
        CUDA_SHARED bool previousReject[BLOCK]
        """
        if (
            denseOutput is not None
            or denseOutput is not None
            or denseOutput is not None
        ):
            if denseOutput is None:
                raise ValueError(
                    "If providing denseOutputLoc/denseOutput/step_num, must provide all."
                )

        if denseOutput is None:
            solOld = self.xp.asarray(condBound)  # boundary conditions
            nODE, numSys = solOld.shape
            denseOutput = self.xp_read_out.zeros((self.max_step, nODE, numSys))
            denseOutputLoc = self.xp_read_out.zeros((self.max_step, numSys))
            step_num = self.xp.zeros(numSys, dtype=int)
            x = self.xp.zeros(numSys)
            if self.use_gpu and self.read_out_to_cpu:
                denseOutput[0, :, :] = solOld.get()
                denseOutputLoc[0, :] = x.get()
            else:
                denseOutput[0, :, :] = solOld
                denseOutputLoc[0, :] = x

        else:
            assert denseOutput.ndim == 3
            max_steps, nODE, numSys = denseOutput.shape
            # adjust max_steps
            self.max_steps = max_steps
            assert denseOutputLoc.shape == (self.max_step, numSys)
            assert step_num.shape[0] == numSys
            x = denseOutputLoc[(step_num, self.xp.arange(numSys))]

            solOld = (
                denseOutput[
                    (
                        self.xp.repeat(step_num, nODE),
                        self.xp.tile(self.xp.arange(4), (numSys, 1)).flatten(),
                        self.xp.repeat(self.xp.arange(numSys), nODE),
                    )
                ]
                .reshape(numSys, nODE)
                .T
            )

        if self.stopping_criterion is not None and hasattr(
            self.stopping_criterion, "setup"
        ):
            self.stopping_criterion.setup(numSys)

        additionalArgs = self.xp.asarray(argsData)

        if hInit is None:
            hInit = 0.01

        # hInit_orig = self.xp.full(numSys, hInit)
        h = self.xp.full_like(x, hInit)

        # xOld = self.xp.zeros_like(x)
        # hOld = self.xp.zeros_like(h)

        # Set pointers to these that can be swapped
        # solNew = self.xp.zeros_like(solOld)

        # Loop while true
        loopFlag = True

        # Set an initial error value
        errOld = self.xp.zeros_like(x)
        errOld[:] = 1e-04

        # Check if the previous step was rejected
        previousReject = self.xp.zeros_like(x, dtype=bool)
        previousReject[:] = False

        # Current output step
        # denseCurrent = 0

        # Current output location
        # denseCurrentLoc = 0.0  # denseOutputLoc[denseCurrent * eqNUM + eqSysNum]

        # Calculate the max and min of a window
        # We need to do this to account for whether we're moving forwards or backwards in the domain
        # double windowMax
        # double windowMin

        # numDensePointsHere = (numDensePoints + 1)/2

        # TODO: dense sampling
        # Dense output variables
        # double s, s1

        # Indexing for output
        # indexOut
        # index

        # for (j = 0 j < N j++)
        # {
        #    denseOutput[j * max_step + step_num] = solOld[j * BLOCK + i]
        # }
        # denseOutputLoc[step_num] = x[i]
        # Use a while loop as it is easier to keep stepping regardless of

        individual_loop_flag = self.xp.ones_like(step_num, dtype=bool)

        # ii = 0
        jj = 0

        stop = self.xp.ones_like(x, dtype=bool)
        self.stop_info = self.xp.zeros_like(x, dtype=int)
        while loopFlag:
            # num_current = np.sum(individual_loop_flag)
            xTemp = x[individual_loop_flag]
            hTemp = h[individual_loop_flag]
            # xOldTemp = self.xp.zeros_like(xTemp)
            # hOldTemp = self.xp.zeros_like(hTemp)
            solOldTemp = solOld[:, individual_loop_flag]  # .flatten()
            # solNewTemp = solNew[:, individual_loop_flag]  # .flatten()
            errOldTemp = errOld[individual_loop_flag]
            previousRejectTemp = previousReject[individual_loop_flag]
            additionalArgsTemp = additionalArgs[:, individual_loop_flag]  # .flatten()

            numSysTemp = xTemp.shape[0]
            # (
            #     k1,
            #     k2,
            #     k3,
            #     k4,
            #     k5,
            #     k6,
            #     k7,
            #     k8,
            #     k9,
            #     k10,
            # ) = [self.xp.zeros((nODE, numSysTemp)) for _ in range(10)]
            # err = self.xp.zeros(numSysTemp)
            flagSuccess = self.xp.zeros(numSysTemp, dtype=bool)
            # 0.000857

            # xCurrent_buffer = self.xp.zeros_like(xTemp)
            # arg_buffer = self.xp.zeros_like(solOldTemp)
            # ak_term_buffer = self.xp.zeros_like(solOldTemp)
            # err = self.xp.zeros_like(xTemp)

            # nargs = nODE
            # numEq = numSysTemp
            # num_add_args = additionalArgs.shape[0]
            # Compute the steps to iterate to the next timestep

            # index_here = self.xp.arange(individual_loop_flag.shape[0])[
            #     individual_loop_flag
            # ].astype(self.xp.int32)

            if fix_step:
                raise NotImplementedError

            flagSuccess = self.take_step(
                xTemp,
                hTemp,
                solOldTemp,
                tMax,
                additionalArgsTemp,
                fix_step=fix_step,
                inds=np.arange(numSys)[individual_loop_flag],
            )

            if fix_step:
                raise NotImplementedError

            """
            self.dormandPrinceSteps(xTemp, solOldTemp, hTemp, additionalArgsTemp, *ks)


        # 0.00344 (~0.00258)
            # TODO: improve
            ks_error = [ks[i - 1] for i in [1, 2, 3, 6, 7, 8, 9, 10]]

            # Compute the error
            self.error(err, solOldTemp, solNewTemp, hTemp, *ks_error)
        # 0.00427 (0.000808)

            # Store the old values


            # # Check if the error was acceptable
            self.controllerSuccess(flagSuccess, err, errOldTemp, previousRejectTemp, hTemp, xTemp)

        # 0.00648 (0.00221)
            """
            x[individual_loop_flag] = xTemp
            h[individual_loop_flag] = hTemp
            solOld[:, individual_loop_flag] = solOldTemp
            errOld[individual_loop_flag] = errOldTemp
            previousReject[individual_loop_flag] = previousRejectTemp

            # 0.00749475707532838 (0.001)

            index_update = self.xp.arange(individual_loop_flag.shape[0])[
                individual_loop_flag
            ][flagSuccess]

            step_num[index_update] += 1

            read_out_step = self.xp.tile(step_num[index_update], nODE)
            read_out_index_update = self.xp.tile(index_update, nODE)
            read_out_ode_dim = self.xp.repeat(self.xp.arange(nODE), len(index_update))

            # 0.007710501374676823 (0.00022)
            if self.use_gpu and self.read_out_to_cpu:
                denseOutput[
                    (
                        read_out_step.get(),
                        read_out_ode_dim.get(),
                        read_out_index_update.get(),
                    )
                ] = solOld[:, index_update].flatten().get()
                denseOutputLoc[(step_num[index_update].get(), index_update.get())] = x[
                    index_update
                ].get()
            else:
                denseOutput[
                    (read_out_step, read_out_ode_dim, read_out_index_update)
                ] = solOld[:, index_update].flatten()
                denseOutputLoc[(step_num[index_update], index_update)] = x[index_update]

            if self.stopping_criterion is not None and len(index_update) > 0:
                stop_temp = self.xp.asarray(
                    self.stopping_criterion(
                        step_num,
                        solOld[:, index_update],
                        additionalArgs[:, index_update],
                        index_update,
                    )
                )
            else:
                stop_temp = self.xp.zeros(len(index_update), dtype=bool)

            # TODO: add max step size 0.05
            # for checking how it stopped
            self.stop_info[index_update] = stop_temp.copy()
            stop[index_update] = stop_temp.astype(bool)

            individual_loop_flag[
                (x >= self.tmax) | (step_num >= self.max_step - 1) | stop
            ] = False  #  | (solNew[0] < 6.0)] = False
            # self.xp.cuda.runtime.deviceSynchronize()
            if self.xp.all(~individual_loop_flag):
                loopFlag = False

            jj += 1
            # print("CHECKING", jj)
            # if jj >= 281:
            #    break
            # if ii % 1 == 0:
            # et = time.perf_counter()
            # print((et - st)/ ii)
            # print(ii)
            # 0.008480359460227191 (0.0.00077)

        if self.stopping_criterion is not None and hasattr(
            self.stopping_criterion, "reset"
        ):
            self.stopping_criterion.reset()

        return (denseOutputLoc, denseOutput, step_num)  # denseDerivOutput


def stopping_criterion(step_num, denseOutput):
    stop = xp.zeros_like(step_num, dtype=bool)
    num_diff = 10
    for i, step in enumerate(step_num):
        if step > num_diff:
            if xp.all(
                xp.abs(
                    denseOutput[step - num_diff : step, :, i]
                    / xp.abs(denseOutput[:, :, i]).max(axis=0)
                )
                < 1e-3
            ):
                stop[i] = True
    return stop


"""
        # If check if we accept the step
        if (flagSuccess[i])

        {
            windowMax = max(xOld[i], xOld[i] + h0)
            windowMin = min(xOld[i], xOld[i] + h0)

            /*
            # Check if our current dense output is in range - we have this here to avoid computing extra steps when not needed
            if (denseCurrentLoc > windowMin && denseCurrentLoc < windowMax)
            {
                # Prepare dense output variables
                densePrepare(arg, xCurrent, newDer, rcont1, rcont2, rcont3, rcont4, rcont5, rcont6, rcont7, rcont8, k1, k2, k3, k6, k7, k8, k9, k10, solOld, solNew, xOld, hOld, additionalArgs)

                # Loop over the current range of xOld to x to fill any dense entries

                while (denseCurrentLoc > windowMin && denseCurrentLoc < windowMax)
                {
                    # Output the data to the desired location
                    double h = h0
                    s = (denseCurrentLoc - xOld[i]) / h
                    s1 = 1.0 - s

                    for (j = 0 j < N j++)
                    {
                        index = j * BLOCK + i
                        indexOut = (denseCurrent * N * eqNUM) + j * eqNUM + eqSysNum

                        # evaluating solution to the 4 ODES at densely sampled points in the libration region
                        denseOutput[indexOut] = rcont1[index] + s * (rcont2[index] + s1 * (rcont3[index] + s * (rcont4[index] + s1 * (rcont5[index] + s * (rcont6[index] + s1 * (rcont7[index] + s * rcont8[index]))))))

                        denseDerivOutput[indexOut] = (rcont2[index] + rcont3[index] - 2*rcont3[index]*s + rcont4[index]*(2 - 3*s)*s - (-1 + s)*s*(rcont5[index]*(2 - 4*s) + rcont6[index]*(3 - 5*s)*s + (-1 + s)*s*(rcont7[index]*(-3 + 6*s) + rcont8[index]*s*(-4 + 7*s))))/h

                        if ((eqSysNum == 0) && ((j == 0) || (j == 2)) && (denseCurrent == 0)) printf("CHECK %d %d %d %.18e, %.18e, %.18e\n", j, eqSysNum, denseCurrent, denseCurrentLoc, denseOutput[indexOut], denseDerivOutput[indexOut])
                        #if ((denseCurrent == 5) && (eqSysNum == 10)) printf("%d, %d, %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n", i, j, denseOutput[indexOut], solOld[index], solNew[index], deriv_out, k1[index], k2[index], k3[index], k4[index], k5[index], k6[index], k7[index], k8[index], k9[index], k10[index])
                    }

                    # Move to next position
                    denseCurrent++

                    if (denseCurrent < numDensePointsHere)
                    {
                        denseCurrentLoc = denseOutputLoc[denseCurrent * eqNUM + eqSysNum]
                    }
                    else
                    {
                        break
                    }
                }
            }*/


            # Update values to the next step
            swap(&solOld, &solNew)
            solOld[:] = solNew

            x[:] = xOld + hOld

            denseOutput[step_num] = solNew

            denseOutputLoc[step_num] = x
        }
        # If we don't accept the step then keep going
        else
        {
            continue
        }


        # If all output now complete we're done
        #if (denseCurrent >= numDensePointsHere)
        if ((x[i] >= tMax) || (step_num >= max_step - 1) || (solNew[0 * BLOCK + i] < 6.0))
        {
            num_steps[eqSysNum] = step_num
            loopFlag = false
            break
        }
        step_num += 1
    }
}
    """


if __name__ == "__main__":
    numSys = 100
    nODE = 2

    xp = np  # xp if gpu_available else np

    x = xp.zeros(numSys)

    b = xp.full(numSys, 0.25)
    c = xp.full(numSys, 5.0)
    additionalArgs = xp.array([b, c])
    solOld = xp.array([xp.random.rand(numSys) * xp.pi - xp.pi / 2.0, xp.zeros(numSys)])
    # self.ode(x, arg, solOld, additionalArgs)
    h = xp.full(numSys, 0.01)

    """
    ks = [xp.zeros((nODE, numSys)) for _ in range(10)]
    dormandPrinceSteps(
        arg,
        x,
        solOld,
        h,
        additionalArgs,
        *ks
    )

    errOld = xp.zeros(numSys)

    err = xp.zeros(numSys)
    solNew = xp.zeros_like(solOld)
    ks_error = [ks[i - 1] for i in [1, 2, 3, 6, 7, 8, 9, 10]]
    error(
        err,
        solOld,
        solNew,
        h,
        *ks_error
    )

    flagSuccess = xp.zeros(numSys, dtype=bool)
    previousReject = xp.zeros(numSys, dtype=bool)
    controllerSuccess(flagSuccess, err, errOld, previousReject, h, x)
    """
    tMax = 10000.0
    max_step = 2000

    # TODO: grow this array as needed might be better

    num_steps = xp.zeros(numSys, dtype=int)

    solOld_check = solOld.copy()

    dopr = DOPR853(
        ODE_pendulum,
        stopping_criterion=None,  # stopping_criterion,
        tmax=tMax,
        max_step=max_step,
    )
    denseOutputLoc, denseOutput, step_num = dopr.integrate(
        solOld,  # Initial condition stored as a vector
        additionalArgs,
    )

    breakpoint()
