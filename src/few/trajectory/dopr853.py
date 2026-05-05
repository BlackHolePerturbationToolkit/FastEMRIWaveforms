import numpy as np

"""
The following coefficients are derived from E. Hairer, S. Nørsett, and G. Wanner, Solving Or-
dinary Differential Equations I: Nonstiff Problems,
ser. Springer Series in Computational Mathematics.
Springer Berlin Heidelberg, 1993, pp. 173–195. doi: 10.1007/978-3-540-78862-1. --- IGNORE ---
"""

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


class DOPR853:
    """
    Stepper class for performing Dormand Prince 8(5,3) integration of ODEs. This is a 12 stage embedded Runge-Kutta method 
    with an error estimate that can be used for adaptive step size control. The class controls steps and error estimates 
    for a system of ODEs, and can be used to take single steps or multiple steps. The class also has a method for preparing the 
    coefficients needed for dense output interpolation, which can be used to evaluate the solution at intermediate points within a step.

    The explicit methods employed within this class are described in: 
    E. Hairer, S. Nørsett, and G. Wanner, Solving Ordinary Differential Equations I: Nonstiff Problems, 
    ser. Springer Series in Computational Mathematics. Springer Berlin Heidelberg, 1993, pp. 173–195. 
    doi: 10.1007/978-3-540-78862-1.

    Arguments:
        ode: A function that evaluates the derivatives of the system of ODEs. The function should have the signature 
        ode(x, y, dydx, additionalArgs), where x is the independent variable, y is the dependent variable(s), 
        dydx is an array to store the derivatives, and additionalArgs is a dictionary of any additional arguments 
        needed for the ODE evaluation.
    """

    def __init__(
        self,
        ode,
    ):
        self.ode = ode

        self.fix_step = False
        self.abstol = 1e-10

    @property
    def xp(self):
        return np

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
