/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 2, n = -5)

- BHPC's maple script `outputC_Slm_Zlmkn.mw`.


25th May 2020; RF
6th June. 2020; Sis


Convention (the phase `B_inc`  is defined in `Zlmkn8.c`):
 Zlmkn8  = ( hat_Zlmkn8 * exp(-I * B_inc) )

 For ell = 2, we have only
 0 <= |k| <= 8
 
 //Shorthad variables for PN amplitudes  in `inspiral_orb_data`
 k = sqrt(1. - q^2 ); y = sqrt(1. -  Y^2) ;
 Ym = Y - 1.0 , Yp = Y + 1.0 ;

PNq[11] = 1, q, q ^2, ..., q ^ 9, q ^ 10
PNe[11] = 1, e, e ^ 2, ..., e ^ 9, e ^ 10
PNv[11] = v ^ 8, v ^ 9, ..., v ^ 17, v ^ 18
PNY[11] = 1, Y, Y ^ 2, ..., Y ^ 9, Y ^ 10
PNYp[11] = 1, Yp, Yp^2,...  Yp^10
PNy[21] = 1, y, y ^ 2, ..., y ^ 19, y ^ 20



 WARGNING !! 
`hZ_2mkM5_5PNe10` stores  only the PN amplitudes that has the index
m + k + n > 0 and m + k + n = 0 with n <= 0

 Other modes should be computed from the symmetry relation in `Zlmkn8.c`: 
 Z8[l, -m, -k, -n] = (-1)^(l + k) * conjugate(Z8_[l, m, k, n])
 
 */


// C headers 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// GSL headers
#include<gsl/gsl_complex.h>

// BHPC headers
//#include "../../Zlmkn8_5PNe10.h"
#include "hZ_2mkM5_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_2mkM5_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/
//static int lmax = 2;
//static int kmax = 6 + 2;

/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_2mkM5_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
gsl_complex hZ_2mkM5(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    gsl_complex hZ_2mkM5 = { 0.0 };
    //GSL_SET_COMPLEX(&hZ_2mkM5, 0.0, 0.0);

    double  Re_2mkM5;
    double  Im_2mkM5;

    // NULL check
    if (PN_orb == NULL) {

        perror("Pointer errors: hZ_2mkM5");
        exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */
    
    double q4 = PN_orb->PNq[4];
 
    double e5 = PN_orb->PNe[5];
    double e7 = PN_orb->PNe[7];
    double e9 = PN_orb->PNe[9];

    double v16 = PN_orb->PNv[8];
    double v17 = PN_orb->PNv[9];
    double v18 = PN_orb->PNv[10];

    double Yp = PN_orb->PNYp[1];

    double y4 = PN_orb->PNy[4];
    double y5 = PN_orb->PNy[5];
    double y6 = PN_orb->PNy[6];
    double y7 = PN_orb->PNy[7];
    double y8 = PN_orb->PNy[8];

if (m == 2 && k == 4) {

 // 1. Z_deg[2][2][4][-5]: 16, 
 Re_2mkM5 = 0.3505204656790814244702e-1 * ((0.1000000000000000000000e1 * e5 - 0.6672378873795639188260e1 * e7 + 0.1898084453722310861059e2 * e9) * v16 + (0.8775896872591757628736e2 * e5 - 0.5884074078219986077039e3 * e7 + 0.1673941486169891232047e4 * e9) * v18) * y4 * Yp * Yp * q4;
 Im_2mkM5 = 0.0e0;

}
 else if (m == 1 && k == 5) {

 // 2. Z_deg[2][1][5][-5]: 16, 
 Re_2mkM5 = 0.0e0;
 Im_2mkM5 = -0.7010409313581628489390e-1 * y5 * ((0.1000000000000000000000e1 * e5 - 0.6672378873795639188262e1 * e7 + 0.1898084453722310861062e2 * e9) * v16 + (0.8775896872591757628742e2 * e5 - 0.5884074078219986077041e3 * e7 + 0.1673941486169891232048e4 * e9) * v18) * q4 * Yp;

}
 else if (m == 0 && k == 6) {

 // 3. Z_deg[2][0][6][-5]: 16, 
 Re_2mkM5 = -0.8585962853164929649124e-1 * ((0.1000000000000000000000e1 * e5 - 0.6672378873795639188263e1 * e7 + 0.1898084453722310861060e2 * e9) * v16 + (0.8775896872591757628736e2 * e5 - 0.5884074078219986077039e3 * e7 + 0.1673941486169891232048e4 * e9) * v18) * y6 * q4;
 Im_2mkM5 = 0.0e0;

}
 else if (m == -1 && k == 7) {

 // 4. Z_deg[2][-1][7][-5]: 16, 
 Re_2mkM5 = 0.0e0;
 Im_2mkM5 = 0.7010409313581628489390e-1 * y7 * ((0.1000000000000000000000e1 * e5 - 0.6672378873795639188262e1 * e7 + 0.1898084453722310861062e2 * e9) * v16 + (0.8775896872591757628742e2 * e5 - 0.5884074078219986077041e3 * e7 + 0.1673941486169891232048e4 * e9) * v18) * q4 / Yp;

}
 else if (m == -2 && k == 8) {

 // 5. Z_deg[2][-2][8][-5]: 16, 
 Re_2mkM5 = 0.3505204656790814244702e-1 * ((0.1000000000000000000000e1 * e5 - 0.6672378873795639188260e1 * e7 + 0.1898084453722310861059e2 * e9) * v16 + (0.8775896872591757628736e2 * e5 - 0.5884074078219986077039e3 * e7 + 0.1673941486169891232047e4 * e9) * v18) * y8 * pow(Yp, -0.2e1) * q4;
 Im_2mkM5 = 0.0e0;

 }
 else {

        perror("Parameter errors: hZ_2mkM5");
        exit(1);

    }

    GSL_SET_COMPLEX(&hZ_2mkM5, Re_2mkM5, Im_2mkM5);
    return hZ_2mkM5;

}

