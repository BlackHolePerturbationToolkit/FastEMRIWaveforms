/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 3, n = -6)

- BHPC's maple script `outputC_Slm_Zlmkn.mw`.


25th May 2020; RF
12th June. 2020; Sis


Convention (the phase `B_inc`  is defined in `Zlmkn8.c`):
 Zlmkn8  = ( hat_Zlmkn8 * exp(-I * B_inc) )

 For ell = 3, we have only
 0 <= |k| <= 10 (jmax = 7)
 
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
`hZ_3mkP0_5PNe10` stores  only the PN amplitudes that has the index
m + k + n > 0 and m + k + n = 0 with n <= 0

 Other modes should be computed from the symmetry relation in `Zlmkn8.c`: 
 Z8[l, -m, -k, -n] = (-1)^(l + k) * conjugate(Z8_[l, m, k, n])
 
 */


// C headers 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// GSL headers
//#include<gsl/cmplx.h>

// BHPC headers
#include "hat_Zlmkn8_5PNe10/ell=3/hZ_3mkM6_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_3mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/
//static int lmax = 2;
//static int kmax = 6 + 2;

/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_3mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
CUDA_CALLABLE_MEMBER
cmplx hZ_3mkM6(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    cmplx hZ_3mkM6 = { 0.0 };

    double  Re_3mkM6 = 0.0;
    double  Im_3mkM6 = 0.0;

    // NULL check
    if (PN_orb == NULL) {

        //perror("Point errors: hZ_3mkM6");
        //exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */
    double q4 = PN_orb->PNq[4];
    double q5 = PN_orb->PNq[5];

    double v17 = PN_orb->PNv[9];
    double v18 = PN_orb->PNv[10];
    
    double e6 = PN_orb->PNe[6];
    double e8 = PN_orb->PNe[8];
    double e10 = PN_orb->PNe[10];

    double Yp = PN_orb->PNYp[1];
    double Yp2 = PN_orb->PNYp[2];
    double Yp3 = PN_orb->PNYp[3];

    double y4 = PN_orb->PNy[4];
    double y5 = PN_orb->PNy[5];
    double y6 = PN_orb->PNy[6];
    double y7 = PN_orb->PNy[7];
    double y8 = PN_orb->PNy[8];
    double y9 = PN_orb->PNy[9];
    double y10 = PN_orb->PNy[10];


if (m == 3 && k == 4) { 

   // 1. Z_deg[3][3][4][-6]: 17, 
   Re_3mkM6  = 0.0e0; 
   Im_3mkM6  = 0.6468891313339304281904e-11 * v17 * Yp3 * y4 * (0.23490473335e11 * e10 - 0.7582561024e10 * e8 + 0.1056063232e10 * e6) * q4; 

} else if (m == 2 && k == 5) { 

   // 2. Z_deg[3][2][5][-6]: 17, 
   Re_3mkM6  = 0.1584548291920382760017e-10 * v17 * y5 * Yp2 * (0.23490473335e11 * e10 - 0.7582561024e10 * e8 + 0.1056063232e10 * e6) * q4; 
   Im_3mkM6  = 0.0e0; 

} else if (m == 1 && k == 6) { 

   // 3. Z_deg[3][1][6][-6]: 17, 
   Re_3mkM6  = 0.0e0; 
   Im_3mkM6  = -0.2505390832498895041781e-10 * v17 * Yp * y6 * (0.23490473335e11 * e10 - 0.7582561024e10 * e8 + 0.1056063232e10 * e6) * q4; 

} else if (m == 0 && k == 7) { 

   // 4. Z_deg[3][0][7][-6]: 17, 
   Re_3mkM6  = -0.2892976143136915293164e-10 * y7 * (0.23490473335e11 * e10 - 0.7582561024e10 * e8 + 0.1056063232e10 * e6) * v17 * q4; 
   Im_3mkM6  = -0.385730152418255372422e-10 * y7 * v18 * (0.23881984e8 * e6 - 0.217651072e9 * e8 + 0.882356593e9 * e10) * q4; 

} else if (m == -1 && k == 8) { 

   // 5. Z_deg[3][-1][8][-6]: 17, 
   Re_3mkM6  = 0.0e0; 
   Im_3mkM6  = 0.2505390832498895041781e-10 * v17 * y8 * (0.23490473335e11 * e10 - 0.7582561024e10 * e8 + 0.1056063232e10 * e6) / Yp * q4; 

} else if (m == -2 && k == 9) { 

   // 6. Z_deg[3][-2][9][-6]: 17, 
   Re_3mkM6  = 0.1584548291920382760017e-10 * v17 * y9 * (0.23490473335e11 * e10 - 0.7582561024e10 * e8 + 0.1056063232e10 * e6) / Yp2 * q4; 
   Im_3mkM6  = 0.0e0; 

} else if (m == -3 && k == 10) { 

   // 7. Z_deg[3][-3][10][-6]: 17, 
   Re_3mkM6  = 0.0e0; 
   Im_3mkM6  = -0.6468891313339304281904e-11 * v17 * y10 * (0.23490473335e11 * e10 - 0.7582561024e10 * e8 + 0.1056063232e10 * e6) / Yp3 * q4; 

 } 
 else {

        //perror("Parameter errors: hZ_3mkM6");
        //exit(1);

    }

    hZ_3mkM6 = cmplx(Re_3mkM6, Im_3mkM6);
    return hZ_3mkM6;

}

