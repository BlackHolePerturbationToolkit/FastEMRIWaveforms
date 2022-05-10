/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 4, n = -7)

- BHPC's maple script `outputC_Slm_Zlmkn.mw`.


25th May 2020; RF
14th June. 2020; Sis


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
`hZ_4mkP0_5PNe10` stores  only the PN amplitudes that has the index
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
#include "hat_Zlmkn8_5PNe10/ell=4/hZ_4mkM7_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_4mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_4mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
CUDA_CALLABLE_MEMBER
cmplx hZ_4mkM7(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    cmplx hZ_4mkM7 = { 0.0 };

    double  Re_4mkM7 = 0.0;
    double  Im_4mkM7 = 0.0;

    // NULL check
    if (PN_orb == NULL) {

        //perror("Point errors: hZ_4mkM7");
        //exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */
    double q4 = PN_orb->PNq[4];
    double q5 = PN_orb->PNq[5];

    double v18 = PN_orb->PNv[10];
    
    double e7 = PN_orb->PNe[7];
    double e9 = PN_orb->PNe[9];

    double Yp = PN_orb->PNYp[1];
    double Yp2 = PN_orb->PNYp[2];
    double Yp3 = PN_orb->PNYp[3];
    double Yp4 = PN_orb->PNYp[4];

    double y4 = PN_orb->PNy[4];
    double y5 = PN_orb->PNy[5];
    double y6 = PN_orb->PNy[6];
    double y7 = PN_orb->PNy[7];
    double y8 = PN_orb->PNy[8];
    double y9 = PN_orb->PNy[9];
    double y10 = PN_orb->PNy[10];
    double y11 = PN_orb->PNy[11];
    double y12 = PN_orb->PNy[12];


if (m == 4 && k == 4) { 

   // 1. Z_deg[4][4][4][-7]: 18, 
   Re_4mkM7  = -0.440152303311217433338e-11 * v18 * y4 * Yp4 * (0.233164000e9 * e7 - 0.1792377397e10 * e9) * q4; 
   Im_4mkM7  = 0.0e0; 

} else if (m == 3 && k == 5) { 

   // 2. Z_deg[4][3][5][-7]: 18, 
   Re_4mkM7  = 0.0e0; 
   Im_4mkM7  = (0.2902748902423032216448e-2 * e7 - 0.2231400011095023878886e-1 * e9) * v18 * Yp3 * y5 * q4; 

} else if (m == 2 && k == 6) { 

   // 3. Z_deg[4][2][6][-7]: 18, 
   Re_4mkM7  = (0.5430545936350544648366e-2 * e7 - 0.4174567167180575430328e-1 * e9) * v18 * y6 * Yp2 * q4; 
   Im_4mkM7  = 0.0e0; 

} else if (m == 1 && k == 7) { 

   // 4. Z_deg[4][1][7][-7]: 18, 
   Re_4mkM7  = 0.0e0; 
   Im_4mkM7  = (-0.7679951714277038828951e-2 * e7 + 0.5903729504864201417501e-1 * e9) * v18 * y7 * q4 * Yp; 

} else if (m == 0 && k == 8) { 

   // 5. Z_deg[4][0][8][-7]: 18, 
   Re_4mkM7  = (-0.8586447048519750483754e-2 * e7 + 0.6600570246823764850132e-1 * e9) * v18 * y8 * q4; 
   Im_4mkM7  = 0.0e0; 

} else if (m == -1 && k == 9) { 

   // 6. Z_deg[4][-1][9][-7]: 18, 
   Re_4mkM7  = 0.0e0; 
   Im_4mkM7  = (0.7679951714277038828951e-2 * e7 - 0.5903729504864201417501e-1 * e9) * v18 * y9 * q4 / Yp; 

} else if (m == -2 && k == 10) { 

   // 7. Z_deg[4][-2][10][-7]: 18, 
   Re_4mkM7  = (0.5430545936350544648366e-2 * e7 - 0.4174567167180575430328e-1 * e9) * v18 * y10 / Yp2 * q4; 
   Im_4mkM7  = 0.0e0; 

} else if (m == -3 && k == 11) { 

   // 8. Z_deg[4][-3][11][-7]: 18, 
   Re_4mkM7  = 0.0e0; 
   Im_4mkM7  = (-0.2902748902423032216448e-2 * e7 + 0.2231400011095023878886e-1 * e9) * v18 * y11 / Yp3 * q4; 

} else if (m == -4 && k == 12) { 

   // 9. Z_deg[4][-4][12][-7]: 18, 
   Re_4mkM7  = -0.440152303311217433338e-11 * v18 * y12 * (0.233164000e9 * e7 - 0.1792377397e10 * e9) / Yp4 * q4; 
   Im_4mkM7  = 0.0e0; 

 } 
 else {

        //perror("Parameter errors: hZ_4mkM7");
        //exit(1);

    }

    hZ_4mkM7 = cmplx(Re_4mkM7, Im_4mkM7);
    return hZ_4mkM7;

}

