/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 4, n = -6)

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
#include "hat_Zlmkn8_5PNe10/ell=4/hZ_4mkM6_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_4mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_4mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
CUDA_CALLABLE_MEMBER
cmplx hZ_4mkM6(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    cmplx hZ_4mkM6 = { 0.0 };

    double  Re_4mkM6 = 0.0;
    double  Im_4mkM6 = 0.0;

    // NULL check
    if (PN_orb == NULL) {

        //perror("Point errors: hZ_4mkM6");
        //exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */
    double q3 = PN_orb->PNq[3];
    double q4 = PN_orb->PNq[4];

    double v18 = PN_orb->PNv[10];
    
    double e6 = PN_orb->PNe[6];
    double e8 = PN_orb->PNe[8];
    double e10 = PN_orb->PNe[10];

    double Yp = PN_orb->PNYp[1];
    double Yp2 = PN_orb->PNYp[2];
    double Yp3 = PN_orb->PNYp[3];
    double Yp4 = PN_orb->PNYp[4];

    double y3 = PN_orb->PNy[3];
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

   // 1. Z_deg[4][4][4][-6]: 18, 
   Re_4mkM6  = -0.901431917181373303476e-8 * v18 * y4 * Yp4 * (0.35717024e8 * e6 - 0.319301888e9 * e8 + 0.1273968635e10 * e10) * q4; 
   Im_4mkM6  = 0.0e0; 

} else if (m == 4 && k == 3) { 

   // 2. Z_deg[4][4][3][-6]: 18, 
   Re_4mkM6  = -0.9603322981335653091012e-14 * v18 * y3 * Yp4 * (0.163806739712e12 * e6 - 0.1421707890176e13 * e8 + 0.5468007909869e13 * e10) * q3; 
   Im_4mkM6  = 0.0e0; 

} else if (m == 3 && k == 5) { 

   // 3. Z_deg[4][3][5][-6]: 18, 
   Re_4mkM6  = 0.0e0; 
   Im_4mkM6  = 0.2549634485667757449386e-7 * v18 * Yp3 * y5 * (0.35717024e8 * e6 - 0.319301888e9 * e8 + 0.1273968635e10 * e10) * q4; 

} else if (m == 3 && k == 4) { 

   // 4. Z_deg[4][3][4][-6]: 18, 
   Re_4mkM6  = 0.0e0; 
   Im_4mkM6  = 0.2716229920810821146363e-13 * v18 * Yp3 * y4 * (0.163806739712e12 * e6 - 0.1421707890176e13 * e8 + 0.5468007909869e13 * e10) * q3; 

} else if (m == 2 && k == 6) { 

   // 5. Z_deg[4][2][6][-6]: 18, 
   Re_4mkM6  = 0.476992935343617172456e-7 * v18 * y6 * Yp2 * (0.35717024e8 * e6 - 0.319301888e9 * e8 + 0.1273968635e10 * e10) * q4; 
   Im_4mkM6  = 0.0e0; 

} else if (m == 2 && k == 5) { 

   // 6. Z_deg[4][2][5][-6]: 18, 
   Re_4mkM6  = 0.5081600873689103399744e-13 * v18 * y5 * Yp2 * (0.163806739712e12 * e6 - 0.1421707890176e13 * e8 + 0.5468007909869e13 * e10) * q3; 
   Im_4mkM6  = 0.0e0; 

} else if (m == 1 && k == 7) { 

   // 7. Z_deg[4][1][7][-6]: 18, 
   Re_4mkM6  = 0.0e0; 
   Im_4mkM6  = -0.6745698783190962379136e-7 * v18 * y7 * Yp * (0.35717024e8 * e6 - 0.319301888e9 * e8 + 0.1273968635e10 * e10) * q4; 

} else if (m == 1 && k == 6) { 

   // 8. Z_deg[4][1][6][-6]: 18, 
   Re_4mkM6  = 0.0e0; 
   Im_4mkM6  = -0.71864688741380991255e-13 * v18 * y6 * Yp * (0.163806739712e12 * e6 - 0.1421707890176e13 * e8 + 0.5468007909869e13 * e10) * q3; 

} else if (m == 0 && k == 8) { 

   // 9. Z_deg[4][0][8][-6]: 18, 
   Re_4mkM6  = -0.7541920517476303799354e-7 * v18 * y8 * (0.35717024e8 * e6 - 0.319301888e9 * e8 + 0.1273968635e10 * e10) * q4; 
   Im_4mkM6  = 0.0e0; 

} else if (m == 0 && k == 7) { 

   // 10. Z_deg[4][0][7][-6]: 18, 
   Re_4mkM6  = -0.267823882012652833784e-13 * v18 * y7 * (0.14979078610031e14 * e10 - 0.3942699341312e13 * e8 + 0.458654932736e12 * e6) * q3; 
   Im_4mkM6  = 0.0e0; 

} else if (m == -1 && k == 9) { 

   // 11. Z_deg[4][-1][9][-6]: 18, 
   Re_4mkM6  = 0.0e0; 
   Im_4mkM6  = 0.6745698783190962379136e-7 * v18 * y9 * (0.35717024e8 * e6 - 0.319301888e9 * e8 + 0.1273968635e10 * e10) / Yp * q4; 

} else if (m == -1 && k == 8) { 

   // 12. Z_deg[4][-1][8][-6]: 18, 
   Re_4mkM6  = 0.0e0; 
   Im_4mkM6  = 0.71864688741380991255e-13 * v18 * y8 * (0.163806739712e12 * e6 - 0.1421707890176e13 * e8 + 0.5468007909869e13 * e10) / Yp * q3; 

} else if (m == -2 && k == 10) { 

   // 13. Z_deg[4][-2][10][-6]: 18, 
   Re_4mkM6  = 0.476992935343617172456e-7 * v18 * y10 * (0.35717024e8 * e6 - 0.319301888e9 * e8 + 0.1273968635e10 * e10) / Yp2 * q4; 
   Im_4mkM6  = 0.0e0; 

} else if (m == -2 && k == 9) { 

   // 14. Z_deg[4][-2][9][-6]: 18, 
   Re_4mkM6  = 0.5081600873689103399744e-13 * v18 * y9 * (0.163806739712e12 * e6 - 0.1421707890176e13 * e8 + 0.5468007909869e13 * e10) / Yp2 * q3; 
   Im_4mkM6  = 0.0e0; 

} else if (m == -3 && k == 11) { 

   // 15. Z_deg[4][-3][11][-6]: 18, 
   Re_4mkM6  = 0.0e0; 
   Im_4mkM6  = -0.2549634485667757449386e-7 * v18 * y11 * (0.35717024e8 * e6 - 0.319301888e9 * e8 + 0.1273968635e10 * e10) / Yp3 * q4; 

} else if (m == -3 && k == 10) { 

   // 16. Z_deg[4][-3][10][-6]: 18, 
   Re_4mkM6  = 0.0e0; 
   Im_4mkM6  = -0.2716229920810821146363e-13 * v18 * y10 * (0.163806739712e12 * e6 - 0.1421707890176e13 * e8 + 0.5468007909869e13 * e10) / Yp3 * q3; 

} else if (m == -4 && k == 12) { 

   // 17. Z_deg[4][-4][12][-6]: 18, 
   Re_4mkM6  = -0.901431917181373303476e-8 * v18 * y12 * (0.35717024e8 * e6 - 0.319301888e9 * e8 + 0.1273968635e10 * e10) / Yp4 * q4; 
   Im_4mkM6  = 0.0e0; 

} else if (m == -4 && k == 11) { 

   // 18. Z_deg[4][-4][11][-6]: 18, 
   Re_4mkM6  = -0.9603322981335653091012e-14 * v18 * y11 * (0.163806739712e12 * e6 - 0.1421707890176e13 * e8 + 0.5468007909869e13 * e10) / Yp4 * q3; 
   Im_4mkM6  = 0.0e0; 

 } 
 else {

        //perror("Parameter errors: hZ_4mkM6");
        //exit(1);

    }

    hZ_4mkM6 = cmplx(Re_4mkM6, Im_4mkM6);
    return hZ_4mkM6;

}

