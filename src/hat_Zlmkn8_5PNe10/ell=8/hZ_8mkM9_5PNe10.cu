/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 8, n = -9)

- BHPC's maple script `outputC_Slm_Zlmkn.mw`.


25th May 2020; RF
17th June. 2020; Sis


Convention (the phase `B_inc`  is defined in `Zlmkn8.c`):
 Zlmkn8  = ( hat_Zlmkn8 * exp(-I * B_inc) )

 For ell = 8, we have only
 0 <= |k| <= 18 (jmax = 10)
 
 //Shorthad variables for PN amplitudes  in `inspiral_orb_data`
 k = sqrt(1. - q^2); y = sqrt(1. - Y^2) ;
 Ym = Y - 1.0 , Yp = Y + 1.0 ;

PNq[11] = 1, q, q ^2, ..., q ^ 9, q ^ 10
PNe[11] = 1, e, e ^ 2, ..., e ^ 9, e ^ 10
PNv[11] = v ^ 8, v ^ 9, ..., v ^ 17, v ^ 18
PNY[11] = 1, Y, Y ^ 2, ..., Y ^ 9, Y ^ 10
PNYp[11] = 1, Yp, Yp^2,...  Yp^10
PNy[21] = 1, y, y ^ 2, ..., y ^ 19, y ^ 20



 WARGNING !! 
`hZ_8mkP0_5PNe10` stores  only the PN amplitudes that has the index
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
#include "hat_Zlmkn8_5PNe10/ell=8/hZ_8mkM9_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_8mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_8mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
CUDA_CALLABLE_MEMBER
cmplx hZ_8mkM9(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    cmplx hZ_8mkM9 = { 0.0 };

    double  Re_8mkM9 = 0.0;
    double  Im_8mkM9 = 0.0;

    // NULL check
    if (PN_orb == NULL) {

        //perror("Point errors: hZ_8mkM9");
        //exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */
    double Ym = PN_orb->PNYm;

    double q2 = PN_orb->PNq[2];
    
    double v18 = PN_orb->PNv[10];
    
    double e9 = PN_orb->PNe[9];

    double Yp = PN_orb->PNYp[1];
    double Yp2 = PN_orb->PNYp[2];
    double Yp3 = PN_orb->PNYp[3];
    double Yp4 = PN_orb->PNYp[4];
    double Yp5 = PN_orb->PNYp[5];
    double Yp6 = PN_orb->PNYp[6];
    double Yp7 = PN_orb->PNYp[7];
    double Yp8 = PN_orb->PNYp[8];

    double y = PN_orb->PNy[1];
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
    double y13 = PN_orb->PNy[13];
    double y14 = PN_orb->PNy[14];
    double y15 = PN_orb->PNy[15];
    double y16 = PN_orb->PNy[16];
    double y17 = PN_orb->PNy[17];
    double y18 = PN_orb->PNy[18];


if (m == 8 && k == 2) { 

   // 1. Z_deg[8][8][2][-9]: 18, 
   Re_8mkM9  = -0.1975809735598240348656e-6 * Yp * Ym * Yp8 * q2 * e9 * v18; 
   Im_8mkM9  = 0.0e0; 

} else if (m == 7 && k == 3) { 

   // 2. Z_deg[8][7][3][-9]: 18, 
   Re_8mkM9  = 0.0e0; 
   Im_8mkM9  = -0.7903238942392961394627e-6 * q2 * y3 * Yp7 * e9 * v18; 

} else if (m == 6 && k == 4) { 

   // 3. Z_deg[8][6][4][-9]: 18, 
   Re_8mkM9  = -0.2164391123050948507059e-5 * e9 * q2 * y4 * Yp6 * v18; 
   Im_8mkM9  = 0.0e0; 

} else if (m == 5 && k == 5) { 

   // 4. Z_deg[8][5][5][-9]: 18, 
   Re_8mkM9  = 0.0e0; 
   Im_8mkM9  = 0.4675619212809658993515e-5 * y5 * Yp5 * q2 * e9 * v18; 

} else if (m == 4 && k == 6) { 

   // 5. Z_deg[8][4][6][-9]: 18, 
   Re_8mkM9  = 0.8429092408164899784911e-5 * y6 * Yp4 * q2 * e9 * v18; 
   Im_8mkM9  = 0.0e0; 

} else if (m == 3 && k == 7) { 

   // 6. Z_deg[8][3][7][-9]: 18, 
   Re_8mkM9  = 0.0e0; 
   Im_8mkM9  = -0.1305829380818641095237e-4 * q2 * Yp3 * y7 * e9 * v18; 

} else if (m == 2 && k == 8) { 

   // 7. Z_deg[8][2][8][-9]: 18, 
   Re_8mkM9  = -0.1768101339945439490086e-4 * e9 * q2 * y8 * Yp2 * v18; 
   Im_8mkM9  = 0.0e0; 

} else if (m == 1 && k == 9) { 

   // 8. Z_deg[8][1][9][-9]: 18, 
   Re_8mkM9  = 0.0e0; 
   Im_8mkM9  = 0.2113285305705265620528e-4 * q2 * Yp * y9 * e9 * v18; 

} else if (m == 0 && k == 10) { 

   // 9. Z_deg[8][0][10][-9]: 18, 
   Re_8mkM9  = 0.2241477555369119174721e-4 * e9 * q2 * y10 * v18; 
   Im_8mkM9  = 0.0e0; 

} else if (m == -1 && k == 11) { 

   // 10. Z_deg[8][-1][11][-9]: 18, 
   Re_8mkM9  = 0.0e0; 
   Im_8mkM9  = -0.2113285305705265620528e-4 / Yp * y11 * q2 * e9 * v18; 

} else if (m == -2 && k == 12) { 

   // 11. Z_deg[8][-2][12][-9]: 18, 
   Re_8mkM9  = -0.1768101339945439490086e-4 * e9 * q2 / Yp2 * y12 * v18; 
   Im_8mkM9  = 0.0e0; 

} else if (m == -3 && k == 13) { 

   // 12. Z_deg[8][-3][13][-9]: 18, 
   Re_8mkM9  = 0.0e0; 
   Im_8mkM9  = 0.1305829380818641095237e-4 * y13 / Yp3 * q2 * e9 * v18; 

} else if (m == -4 && k == 14) { 

   // 13. Z_deg[8][-4][14][-9]: 18, 
   Re_8mkM9  = 0.8429092408164899784911e-5 / Yp4 * y14 * q2 * e9 * v18; 
   Im_8mkM9  = 0.0e0; 

} else if (m == -5 && k == 15) { 

   // 14. Z_deg[8][-5][15][-9]: 18, 
   Re_8mkM9  = 0.0e0; 
   Im_8mkM9  = -0.4675619212809658993515e-5 * q2 * y15 / Yp5 * e9 * v18; 

} else if (m == -6 && k == 16) { 

   // 15. Z_deg[8][-6][16][-9]: 18, 
   Re_8mkM9  = -0.2164391123050948507059e-5 * e9 * q2 / Yp6 * y16 * v18; 
   Im_8mkM9  = 0.0e0; 

} else if (m == -7 && k == 17) { 

   // 16. Z_deg[8][-7][17][-9]: 18, 
   Re_8mkM9  = 0.0e0; 
   Im_8mkM9  = 0.7903238942392961394627e-6 / Yp7 * y17 * q2 * e9 * v18; 

} else if (m == -8 && k == 18) { 

   // 17. Z_deg[8][-8][18][-9]: 18, 
   Re_8mkM9  = 0.1975809735598240348656e-6 / Yp8 * y18 * q2 * e9 * v18; 
   Im_8mkM9  = 0.0e0; 

 } 

 else {

        //perror("Parameter errors: hZ_8mkM9");
        //exit(1);

    }

    hZ_8mkM9 = cmplx(Re_8mkM9, Im_8mkM9);
    return hZ_8mkM9;

}

