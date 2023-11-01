/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 7, n = -8)

- BHPC's maple script `outputC_Slm_Zlmkn.mw`.


25th May 2020; RF
17th June. 2020; Sis


Convention (the phase `B_inc`  is defined in `Zlmkn8.c`):
 Zlmkn8  = ( hat_Zlmkn8 * exp(-I * B_inc) )

 For ell = 7, we have only
 0 <= |k| <= 16 (jmax = 9)
 
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
`hZ_7mkP0_5PNe10` stores  only the PN amplitudes that has the index
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
#include "hat_Zlmkn8_5PNe10/ell=7/hZ_7mkM8_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_7mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_7mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
CUDA_CALLABLE_MEMBER
cmplx hZ_7mkM8(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    cmplx hZ_7mkM8 = { 0.0 };

    double  Re_7mkM8 = 0.0;
    double  Im_7mkM8 = 0.0;

    // NULL check
    if (PN_orb == NULL) {

        //perror("Point errors: hZ_7mkM8");
        //exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */
    double Ym = PN_orb->PNYm;

    double q2 = PN_orb->PNq[2];
    
    double v17 = PN_orb->PNv[9];
    double v18 = PN_orb->PNv[10];
    
    double e8 = PN_orb->PNe[8];
    double e10 = PN_orb->PNe[10];

    double Yp = PN_orb->PNYp[1];
    double Yp2 = PN_orb->PNYp[2];
    double Yp3 = PN_orb->PNYp[3];
    double Yp4 = PN_orb->PNYp[4];
    double Yp5 = PN_orb->PNYp[5];
    double Yp6 = PN_orb->PNYp[6];
    double Yp7 = PN_orb->PNYp[7];

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


if (m == 7 && k == 2) { 

   // 1. Z_deg[7][7][2][-8]: 17, 
   Re_7mkM8  = 0.0e0; 
   Im_7mkM8  = (0.2528510400780596767852e-4 * e10 - 0.3065692161742771002489e-5 * e8) * v17 * Yp7 * q2 * Yp * Ym; 

} else if (m == 6 && k == 3) { 

   // 2. Z_deg[7][6][3][-8]: 17, 
   Re_7mkM8  = (-0.9460819618615458904965e-4 * e10 + 0.1147076972255981179300e-4 * e8) * v17 * y3 * Yp6 * q2; 
   Im_7mkM8  = 0.0e0; 

} else if (m == 5 && k == 4) { 

   // 3. Z_deg[7][5][4][-8]: 17, 
   Re_7mkM8  = 0.0e0; 
   Im_7mkM8  = (0.2412045192495083667471e-3 * e10 - 0.2924483932563088746122e-4 * e8) * v17 * Yp5 * y4 * q2; 

} else if (m == 4 && k == 5) { 

   // 4. Z_deg[7][4][5][-8]: 17, 
   Re_7mkM8  = (0.4824090384990167334943e-3 * e10 - 0.5848967865126177492244e-4 * e8) * v17 * y5 * Yp4 * q2; 
   Im_7mkM8  = 0.0e0; 

} else if (m == 3 && k == 6) { 

   // 5. Z_deg[7][3][6][-8]: 17, 
   Re_7mkM8  = 0.0e0; 
   Im_7mkM8  = (-0.7999848880886756942129e-3 * e10 + 0.9699415909734789522727e-4 * e8) * v17 * y6 * Yp3 * q2; 

} else if (m == 2 && k == 7) { 

   // 6. Z_deg[7][2][7][-8]: 17, 
   Re_7mkM8  = (-0.1131349478428527826237e-2 * e10 + 0.1371704552664431121844e-3 * e8) * v17 * y7 * Yp2 * q2; 
   Im_7mkM8  = 0.0e0; 

} else if (m == 1 && k == 8) { 

   // 7. Z_deg[7][1][8][-8]: 17, 
   Re_7mkM8  = 0.0e0; 
   Im_7mkM8  = (0.1385614471456888661721e-2 * e10 - 0.1679988115940255882034e-3 * e8) * v17 * y8 * q2 * Yp; 

} else if (m == 0 && k == 9) { 

   // 8. Z_deg[7][0][9][-8]: 17, 
   Re_7mkM8  = (0.1481284177813582293038e-2 * e10 - 0.1795982841057227167416e-3 * e8) * y9 * v17 * q2; 
   Im_7mkM8  = (0.7601382869361422213456e-4 * e10 - 0.7542141231343552250112e-5 * e8) * y9 * v18 * q2; 

} else if (m == -1 && k == 10) { 

   // 9. Z_deg[7][-1][10][-8]: 17, 
   Re_7mkM8  = 0.0e0; 
   Im_7mkM8  = (-0.1385614471456888661721e-2 * e10 + 0.1679988115940255882034e-3 * e8) * v17 * y10 * q2 / Yp; 

} else if (m == -2 && k == 11) { 

   // 10. Z_deg[7][-2][11][-8]: 17, 
   Re_7mkM8  = (-0.1131349478428527826237e-2 * e10 + 0.1371704552664431121844e-3 * e8) * v17 * y11 / Yp2 * q2; 
   Im_7mkM8  = 0.0e0; 

} else if (m == -3 && k == 12) { 

   // 11. Z_deg[7][-3][12][-8]: 17, 
   Re_7mkM8  = 0.0e0; 
   Im_7mkM8  = (0.7999848880886756942129e-3 * e10 - 0.9699415909734789522727e-4 * e8) * v17 * y12 / Yp3 * q2; 

} else if (m == -4 && k == 13) { 

   // 12. Z_deg[7][-4][13][-8]: 17, 
   Re_7mkM8  = (0.4824090384990167334943e-3 * e10 - 0.5848967865126177492244e-4 * e8) * v17 * y13 / Yp4 * q2; 
   Im_7mkM8  = 0.0e0; 

} else if (m == -5 && k == 14) { 

   // 13. Z_deg[7][-5][14][-8]: 17, 
   Re_7mkM8  = 0.0e0; 
   Im_7mkM8  = (-0.2412045192495083667471e-3 * e10 + 0.2924483932563088746122e-4 * e8) * v17 * y14 / Yp5 * q2; 

} else if (m == -6 && k == 15) { 

   // 14. Z_deg[7][-6][15][-8]: 17, 
   Re_7mkM8  = (-0.9460819618615458904965e-4 * e10 + 0.1147076972255981179300e-4 * e8) * v17 * y15 / Yp6 * q2; 
   Im_7mkM8  = 0.0e0; 

} else if (m == -7 && k == 16) { 

   // 15. Z_deg[7][-7][16][-8]: 17, 
   Re_7mkM8  = 0.0e0; 
   Im_7mkM8  = (0.2528510400780596767852e-4 * e10 - 0.3065692161742771002489e-5 * e8) * v17 * y16 / Yp7 * q2; 

 }  
 else {

        //perror("Parameter errors: hZ_7mkM8");
        //exit(1);

    }

    hZ_7mkM8 = cmplx(Re_7mkM8, Im_7mkM8);
    return hZ_7mkM8;

}

