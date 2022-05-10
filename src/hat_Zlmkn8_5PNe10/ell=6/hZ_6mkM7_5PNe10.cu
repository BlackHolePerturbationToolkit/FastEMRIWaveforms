/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 6, n = -7)

- BHPC's maple script `outputC_Slm_Zlmkn.mw`.


25th May 2020; RF
17th June. 2020; Sis


Convention (the phase `B_inc`  is defined in `Zlmkn8.c`):
 Zlmkn8  = ( hat_Zlmkn8 * exp(-I * B_inc) )

 For ell = 6, we have only
 0 <= |k| <= 14 (jmax = 8)
 
 //Shorthad variables for PN amplitudes  in `inspiral_orb_data`
 k = sqrt(1. - q^2); y = sqrt(1. -  Y^2) ;
 Ym = Y - 1.0 , Yp = Y + 1.0 ;

PNq[11] = 1, q, q ^2, ..., q ^ 9, q ^ 10
PNe[11] = 1, e, e ^ 2, ..., e ^ 9, e ^ 10
PNv[11] = v ^ 8, v ^ 9, ..., v ^ 17, v ^ 18
PNY[11] = 1, Y, Y ^ 2, ..., Y ^ 9, Y ^ 10
PNYp[11] = 1, Yp, Yp^2,...  Yp^10
PNy[21] = 1, y, y ^ 2, ..., y ^ 19, y ^ 20



 WARGNING !! 
`hZ_6mkP0_5PNe10` stores  only the PN amplitudes that has the index
m + k + n > 0 and m + k + n = 0 with n <= 0

 Other modes should be computed from the symmetry relation in `Zlmkn8.c`: 
 Z8[l, -m, -k, -n] = (-1)^(l + k) * conjugate(Z8_[l, m, k, n])
 
 */


// C headers 
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// GSL headers
#include<gsl/gsl_complex.h>

// BHPC headers
#include "hZ_6mkM7_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_6mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_6mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
gsl_complex hZ_6mkM7(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    gsl_complex hZ_6mkM7 = { 0.0 };

    double  Re_6mkM7;
    double  Im_6mkM7;

    // NULL check
    if (PN_orb == NULL) {

        perror("Pointer errors: hZ_6mkM7");
        exit(1);

    }

    //printf("Z6mk7[%d, %d]\n", m, k);


    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */
    double Ym = PN_orb->PNYm;

    double q = PN_orb->PNq[1];
    double q2 = PN_orb->PNq[2];
 
    double v16 = PN_orb->PNv[8];
    double v18 = PN_orb->PNv[10];
    
    double e7 = PN_orb->PNe[7];
    double e9 = PN_orb->PNe[9];

    double Y = PN_orb->PNY[1];
 
    double Yp = PN_orb->PNYp[1];
    double Yp2 = PN_orb->PNYp[2];
    double Yp3 = PN_orb->PNYp[3];
    double Yp4 = PN_orb->PNYp[4];
    double Yp5 = PN_orb->PNYp[5];
    double Yp6 = PN_orb->PNYp[6];

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

if (m == 6 && k == 2) { 

   // 1. Z_deg[6][6][2][-7]: 16, 
   Re_6mkM7  = -0.1659088947748562147203e-14 * Yp * Ym * Yp6 * (0.25029559296e11 * v16 * e7 + 0.4850010416928e13 * v18 * e7 - 0.193893633864e12 * v16 * e9 - 0.37410812945489e14 * v18 * e9) * q2; 
   Im_6mkM7  = 0.0e0; 

} else if (m == 5 && k == 3) { 

   // 2. Z_deg[6][5][3][-7]: 16, 
   Re_6mkM7  = 0.0e0; 
   Im_6mkM7  = -0.5747252703552991860938e-14 * Yp5 * y3 * (0.25029559296e11 * v16 * e7 + 0.4850010416928e13 * v18 * e7 - 0.193893633864e12 * v16 * e9 - 0.37410812945489e14 * v18 * e9) * q2; 

} else if (m == 4 && k == 4) { 

   // 3. Z_deg[6][4][4][-7]: 16, 
   Re_6mkM7  = -0.1347850232821638302491e-13 * y4 * Yp4 * (0.25029559296e11 * v16 * e7 + 0.4850010416928e13 * v18 * e7 - 0.193893633864e12 * v16 * e9 - 0.37410812945489e14 * v18 * e9) * q2; 
   Im_6mkM7  = 0.0e0; 

} else if (m == 3 && k == 5) { 

   // 4. Z_deg[6][3][5][-7]: 16, 
   Re_6mkM7  = 0.0e0; 
   Im_6mkM7  = 0.2460826588850004398686e-13 * Yp3 * y5 * (0.25029559296e11 * v16 * e7 + 0.4850010416928e13 * v18 * e7 - 0.193893633864e12 * v16 * e9 - 0.37410812945489e14 * v18 * e9) * q2; 

} else if (m == 2 && k == 6) { 

   // 5. Z_deg[6][2][6][-7]: 16, 
   Re_6mkM7  = 0.3691239883275006598028e-13 * y6 * Yp2 * (0.25029559296e11 * v16 * e7 + 0.4850010416928e13 * v18 * e7 - 0.193893633864e12 * v16 * e9 - 0.37410812945489e14 * v18 * e9) * q2; 
   Im_6mkM7  = 0.0e0; 

} else if (m == 1 && k == 7) { 

   // 6. Z_deg[6][1][7][-7]: 16, 
   Re_6mkM7  = 0.0e0; 
   Im_6mkM7  = -0.4669090168481235802835e-13 * Yp * y7 * (0.25029559296e11 * v16 * e7 + 0.4850010416928e13 * v18 * e7 - 0.193893633864e12 * v16 * e9 - 0.37410812945489e14 * v18 * e9) * q2; 

} else if (m == 0 && k == 8) { 

   // 7. Z_deg[6][0][8][-7]: 16, 
   Re_6mkM7  = -0.5043193779902059651883e-13 * y8 * (0.25029559296e11 * v16 * e7 + 0.4850010416928e13 * v18 * e7 - 0.193893633864e12 * v16 * e9 - 0.37410812945489e14 * v18 * e9) * q2; 
   Im_6mkM7  = 0.0e0; 

} else if (m == -1 && k == 9) { 

   // 8. Z_deg[6][-1][9][-7]: 16, 
   Re_6mkM7  = 0.0e0; 
   Im_6mkM7  = 0.4669090168481235802835e-13 * y9 * (0.25029559296e11 * v16 * e7 + 0.4850010416928e13 * v18 * e7 - 0.193893633864e12 * v16 * e9 - 0.37410812945489e14 * v18 * e9) / Yp * q2; 

} else if (m == -2 && k == 10) { 

   // 9. Z_deg[6][-2][10][-7]: 16, 
   Re_6mkM7  = 0.3691239883275006598028e-13 * y10 * (0.25029559296e11 * v16 * e7 + 0.4850010416928e13 * v18 * e7 - 0.193893633864e12 * v16 * e9 - 0.37410812945489e14 * v18 * e9) / Yp2 * q2; 
   Im_6mkM7  = 0.0e0; 

} else if (m == -3 && k == 11) { 

   // 10. Z_deg[6][-3][11][-7]: 16, 
   Re_6mkM7  = 0.0e0; 
   Im_6mkM7  = -0.2460826588850004398686e-13 * y11 * (0.25029559296e11 * v16 * e7 + 0.4850010416928e13 * v18 * e7 - 0.193893633864e12 * v16 * e9 - 0.37410812945489e14 * v18 * e9) / Yp3 * q2; 

} else if (m == -4 && k == 12) { 

   // 11. Z_deg[6][-4][12][-7]: 16, 
   Re_6mkM7  = -0.1347850232821638302491e-13 * y12 * (0.25029559296e11 * v16 * e7 + 0.4850010416928e13 * v18 * e7 - 0.193893633864e12 * v16 * e9 - 0.37410812945489e14 * v18 * e9) / Yp4 * q2; 
   Im_6mkM7  = 0.0e0; 

} else if (m == -5 && k == 13) { 

   // 12. Z_deg[6][-5][13][-7]: 16, 
   Re_6mkM7  = 0.0e0; 
   Im_6mkM7  = 0.5747252703552991860938e-14 * y13 * (0.25029559296e11 * v16 * e7 + 0.4850010416928e13 * v18 * e7 - 0.193893633864e12 * v16 * e9 - 0.37410812945489e14 * v18 * e9) / Yp5 * q2; 

} else if (m == -6 && k == 14) { 

   // 13. Z_deg[6][-6][14][-7]: 16, 
   Re_6mkM7  = 0.1659088947748562147203e-14 * y14 * (0.25029559296e11 * v16 * e7 + 0.4850010416928e13 * v18 * e7 - 0.193893633864e12 * v16 * e9 - 0.37410812945489e14 * v18 * e9) / Yp6 * q2; 
   Im_6mkM7  = 0.0e0; 

 } 
 else {

        perror("Parameter errors: hZ_6mkM7");
        exit(1);

    }

    GSL_SET_COMPLEX(&hZ_6mkM7, Re_6mkM7, Im_6mkM7);
    return hZ_6mkM7;

}

