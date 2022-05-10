/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 9, n = -9)

- BHPC's maple script `outputC_Slm_Zlmkn.mw`.


25th May 2020; RF
20th June. 2020; Sis


Convention (the phase `B_inc`  is defined in `Zlmkn8.c`):
 Zlmkn8  = ( hat_Zlmkn8 * exp(-I * B_inc) )

 For ell = 9, we have only
 0 <= |k| <= 18 (jmax = 9)
 
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
`hZ_9mkP0_5PNe10` stores  only the PN amplitudes that has the index
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
#include "hat_Zlmkn8_5PNe10/ell=9/hZ_9mkM9_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_9mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_9mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
CUDA_CALLABLE_MEMBER
cmplx hZ_9mkM9(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    cmplx hZ_9mkM9 = { 0.0 };

    double  Re_9mkM9 = 0.0;
    double  Im_9mkM9 = 0.0;

    // NULL check
    if (PN_orb == NULL) {

        //perror("Point errors: hZ_9mkM9");
        //exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */
    double K = PN_orb->PNK;
    double Ym = PN_orb->PNYm;

    double q = PN_orb->PNq[1];
    //double q2 = PN_orb->PNq[2];
    
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
    double Yp9 = PN_orb->PNYp[9];

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


if (m == 9 && k == 1) { 

   // 1. Z_deg[9][9][1][-9]: 18, 
   Re_9mkM9  = -0.2088580805792932206798e-11 * e9 * y * q * Yp9 * v18; 
   Im_9mkM9  = 0.0e0; 

} else if (m == 8 && k == 2) { 

   // 2. Z_deg[9][8][2][-9]: 18, 
   Re_9mkM9  = 0.0e0; 
   Im_9mkM9  = -0.8861097904993476144797e-11 * Yp8 * q * Yp * Ym * e9 * v18; 

} else if (m == 7 && k == 3) { 

   // 3. Z_deg[9][7][3][-9]: 18, 
   Re_9mkM9  = 0.2583431780976571600785e-10 * e9 * q * y3 * Yp7 * v18; 
   Im_9mkM9  = 0.0e0; 

} else if (m == 6 && k == 4) { 

   // 4. Z_deg[9][6][4][-9]: 18, 
   Re_9mkM9  = 0.0e0; 
   Im_9mkM9  = -0.5966180136719431691743e-10 * y4 * Yp6 * q * e9 * v18; 

} else if (m == 5 && k == 5) { 

   // 5. Z_deg[9][5][5][-9]: 18, 
   Re_9mkM9  = -0.1155345815499392425854e-9 * e9 * y5 * q * Yp5 * v18; 
   Im_9mkM9  = 0.0e0; 

} else if (m == 4 && k == 6) { 

   // 6. Z_deg[9][4][6][-9]: 18, 
   Re_9mkM9  = 0.0e0; 
   Im_9mkM9  = 0.1933263321303509639032e-9 * Yp4 * q * y6 * e9 * v18; 

} else if (m == 3 && k == 7) { 

   // 7. Z_deg[9][3][7][-9]: 18, 
   Re_9mkM9  = 0.2845686557565889139631e-9 * e9 * q * y7 * Yp3 * v18; 
   Im_9mkM9  = 0.0e0; 

} else if (m == 2 && k == 8) { 

   // 8. Z_deg[9][2][8][-9]: 18, 
   Re_9mkM9  = 0.0e0; 
   Im_9mkM9  = -0.3725878301189713266486e-9 * e9 * Yp2 * y8 * q * v18; 

} else if (m == 1 && k == 9) { 

   // 9. Z_deg[9][1][9][-9]: 18, 
   Re_9mkM9  = -0.4368979575771094465933e-9 * y9 * q * Yp * e9 * v18; 
   Im_9mkM9  = 0.0e0; 

} else if (m == 0 && k == 10) { 

   // 10. Z_deg[9][0][10][-9]: 18, 
   Re_9mkM9  = 0.0e0; 
   Im_9mkM9  = 0.4605308836730951721952e-9 * e9 * q * y10 * v18; 

} else if (m == -1 && k == 11) { 

   // 11. Z_deg[9][-1][11][-9]: 18, 
   Re_9mkM9  = 0.4368979575771094465933e-9 * q * y11 / Yp * e9 * v18; 
   Im_9mkM9  = 0.0e0; 

} else if (m == -2 && k == 12) { 

   // 12. Z_deg[9][-2][12][-9]: 18, 
   Re_9mkM9  = 0.0e0; 
   Im_9mkM9  = -0.3725878301189713266486e-9 * q * e9 / Yp2 * y12 * v18; 

} else if (m == -3 && k == 13) { 

   // 13. Z_deg[9][-3][13][-9]: 18, 
   Re_9mkM9  = -0.2845686557565889139631e-9 * e9 * y13 * q / Yp3 * v18; 
   Im_9mkM9  = 0.0e0; 

} else if (m == -4 && k == 14) { 

   // 14. Z_deg[9][-4][14][-9]: 18, 
   Re_9mkM9  = 0.0e0; 
   Im_9mkM9  = 0.1933263321303509639032e-9 * q / Yp4 * y14 * e9 * v18; 

} else if (m == -5 && k == 15) { 

   // 15. Z_deg[9][-5][15][-9]: 18, 
   Re_9mkM9  = 0.1155345815499392425854e-9 * e9 * q * y15 / Yp5 * v18; 
   Im_9mkM9  = 0.0e0; 

} else if (m == -6 && k == 16) { 

   // 16. Z_deg[9][-6][16][-9]: 18, 
   Re_9mkM9  = 0.0e0; 
   Im_9mkM9  = -0.5966180136719431691743e-10 * e9 * y16 / Yp6 * q * v18; 

} else if (m == -7 && k == 17) { 

   // 17. Z_deg[9][-7][17][-9]: 18, 
   Re_9mkM9  = -0.2583431780976571600785e-10 * e9 * q * y17 / Yp7 * v18; 
   Im_9mkM9  = 0.0e0; 

} else if (m == -8 && k == 18) { 

   // 18. Z_deg[9][-8][18][-9]: 18, 
   Re_9mkM9  = 0.0e0; 
   Im_9mkM9  = 0.8861097904993476144797e-11 / Yp8 * q * y18 * e9 * v18; 

 } 


 else {

        //perror("Parameter errors: hZ_9mkM9");
        //exit(1);

    }

    hZ_9mkM9 = cmplx(Re_9mkM9, Im_9mkM9);
    return hZ_9mkM9;

}

