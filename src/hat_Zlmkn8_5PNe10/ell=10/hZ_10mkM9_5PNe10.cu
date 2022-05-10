/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 10, n = -9)

- BHPC's maple script `outputC_Slm_Zlmkn.mw`.


25th May 2020; RF
20th June. 2020; Sis


Convention (the phase `B_inc`  is defined in `Zlmkn8.c`):
 Zlmkn8  = ( hat_Zlmkn8 * exp(-I * B_inc) )

 For ell = 10, we have only
 0 <= |k| <= 20 (jmax = 10)
 
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
`hZ_10mkP0_5PNe10` stores  only the PN amplitudes that has the index
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
#include "hZ_10mkM9_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_10mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_10mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
gsl_complex hZ_10mkM9(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    gsl_complex hZ_10mkM9 = { 0.0 };

    double  Re_10mkM9;
    double  Im_10mkM9;

    // NULL check
    if (PN_orb == NULL) {

        perror("Pointer errors: hZ_10mkM9");
        exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */
   // double K = PN_orb->PNK;
    double Ym = PN_orb->PNYm;

    //double q = PN_orb->PNq[1];
    //double q2 = PN_orb->PNq[2];
    
    double v16 = PN_orb->PNv[8];
    double v17 = PN_orb->PNv[9];
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
    double Yp10 = PN_orb->PNYp[10];

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
    double y19 = PN_orb->PNy[19];
    double y20 = PN_orb->PNy[20];

    //printf("10mkM9[%d, %d] \n", m, k);

    if (m == 10 && k == 0) { 

   // 1. Z_deg[10][10][0][-9]: 16, 
   Re_10mkM9  = (-0.6477791771879361383649e-9 * v16 - 0.2278637479329181613845e-6 * v18) * Yp10 * e9; 
   Im_10mkM9  = -0.9077569207237795060298e-13 * Yp10 * e9 * v17; 

} else if (m == 9 && k == 1) { 

   // 2. Z_deg[10][9][1][-9]: 16, 
   Re_10mkM9  = -0.4059612363568517143327e-12 * Yp9 * e9 * v17 * y; 
   Im_10mkM9  = (0.2896956549202212536051e-8 * v16 + 0.1019037659971764396471e-5 * v18) * Yp9 * e9 * y; 

} else if (m == 8 && k == 2) { 

   // 3. Z_deg[10][8][2][-9]: 16, 
   Re_10mkM9  = (-0.8929019758947401780092e-8 * v16 - 0.3140885010341341432341e-5 * v18) * e9 * Yp8 * Yp * Ym; 
   Im_10mkM9  = -0.1251256565030387527107e-11 * e9 * Yp * Ym * Yp8 * v17; 

} else if (m == 7 && k == 3) { 

   // 4. Z_deg[10][7][3][-9]: 16, 
   Re_10mkM9  = 0.3064940121632046902892e-11 * y3 * Yp7 * e9 * v17; 
   Im_10mkM9  = (-0.2187154231264998609147e-7 * v16 - 0.7693565616092552106097e-5 * v18) * y3 * Yp7 * e9; 

} else if (m == 6 && k == 4) { 

   // 5. Z_deg[10][6][4][-9]: 16, 
   Re_10mkM9  = (-0.4508933957511092758007e-7 * v16 - 0.1586069183638490203888e-4 * v18) * y4 * Yp6 * e9; 
   Im_10mkM9  = -0.6318535928841184683136e-11 * y4 * Yp6 * e9 * v17; 

} else if (m == 5 && k == 5) { 

   // 6. Z_deg[10][5][5][-9]: 16, 
   Re_10mkM9  = -0.1130294068413093035387e-10 * y5 * Yp5 * e9 * v17; 
   Im_10mkM9  = (0.8065826268041561497770e-7 * v16 + 0.2837246809306609060254e-4 * v18) * y5 * Yp5 * e9; 

} else if (m == 4 && k == 6) { 

   // 7. Z_deg[10][4][6][-9]: 16, 
   Re_10mkM9  = (0.1275319110911356015748e-6 * v16 + 0.4486081100727151822032e-4 * v18) * e9 * y6 * Yp4; 
   Im_10mkM9  = 0.1787151840981776958754e-10 * e9 * y6 * Yp4 * v17; 

} else if (m == 3 && k == 7) { 

   // 8. Z_deg[10][3][7][-9]: 16, 
   Re_10mkM9  = 0.2527414371536473873255e-10 * y7 * Yp3 * e9 * v17; 
   Im_10mkM9  = (-0.1803573583004437103203e-6 * v16 - 0.6344276734553960815552e-4 * v18) * y7 * Yp3 * e9; 

} else if (m == 2 && k == 8) { 

   // 9. Z_deg[10][2][8][-9]: 16, 
   Re_10mkM9  = (-0.2299114223485020253665e-6 * v16 - 0.8087397717280839638922e-4 * v18) * e9 * y8 * Yp2; 
   Im_10mkM9  = -0.3221833799849831242396e-10 * e9 * y8 * Yp2 * v17; 

} else if (m == 1 && k == 9) { 

   // 10. Z_deg[10][1][9][-9]: 16, 
   Re_10mkM9  = -0.372025322325506984964e-10 * y9 * e9 * Yp * v17; 
   Im_10mkM9  = (0.2654788431653547706493e-6 * v16 + 0.9338522498231315378214e-4 * v18) * y9 * e9 * Yp; 

} else if (m == 0 && k == 10) { 

   // 11. Z_deg[10][0][10][-9]: 16, 
   Re_10mkM9  = (0.2784365597138000463740e-6 * v16 + 0.9794325024981031969037e-4 * v18) * y10 * e9; 
   Im_10mkM9  = 0.3901834497983443460297e-10 * y10 * e9 * v17; 

} else if (m == -1 && k == 11) { 

   // 12. Z_deg[10][-1][11][-9]: 16, 
   Re_10mkM9  = 0.372025322325506984964e-10 * y11 / Yp * e9 * v17; 
   Im_10mkM9  = (-0.2654788431653547706493e-6 * v16 - 0.9338522498231315378214e-4 * v18) * y11 * e9 / Yp; 

} else if (m == -2 && k == 12) { 

   // 13. Z_deg[10][-2][12][-9]: 16, 
   Re_10mkM9  = (-0.2299114223485020253665e-6 * v16 - 0.8087397717280839638922e-4 * v18) * e9 * y12 / Yp2; 
   Im_10mkM9  = -0.3221833799849831242396e-10 * y12 * e9 * v17 / Yp2; 

} else if (m == -3 && k == 13) { 

   // 14. Z_deg[10][-3][13][-9]: 16, 
   Re_10mkM9  = -0.2527414371536473873255e-10 * y13 * e9 * v17 / Yp3; 
   Im_10mkM9  = (0.1803573583004437103203e-6 * v16 + 0.6344276734553960815552e-4 * v18) * y13 * e9 / Yp3; 

} else if (m == -4 && k == 14) { 

   // 15. Z_deg[10][-4][14][-9]: 16, 
   Re_10mkM9  = (0.1275319110911356015748e-6 * v16 + 0.4486081100727151822032e-4 * v18) * e9 * y14 / Yp4; 
   Im_10mkM9  = 0.1787151840981776958754e-10 * e9 * y14 * v17 / Yp4; 

} else if (m == -5 && k == 15) { 

   // 16. Z_deg[10][-5][15][-9]: 16, 
   Re_10mkM9  = 0.1130294068413093035387e-10 * y15 * e9 * v17 / Yp5; 
   Im_10mkM9  = (-0.8065826268041561497770e-7 * v16 - 0.2837246809306609060254e-4 * v18) * y15 * e9 / Yp5; 

} else if (m == -6 && k == 16) { 

   // 17. Z_deg[10][-6][16][-9]: 16, 
   Re_10mkM9  = (-0.4508933957511092758007e-7 * v16 - 0.1586069183638490203888e-4 * v18) * y16 * e9 / Yp6; 
   Im_10mkM9  = -0.6318535928841184683136e-11 * y16 * e9 * v17 / Yp6; 

} else if (m == -7 && k == 17) { 

   // 18. Z_deg[10][-7][17][-9]: 16, 
   Re_10mkM9  = -0.3064940121632046902892e-11 * y17 * e9 * v17 / Yp7; 
   Im_10mkM9  = (0.2187154231264998609147e-7 * v16 + 0.7693565616092552106097e-5 * v18) * y17 * e9 / Yp7; 

} else if (m == -8 && k == 18) { 

   // 19. Z_deg[10][-8][18][-9]: 16, 
   Re_10mkM9  = (0.8929019758947401780092e-8 * v16 + 0.3140885010341341432341e-5 * v18) * e9 * y18 / Yp8; 
   Im_10mkM9  = 0.1251256565030387527107e-11 * e9 * y18 * v17 / Yp8; 

} else if (m == -9 && k == 19) { 

   // 20. Z_deg[10][-9][19][-9]: 16, 
   Re_10mkM9  = 0.4059612363568517143327e-12 * y19 * e9 * v17 / Yp9; 
   Im_10mkM9  = (-0.2896956549202212536051e-8 * v16 - 0.1019037659971764396471e-5 * v18) * y19 * e9 / Yp9; 

} else if (m == -10 && k == 20) { 

   // 21. Z_deg[10][-10][20][-9]: 16, 
   Re_10mkM9  = (-0.6477791771879361383649e-9 * v16 - 0.2278637479329181613845e-6 * v18) * y20 * e9 / Yp10; 
   Im_10mkM9  = -0.9077569207237795060298e-13 * y20 * e9 * v17 / Yp10; 

 } 

 else {

        perror("Parameter errors: hZ_10mkM9");
        exit(1);

    }

    GSL_SET_COMPLEX(&hZ_10mkM9, Re_10mkM9, Im_10mkM9);
    return hZ_10mkM9;

}

