/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 12, n = -10)

- BHPC's maple script `outputC_Slm_Zlmkn.mw`.


25th May 2020; RF
8th Nov 2020; Sis


Convention (the phase `B_inc`  is defined in `Zlmkn8.c`):
 Zlmkn8  = ( hat_Zlmkn8 * exp(-I * B_inc) )

 For ell = 12, we have only
 0 <= |k| <= 24 (jmax = 12)
 
 //Shorthad variables for PN amplitudes  in `inspiral_orb_data`
 k = sqrt(1. - q^2); y = sqrt(1. - Y^2) ;
 Ym = Y - 1.0 , Yp = Y + 1.0 ;

PNq[11] = 1, q, q ^2, ..., q ^ 9, q ^ 10
PNe[11] = 1, e, e ^ 2, ..., e ^ 9, e ^ 10
PNv[11] = v ^ 8, v ^ 9, ..., v ^ 17, v ^ 18

PNY[13] = 1, Y, Y ^ 2, ..., Y ^ 11, Y ^ 12
PNYp[13] = 1, Yp, Yp^2,...  Yp^12
PNy[25] = 1, y, y ^ 2, ..., y ^ 23, y ^ 24



 WARGNING !! 
`hZ_12mkP0_5PNe10` stores  only the PN amplitudes that has the index
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
#include "hat_Zlmkn8_5PNe10/ell=12/hZ_12mkM10_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_12mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_12mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
CUDA_CALLABLE_MEMBER
cmplx hZ_12mkM10(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    cmplx hZ_12mkM10 = { 0.0 };

    double  Re_12mkM10 = 0.0;
    double  Im_12mkM10 = 0.0;

    // NULL check
    if (PN_orb == NULL) {

        //perror("Point errors: hZ_12mkM10");
        //exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */

   // double K = PN_orb->PNK;
    double Ym = PN_orb->PNYm;

    //double q = PN_orb->PNq[1];
    
    double v18 = PN_orb->PNv[10];
    
    double e10 = PN_orb->PNe[10];

    double Y = PN_orb->PNY[1];
    double Y2 = PN_orb->PNY[2];
    double Y3 = PN_orb->PNY[3];
    double Y4 = PN_orb->PNY[4];
    double Y5 = PN_orb->PNY[5];
    double Y6 = PN_orb->PNY[6];
    double Y7 = PN_orb->PNY[7];
    double Y8 = PN_orb->PNY[8];
    double Y9 = PN_orb->PNY[9];
    double Y10 = PN_orb->PNY[10];
    double Y11 = PN_orb->PNY[11];
    double Y12 = PN_orb->PNY[12];

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
    double Yp11 = PN_orb->PNYp[11];
    double Yp12 = PN_orb->PNYp[12];

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
    double y21 = PN_orb->PNy[21];
    double y22 = PN_orb->PNy[22];
    double y23 = PN_orb->PNy[23];
    double y24 = PN_orb->PNy[24];

   // printf("12mkM10[%d, %d] \n", m, k);


    if (m == 12 && k == 0) {

        // 1. Z_deg[12][12][0][-10]: 18, 
        Re_12mkM10 = -0.4248432897864526439605e-7 * Yp12 * e10 * v18;
        Im_12mkM10 = 0.0e0;

    }
    else if (m == 11 && k == 1) {

        // 2. Z_deg[12][11][1][-10]: 18, 
        Re_12mkM10 = 0.0e0;
        Im_12mkM10 = 0.2081298561244354163354e-6 * e10 * y * Yp11 * v18;

    }
    else if (m == 10 && k == 2) {

        // 3. Z_deg[12][10][2][-10]: 18, 
        Re_12mkM10 = -0.7058026817881532714592e-6 * Yp * Ym * Yp10 * e10 * v18;
        Im_12mkM10 = 0.0e0;

    }
    else if (m == 9 && k == 3) {

        // 4. Z_deg[12][9][3][-10]: 18, 
        Re_12mkM10 = 0.0e0;
        Im_12mkM10 = -0.1911322697647337032499e-5 * y3 * Yp9 * e10 * v18;

    }
    else if (m == 8 && k == 4) {

        // 5. Z_deg[12][8][4][-10]: 18, 
        Re_12mkM10 = -0.4379390469728058184435e-5 * e10 * y4 * Yp8 * v18;
        Im_12mkM10 = 0.0e0;

    }
    else if (m == 7 && k == 5) {

        // 6. Z_deg[12][7][5][-10]: 18, 
        Re_12mkM10 = 0.0e0;
        Im_12mkM10 = 0.8758780939456116368873e-5 * e10 * y5 * Yp7 * v18;

    }
    else if (m == 6 && k == 6) {

        // 7. Z_deg[12][6][6][-10]: 18, 
        Re_12mkM10 = 0.1558636491382887947115e-4 * y6 * Yp6 * e10 * v18;
        Im_12mkM10 = 0.0e0;

    }
    else if (m == 5 && k == 7) {

        // 8. Z_deg[12][5][7][-10]: 18, 
        Re_12mkM10 = 0.0e0;
        Im_12mkM10 = -0.2499378746262086283039e-4 * Yp5 * y7 * e10 * v18;

    }
    else if (m == 4 && k == 8) {

        // 9. Z_deg[12][4][8][-10]: 18, 
        Re_12mkM10 = -0.3643439309113245865898e-4 * y8 * Yp4 * e10 * v18;
        Im_12mkM10 = 0.0e0;

    }
    else if (m == 3 && k == 9) {

        // 10. Z_deg[12][3][9][-10]: 18, 
        Re_12mkM10 = 0.0e0;
        Im_12mkM10 = 0.4857919078817661154532e-4 * y9 * Yp3 * e10 * v18;

    }
    else if (m == 2 && k == 10) {

        // 11. Z_deg[12][2][10][-10]: 18, 
        Re_12mkM10 = 0.59497114774172831559e-4 * e10 * y10 * Yp2 * v18;
        Im_12mkM10 = 0.0e0;

    }
    else if (m == 1 && k == 11) {

        // 12. Z_deg[12][1][11][-10]: 18, 
        Re_12mkM10 = 0.0e0;
        Im_12mkM10 = -0.6712179792959507482705e-4 * y11 * Yp * e10 * v18;

    }
    else if (m == 0 && k == 12) {

        // 13. Z_deg[12][0][12][-10]: 18, 
        Re_12mkM10 = -0.6986258228653716518568e-4 * y12 * e10 * v18;
        Im_12mkM10 = 0.0e0;

    }
    else if (m == -1 && k == 13) {

        // 14. Z_deg[12][-1][13][-10]: 18, 
        Re_12mkM10 = 0.0e0;
        Im_12mkM10 = 0.6712179792959507482705e-4 / Yp * y13 * e10 * v18;

    }
    else if (m == -2 && k == 14) {

        // 15. Z_deg[12][-2][14][-10]: 18, 
        Re_12mkM10 = 0.59497114774172831559e-4 * e10 / Yp2 * y14 * v18;
        Im_12mkM10 = 0.0e0;

    }
    else if (m == -3 && k == 15) {

        // 16. Z_deg[12][-3][15][-10]: 18, 
        Re_12mkM10 = 0.0e0;
        Im_12mkM10 = -0.4857919078817661154532e-4 * y15 / Yp3 * e10 * v18;

    }
    else if (m == -4 && k == 16) {

        // 17. Z_deg[12][-4][16][-10]: 18, 
        Re_12mkM10 = -0.3643439309113245865898e-4 / Yp4 * y16 * e10 * v18;
        Im_12mkM10 = 0.0e0;

    }
    else if (m == -5 && k == 17) {

        // 18. Z_deg[12][-5][17][-10]: 18, 
        Re_12mkM10 = 0.0e0;
        Im_12mkM10 = 0.2499378746262086283039e-4 * y17 / Yp5 * e10 * v18;

    }
    else if (m == -6 && k == 18) {

        // 19. Z_deg[12][-6][18][-10]: 18, 
        Re_12mkM10 = 0.1558636491382887947115e-4 / Yp6 * y18 * e10 * v18;
        Im_12mkM10 = 0.0e0;

    }
    else if (m == -7 && k == 19) {

        // 20. Z_deg[12][-7][19][-10]: 18, 
        Re_12mkM10 = 0.0e0;
        Im_12mkM10 = -0.8758780939456116368873e-5 * y19 / Yp7 * e10 * v18;

    }
    else if (m == -8 && k == 20) {

        // 21. Z_deg[12][-8][20][-10]: 18, 
        Re_12mkM10 = -0.4379390469728058184435e-5 * e10 / Yp8 * y20 * v18;
        Im_12mkM10 = 0.0e0;

    }
    else if (m == -9 && k == 21) {

        // 22. Z_deg[12][-9][21][-10]: 18, 
        Re_12mkM10 = 0.0e0;
        Im_12mkM10 = 0.1911322697647337032499e-5 / Yp9 * y21 * e10 * v18;

    }
    else if (m == -10 && k == 22) {

        // 23. Z_deg[12][-10][22][-10]: 18, 
        Re_12mkM10 = 0.7058026817881532714592e-6 / Yp10 * y22 * e10 * v18;
        Im_12mkM10 = 0.0e0;

    }
    else if (m == -11 && k == 23) {

        // 24. Z_deg[12][-11][23][-10]: 18, 
        Re_12mkM10 = 0.0e0;
        Im_12mkM10 = -0.2081298561244354163354e-6 * y23 / Yp11 * e10 * v18;

    }
    else if (m == -12 && k == 24) {

        // 25. Z_deg[12][-12][24][-10]: 18, 
        Re_12mkM10 = -0.4248432897864526439605e-7 * y24 / Yp12 * e10 * v18;
        Im_12mkM10 = 0.0e0;

    }
    else {

        //perror("Parameter errors: hZ_12mkM10");
        //exit(1);

    }

    hZ_12mkM10 = cmplx(Re_12mkM10, Im_12mkM10);
    return hZ_12mkM10;

}

