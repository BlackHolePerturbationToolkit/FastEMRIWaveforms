/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 11, n = -10)

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
`hZ_11mkP0_5PNe10` stores  only the PN amplitudes that has the index
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
#include "hZ_11mkM10_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_11mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_11mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
gsl_complex hZ_11mkM10(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    gsl_complex hZ_11mkM10 = { 0.0 };

    double  Re_11mkM10;
    double  Im_11mkM10;

    // NULL check
    if (PN_orb == NULL) {

        perror("Pointer errors: hZ_11mkM10");
        exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */

   // double K = PN_orb->PNK;
    double Ym = PN_orb->PNYm;

    //double q = PN_orb->PNq[1];
    
    double v17 = PN_orb->PNv[9];
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

    //printf("11mkM10[%d, %d] \n", m, k);

    if (m == 11 && k == 0) {

        // 1. Z_deg[11][11][0][-10]: 17, 
        Re_11mkM10 = -0.362271246833543850919e-10 * Yp11 * e10 * v18;
        Im_11mkM10 = 0.2818280492805642463193e-10 * Yp11 * e10 * v17;

    }
    else if (m == 10 && k == 1) {

        // 2. Z_deg[11][10][1][-10]: 17, 
        Re_11mkM10 = 0.1321890723905852698432e-9 * Yp10 * e10 * v17 * y;
        Im_11mkM10 = 0.169920276547893777958e-9 * Yp10 * e10 * v18 * y;

    }
    else if (m == 9 && k == 2) {

        // 3. Z_deg[11][9][2][-10]: 17, 
        Re_11mkM10 = -0.5506046258543269381157e-9 * e10 * Yp * Ym * Yp9 * v18;
        Im_11mkM10 = 0.4283415506632243879539e-9 * e10 * Yp * Ym * Yp9 * v17;

    }
    else if (m == 8 && k == 3) {

        // 4. Z_deg[11][8][3][-10]: 17, 
        Re_11mkM10 = -0.1105973128138219052959e-8 * e10 * y3 * Yp8 * v17;
        Im_11mkM10 = -0.1421655030852382633215e-8 * y3 * Yp8 * e10 * v18;

    }
    else if (m == 7 && k == 4) {

        // 5. Z_deg[11][7][4][-10]: 17, 
        Re_11mkM10 = -0.3098425306030867162268e-8 * y4 * Yp7 * e10 * v18;
        Im_11mkM10 = 0.2410412549913028503856e-8 * y4 * Yp7 * e10 * v17;

    }
    else if (m == 6 && k == 5) {

        // 6. Z_deg[11][6][5][-10]: 17, 
        Re_11mkM10 = 0.4573436255027681181409e-8 * y5 * Yp6 * e10 * v17;
        Im_11mkM10 = 0.5878848676177071168566e-8 * y5 * Yp6 * e10 * v18;

    }
    else if (m == 5 && k == 6) {

        // 7. Z_deg[11][5][6][-10]: 17, 
        Re_11mkM10 = 0.9895576872822282454356e-8 * y6 * Yp5 * e10 * v18;
        Im_11mkM10 = -0.7698240340489372309568e-8 * y6 * Yp5 * e10 * v17;

    }
    else if (m == 4 && k == 7) {

        // 8. Z_deg[11][4][7][-10]: 17, 
        Re_11mkM10 = -0.1163864541356575831944e-7 * e10 * y7 * Yp4 * v17;
        Im_11mkM10 = -0.1496070599143428298697e-7 * y7 * Yp4 * e10 * v18;

    }
    else if (m == 3 && k == 8) {

        // 9. Z_deg[11][3][8][-10]: 17, 
        Re_11mkM10 = -0.2048579036927811818836e-7 * y8 * Yp3 * e10 * v18;
        Im_11mkM10 = 0.159368715795350222596e-7 * y8 * Yp3 * e10 * v17;

    }
    else if (m == 2 && k == 9) {

        // 10. Z_deg[11][2][9][-10]: 17, 
        Re_11mkM10 = 0.1987677108921163565182e-7 * e10 * y9 * Yp2 * v17;
        Im_11mkM10 = 0.2555026961970397979695e-7 * y9 * Yp2 * e10 * v18;

    }
    else if (m == 1 && k == 10) {

        // 11. Z_deg[11][1][10][-10]: 17, 
        Re_11mkM10 = 0.969783173356521015893e-11 * y10 * Yp * e10 * v18;
        Im_11mkM10 = -0.226630059262401325118e-7 * y10 * Yp * e10 * v17;

    }
    else if (m == 0 && k == 11) {

        // 12. Z_deg[11][0][11][-10]: 17, 
        Re_11mkM10 = -0.236707376912998260291e-7 * y11 * e10 * v17;
        Im_11mkM10 = -0.1012905489619088038095e-10 * y11 * e10 * v18;

    }
    else if (m == -1 && k == 12) {

        // 13. Z_deg[11][-1][12][-10]: 17, 
        Re_11mkM10 = -0.969783173356521015893e-11 / Yp * y12 * e10 * v18;
        Im_11mkM10 = 0.226630059262401325118e-7 / Yp * y12 * e10 * v17;

    }
    else if (m == -2 && k == 13) {

        // 14. Z_deg[11][-2][13][-10]: 17, 
        Re_11mkM10 = 0.1987677108921163565182e-7 * e10 * y13 * v17 / Yp2;
        Im_11mkM10 = 0.2555026961970397979695e-7 * e10 * y13 * v18 / Yp2;

    }
    else if (m == -3 && k == 14) {

        // 15. Z_deg[11][-3][14][-10]: 17, 
        Re_11mkM10 = 0.2048579036927811818836e-7 * y14 * e10 * v18 / Yp3;
        Im_11mkM10 = -0.159368715795350222596e-7 * y14 * e10 * v17 / Yp3;

    }
    else if (m == -4 && k == 15) {

        // 16. Z_deg[11][-4][15][-10]: 17, 
        Re_11mkM10 = -0.1163864541356575831944e-7 * e10 * y15 * v17 / Yp4;
        Im_11mkM10 = -0.1496070599143428298697e-7 * y15 / Yp4 * e10 * v18;

    }
    else if (m == -5 && k == 16) {

        // 17. Z_deg[11][-5][16][-10]: 17, 
        Re_11mkM10 = -0.9895576872822282454356e-8 * y16 * e10 * v18 / Yp5;
        Im_11mkM10 = 0.7698240340489372309568e-8 * y16 * e10 * v17 / Yp5;

    }
    else if (m == -6 && k == 17) {

        // 18. Z_deg[11][-6][17][-10]: 17, 
        Re_11mkM10 = 0.4573436255027681181409e-8 * y17 * e10 * v17 / Yp6;
        Im_11mkM10 = 0.5878848676177071168566e-8 * y17 * e10 * v18 / Yp6;

    }
    else if (m == -7 && k == 18) {

        // 19. Z_deg[11][-7][18][-10]: 17, 
        Re_11mkM10 = 0.3098425306030867162268e-8 * y18 * e10 * v18 / Yp7;
        Im_11mkM10 = -0.2410412549913028503856e-8 * y18 * e10 * v17 / Yp7;

    }
    else if (m == -8 && k == 19) {

        // 20. Z_deg[11][-8][19][-10]: 17, 
        Re_11mkM10 = -0.1105973128138219052959e-8 * e10 * y19 * v17 / Yp8;
        Im_11mkM10 = -0.1421655030852382633215e-8 * e10 * y19 * v18 / Yp8;

    }
    else if (m == -9 && k == 20) {

        // 21. Z_deg[11][-9][20][-10]: 17, 
        Re_11mkM10 = -0.5506046258543269381157e-9 * e10 * y20 * v18 / Yp9;
        Im_11mkM10 = 0.4283415506632243879539e-9 * e10 * y20 * v17 / Yp9;

    }
    else if (m == -10 && k == 21) {

        // 22. Z_deg[11][-10][21][-10]: 17, 
        Re_11mkM10 = 0.1321890723905852698432e-9 * y21 * e10 * v17 / Yp10;
        Im_11mkM10 = 0.169920276547893777958e-9 * y21 * e10 * v18 / Yp10;

    }
    else if (m == -11 && k == 22) {

        // 23. Z_deg[11][-11][22][-10]: 17, 
        Re_11mkM10 = 0.362271246833543850919e-10 * y22 * e10 * v18 / Yp11;
        Im_11mkM10 = -0.2818280492805642463193e-10 * y22 * e10 * v17 / Yp11;

    }
    else {

        perror("Parameter errors: hZ_11mkM10");
        exit(1);

    }

    GSL_SET_COMPLEX(&hZ_11mkM10, Re_11mkM10, Im_11mkM10);
    return hZ_11mkM10;

}

