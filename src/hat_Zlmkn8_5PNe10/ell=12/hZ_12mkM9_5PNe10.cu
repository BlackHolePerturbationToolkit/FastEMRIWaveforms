/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 12, n = -9)

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
#include "hat_Zlmkn8_5PNe10/ell=12/hZ_12mkM9_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_12mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_12mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
CUDA_CALLABLE_MEMBER
cmplx hZ_12mkM9(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    cmplx hZ_12mkM9 = { 0.0 };

    double  Re_12mkM9 = 0.0;
    double  Im_12mkM9 = 0.0;

    // NULL check
    if (PN_orb == NULL) {

        //perror("Point errors: hZ_12mkM9");
        //exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */

   // double K = PN_orb->PNK;
    double Ym = PN_orb->PNYm;

    //double q = PN_orb->PNq[1];
    
    double v18 = PN_orb->PNv[10];
    
    double e9 = PN_orb->PNe[9];

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

    if (m == 12 && k == 0) {

        // 1. Z_deg[12][12][0][-9]: 18, 
        Re_12mkM9 = 0.2038048637966635060432e-4 * Yp12 * e9 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == 12 && k == -2) {

        // 2. Z_deg[12][12][-2][-9]: 18, 
        Re_12mkM9 = -0.8758944199151798832758e-11 * Yp * Ym * Yp10 * e9 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == 11 && k == 1) {

        // 3. Z_deg[12][11][1][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = -0.9984358467984998750205e-4 * e9 * y * Yp11 * v18;

    }
    else if (m == 11 && k == -1) {

        // 4. Z_deg[12][11][-1][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = 0.429098879468650992036e-10 * e9 * y * (-0.8333333333333333333333e0 + Y) * Yp10 * v18;

    }
    else if (m == 10 && k == 2) {

        // 5. Z_deg[12][10][2][-9]: 18, 
        Re_12mkM9 = 0.3385860689984266231136e-3 * Yp * Ym * Yp10 * e9 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == 10 && k == 0) {

        // 6. Z_deg[12][10][0][-9]: 18, 
        Re_12mkM9 = 0.2108905939091077644894e-11 * v18 * e9 * (-0.69e2 * Y2 + 0.115e3 * Y - 0.47e2) * Yp10;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == 9 && k == 3) {

        // 7. Z_deg[12][9][3][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = 0.9168954092726463400794e-3 * y3 * Yp9 * e9 * v18;

    }
    else if (m == 9 && k == 1) {

        // 8. Z_deg[12][9][1][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = -0.8566416420470390111034e-11 * y * e9 * v18 * (-0.46e2 * Y2 + 0.69e2 * Y - 0.25e2) * Yp9;

    }
    else if (m == 8 && k == 4) {

        // 9. Z_deg[12][8][4][-9]: 18, 
        Re_12mkM9 = 0.2100871308674708325573e-2 * e9 * y4 * Yp8 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == 8 && k == 2) {

        // 10. Z_deg[12][8][2][-9]: 18, 
        Re_12mkM9 = 0.1308541722710607243244e-10 * v18 * Yp * Ym * e9 * (-0.69e2 * Y2 + 0.92e2 * Y - 0.29e2) * Yp8;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == 7 && k == 5) {

        // 11. Z_deg[12][7][5][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = -0.4201742617349416651148e-2 * e9 * y5 * Yp7 * v18;

    }
    else if (m == 7 && k == 3) {

        // 12. Z_deg[12][7][3][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = 0.1308541722710607243244e-10 * y3 * e9 * v18 * (-0.138e3 * Y2 + 0.161e3 * Y - 0.43e2) * Yp7;

    }
    else if (m == 6 && k == 6) {

        // 13. Z_deg[12][6][6][-9]: 18, 
        Re_12mkM9 = -0.747705578672242934329e-2 * y6 * Yp6 * e9 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == 6 && k == 4) {

        // 14. Z_deg[12][6][4][-9]: 18, 
        Re_12mkM9 = 0.3213422544968672555825e-8 * y4 * e9 * v18 * (Y - 0.1e1 * Y2 - 0.2173913043478260869565e0) * Yp6;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == 5 && k == 7) {

        // 15. Z_deg[12][5][7][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = 0.1198996329244749488564e-1 * Yp5 * y7 * e9 * v18;

    }
    else if (m == 5 && k == 5) {

        // 16. Z_deg[12][5][5][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = -0.3734014348511786181377e-10 * y5 * e9 * v18 * (-0.138e3 * Y2 + 0.115e3 * Y - 0.19e2) * Yp5;

    }
    else if (m == 4 && k == 8) {

        // 17. Z_deg[12][4][8][-9]: 18, 
        Re_12mkM9 = 0.1747822479480557945397e-1 * v18 * y8 * Yp4 * e9;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == 4 && k == 6) {

        // 18. Z_deg[12][4][6][-9]: 18, 
        Re_12mkM9 = -0.1088642902041716990134e-9 * v18 * y6 * e9 * (-0.69e2 * Y2 + 0.46e2 * Y - 0.5e1) * Yp4;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == 3 && k == 9) {

        // 19. Z_deg[12][3][9][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = -0.2330429972640743927195e-1 * y9 * Yp3 * e9 * v18;

    }
    else if (m == 3 && k == 7) {

        // 20. Z_deg[12][3][7][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = 0.5007757349391898154615e-8 * y7 * e9 * v18 * (Y - 0.2e1 * Y2 - 0.434782608695652173913e-1) * Yp3;

    }
    else if (m == 2 && k == 10) {

        // 21. Z_deg[12][2][10][-9]: 18, 
        Re_12mkM9 = -0.2854182157128992307434e-1 * e9 * y10 * Yp2 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == 2 && k == 8) {

        // 22. Z_deg[12][2][8][-9]: 18, 
        Re_12mkM9 = 0.408881675389417678159e-8 * y8 * e9 * v18 * (Y - 0.3e1 * Y2 + 0.434782608695652173913e-1) * Yp2;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == 1 && k == 11) {

        // 23. Z_deg[12][1][11][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = 0.3219951736016453629084e-1 * y11 * Yp * e9 * v18;

    }
    else if (m == 1 && k == 9) {

        // 24. Z_deg[12][1][9][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = -0.2306403705185747626201e-8 * y9 * e9 * v18 * (Y - 0.6e1 * Y2 + 0.2173913043478260869565e0) * Yp;

    }
    else if (m == 0 && k == 12) {

        // 25. Z_deg[12][0][12][-9]: 18, 
        Re_12mkM9 = 0.3351432024393700067207e-1 * y12 * e9 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == 0 && k == 10) {

        // 26. Z_deg[12][0][10][-9]: 18, 
        Re_12mkM9 = 0.1440348652238364324187e-7 * e9 * y10 * v18 * (Y2 - 0.4347826086956521739130e-1);
        Im_12mkM9 = 0.0e0;

    }
    else if (m == -1 && k == 13) {

        // 27. Z_deg[12][-1][13][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = -0.3219951736016453629084e-1 / Yp * y13 * e9 * v18;

    }
    else if (m == -1 && k == 11) {

        // 28. Z_deg[12][-1][11][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = -0.2306403705185747626201e-8 * v18 * e9 * (Y + 0.6e1 * Y2 - 0.2173913043478260869565e0) * y11 / Yp;

    }
    else if (m == -2 && k == 14) {

        // 29. Z_deg[12][-2][14][-9]: 18, 
        Re_12mkM9 = -0.2854182157128992307434e-1 * e9 / Yp2 * y14 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == -2 && k == 12) {

        // 30. Z_deg[12][-2][12][-9]: 18, 
        Re_12mkM9 = -0.408881675389417678159e-8 * (Y + 0.3e1 * Y2 - 0.434782608695652173913e-1) * v18 * e9 * y12 / Yp2;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == -3 && k == 15) {

        // 31. Z_deg[12][-3][15][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = 0.2330429972640743927195e-1 * y15 / Yp3 * e9 * v18;

    }
    else if (m == -3 && k == 13) {

        // 32. Z_deg[12][-3][13][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = 0.5007757349391898154615e-8 * v18 * e9 * y13 * (Y + 0.2e1 * Y2 + 0.434782608695652173913e-1) / Yp3;

    }
    else if (m == -4 && k == 16) {

        // 33. Z_deg[12][-4][16][-9]: 18, 
        Re_12mkM9 = 0.1747822479480557945397e-1 / Yp4 * y16 * e9 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == -4 && k == 14) {

        // 34. Z_deg[12][-4][14][-9]: 18, 
        Re_12mkM9 = 0.1088642902041716990134e-9 * y14 * e9 * v18 * (0.69e2 * Y2 + 0.46e2 * Y + 0.5e1) / Yp4;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == -5 && k == 17) {

        // 35. Z_deg[12][-5][17][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = -0.1198996329244749488564e-1 * y17 / Yp5 * e9 * v18;

    }
    else if (m == -5 && k == 15) {

        // 36. Z_deg[12][-5][15][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = -0.3734014348511786181377e-10 * y15 * e9 * v18 * (0.138e3 * Y2 + 0.115e3 * Y + 0.19e2) / Yp5;

    }
    else if (m == -6 && k == 18) {

        // 37. Z_deg[12][-6][18][-9]: 18, 
        Re_12mkM9 = -0.747705578672242934329e-2 / Yp6 * y18 * e9 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == -6 && k == 16) {

        // 38. Z_deg[12][-6][16][-9]: 18, 
        Re_12mkM9 = -0.3213422544968672555825e-8 * v18 * e9 * y16 * (Y + Y2 + 0.2173913043478260869565e0) / Yp6;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == -7 && k == 19) {

        // 39. Z_deg[12][-7][19][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = 0.4201742617349416651148e-2 * y19 / Yp7 * e9 * v18;

    }
    else if (m == -7 && k == 17) {

        // 40. Z_deg[12][-7][17][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = 0.1308541722710607243244e-10 * y17 * e9 * v18 * (0.138e3 * Y2 + 0.161e3 * Y + 0.43e2) / Yp7;

    }
    else if (m == -8 && k == 20) {

        // 41. Z_deg[12][-8][20][-9]: 18, 
        Re_12mkM9 = 0.2100871308674708325573e-2 * e9 / Yp8 * y20 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == -8 && k == 18) {

        // 42. Z_deg[12][-8][18][-9]: 18, 
        Re_12mkM9 = 0.1308541722710607243244e-10 * y18 * e9 * v18 * (0.69e2 * Y2 + 0.92e2 * Y + 0.29e2) / Yp8;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == -9 && k == 21) {

        // 43. Z_deg[12][-9][21][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = -0.9168954092726463400794e-3 / Yp9 * y21 * e9 * v18;

    }
    else if (m == -9 && k == 19) {

        // 44. Z_deg[12][-9][19][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = -0.8566416420470390111034e-11 * y19 * e9 * v18 * (0.46e2 * Y2 + 0.69e2 * Y + 0.25e2) / Yp9;

    }
    else if (m == -10 && k == 22) {

        // 45. Z_deg[12][-10][22][-9]: 18, 
        Re_12mkM9 = -0.3385860689984266231136e-3 / Yp10 * y22 * e9 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == -10 && k == 20) {

        // 46. Z_deg[12][-10][20][-9]: 18, 
        Re_12mkM9 = -0.2108905939091077644894e-11 * y20 * e9 * v18 * (0.69e2 * Y2 + 0.115e3 * Y + 0.47e2) / Yp10;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == -11 && k == 23) {

        // 47. Z_deg[12][-11][23][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = 0.9984358467984998750205e-4 * y23 / Yp11 * e9 * v18;

    }
    else if (m == -11 && k == 21) {

        // 48. Z_deg[12][-11][21][-9]: 18, 
        Re_12mkM9 = 0.0e0;
        Im_12mkM9 = 0.429098879468650992036e-10 * y21 / Yp10 * (0.8333333333333333333333e0 + Y) * e9 * v18;

    }
    else if (m == -12 && k == 24) {

        // 49. Z_deg[12][-12][24][-9]: 18, 
        Re_12mkM9 = 0.2038048637966635060432e-4 * y24 / Yp12 * e9 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else if (m == -12 && k == 22) {

        // 50. Z_deg[12][-12][22][-9]: 18, 
        Re_12mkM9 = 0.8758944199151798832758e-11 / Yp10 * y22 * e9 * v18;
        Im_12mkM9 = 0.0e0;

    }
    else {

        //perror("Parameter errors: hZ_12mkM9");
        //exit(1);

    }

    hZ_12mkM9 = cmplx(Re_12mkM9, Im_12mkM9);
    return hZ_12mkM9;

}

