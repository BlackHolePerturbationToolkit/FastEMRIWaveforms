/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 11, n = -9)

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
#include "hZ_11mkM9_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_11mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_11mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
gsl_complex hZ_11mkM9(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    gsl_complex hZ_11mkM9 = { 0.0 };

    double  Re_11mkM9;
    double  Im_11mkM9;

    // NULL check
    if (PN_orb == NULL) {

        perror("Pointer errors: hZ_11mkM9");
        exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */

   // double K = PN_orb->PNK;
    double Ym = PN_orb->PNYm;

    //double q = PN_orb->PNq[1];
    
    double v17 = PN_orb->PNv[9];
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

    if (m == 11 && k == 0) {

        // 1. Z_deg[11][11][0][-9]: 17, 
        Re_11mkM9 = 0.7098261457420962889457e-6 * Yp11 * e9 * v18;
        Im_11mkM9 = -0.5020560715531403881159e-6 * Yp11 * e9 * v17;

    }
    else if (m == 11 && k == -1) {

        // 2. Z_deg[11][11][-1][-9]: 18, 
        Re_11mkM9 = 0.0e0;
        Im_11mkM9 = -0.9439102803316565652998e-11 * Yp10 * e9 * y * v18;

    }
    else if (m == 10 && k == 1) {

        // 3. Z_deg[11][10][1][-9]: 17, 
        Re_11mkM9 = -0.2354851710327889089725e-5 * Yp10 * e9 * v17 * y;
        Im_11mkM9 = -0.3329379740723451010374e-5 * Yp10 * e9 * y * v18;

    }
    else if (m == 10 && k == 0) {

        // 4. Z_deg[11][10][0][-9]: 18, 
        Re_11mkM9 = 0.4427331654726953322143e-10 * e9 * (-0.9090909090909090909091e0 + Y) * v18 * Yp10;
        Im_11mkM9 = 0.0e0;

    }
    else if (m == 9 && k == 2) {

        // 5. Z_deg[11][9][2][-9]: 17, 
        Re_11mkM9 = 0.1078842339308053925798e-4 * e9 * Yp * Ym * Yp9 * v18;
        Im_11mkM9 = -0.7630591658918654055336e-5 * e9 * Yp * Ym * Yp9 * v17;

    }
    else if (m == 9 && k == 1) {

        // 6. Z_deg[11][9][1][-9]: 18, 
        Re_11mkM9 = 0.0e0;
        Im_11mkM9 = -0.143461942200691914918e-9 * (Y - 0.8181818181818181818179e0) * e9 * Yp9 * v18 * y;

    }
    else if (m == 8 && k == 3) {

        // 7. Z_deg[11][8][3][-9]: 17, 
        Re_11mkM9 = 0.1970210294446744871935e-4 * e9 * y3 * Yp8 * v17;
        Im_11mkM9 = 0.2785558942215696090604e-4 * e9 * y3 * Yp8 * v18;

    }
    else if (m == 8 && k == 2) {

        // 8. Z_deg[11][8][2][-9]: 18, 
        Re_11mkM9 = 0.3704171419719005368983e-9 * (Y - 0.7272727272727272727272e0) * Yp8 * e9 * v18 * Yp * Ym;
        Im_11mkM9 = 0.0e0;

    }
    else if (m == 7 && k == 4) {

        // 9. Z_deg[11][7][4][-9]: 17, 
        Re_11mkM9 = 0.6070984965197136907797e-4 * Yp7 * y4 * e9 * v18;
        Im_11mkM9 = -0.429397378550843779531e-4 * Yp7 * y4 * e9 * v17;

    }
    else if (m == 7 && k == 3) {

        // 10. Z_deg[11][7][3][-9]: 18, 
        Re_11mkM9 = 0.0e0;
        Im_11mkM9 = 0.8073054444053364689731e-9 * (Y - 0.6363636363636363636363e0) * Yp7 * y3 * e9 * v18;

    }
    else if (m == 6 && k == 5) {

        // 11. Z_deg[11][6][5][-9]: 17, 
        Re_11mkM9 = -0.8147242425157188611536e-4 * y5 * Yp6 * e9 * v17;
        Im_11mkM9 = -0.1151888407839660715949e-3 * y5 * Yp6 * e9 * v18;

    }
    else if (m == 6 && k == 4) {

        // 12. Z_deg[11][6][4][-9]: 18, 
        Re_11mkM9 = 0.1531754383065180631239e-8 * (-0.5454545454545454545455e0 + Y) * Yp6 * y4 * e9 * v18;
        Im_11mkM9 = 0.0e0;

    }
    else if (m == 5 && k == 6) {

        // 13. Z_deg[11][5][6][-9]: 17, 
        Re_11mkM9 = -0.1938917110569780785033e-3 * y6 * Yp5 * e9 * v18;
        Im_11mkM9 = 0.1371385251781800977855e-3 * y6 * Yp5 * e9 * v17;

    }
    else if (m == 5 && k == 5) {

        // 14. Z_deg[11][5][5][-9]: 18, 
        Re_11mkM9 = 0.0e0;
        Im_11mkM9 = -0.2578326826020758308856e-8 * (Y - 0.4545454545454545454547e0) * y5 * Yp5 * e9 * v18;

    }
    else if (m == 4 && k == 7) {

        // 15. Z_deg[11][4][7][-9]: 17, 
        Re_11mkM9 = 0.207333961592933920361e-3 * e9 * y7 * Yp4 * v17;
        Im_11mkM9 = 0.2931367135620323011479e-3 * e9 * y7 * Yp4 * v18;

    }
    else if (m == 4 && k == 6) {

        // 16. Z_deg[11][4][6][-9]: 18, 
        Re_11mkM9 = -0.3898063760169957634896e-8 * (Y - 0.3636363636363636363637e0) * y6 * Yp4 * e9 * v18;
        Im_11mkM9 = 0.0e0;

    }
    else if (m == 3 && k == 8) {

        // 17. Z_deg[11][3][8][-9]: 17, 
        Re_11mkM9 = 0.4013939761271391110074e-3 * Yp3 * y8 * e9 * v18;
        Im_11mkM9 = -0.2839037192533991288942e-3 * Yp3 * y8 * e9 * v17;

    }
    else if (m == 3 && k == 7) {

        // 18. Z_deg[11][3][7][-9]: 18, 
        Re_11mkM9 = 0.0e0;
        Im_11mkM9 = 0.5337643630096234175248e-8 * (Y - 0.2727272727272727272726e0) * y7 * Yp3 * e9 * v18;

    }
    else if (m == 2 && k == 9) {

        // 19. Z_deg[11][2][9][-9]: 17, 
        Re_11mkM9 = -0.3540901494256920313614e-3 * e9 * y9 * Yp2 * v17;
        Im_11mkM9 = -0.5006262452608910465998e-3 * e9 * y9 * Yp2 * v18;

    }
    else if (m == 2 && k == 8) {

        // 20. Z_deg[11][2][8][-9]: 18, 
        Re_11mkM9 = 0.6657211238838816599811e-8 * Yp2 * (-0.1818181818181818181818e0 + Y) * y8 * e9 * v18;
        Im_11mkM9 = 0.0e0;

    }
    else if (m == 1 && k == 10) {

        // 21. Z_deg[11][1][10][-9]: 17, 
        Re_11mkM9 = -0.2396234057256072299737e-7 * y10 * Yp * e9 * v18;
        Im_11mkM9 = 0.4037248866448556996227e-3 * y10 * Yp * e9 * v17;

    }
    else if (m == 1 && k == 9) {

        // 22. Z_deg[11][1][9][-9]: 18, 
        Re_11mkM9 = 0.0e0;
        Im_11mkM9 = 0.4174713759819793680648e-7 * (Y - 0.9090909090909090909092e-1) * y9 * e9 * v18 * Yp;

    }
    else if (m == 0 && k == 11) {

        // 23. Z_deg[11][0][11][-9]: 17, 
        Re_11mkM9 = 0.4216768915086977963315e-3 * y11 * e9 * v17;
        Im_11mkM9 = 0.2502784846850091052411e-7 * y11 * e9 * v18;

    }
    else if (m == 0 && k == 10) {

        // 24. Z_deg[11][0][10][-9]: 18, 
        Re_11mkM9 = 0.436034631357256693068e-7 * y10 * Y * e9 * v18;
        Im_11mkM9 = 0.0e0;

    }
    else if (m == -1 && k == 12) {

        // 25. Z_deg[11][-1][12][-9]: 17, 
        Re_11mkM9 = 0.2396234057256072299737e-7 / Yp * y12 * e9 * v18;
        Im_11mkM9 = -0.4037248866448556996227e-3 / Yp * y12 * e9 * v17;

    }
    else if (m == -1 && k == 11) {

        // 26. Z_deg[11][-1][11][-9]: 18, 
        Re_11mkM9 = 0.0e0;
        Im_11mkM9 = -0.4174713759819793680648e-7 * (Y + 0.9090909090909090909092e-1) * y11 * e9 * v18 / Yp;

    }
    else if (m == -2 && k == 13) {

        // 27. Z_deg[11][-2][13][-9]: 17, 
        Re_11mkM9 = -0.3540901494256920313614e-3 * e9 * y13 * v17 / Yp2;
        Im_11mkM9 = -0.5006262452608910465998e-3 * e9 * y13 * v18 / Yp2;

    }
    else if (m == -2 && k == 12) {

        // 28. Z_deg[11][-2][12][-9]: 18, 
        Re_11mkM9 = 0.6657211238838816599811e-8 * e9 * y12 * v18 * (0.1818181818181818181818e0 + Y) / Yp2;
        Im_11mkM9 = 0.0e0;

    }
    else if (m == -3 && k == 14) {

        // 29. Z_deg[11][-3][14][-9]: 17, 
        Re_11mkM9 = -0.4013939761271391110074e-3 * y14 * e9 * v18 / Yp3;
        Im_11mkM9 = 0.2839037192533991288942e-3 * y14 * e9 * v17 / Yp3;

    }
    else if (m == -3 && k == 13) {

        // 30. Z_deg[11][-3][13][-9]: 18, 
        Re_11mkM9 = 0.0e0;
        Im_11mkM9 = -0.5337643630096234175248e-8 * (Y + 0.2727272727272727272726e0) * e9 * y13 * v18 / Yp3;

    }
    else if (m == -4 && k == 15) {

        // 31. Z_deg[11][-4][15][-9]: 17, 
        Re_11mkM9 = 0.207333961592933920361e-3 * e9 * y15 * v17 / Yp4;
        Im_11mkM9 = 0.2931367135620323011479e-3 * e9 * y15 * v18 / Yp4;

    }
    else if (m == -4 && k == 14) {

        // 32. Z_deg[11][-4][14][-9]: 18, 
        Re_11mkM9 = -0.3898063760169957634896e-8 * (Y + 0.3636363636363636363637e0) * e9 * y14 * v18 / Yp4;
        Im_11mkM9 = 0.0e0;

    }
    else if (m == -5 && k == 16) {

        // 33. Z_deg[11][-5][16][-9]: 17, 
        Re_11mkM9 = 0.1938917110569780785033e-3 * e9 * y16 * v18 / Yp5;
        Im_11mkM9 = -0.1371385251781800977855e-3 * e9 * y16 * v17 / Yp5;

    }
    else if (m == -5 && k == 15) {

        // 34. Z_deg[11][-5][15][-9]: 18, 
        Re_11mkM9 = 0.0e0;
        Im_11mkM9 = 0.2578326826020758308856e-8 * (Y + 0.4545454545454545454547e0) * e9 * y15 * v18 / Yp5;

    }
    else if (m == -6 && k == 17) {

        // 35. Z_deg[11][-6][17][-9]: 17, 
        Re_11mkM9 = -0.8147242425157188611536e-4 * y17 * e9 * v17 / Yp6;
        Im_11mkM9 = -0.1151888407839660715949e-3 * y17 * e9 * v18 / Yp6;

    }
    else if (m == -6 && k == 16) {

        // 36. Z_deg[11][-6][16][-9]: 18, 
        Re_11mkM9 = 0.1531754383065180631239e-8 * (0.5454545454545454545455e0 + Y) * e9 * y16 * v18 / Yp6;
        Im_11mkM9 = 0.0e0;

    }
    else if (m == -7 && k == 18) {

        // 37. Z_deg[11][-7][18][-9]: 17, 
        Re_11mkM9 = -0.6070984965197136907797e-4 * y18 * e9 * v18 / Yp7;
        Im_11mkM9 = 0.429397378550843779531e-4 * y18 * e9 * v17 / Yp7;

    }
    else if (m == -7 && k == 17) {

        // 38. Z_deg[11][-7][17][-9]: 18, 
        Re_11mkM9 = 0.0e0;
        Im_11mkM9 = -0.8073054444053364689731e-9 * (Y + 0.6363636363636363636363e0) * e9 * y17 * v18 / Yp7;

    }
    else if (m == -8 && k == 19) {

        // 39. Z_deg[11][-8][19][-9]: 17, 
        Re_11mkM9 = 0.1970210294446744871935e-4 * e9 * y19 * v17 / Yp8;
        Im_11mkM9 = 0.2785558942215696090604e-4 * e9 * y19 * v18 / Yp8;

    }
    else if (m == -8 && k == 18) {

        // 40. Z_deg[11][-8][18][-9]: 18, 
        Re_11mkM9 = -0.3704171419719005368983e-9 * (Y + 0.7272727272727272727272e0) * e9 * y18 * v18 / Yp8;
        Im_11mkM9 = 0.0e0;

    }
    else if (m == -9 && k == 20) {

        // 41. Z_deg[11][-9][20][-9]: 17, 
        Re_11mkM9 = 0.1078842339308053925798e-4 * e9 * y20 * v18 / Yp9;
        Im_11mkM9 = -0.7630591658918654055336e-5 * e9 * y20 * v17 / Yp9;

    }
    else if (m == -9 && k == 19) {

        // 42. Z_deg[11][-9][19][-9]: 18, 
        Re_11mkM9 = 0.0e0;
        Im_11mkM9 = 0.143461942200691914918e-9 * (Y + 0.8181818181818181818179e0) * e9 * y19 * v18 / Yp9;

    }
    else if (m == -10 && k == 21) {

        // 43. Z_deg[11][-10][21][-9]: 17, 
        Re_11mkM9 = -0.2354851710327889089725e-5 * y21 * e9 * v17 / Yp10;
        Im_11mkM9 = -0.3329379740723451010374e-5 * y21 * e9 * v18 / Yp10;

    }
    else if (m == -10 && k == 20) {

        // 44. Z_deg[11][-10][20][-9]: 18, 
        Re_11mkM9 = 0.4427331654726953322143e-10 * (0.9090909090909090909091e0 + Y) * e9 * y20 * v18 / Yp10;
        Im_11mkM9 = 0.0e0;

    }
    else if (m == -11 && k == 22) {

        // 45. Z_deg[11][-11][22][-9]: 17, 
        Re_11mkM9 = -0.7098261457420962889457e-6 * y22 * e9 * v18 / Yp11;
        Im_11mkM9 = 0.5020560715531403881159e-6 * y22 * e9 * v17 / Yp11;

    }
    else if (m == -11 && k == 21) {

        // 46. Z_deg[11][-11][21][-9]: 18, 
        Re_11mkM9 = 0.0e0;
        Im_11mkM9 = -0.9439102803316565652998e-11 * y21 * e9 * v18 / Yp10;

    }
    else {

        perror("Parameter errors: hZ_11mkM9");
        exit(1);

    }

    GSL_SET_COMPLEX(&hZ_11mkM9, Re_11mkM9, Im_11mkM9);
    return hZ_11mkM9;

}

