/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 12, n = -8)

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
#include<gsl/gsl_complex.h>

// BHPC headers
#include "hZ_12mkM8_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_12mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_12mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
gsl_complex hZ_12mkM8(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    gsl_complex hZ_12mkM8 = { 0.0 };

    double  Re_12mkM8;
    double  Im_12mkM8;

    // NULL check
    if (PN_orb == NULL) {

        perror("Pointer errors: hZ_12mkM8");
        exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */

   // double K = PN_orb->PNK;
    double Ym = PN_orb->PNYm;

    //double q = PN_orb->PNq[1];
    
    double v18 = PN_orb->PNv[10];
    
    double e8 = PN_orb->PNe[8];
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

    if (m == 12 && k == 0) {

        // 1. Z_deg[12][12][0][-8]: 18, 
        Re_12mkM8 = (0.1734603647176403445471e-1 * e10 - 0.1391849870730755885475e-2 * e8) * v18 * Yp12;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == 12 && k == -2) {

        // 2. Z_deg[12][12][-2][-8]: 18, 
        Re_12mkM8 = (-0.2438008826049839082087e-5 * e10 + 0.2444960996759284732962e-6 * e8) * v18 * Yp10 * Yp * Ym;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == 11 && k == 1) {

        // 3. Z_deg[12][11][1][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = (-0.8497787683105782179312e-1 * e10 + 0.6818643963698157840224e-2 * e8) * v18 * Yp11 * y;

    }
    else if (m == 11 && k == -1) {

        // 4. Z_deg[12][11][-1][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = 0.2675313499760679529564e-17 * y * v18 * (-0.5e1 + 0.6e1 * Y) * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * Yp10;

    }
    else if (m == 10 && k == 2) {

        // 5. Z_deg[12][10][2][-8]: 18, 
        Re_12mkM8 = (0.2881740009668047554231e0 * e10 - 0.2312314669962306916094e-1 * e8) * v18 * Yp10 * Yp * Ym;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == 10 && k == 0) {

        // 6. Z_deg[12][10][0][-8]: 18, 
        Re_12mkM8 = 0.7889069114646370278842e-18 * (-0.69e2 * Y2 + 0.115e3 * Y - 0.47e2) * v18 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * Yp10;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == 9 && k == 3) {

        // 7. Z_deg[12][9][3][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = (0.7803788836906407345858e0 * e10 - 0.6261777727458969029931e-1 * e8) * v18 * y3 * Yp9;

    }
    else if (m == 9 && k == 1) {

        // 8. Z_deg[12][9][1][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = -0.3204555023210726313118e-17 * v18 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * y * (-0.46e2 * Y2 + 0.69e2 * Y - 0.25e2) * Yp9;

    }
    else if (m == 8 && k == 4) {

        // 9. Z_deg[12][8][4][-8]: 18, 
        Re_12mkM8 = (0.1788072652628750301464e1 * e10 - 0.1434753521053464276157e0 * e8) * v18 * y4 * Yp8;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == 8 && k == 2) {

        // 10. Z_deg[12][8][2][-8]: 18, 
        Re_12mkM8 = 0.4895038654171374045694e-17 * (-0.69e2 * Y2 + 0.92e2 * Y - 0.29e2) * v18 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * Yp * Ym * Yp8;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == 7 && k == 5) {

        // 11. Z_deg[12][7][5][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = (-0.3576145305257500602928e1 * e10 + 0.2869507042106928552313e0 * e8) * v18 * y5 * Yp7;

    }
    else if (m == 7 && k == 3) {

        // 12. Z_deg[12][7][3][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = 0.4895038654171374045694e-17 * v18 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * y3 * (-0.138e3 * Y2 + 0.161e3 * Y - 0.43e2) * Yp7;

    }
    else if (m == 6 && k == 6) {

        // 13. Z_deg[12][6][6][-8]: 18, 
        Re_12mkM8 = (-0.6363797210811455731157e1 * e10 + 0.5106325205555097140723e0 * e8) * v18 * y6 * Yp6;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == 6 && k == 4) {

        // 14. Z_deg[12][6][4][-8]: 18, 
        Re_12mkM8 = 0.5226471075730579649147e-16 * v18 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * (-0.23e2 * Y2 + 0.23e2 * Y - 0.5e1) * y4 * Yp6;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == 5 && k == 7) {

        // 15. Z_deg[12][5][7][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = (0.1020477807504175218034e2 * e10 - 0.8188336896272225936297e0 * e8) * v18 * Yp5 * y7;

    }
    else if (m == 5 && k == 5) {

        // 16. Z_deg[12][5][5][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = -0.1396833150519119354025e-16 * (-0.138e3 * Y2 + 0.115e3 * Y - 0.19e2) * v18 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * y5 * Yp5;

    }
    else if (m == 4 && k == 8) {

        // 17. Z_deg[12][4][8][-8]: 18, 
        Re_12mkM8 = (0.1487589251328512067926e2 * e10 - 0.1193644963523755559924e1 * e8) * v18 * y8 * Yp4;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == 4 && k == 6) {

        // 18. Z_deg[12][4][6][-8]: 18, 
        Re_12mkM8 = -0.4072433452901094900216e-16 * v18 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * (-0.69e2 * Y2 + 0.46e2 * Y - 0.5e1) * y6 * Yp4;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == 3 && k == 9) {

        // 19. Z_deg[12][3][9][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = (-0.1983452335104682757235e2 * e10 + 0.1591526618031674079898e1 * e8) * v18 * y9 * Yp3;

    }
    else if (m == 3 && k == 7) {

        // 20. Z_deg[12][3][7][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = 0.8144866905802189800431e-16 * v18 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * y7 * (-0.46e2 * Y2 + 0.23e2 * Y - 0.1e1) * Yp3;

    }
    else if (m == 2 && k == 10) {

        // 21. Z_deg[12][2][10][-8]: 18, 
        Re_12mkM8 = (-0.2429223075069131668842e2 * e10 + 0.1949214063117493339869e1 * e8) * v18 * y10 * Yp2;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == 2 && k == 8) {

        // 22. Z_deg[12][2][8][-8]: 18, 
        Re_12mkM8 = 0.6650255980698875190217e-16 * v18 * (0.23e2 * Y - 0.69e2 * Y2 + 0.1e1) * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * y8 * Yp2;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == 1 && k == 11) {

        // 23. Z_deg[12][1][11][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = (0.2740533234083479230116e2 * e10 - 0.2199010035405810477268e1 * e8) * v18 * y11 * Yp;

    }
    else if (m == 1 && k == 9) {

        // 24. Z_deg[12][1][9][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = -0.3751250290125995915164e-16 * v18 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * y9 * (-0.138e3 * Y2 + 0.23e2 * Y + 0.5e1) * Yp;

    }
    else if (m == 0 && k == 12) {

        // 25. Z_deg[12][0][12][-8]: 18, 
        Re_12mkM8 = (0.2852437426899269446746e2 * e10 - 0.2288802211594546200143e1 * e8) * v18 * y12;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == 0 && k == 10) {

        // 26. Z_deg[12][0][10][-8]: 18, 
        Re_12mkM8 = 0.2342655055332825504323e-15 * (0.23e2 * Y2 - 0.1e1) * v18 * y10 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8);
        Im_12mkM8 = 0.0e0;

    }
    else if (m == -1 && k == 13) {

        // 27. Z_deg[12][-1][13][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = (-0.2740533234083479230116e2 * e10 + 0.2199010035405810477268e1 * e8) * v18 * y13 / Yp;

    }
    else if (m == -1 && k == 11) {

        // 28. Z_deg[12][-1][11][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = -0.3751250290125995915164e-16 * y11 * (0.138e3 * Y2 + 0.23e2 * Y - 0.5e1) * v18 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) / Yp;

    }
    else if (m == -2 && k == 14) {

        // 29. Z_deg[12][-2][14][-8]: 18, 
        Re_12mkM8 = (-0.2429223075069131668842e2 * e10 + 0.1949214063117493339869e1 * e8) * v18 * y14 / Yp2;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == -2 && k == 12) {

        // 30. Z_deg[12][-2][12][-8]: 18, 
        Re_12mkM8 = -0.6650255980698875190217e-16 * y12 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * (0.23e2 * Y + 0.69e2 * Y2 - 0.1e1) * v18 / Yp2;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == -3 && k == 15) {

        // 31. Z_deg[12][-3][15][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = (0.1983452335104682757235e2 * e10 - 0.1591526618031674079898e1 * e8) * v18 * y15 / Yp3;

    }
    else if (m == -3 && k == 13) {

        // 32. Z_deg[12][-3][13][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = 0.8144866905802189800431e-16 * y13 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * (0.46e2 * Y2 + 0.23e2 * Y + 0.1e1) * v18 / Yp3;

    }
    else if (m == -4 && k == 16) {

        // 33. Z_deg[12][-4][16][-8]: 18, 
        Re_12mkM8 = (0.1487589251328512067926e2 * e10 - 0.1193644963523755559924e1 * e8) * v18 * y16 / Yp4;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == -4 && k == 14) {

        // 34. Z_deg[12][-4][14][-8]: 18, 
        Re_12mkM8 = 0.4072433452901094900216e-16 * y14 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * v18 * (0.69e2 * Y2 + 0.46e2 * Y + 0.5e1) / Yp4;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == -5 && k == 17) {

        // 35. Z_deg[12][-5][17][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = (-0.1020477807504175218034e2 * e10 + 0.8188336896272225936297e0 * e8) * v18 * y17 / Yp5;

    }
    else if (m == -5 && k == 15) {

        // 36. Z_deg[12][-5][15][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = -0.1396833150519119354025e-16 * (0.138e3 * Y2 + 0.115e3 * Y + 0.19e2) * y15 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * v18 / Yp5;

    }
    else if (m == -6 && k == 18) {

        // 37. Z_deg[12][-6][18][-8]: 18, 
        Re_12mkM8 = (-0.6363797210811455731157e1 * e10 + 0.5106325205555097140723e0 * e8) * v18 * y18 / Yp6;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == -6 && k == 16) {

        // 38. Z_deg[12][-6][16][-8]: 18, 
        Re_12mkM8 = -0.5226471075730579649147e-16 * y16 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * (0.23e2 * Y2 + 0.23e2 * Y + 0.5e1) * v18 / Yp6;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == -7 && k == 19) {

        // 39. Z_deg[12][-7][19][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = (0.3576145305257500602928e1 * e10 - 0.2869507042106928552313e0 * e8) * v18 * y19 / Yp7;

    }
    else if (m == -7 && k == 17) {

        // 40. Z_deg[12][-7][17][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = 0.4895038654171374045694e-17 * (0.138e3 * Y2 + 0.161e3 * Y + 0.43e2) * y17 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * v18 / Yp7;

    }
    else if (m == -8 && k == 20) {

        // 41. Z_deg[12][-8][20][-8]: 18, 
        Re_12mkM8 = (0.1788072652628750301464e1 * e10 - 0.1434753521053464276157e0 * e8) * v18 * y20 / Yp8;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == -8 && k == 18) {

        // 42. Z_deg[12][-8][18][-8]: 18, 
        Re_12mkM8 = 0.4895038654171374045694e-17 * y18 * (0.69e2 * Y2 + 0.92e2 * Y + 0.29e2) * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * v18 / Yp8;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == -9 && k == 21) {

        // 43. Z_deg[12][-9][21][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = (-0.7803788836906407345858e0 * e10 + 0.6261777727458969029931e-1 * e8) * v18 * y21 / Yp9;

    }
    else if (m == -9 && k == 19) {

        // 44. Z_deg[12][-9][19][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = -0.3204555023210726313118e-17 * y19 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * (0.46e2 * Y2 + 0.69e2 * Y + 0.25e2) * v18 / Yp9;

    }
    else if (m == -10 && k == 22) {

        // 45. Z_deg[12][-10][22][-8]: 18, 
        Re_12mkM8 = (-0.2881740009668047554231e0 * e10 + 0.2312314669962306916094e-1 * e8) * v18 * y22 / Yp10;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == -10 && k == 20) {

        // 46. Z_deg[12][-10][20][-8]: 18, 
        Re_12mkM8 = -0.7889069114646370278842e-18 * y20 * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) * (0.69e2 * Y2 + 0.115e3 * Y + 0.47e2) * v18 / Yp10;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == -11 && k == 23) {

        // 47. Z_deg[12][-11][23][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = (0.8497787683105782179312e-1 * e10 - 0.6818643963698157840224e-2 * e8) * v18 * y23 / Yp11;

    }
    else if (m == -11 && k == 21) {

        // 48. Z_deg[12][-11][21][-8]: 18, 
        Re_12mkM8 = 0.0e0;
        Im_12mkM8 = 0.2675313499760679529564e-17 * v18 * y21 * (0.5e1 + 0.6e1 * Y) * (0.744072001625e12 * e10 - 0.74619378048e11 * e8) / Yp10;

    }
    else if (m == -12 && k == 24) {

        // 49. Z_deg[12][-12][24][-8]: 18, 
        Re_12mkM8 = (0.1734603647176403445471e-1 * e10 - 0.1391849870730755885475e-2 * e8) * v18 * y24 / Yp12;
        Im_12mkM8 = 0.0e0;

    }
    else if (m == -12 && k == 22) {

        // 50. Z_deg[12][-12][22][-8]: 18, 
        Re_12mkM8 = (0.2438008826049839082087e-5 * e10 - 0.2444960996759284732962e-6 * e8) * v18 * y22 / Yp10;
        Im_12mkM8 = 0.0e0;

    }
    else {

        perror("Parameter errors: hZ_12mkM8");
        exit(1);

    }

    GSL_SET_COMPLEX(&hZ_12mkM8, Re_12mkM8, Im_12mkM8);
    return hZ_12mkM8;

}

