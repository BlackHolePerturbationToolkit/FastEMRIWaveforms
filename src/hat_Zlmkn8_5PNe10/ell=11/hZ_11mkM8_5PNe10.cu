/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 11, n = -8)

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
//#include<gsl/cmplx.h>

// BHPC headers
#include "hat_Zlmkn8_5PNe10/ell=11/hZ_11mkM8_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_11mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_11mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
CUDA_CALLABLE_MEMBER
cmplx hZ_11mkM8(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    cmplx hZ_11mkM8 = { 0.0 };

    double  Re_11mkM8 = 0.0;
    double  Im_11mkM8 = 0.0;

    // NULL check
    if (PN_orb == NULL) {

        //perror("Point errors: hZ_11mkM8");
        //exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */

   // double K = PN_orb->PNK;
    double Ym = PN_orb->PNYm;

    //double q = PN_orb->PNq[1];
    
    double v17 = PN_orb->PNv[9];
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

        // 1. Z_deg[11][11][0][-8]: 17, 
        Re_11mkM8 = (0.2747314524353989349503e-2 * e10 - 0.2272599496688495605981e-3 * e8) * Yp11 * v18;
        Im_11mkM8 = (-0.1618838487562541827526e-2 * e10 + 0.1489212548269877804354e-3 * e8) * Yp11 * v17;

    }
    else if (m == 11 && k == -1) {

        // 2. Z_deg[11][11][-1][-8]: 18, 
        Re_11mkM8 = 0.0e0;
        Im_11mkM8 = (0.1575544304981868651162e-6 * e8 - 0.1723410390216762218736e-5 * e10) * v18 * Yp10 * y;

    }
    else if (m == 11 && k == -2) {

        // 3. Z_deg[11][11][-2][-8]: 17, 
        Re_11mkM8 = (-0.2206919269024700075847e-8 * e10 + 0.2203706800259446967861e-9 * e8) * Yp9 * v18 * Yp * Ym;
        Im_11mkM8 = (0.1704897597457774728655e-8 * e10 - 0.1931086304706870503536e-9 * e8) * Yp9 * v17 * Yp * Ym;

    }
    else if (m == 10 && k == 1) {

        // 4. Z_deg[11][10][1][-8]: 17, 
        Re_11mkM8 = (-0.7593025554672071140651e-2 * e10 + 0.6985026006131844663818e-3 * e8) * Yp10 * v17 * y;
        Im_11mkM8 = (-0.1288604734222176091447e-1 * e10 + 0.1065943649503451369556e-2 * e8) * v18 * Yp10 * y;

    }
    else if (m == 10 && k == 0) {

        // 5. Z_deg[11][10][0][-8]: 18, 
        Re_11mkM8 = 0.1928232125111649993485e-17 * (-0.10e2 + 0.11e2 * Y) * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * Yp10;
        Im_11mkM8 = 0.0e0;

    }
    else if (m == 10 && k == -1) {

        // 6. Z_deg[11][10][-1][-8]: 17, 
        Re_11mkM8 = 0.2912826238800862874679e-20 * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * y * (-0.9e1 + 0.11e2 * Y) * Yp9;
        Im_11mkM8 = 0.2648023853455329886071e-21 * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) * y * (-0.9e1 + 0.11e2 * Y) * Yp9;

    }
    else if (m == 9 && k == 2) {

        // 7. Z_deg[11][9][2][-8]: 17, 
        Re_11mkM8 = (0.4175556572617350297242e-1 * e10 - 0.3454052195773210402613e-2 * e8) * Yp9 * v18 * Yp * Ym;
        Im_11mkM8 = (-0.2460421486810710432081e-1 * e10 + 0.2263407115868797879339e-2 * e8) * Yp9 * v17 * Yp * Ym;

    }
    else if (m == 9 && k == 1) {

        // 8. Z_deg[11][9][1][-8]: 18, 
        Re_11mkM8 = 0.0e0;
        Im_11mkM8 = -0.6248186204594273553305e-17 * y * (-0.9e1 + 0.11e2 * Y) * v18 * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * Yp9;

    }
    else if (m == 9 && k == 0) {

        // 9. Z_deg[11][9][0][-8]: 17, 
        Re_11mkM8 = 0.4085989513676849509083e-22 * (-0.231e3 * Y2 + 0.378e3 * Y - 0.151e3) * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) * Yp9;
        Im_11mkM8 = -0.4494588465044534459991e-21 * (-0.231e3 * Y2 + 0.378e3 * Y - 0.151e3) * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * Yp9;

    }
    else if (m == 8 && k == 3) {

        // 10. Z_deg[11][8][3][-8]: 17, 
        Re_11mkM8 = (0.6352780962045848745937e-1 * e10 - 0.5844092043631476907727e-2 * e8) * y3 * Yp8 * v17;
        Im_11mkM8 = (0.1078124071126261218004e0 * e10 - 0.8918324420773869486649e-2 * e8) * y3 * Yp8 * v18;

    }
    else if (m == 8 && k == 2) {

        // 11. Z_deg[11][8][2][-8]: 18, 
        Re_11mkM8 = 0.16132747409597699653e-16 * (-0.8e1 + 0.11e2 * Y) * Yp * Ym * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * Yp8;
        Im_11mkM8 = 0.0e0;

    }
    else if (m == 8 && k == 1) {

        // 12. Z_deg[11][8][1][-8]: 17, 
        Re_11mkM8 = -0.3481493254634687730291e-20 * (-0.77e2 * Y2 + 0.112e3 * Y - 0.39e2) * y * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * Yp8;
        Im_11mkM8 = -0.3164993867849716118446e-21 * (-0.77e2 * Y2 + 0.112e3 * Y - 0.39e2) * y * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) * Yp8;

    }
    else if (m == 7 && k == 4) {

        // 13. Z_deg[11][7][4][-8]: 17, 
        Re_11mkM8 = (0.2349716937319015006992e0 * e10 - 0.1943703744793210454597e-1 * e8) * Yp7 * y4 * v18;
        Im_11mkM8 = (-0.1384556511200347693276e0 * e10 + 0.1273690331746985028972e-1 * e8) * Yp7 * y4 * v17;

    }
    else if (m == 7 && k == 3) {

        // 14. Z_deg[11][7][3][-8]: 18, 
        Re_11mkM8 = 0.0e0;
        Im_11mkM8 = 0.3516050782005197546098e-16 * y3 * (-0.7e1 + 0.11e2 * Y) * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * Yp7;

    }
    else if (m == 7 && k == 2) {

        // 15. Z_deg[11][7][2][-8]: 17, 
        Re_11mkM8 = 0.6897944213441418874441e-21 * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) * (-0.77e2 * Y2 + 0.98e2 * Y - 0.29e2) * Yp7 * Yp * Ym;
        Im_11mkM8 = -0.7587738634785560761883e-20 * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * (-0.77e2 * Y2 + 0.98e2 * Y - 0.29e2) * Yp7 * Yp * Ym;

    }
    else if (m == 6 && k == 5) {

        // 16. Z_deg[11][6][5][-8]: 17, 
        Re_11mkM8 = (-0.2627011274765717997090e0 * e10 + 0.2416657489233565594023e-1 * e8) * y5 * Yp6 * v17;
        Im_11mkM8 = (-0.4458274427161911130863e0 * e10 + 0.3687918558087114167040e-1 * e8) * y5 * Yp6 * v18;

    }
    else if (m == 6 && k == 4) {

        // 17. Z_deg[11][6][4][-8]: 18, 
        Re_11mkM8 = 0.6671237303971557891514e-16 * y4 * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * (-0.6e1 + 0.11e2 * Y) * Yp6;
        Im_11mkM8 = 0.0e0;

    }
    else if (m == 6 && k == 3) {

        // 18. Z_deg[11][6][3][-8]: 17, 
        Re_11mkM8 = 0.4798907275195779210117e-20 * (-0.231e3 * Y2 + 0.252e3 * Y - 0.61e2) * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * y3 * Yp6;
        Im_11mkM8 = 0.4362642977450708372834e-21 * (-0.231e3 * Y2 + 0.252e3 * Y - 0.61e2) * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) * y3 * Yp6;

    }
    else if (m == 5 && k == 6) {

        // 19. Z_deg[11][5][6][-8]: 17, 
        Re_11mkM8 = (-0.7504394098949180945300e0 * e10 + 0.6207691948279660561314e-1 * e8) * y6 * Yp5 * v18;
        Im_11mkM8 = (0.4421918890438204428426e0 * e10 - 0.4067840707807349324977e-1 * e8) * y6 * Yp5 * v17;

    }
    else if (m == 5 && k == 5) {

        // 20. Z_deg[11][5][5][-8]: 18, 
        Re_11mkM8 = 0.0e0;
        Im_11mkM8 = -0.1122936568274101064772e-15 * (-0.5e1 + 0.11e2 * Y) * y5 * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * Yp5;

    }
    else if (m == 5 && k == 4) {

        // 21. Z_deg[11][5][4][-8]: 17, 
        Re_11mkM8 = 0.2203026714753703435179e-20 * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) * (-0.77e2 * Y2 + 0.70e2 * Y - 0.13e2) * y4 * Yp5;
        Im_11mkM8 = -0.2423329386229073778697e-19 * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * (-0.77e2 * Y2 + 0.70e2 * Y - 0.13e2) * y4 * Yp5;

    }
    else if (m == 4 && k == 7) {

        // 22. Z_deg[11][4][7][-8]: 17, 
        Re_11mkM8 = (0.6685312972456090904730e0 * e10 - 0.6149997077647546654502e-1 * e8) * y7 * Yp4 * v17;
        Im_11mkM8 = (0.1134557744345152711692e1 * e10 - 0.9385148063340579779042e-1 * e8) * y7 * Yp4 * v18;

    }
    else if (m == 4 && k == 6) {

        // 23. Z_deg[11][4][6][-8]: 18, 
        Re_11mkM8 = -0.1697720513002042877433e-15 * y6 * (-0.4e1 + 0.11e2 * Y) * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * Yp4;
        Im_11mkM8 = 0.0e0;

    }
    else if (m == 4 && k == 5) {

        // 24. Z_deg[11][4][5][-8]: 17, 
        Re_11mkM8 = -0.2564610760302768628935e-18 * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * (-0.11e2 * Y2 + 0.8e1 * Y - 0.1e1) * y5 * Yp4;
        Im_11mkM8 = -0.233146432754797148085e-19 * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) * (-0.11e2 * Y2 + 0.8e1 * Y - 0.1e1) * y5 * Yp4;

    }
    else if (m == 3 && k == 8) {

        // 25. Z_deg[11][3][8][-8]: 17, 
        Re_11mkM8 = (0.1553557173425048650017e1 * e10 - 0.1285114324954389772454e0 * e8) * Yp3 * y8 * v18;
        Im_11mkM8 = (-0.9154241797490285630257e0 * e10 + 0.8421230320046029799670e-1 * e8) * Yp3 * y8 * v17;

    }
    else if (m == 3 && k == 7) {

        // 26. Z_deg[11][3][7][-8]: 18, 
        Re_11mkM8 = 0.0e0;
        Im_11mkM8 = 0.2324699553276153860852e-15 * y7 * (-0.3e1 + 0.11e2 * Y) * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * Yp3;

    }
    else if (m == 3 && k == 6) {

        // 27. Z_deg[11][3][6][-8]: 17, 
        Re_11mkM8 = -0.1064163003513864377221e-19 * (-0.33e2 * Y2 + 0.18e2 * Y - 0.1e1) * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) * y6 * Yp3;
        Im_11mkM8 = 0.1170579303865250814943e-18 * (-0.33e2 * Y2 + 0.18e2 * Y - 0.1e1) * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * y6 * Yp3;

    }
    else if (m == 2 && k == 9) {

        // 28. Z_deg[11][2][9][-8]: 17, 
        Re_11mkM8 = (-0.1141734548063143002414e1 * e10 + 0.1050311954424163664070e0 * e8) * y9 * Yp2 * v17;
        Im_11mkM8 = (-0.1937626224573826130043e1 * e10 + 0.1602819168938199922171e0 * e8) * y9 * Yp2 * v18;

    }
    else if (m == 2 && k == 8) {

        // 29. Z_deg[11][2][8][-8]: 18, 
        Re_11mkM8 = 0.2899409751848600928152e-15 * (-0.2e1 + 0.11e2 * Y) * y8 * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * Yp2;
        Im_11mkM8 = 0.0e0;

    }
    else if (m == 2 && k == 7) {

        // 30. Z_deg[11][2][7][-8]: 17, 
        Re_11mkM8 = 0.6257009570160162612633e-19 * (-0.77e2 * Y2 + 0.28e2 * Y + 0.1e1) * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * y7 * Yp2;
        Im_11mkM8 = 0.5688190518327420556939e-20 * (-0.77e2 * Y2 + 0.28e2 * Y + 0.1e1) * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) * y7 * Yp2;

    }
    else if (m == 1 && k == 10) {

        // 31. Z_deg[11][1][10][-8]: 17, 
        Re_11mkM8 = 0.5195677290837730263227e-3 * y10 * Yp * e10 * v18;
        Im_11mkM8 = (0.1301777673688266255418e1 * e10 - 0.1197539879122277240575e0 * e8) * y10 * v17 * Yp;

    }
    else if (m == 1 && k == 9) {

        // 32. Z_deg[11][1][9][-8]: 18, 
        Re_11mkM8 = 0.0e0;
        Im_11mkM8 = 0.1818209660492810597318e-14 * y9 * (-0.1e1 + 0.11e2 * Y) * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * Yp;

    }
    else if (m == 1 && k == 8) {

        // 33. Z_deg[11][1][8][-8]: 17, 
        Re_11mkM8 = -0.1256644671084540754675e-10 * (-0.77e2 * Y2 + 0.14e2 * Y + 0.3e1) * y8 * v18 * e10 * Yp;
        Im_11mkM8 = -0.713408854650673800915e-19 * (-0.77e2 * Y2 + 0.14e2 * Y + 0.3e1) * y8 * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * Yp;

    }
    else if (m == 0 && k == 11) {

        // 34. Z_deg[11][0][11][-8]: 17, 
        Re_11mkM8 = (0.1359662436066688195741e1 * e10 - 0.1250789610426454523445e0 * e8) * y11 * v17;
        Im_11mkM8 = -0.5426707943347695746794e-3 * y11 * e10 * v18;

    }
    else if (m == 0 && k == 10) {

        // 35. Z_deg[11][0][10][-8]: 18, 
        Re_11mkM8 = (-0.7278148088062032204127e-3 * e8 + 0.7961208070658930936039e-2 * e10) * v18 * y10 * Y;
        Im_11mkM8 = 0.0e0;

    }
    else if (m == 0 && k == 9) {

        // 36. Z_deg[11][0][9][-8]: 17, 
        Re_11mkM8 = 0.2732147905423825670307e-18 * y9 * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * (0.21e2 * Y2 - 0.1e1);
        Im_11mkM8 = -0.1010642365247664796593e-8 * y9 * e10 * v18 * (Y2 - 0.4761904761904761904762e-1);

    }
    else if (m == -1 && k == 12) {

        // 37. Z_deg[11][-1][12][-8]: 17, 
        Re_11mkM8 = -0.5195677290837730263227e-3 / Yp * y12 * e10 * v18;
        Im_11mkM8 = (-0.1301777673688266255418e1 * e10 + 0.1197539879122277240575e0 * e8) * y12 * v17 / Yp;

    }
    else if (m == -1 && k == 11) {

        // 38. Z_deg[11][-1][11][-8]: 18, 
        Re_11mkM8 = 0.0e0;
        Im_11mkM8 = -0.1818209660492810597318e-14 * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * (0.1e1 + 0.11e2 * Y) * v18 * y11 / Yp;

    }
    else if (m == -1 && k == 10) {

        // 39. Z_deg[11][-1][10][-8]: 17, 
        Re_11mkM8 = -0.1256644671084540754675e-10 * (0.77e2 * Y2 + 0.14e2 * Y - 0.3e1) * y10 * v18 * e10 / Yp;
        Im_11mkM8 = -0.713408854650673800915e-19 * (0.77e2 * Y2 + 0.14e2 * Y - 0.3e1) * y10 * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) / Yp;

    }
    else if (m == -2 && k == 13) {

        // 40. Z_deg[11][-2][13][-8]: 17, 
        Re_11mkM8 = (-0.1141734548063143002414e1 * e10 + 0.1050311954424163664070e0 * e8) * y13 * v17 / Yp2;
        Im_11mkM8 = (-0.1937626224573826130043e1 * e10 + 0.1602819168938199922171e0 * e8) * y13 * v18 / Yp2;

    }
    else if (m == -2 && k == 12) {

        // 41. Z_deg[11][-2][12][-8]: 18, 
        Re_11mkM8 = 0.2899409751848600928152e-15 * (0.2e1 + 0.11e2 * Y) * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * y12 / Yp2;
        Im_11mkM8 = 0.0e0;

    }
    else if (m == -2 && k == 11) {

        // 42. Z_deg[11][-2][11][-8]: 17, 
        Re_11mkM8 = -0.6257009570160162612633e-19 * y11 * (0.77e2 * Y2 + 0.28e2 * Y - 0.1e1) * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) / Yp2;
        Im_11mkM8 = -0.5688190518327420556939e-20 * y11 * (0.77e2 * Y2 + 0.28e2 * Y - 0.1e1) * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) / Yp2;

    }
    else if (m == -3 && k == 14) {

        // 43. Z_deg[11][-3][14][-8]: 17, 
        Re_11mkM8 = (-0.1553557173425048650017e1 * e10 + 0.1285114324954389772454e0 * e8) * y14 * v18 / Yp3;
        Im_11mkM8 = (0.9154241797490285630257e0 * e10 - 0.8421230320046029799670e-1 * e8) * y14 * v17 / Yp3;

    }
    else if (m == -3 && k == 13) {

        // 44. Z_deg[11][-3][13][-8]: 18, 
        Re_11mkM8 = 0.0e0;
        Im_11mkM8 = -0.2324699553276153860852e-15 * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * (0.3e1 + 0.11e2 * Y) * v18 * y13 / Yp3;

    }
    else if (m == -3 && k == 12) {

        // 45. Z_deg[11][-3][12][-8]: 17, 
        Re_11mkM8 = -0.1064163003513864377221e-19 * (0.33e2 * Y2 + 0.18e2 * Y + 0.1e1) * y12 * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) / Yp3;
        Im_11mkM8 = 0.1170579303865250814943e-18 * (0.33e2 * Y2 + 0.18e2 * Y + 0.1e1) * y12 * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) / Yp3;

    }
    else if (m == -4 && k == 15) {

        // 46. Z_deg[11][-4][15][-8]: 17, 
        Re_11mkM8 = (0.6685312972456090904730e0 * e10 - 0.6149997077647546654502e-1 * e8) * y15 * v17 / Yp4;
        Im_11mkM8 = (0.1134557744345152711692e1 * e10 - 0.9385148063340579779042e-1 * e8) * y15 * v18 / Yp4;

    }
    else if (m == -4 && k == 14) {

        // 47. Z_deg[11][-4][14][-8]: 18, 
        Re_11mkM8 = -0.1697720513002042877433e-15 * (0.4e1 + 0.11e2 * Y) * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * y14 / Yp4;
        Im_11mkM8 = 0.0e0;

    }
    else if (m == -4 && k == 13) {

        // 48. Z_deg[11][-4][13][-8]: 17, 
        Re_11mkM8 = 0.2564610760302768628935e-18 * y13 * (0.11e2 * Y2 + 0.8e1 * Y + 0.1e1) * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) / Yp4;
        Im_11mkM8 = 0.233146432754797148085e-19 * y13 * (0.11e2 * Y2 + 0.8e1 * Y + 0.1e1) * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) / Yp4;

    }
    else if (m == -5 && k == 16) {

        // 49. Z_deg[11][-5][16][-8]: 17, 
        Re_11mkM8 = (0.7504394098949180945300e0 * e10 - 0.6207691948279660561314e-1 * e8) * y16 * v18 / Yp5;
        Im_11mkM8 = (-0.4421918890438204428426e0 * e10 + 0.4067840707807349324977e-1 * e8) * y16 * v17 / Yp5;

    }
    else if (m == -5 && k == 15) {

        // 50. Z_deg[11][-5][15][-8]: 18, 
        Re_11mkM8 = 0.0e0;
        Im_11mkM8 = 0.1122936568274101064772e-15 * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * y15 * (0.5e1 + 0.11e2 * Y) / Yp5;

    }
    else if (m == -5 && k == 14) {

        // 51. Z_deg[11][-5][14][-8]: 17, 
        Re_11mkM8 = 0.2203026714753703435179e-20 * y14 * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) * (0.77e2 * Y2 + 0.70e2 * Y + 0.13e2) / Yp5;
        Im_11mkM8 = -0.2423329386229073778697e-19 * y14 * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * (0.77e2 * Y2 + 0.70e2 * Y + 0.13e2) / Yp5;

    }
    else if (m == -6 && k == 17) {

        // 52. Z_deg[11][-6][17][-8]: 17, 
        Re_11mkM8 = (-0.2627011274765717997090e0 * e10 + 0.2416657489233565594023e-1 * e8) * y17 * v17 / Yp6;
        Im_11mkM8 = (-0.4458274427161911130863e0 * e10 + 0.3687918558087114167040e-1 * e8) * y17 * v18 / Yp6;

    }
    else if (m == -6 && k == 16) {

        // 53. Z_deg[11][-6][16][-8]: 18, 
        Re_11mkM8 = 0.6671237303971557891514e-16 * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * y16 * (0.6e1 + 0.11e2 * Y) / Yp6;
        Im_11mkM8 = 0.0e0;

    }
    else if (m == -6 && k == 15) {

        // 54. Z_deg[11][-6][15][-8]: 17, 
        Re_11mkM8 = -0.4798907275195779210117e-20 * y15 * (0.231e3 * Y2 + 0.252e3 * Y + 0.61e2) * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) / Yp6;
        Im_11mkM8 = -0.4362642977450708372834e-21 * y15 * (0.231e3 * Y2 + 0.252e3 * Y + 0.61e2) * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) / Yp6;

    }
    else if (m == -7 && k == 18) {

        // 55. Z_deg[11][-7][18][-8]: 17, 
        Re_11mkM8 = (-0.2349716937319015006992e0 * e10 + 0.1943703744793210454597e-1 * e8) * y18 * v18 / Yp7;
        Im_11mkM8 = (0.1384556511200347693276e0 * e10 - 0.1273690331746985028972e-1 * e8) * y18 * v17 / Yp7;

    }
    else if (m == -7 && k == 17) {

        // 56. Z_deg[11][-7][17][-8]: 18, 
        Re_11mkM8 = 0.0e0;
        Im_11mkM8 = -0.3516050782005197546098e-16 * (0.7e1 + 0.11e2 * Y) * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * y17 / Yp7;

    }
    else if (m == -7 && k == 16) {

        // 57. Z_deg[11][-7][16][-8]: 17, 
        Re_11mkM8 = -0.6897944213441418874441e-21 * y16 * (0.77e2 * Y2 + 0.98e2 * Y + 0.29e2) * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) / Yp7;
        Im_11mkM8 = 0.7587738634785560761883e-20 * y16 * (0.77e2 * Y2 + 0.98e2 * Y + 0.29e2) * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) / Yp7;

    }
    else if (m == -8 && k == 19) {

        // 58. Z_deg[11][-8][19][-8]: 17, 
        Re_11mkM8 = (0.6352780962045848745937e-1 * e10 - 0.5844092043631476907727e-2 * e8) * y19 * v17 / Yp8;
        Im_11mkM8 = (0.1078124071126261218004e0 * e10 - 0.8918324420773869486649e-2 * e8) * y19 * v18 / Yp8;

    }
    else if (m == -8 && k == 18) {

        // 59. Z_deg[11][-8][18][-8]: 18, 
        Re_11mkM8 = -0.16132747409597699653e-16 * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * (0.8e1 + 0.11e2 * Y) * v18 * y18 / Yp8;
        Im_11mkM8 = 0.0e0;

    }
    else if (m == -8 && k == 17) {

        // 60. Z_deg[11][-8][17][-8]: 17, 
        Re_11mkM8 = 0.3481493254634687730291e-20 * y17 * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * (0.77e2 * Y2 + 0.112e3 * Y + 0.39e2) / Yp8;
        Im_11mkM8 = 0.3164993867849716118446e-21 * y17 * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) * (0.77e2 * Y2 + 0.112e3 * Y + 0.39e2) / Yp8;

    }
    else if (m == -9 && k == 20) {

        // 61. Z_deg[11][-9][20][-8]: 17, 
        Re_11mkM8 = (0.4175556572617350297242e-1 * e10 - 0.3454052195773210402613e-2 * e8) * y20 * v18 / Yp9;
        Im_11mkM8 = (-0.2460421486810710432081e-1 * e10 + 0.2263407115868797879339e-2 * e8) * y20 * v17 / Yp9;

    }
    else if (m == -9 && k == 19) {

        // 62. Z_deg[11][-9][19][-8]: 18, 
        Re_11mkM8 = 0.0e0;
        Im_11mkM8 = 0.6248186204594273553305e-17 * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * (0.9e1 + 0.11e2 * Y) * y19 / Yp9;

    }
    else if (m == -9 && k == 18) {

        // 63. Z_deg[11][-9][18][-8]: 17, 
        Re_11mkM8 = 0.4085989513676849509083e-22 * (0.231e3 * Y2 + 0.378e3 * Y + 0.151e3) * y18 * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) / Yp9;
        Im_11mkM8 = -0.4494588465044534459991e-21 * (0.231e3 * Y2 + 0.378e3 * Y + 0.151e3) * y18 * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) / Yp9;

    }
    else if (m == -10 && k == 21) {

        // 64. Z_deg[11][-10][21][-8]: 17, 
        Re_11mkM8 = (-0.7593025554672071140651e-2 * e10 + 0.6985026006131844663818e-3 * e8) * y21 * v17 / Yp10;
        Im_11mkM8 = (-0.1288604734222176091447e-1 * e10 + 0.1065943649503451369556e-2 * e8) * y21 * v18 / Yp10;

    }
    else if (m == -10 && k == 20) {

        // 65. Z_deg[11][-10][20][-8]: 18, 
        Re_11mkM8 = 0.1928232125111649993485e-17 * (0.10e2 + 0.11e2 * Y) * (-0.34840947840e11 * e8 + 0.381107984857e12 * e10) * v18 * y20 / Yp10;
        Im_11mkM8 = 0.0e0;

    }
    else if (m == -10 && k == 19) {

        // 66. Z_deg[11][-10][19][-8]: 17, 
        Re_11mkM8 = -0.2912826238800862874679e-20 * y19 * v17 * (0.249575744855e12 * e10 - 0.28268695058e11 * e8) * (0.9e1 + 0.11e2 * Y) / Yp9;
        Im_11mkM8 = -0.2648023853455329886071e-21 * y19 * v18 * (0.3553720020166e13 * e10 - 0.354854710119e12 * e8) * (0.9e1 + 0.11e2 * Y) / Yp9;

    }
    else if (m == -11 && k == 22) {

        // 67. Z_deg[11][-11][22][-8]: 17, 
        Re_11mkM8 = (-0.2747314524353989349503e-2 * e10 + 0.2272599496688495605981e-3 * e8) * y22 * v18 / Yp11;
        Im_11mkM8 = (0.1618838487562541827526e-2 * e10 - 0.1489212548269877804354e-3 * e8) * y22 * v17 / Yp11;

    }
    else if (m == -11 && k == 21) {

        // 68. Z_deg[11][-11][21][-8]: 18, 
        Re_11mkM8 = 0.0e0;
        Im_11mkM8 = (0.1575544304981868651162e-6 * e8 - 0.1723410390216762218736e-5 * e10) * y21 * v18 / Yp10;

    }
    else if (m == -11 && k == 20) {

        // 69. Z_deg[11][-11][20][-8]: 17, 
        Re_11mkM8 = (-0.2206919269024700075847e-8 * e10 + 0.2203706800259446967861e-9 * e8) * y20 * v18 / Yp9;
        Im_11mkM8 = (0.1704897597457774728655e-8 * e10 - 0.1931086304706870503536e-9 * e8) * y20 * v17 / Yp9;

    }
    else {

        //perror("Parameter errors: hZ_11mkM8");
        //exit(1);

    }

    hZ_11mkM8 = cmplx(Re_11mkM8, Im_11mkM8);
    return hZ_11mkM8;

}
