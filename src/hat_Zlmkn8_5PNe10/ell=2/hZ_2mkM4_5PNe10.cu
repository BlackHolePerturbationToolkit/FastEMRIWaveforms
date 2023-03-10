/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 2, n = -4)

- BHPC's maple script `outputC_Slm_Zlmkn.mw`.


25th May 2020; RF
6th June. 2020; Sis


Convention (the phase `B_inc`  is defined in `Zlmkn8.c`):
 Zlmkn8  = ( hat_Zlmkn8 * exp(-I * B_inc) )

 For ell = 2, we have only
 0 <= |k| <= 8
 
 //Shorthad variables for PN amplitudes  in `inspiral_orb_data`
 k = sqrt(1. - q^2 ); y = sqrt(1. -  Y^2) ;
 Ym = Y - 1.0 , Yp = Y + 1.0 ;

PNq[11] = 1, q, q ^2, ..., q ^ 9, q ^ 10
PNe[11] = 1, e, e ^ 2, ..., e ^ 9, e ^ 10
PNv[11] = v ^ 8, v ^ 9, ..., v ^ 17, v ^ 18
PNY[11] = 1, Y, Y ^ 2, ..., Y ^ 9, Y ^ 10
PNYp[11] = 1, Yp, Yp^2,...  Yp^10
PNy[21] = 1, y, y ^ 2, ..., y ^ 19, y ^ 20



 WARGNING !! 
`hZ_2mkM5_5PNe10` stores  only the PN amplitudes that has the index
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

#include "global.h"

// BHPC headers
#include "Zlmkn8_5PNe10.h"
#include "hat_Zlmkn8_5PNe10/ell=2/hZ_2mkM4_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_2mkM5_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/
//static int lmax = 2;
//static int kmax = 6 + 2;

/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_2mkM5_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
CUDA_CALLABLE_MEMBER
cmplx hZ_2mkM4(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    cmplx hZ_2mkM4 = { 0.0 };
    //&hZ_2mkM4, 0.0, 0.0);

    double  Re_2mkM4 = 0.0;
    double  Im_2mkM4 = 0.0;

    // NULL check
    if (PN_orb == NULL) {

        //perror("Point errors: hZ_2mkM4");
        //exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */
   
    double q3 = PN_orb->PNq[3];
    double q4 = PN_orb->PNq[4];
    double q5 = PN_orb->PNq[5];

    double e4 = PN_orb->PNe[4];
    double e6 = PN_orb->PNe[6];
    double e8 = PN_orb->PNe[8];
    double e10 = PN_orb->PNe[10];

    double v16 = PN_orb->PNv[8];
    double v17 = PN_orb->PNv[9];
    double v18 = PN_orb->PNv[10];

    double Y = PN_orb->PNY[1];
  
    double Yp = PN_orb->PNYp[1];

    double y = PN_orb->PNy[1];
    double y2 = PN_orb->PNy[2];
    double y3 = PN_orb->PNy[3];
    double y4 = PN_orb->PNy[4];
    double y5 = PN_orb->PNy[5];
   double y6 = PN_orb->PNy[6];
    double y7 = PN_orb->PNy[7];
    double y8 = PN_orb->PNy[8];

if (m == 2 && k == 4) {

 // 1. Z_deg[2][2][4][-4]: 16, 
 Re_2mkM4 = 0.2852769961089326677313e1 * y4 * ((0.1000000000000000000000e1 * e4 - 0.8048227206946454413895e1 * e6 + 0.2847567082730342498794e2 * e8 - 0.5803174753922311809433e2 * e10) * v16 + (0.3631590474467645234650e2 * e4 - 0.2961607343136241471987e3 * e6 + 0.1047475117233262697264e4 * e8 - 0.2086364026414767245538e4 * e10) * v18) * Yp * Yp * q4;
 Im_2mkM4 = 0.0e0;

}
 else if (m == 2 && k == 3) {

 // 2. Z_deg[2][2][3][-4]: 16, 
 Re_2mkM4 = 0.19927373985765900101e-1 * (((-0.4871118368268312630331e2 * Y + 0.2435559184134156315166e3) * e10 + (-0.5e1 + Y) * e4 + (-0.7667089290531427029943e1 * Y + 0.3833544645265713514971e2) * e6 + (0.2562010301456703533854e2 * Y - 0.1281005150728351766927e3) * e8) * q4 * v17 + ((-0.177819430776761743273e3 * v16 - 0.1280648519952372065732e5 * v18) * e10 + (0.3662568885120813904197e1 * e4 - 0.2805803691857104320013e2 * e6 + 0.9365744986961604172289e2 * e8) * v16 + (0.6868614355109369327967e4 * e8 - 0.2073872841028627378777e4 * e6 + 0.2712391186130058533619e3 * e4) * v18) * q3) * y3 * Yp * Yp;
 Im_2mkM4 = 0.0e0;

}
 else if (m == 1 && k == 5) {

 // 3. Z_deg[2][1][5][-4]: 16, 
 Re_2mkM4 = 0.0e0;
 Im_2mkM4 = -0.5705539922178653354625e1 * ((0.1000000000000000000000e1 * e4 - 0.8048227206946454413901e1 * e6 + 0.2847567082730342498796e2 * e8 - 0.5803174753922311809438e2 * e10) * v16 + (0.3631590474467645234651e2 * e4 - 0.2961607343136241471988e3 * e6 + 0.1047475117233262697265e4 * e8 - 0.2086364026414767245538e4 * e10) * v18) * y5 * q4 * Yp;

}
 else if (m == 1 && k == 4) {

 // 4. Z_deg[2][1][4][-4]: 16, 
 Re_2mkM4 = 0.0e0;
 Im_2mkM4 = -0.3985474797153180020201e-1 * (((-0.487111836826831263033e2 * Y + 0.1217779592067078157583e3) * e10 + (-0.25e1 + Y) * e4 + (-0.7667089290531427029941e1 * Y + 0.1916772322632856757485e2) * e6 + (0.2562010301456703533855e2 * Y - 0.6405025753641758834635e2) * e8) * q4 * v17 + ((-0.2073872841028627378776e4 * e6 + 0.271239118613005853362e3 * e4 + 0.6868614355109369327966e4 * e8 - 0.1280648519952372065731e5 * e10) * q3 - 0.7500000000000000000001e0 * q5 * e4 + 0.3653338776201234472747e2 * q5 * e10 - 0.1921507726092527650391e2 * q5 * e8 + 0.5750316967898570272458e1 * e6 * q5) * v18 + (-0.2805803691857104320013e2 * v16 * e6 + 0.9365744986961604172287e2 * v16 * e8 - 0.1778194307767617432729e3 * v16 * e10 + 0.3662568885120813904196e1 * v16 * e4) * q3) * y4 * Yp;

}
 else if (m == 0 && k == 6) {

 // 5. Z_deg[2][0][6][-4]: 16, 
 Re_2mkM4 = -0.6987830758208271794486e1 * y6 * ((0.1000000000000000000000e1 * e4 - 0.8048227206946454413903e1 * e6 + 0.2847567082730342498795e2 * e8 - 0.5803174753922311809438e2 * e10) * v16 + (0.3631590474467645234655e2 * e4 - 0.2961607343136241471990e3 * e6 + 0.1047475117233262697264e4 * e8 - 0.2086364026414767245539e4 * e10) * v18) * q4;
 Im_2mkM4 = 0.0e0;

}
 else if (m == 0 && k == 5) {

 // 6. Z_deg[2][0][5][-4]: 16, 
 Re_2mkM4 = -0.4881189817873790917106e-1 * (((-0.1293314089065249360042e5 * e10 + 0.6915243015066293379843e4 * e8 - 0.2084078603614211922817e4 * e6 + 0.2722467361191396799706e3 * e4) * q3 + 0.487111836826831263033e2 * q5 * e10 - 0.2562010301456703533854e2 * q5 * e8 + 0.7667089290531427029942e1 * e6 * q5 - 0.1e1 * q5 * e4) * v18 + (-0.177819430776761743273e3 * v16 * e10 + (0.3662568885120813904196e1 * e4 - 0.2805803691857104320013e2 * e6 + 0.9365744986961604172285e2 * e8) * v16) * q3 - 0.4871118368268312630332e2 * v17 * q4 * e10 * Y + (q4 * e4 * Y - 0.7667089290531427029942e1 * q4 * e6 * Y + 0.2562010301456703533854e2 * q4 * e8 * Y) * v17) * y5;
 Im_2mkM4 = 0.0e0;

}
 else if (m == -1 && k == 7) {

 // 7. Z_deg[2][-1][7][-4]: 16, 
 Re_2mkM4 = 0.0e0;
 Im_2mkM4 = 0.5705539922178653354625e1 * ((0.1000000000000000000000e1 * e4 - 0.8048227206946454413901e1 * e6 + 0.2847567082730342498796e2 * e8 - 0.5803174753922311809438e2 * e10) * v16 + (0.3631590474467645234651e2 * e4 - 0.2961607343136241471988e3 * e6 + 0.1047475117233262697265e4 * e8 - 0.2086364026414767245538e4 * e10) * v18) * y7 * q4 / Yp;

}
 else if (m == -1 && k == 6) {

 // 8. Z_deg[2][-1][6][-4]: 16, 
 Re_2mkM4 = 0.0e0;
 Im_2mkM4 = 0.3985474797153180020201e-1 * (((-0.487111836826831263033e2 * Y - 0.1217779592067078157583e3) * e10 + (0.25e1 + Y) * e4 + (-0.7667089290531427029941e1 * Y - 0.1916772322632856757485e2) * e6 + (0.2562010301456703533855e2 * Y + 0.6405025753641758834635e2) * e8) * q4 * v17 + ((-0.2073872841028627378776e4 * e6 + 0.271239118613005853362e3 * e4 + 0.6868614355109369327966e4 * e8 - 0.1280648519952372065731e5 * e10) * q3 - 0.7500000000000000000001e0 * q5 * e4 + 0.3653338776201234472747e2 * q5 * e10 - 0.1921507726092527650391e2 * q5 * e8 + 0.5750316967898570272458e1 * e6 * q5) * v18 + (-0.2805803691857104320013e2 * v16 * e6 + 0.9365744986961604172287e2 * v16 * e8 - 0.1778194307767617432729e3 * v16 * e10 + 0.3662568885120813904196e1 * v16 * e4) * q3) * y6 / Yp;

}
 else if (m == -2 && k == 8) {

 // 9. Z_deg[2][-2][8][-4]: 16, 
 Re_2mkM4 = 0.2852769961089326677313e1 * y8 * ((0.1000000000000000000000e1 * e4 - 0.8048227206946454413895e1 * e6 + 0.2847567082730342498794e2 * e8 - 0.5803174753922311809433e2 * e10) * v16 + (0.3631590474467645234650e2 * e4 - 0.2961607343136241471987e3 * e6 + 0.1047475117233262697264e4 * e8 - 0.2086364026414767245538e4 * e10) * v18) * pow(Yp, -0.2e1) * q4;
 Im_2mkM4 = 0.0e0;

}
 else if (m == -2 && k == 7) {

 // 10. Z_deg[2][-2][7][-4]: 16, 
 Re_2mkM4 = 0.19927373985765900101e-1 * (((-0.4871118368268312630331e2 * Y - 0.2435559184134156315166e3) * e10 + (0.5e1 + Y) * e4 + (-0.7667089290531427029943e1 * Y - 0.3833544645265713514971e2) * e6 + (0.2562010301456703533854e2 * Y + 0.1281005150728351766927e3) * e8) * q4 * v17 + ((-0.177819430776761743273e3 * v16 - 0.1280648519952372065732e5 * v18) * e10 + (0.3662568885120813904197e1 * e4 - 0.2805803691857104320013e2 * e6 + 0.9365744986961604172289e2 * e8) * v16 + (0.6868614355109369327967e4 * e8 - 0.2073872841028627378777e4 * e6 + 0.2712391186130058533619e3 * e4) * v18) * q3) * y7 * pow(Yp, -0.2e1);
 Im_2mkM4 = 0.0e0;

 }
 else {

        //perror("Parameter errors: hZ_2mkM4");
        //exit(1);

    }

    hZ_2mkM4 = cmplx(Re_2mkM4, Im_2mkM4);
    return hZ_2mkM4;

}
