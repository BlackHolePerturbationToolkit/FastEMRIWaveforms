/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 3, n = -5)

- BHPC's maple script `outputC_Slm_Zlmkn.mw`.


25th May 2020; RF
12th June. 2020; Sis


Convention (the phase `B_inc`  is defined in `Zlmkn8.c`):
 Zlmkn8  = ( hat_Zlmkn8 * exp(-I * B_inc) )

 For ell = 3, we have only
 0 <= |k| <= 10 (jmax = 7)
 
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
`hZ_3mkP0_5PNe10` stores  only the PN amplitudes that has the index
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
#include "hat_Zlmkn8_5PNe10/ell=3/hZ_3mkM5_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_3mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/
//static int lmax = 2;
//static int kmax = 6 + 2;

/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_3mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
CUDA_CALLABLE_MEMBER
cmplx hZ_3mkM5(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    cmplx hZ_3mkM5 = { 0.0 };

    double  Re_3mkM5 = 0.0;
    double  Im_3mkM5 = 0.0;

    // NULL check
    if (PN_orb == NULL) {

        //perror("Point errors: hZ_3mkM5");
        //exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */
    double q3 = PN_orb->PNq[3];
    double q4 = PN_orb->PNq[4];

    double v17 = PN_orb->PNv[9];
    double v18 = PN_orb->PNv[10];
    
    double e5 = PN_orb->PNe[5];
    double e7 = PN_orb->PNe[7];
    double e9 = PN_orb->PNe[9];

    double Y = PN_orb->PNY[1];
 
    double Yp = PN_orb->PNYp[1];
    double Yp2 = PN_orb->PNYp[2];
    double Yp3 = PN_orb->PNYp[3];

    double y3 = PN_orb->PNy[3];
    double y4 = PN_orb->PNy[4];
    double y5 = PN_orb->PNy[5];
    double y6 = PN_orb->PNy[6];
    double y7 = PN_orb->PNy[7];
    double y8 = PN_orb->PNy[8];
    double y9 = PN_orb->PNy[9];
    double y10 = PN_orb->PNy[10];


if (m == 3 && k == 4) { 

   // 1. Z_deg[3][3][4][-5]: 17, 
   Re_3mkM5  = 0.0e0; 
   Im_3mkM5  = 0.1059863152777511613547e-6 * v17 * Yp3 * y4 * (0.326989645e9 * e9 - 0.86916284e8 * e7 + 0.10238928e8 * e5) * q4; 

} else if (m == 3 && k == 3) { 

   // 2. Z_deg[3][3][3][-5]: 17, 
   Re_3mkM5  = 0.0e0; 
   Im_3mkM5  = 0.6047052530244444940817e-2 * y3 * (((-0.3e1 + Y) * e5 + (0.2451713662138691756479e2 - 0.8172378873795639188262e1 * Y) * e7 + (-0.8809323854374970217897e2 + 0.2936441284791656739299e2 * Y) * e9) * q4 * v18 + (0.6075987688223644216787e2 * q3 * e9 - 0.1691578473609391128168e2 * q3 * e7 + 0.2070405345131883819039e1 * q3 * e5) * v17) * Yp3; 

} else if (m == 2 && k == 5) { 

   // 3. Z_deg[3][2][5][-5]: 17, 
   Re_3mkM5  = 0.2596123921482355114012e-6 * v17 * y5 * Yp2 * (0.326989645e9 * e9 - 0.86916284e8 * e7 + 0.10238928e8 * e5) * q4; 
   Im_3mkM5  = 0.0e0; 

} else if (m == 2 && k == 4) { 

   // 4. Z_deg[3][2][4][-5]: 17, 
   Re_3mkM5  = 0.1481219314690483173535e-1 * y4 * Yp2 * (((-0.2e1 + Y) * e5 + (0.1634475774759127837652e2 - 0.8172378873795639188262e1 * Y) * e7 + (-0.5872882569583313478598e2 + 0.2936441284791656739299e2 * Y) * e9) * q4 * v18 + (0.6075987688223644216787e2 * q3 * e9 - 0.1691578473609391128168e2 * q3 * e7 + 0.2070405345131883819039e1 * q3 * e5) * v17); 
   Im_3mkM5  = 0.0e0; 

} else if (m == 1 && k == 6) { 

   // 5. Z_deg[3][1][6][-5]: 17, 
   Re_3mkM5  = 0.0e0; 
   Im_3mkM5  = -0.4104832339966189636454e-6 * v17 * Yp * y6 * (0.326989645e9 * e9 - 0.86916284e8 * e7 + 0.10238928e8 * e5) * q4; 

} else if (m == 1 && k == 5) { 

   // 6. Z_deg[3][1][5][-5]: 17, 
   Re_3mkM5  = 0.0e0; 
   Im_3mkM5  = 0.2004312665999116033425e-9 * v18 * y7 * (0.3431193585e10 * e9 - 0.954931880e9 * e7 + 0.116848704e9 * e5) * q4 - 0.6235639405330583215101e-9 * v17 * Yp * y5 * (0.2282050565e10 * e9 - 0.635331704e9 * e7 + 0.77761344e8 * e5) * q3; 

} else if (m == 0 && k == 7) { 

   // 7. Z_deg[3][0][7][-5]: 17, 
   Re_3mkM5  = -0.4739852112915522016321e-6 * y7 * (0.326989645e9 * e9 - 0.86916284e8 * e7 + 0.10238928e8 * e5) * v17 * q4; 
   Im_3mkM5  = -0.1011168450755311363482e-4 * y7 * v18 * (0.31584e5 * e5 - 0.331492e6 * e7 + 0.1573613e7 * e9) * q4; 

} else if (m == 0 && k == 6) { 

   // 8. Z_deg[3][0][6][-5]: 17, 
   Re_3mkM5  = -0.2447850706397877476209e-1 * y6 * ((0.6712598901276036188167e2 * e9 - 0.1868813497660702292893e2 * e7 + 0.2287331929895899957784e1 * e5) * q3 * v17 + (0.9999999999999999999999e0 * e5 - 0.8056483591500873265644e1 * e7 + 0.2841751412400533540198e2 * e9) * Y * q4 * v18); 
   Im_3mkM5  = 0.0e0; 

} else if (m == -1 && k == 8) { 

   // 9. Z_deg[3][-1][8][-5]: 17, 
   Re_3mkM5  = 0.0e0; 
   Im_3mkM5  = 0.4104832339966189636454e-6 * v17 * y8 * (0.326989645e9 * e9 - 0.86916284e8 * e7 + 0.10238928e8 * e5) / Yp * q4; 

} else if (m == -1 && k == 7) { 

   // 10. Z_deg[3][-1][7][-5]: 17, 
   Re_3mkM5  = 0.0e0; 
   Im_3mkM5  = 0.2004312665999116033425e-9 * v18 * y7 * (0.3431193585e10 * e9 - 0.954931880e9 * e7 + 0.116848704e9 * e5) * q4 + 0.6235639405330583215101e-9 * v17 * y7 * (0.2282050565e10 * e9 - 0.635331704e9 * e7 + 0.77761344e8 * e5) / Yp * q3; 

} else if (m == -2 && k == 9) { 

   // 11. Z_deg[3][-2][9][-5]: 17, 
   Re_3mkM5  = 0.2596123921482355114012e-6 * v17 * y9 * (0.326989645e9 * e9 - 0.86916284e8 * e7 + 0.10238928e8 * e5) / Yp2 * q4; 
   Im_3mkM5  = 0.0e0; 

} else if (m == -2 && k == 8) { 

   // 12. Z_deg[3][-2][8][-5]: 17, 
   Re_3mkM5  = 0.1481219314690483173535e-1 * y8 * (((0.2e1 + Y) * e5 + (-0.1634475774759127837652e2 - 0.8172378873795639188262e1 * Y) * e7 + (0.5872882569583313478598e2 + 0.2936441284791656739299e2 * Y) * e9) * q4 * v18 + (0.6075987688223644216787e2 * q3 * e9 - 0.1691578473609391128168e2 * q3 * e7 + 0.2070405345131883819039e1 * q3 * e5) * v17) / Yp2; 
   Im_3mkM5  = 0.0e0; 

} else if (m == -3 && k == 10) { 

   // 13. Z_deg[3][-3][10][-5]: 17, 
   Re_3mkM5  = 0.0e0; 
   Im_3mkM5  = -0.1059863152777511613547e-6 * v17 * y10 * (0.326989645e9 * e9 - 0.86916284e8 * e7 + 0.10238928e8 * e5) / Yp3 * q4; 

} else if (m == -3 && k == 9) { 

   // 14. Z_deg[3][-3][9][-5]: 17, 
   Re_3mkM5  = 0.0e0; 
   Im_3mkM5  = -0.6047052530244444940817e-2 * y9 * (((0.3e1 + Y) * e5 + (-0.2451713662138691756479e2 - 0.8172378873795639188262e1 * Y) * e7 + (0.8809323854374970217897e2 + 0.2936441284791656739299e2 * Y) * e9) * q4 * v18 + (0.6075987688223644216787e2 * q3 * e9 - 0.1691578473609391128168e2 * q3 * e7 + 0.2070405345131883819039e1 * q3 * e5) * v17) / Yp3; 

 } 

 else {

        //perror("Parameter errors: hZ_3mkM5");
        //exit(1);

    }

    hZ_3mkM5 = cmplx(Re_3mkM5, Im_3mkM5);
    return hZ_3mkM5;

}

