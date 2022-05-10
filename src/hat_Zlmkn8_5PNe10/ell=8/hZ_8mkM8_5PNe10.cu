/*

PN Teukolsky amplitude hat_Zlmkn8: (ell = 8, n = -8)

- BHPC's maple script `outputC_Slm_Zlmkn.mw`.


25th May 2020; RF
17th June. 2020; Sis


Convention (the phase `B_inc`  is defined in `Zlmkn8.c`):
 Zlmkn8  = ( hat_Zlmkn8 * exp(-I * B_inc) )

 For ell = 8, we have only
 0 <= |k| <= 18 (jmax = 10)
 
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
`hZ_8mkP0_5PNe10` stores  only the PN amplitudes that has the index
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
#include "hat_Zlmkn8_5PNe10/ell=8/hZ_8mkM8_5PNe10.h"

/*-*-*-*-*-*-*-*-*-*-*-* Global variables (but used only within hZ_8mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* Global functions (used only within hZ_8mkP0_5PNe10.c) *-*-*-*-*-*-*-*-*-*-*-*/


/*-*-*-*-*-*-*-*-*-*-*-* External functions (can be refered by other source files) *-*-*-*-*-*-*-*-*-*-*-*/
CUDA_CALLABLE_MEMBER
cmplx hZ_8mkM8(const int m, const int k, inspiral_orb_PNvar* PN_orb) { //

    cmplx hZ_8mkM8 = { 0.0 };

    double  Re_8mkM8 = 0.0;
    double  Im_8mkM8 = 0.0;

    // NULL check
    if (PN_orb == NULL) {

        //perror("Point errors: hZ_8mkM8");
        //exit(1);

    }

    /* Read out the PN variables from `inspiral_orb_PNvar`*/
    /* PNq, PNe, PNv, PNY, PNy */
    double Ym = PN_orb->PNYm;

    double q = PN_orb->PNq[1];
    double q2 = PN_orb->PNq[2];
    
    double v17 = PN_orb->PNv[9];
    double v18 = PN_orb->PNv[10];
    
    double e8 = PN_orb->PNe[8];
    double e10 = PN_orb->PNe[10];

    double Yp = PN_orb->PNYp[1];
    double Yp2 = PN_orb->PNYp[2];
    double Yp3 = PN_orb->PNYp[3];
    double Yp4 = PN_orb->PNYp[4];
    double Yp5 = PN_orb->PNYp[5];
    double Yp6 = PN_orb->PNYp[6];
    double Yp7 = PN_orb->PNYp[7];
    double Yp8 = PN_orb->PNYp[8];

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


if (m == 8 && k == 2) { 

   // 1. Z_deg[8][8][2][-8]: 18, 
   Re_8mkM8  = (-0.5880209115911986060182e-2 * e10 + 0.6040570200197838816889e-3 * e8) * v18 * Yp8 * q2 * Yp * Ym; 
   Im_8mkM8  = 0.0e0; 

} else if (m == 8 && k == 1) { 

   // 2. Z_deg[8][8][1][-8]: 17, 
   Re_8mkM8  = (0.8323334419141616456905e-6 * e10 - 0.8522081890439845244749e-7 * e8) * Yp8 * v18 * q * y; 
   Im_8mkM8  = -0.3698888389339641459872e-8 * Yp8 * e10 * v17 * y * q; 

} else if (m == 7 && k == 3) { 

   // 3. Z_deg[8][7][3][-8]: 18, 
   Re_8mkM8  = 0.0e0; 
   Im_8mkM8  = (-0.2352083646364794424073e-1 * e10 + 0.2416228080079135526756e-2 * e8) * v18 * y3 * Yp7 * q2; 

} else if (m == 7 && k == 2) { 

   // 4. Z_deg[8][7][2][-8]: 17, 
   Re_8mkM8  = 0.1479555355735856583949e-7 * Yp * Ym * Yp7 * e10 * v17 * q; 
   Im_8mkM8  = (0.3329333767656646582764e-5 * e10 - 0.3408832756175938097902e-6 * e8) * Yp7 * v18 * q * Yp * Ym; 

} else if (m == 6 && k == 4) { 

   // 5. Z_deg[8][6][4][-8]: 18, 
   Re_8mkM8  = (-0.6441446351265009554462e-1 * e10 + 0.6617113117683707107517e-2 * e8) * v18 * y4 * Yp6 * q2; 
   Im_8mkM8  = 0.0e0; 

} else if (m == 6 && k == 3) { 

   // 6. Z_deg[8][6][3][-8]: 17, 
   Re_8mkM8  = (-0.9117756030046044820820e-5 * e10 + 0.9335472976600345758589e-6 * e8) * y3 * Yp6 * v18 * q; 
   Im_8mkM8  = 0.4051929217070546067236e-7 * y3 * Yp6 * e10 * v17 * q; 

} else if (m == 5 && k == 5) { 

   // 7. Z_deg[8][5][5][-8]: 18, 
   Re_8mkM8  = 0.0e0; 
   Im_8mkM8  = (0.1391511450841798699985e0 * e10 - 0.1429459809591377380923e-1 * e8) * v18 * y5 * Yp5 * q2; 

} else if (m == 5 && k == 4) { 

   // 8. Z_deg[8][5][4][-8]: 17, 
   Re_8mkM8  = 0.8753167528045661673781e-7 * y4 * Yp5 * e10 * v17 * q; 
   Im_8mkM8  = (0.1969660419402436118817e-4 * e10 - 0.2016692655278021020098e-5 * e8) * y4 * Yp5 * v18 * q; 

} else if (m == 4 && k == 6) { 

   // 9. Z_deg[8][4][6][-8]: 18, 
   Re_8mkM8  = (0.2508582943202696770234e0 * e10 - 0.2576995319848350995695e-1 * e8) * v18 * y6 * Yp4 * q2; 
   Im_8mkM8  = 0.0e0; 

} else if (m == 4 && k == 5) { 

   // 10. Z_deg[8][4][5][-8]: 17, 
   Re_8mkM8  = (0.3550855818703694815996e-4 * e10 - 0.3635644387728263984028e-5 * e8) * y5 * Yp4 * v18 * q; 
   Im_8mkM8  = -0.1577999717254750486001e-6 * y5 * Yp4 * e10 * v17 * q; 

} else if (m == 3 && k == 7) { 

   // 11. Z_deg[8][3][7][-8]: 18, 
   Re_8mkM8  = 0.0e0; 
   Im_8mkM8  = (-0.3886279984641612381479e0 * e10 + 0.3992263982810847597702e-1 * e8) * v18 * Yp3 * y7 * q2; 

} else if (m == 3 && k == 6) { 

   // 12. Z_deg[8][3][6][-8]: 17, 
   Re_8mkM8  = -0.2444626650099064503134e-6 * Yp3 * y6 * e10 * v17 * q; 
   Im_8mkM8  = (-0.5500962180249245113864e-4 * e10 + 0.5632316066561610887263e-5 * e8) * Yp3 * y6 * v18 * q; 

} else if (m == 2 && k == 8) { 

   // 13. Z_deg[8][2][8][-8]: 18, 
   Re_8mkM8  = (-0.5262047974399418197249e0 * e10 + 0.5405550986300040565513e-1 * e8) * v18 * y8 * Yp2 * q2; 
   Im_8mkM8  = 0.0e0; 

} else if (m == 2 && k == 7) { 

   // 14. Z_deg[8][2][7][-8]: 17, 
   Re_8mkM8  = (-0.7448338002465805248009e-4 * e10 + 0.7626192005299112806023e-5 * e8) * y7 * Yp2 * v18 * q; 
   Im_8mkM8  = 0.3310040131733559345586e-6 * y7 * Yp2 * e10 * v17 * q; 

} else if (m == 1 && k == 9) { 

   // 15. Z_deg[8][1][9][-8]: 18, 
   Re_8mkM8  = 0.0e0; 
   Im_8mkM8  = (0.6289350282692279598156e0 * e10 - 0.6460869188041557412107e-1 * e8) * v18 * y9 * q2 * Yp; 

} else if (m == 1 && k == 8) { 

   // 16. Z_deg[8][1][8][-8]: 17, 
   Re_8mkM8  = 0.3956254663492935263337e-6 * Yp * y8 * q * e10 * v17; 
   Im_8mkM8  = (0.8902466672539719833005e-4 * e10 - 0.9115042865010729331829e-5 * e8) * y8 * v18 * q * Yp; 

} else if (m == 0 && k == 10) { 

   // 17. Z_deg[8][0][10][-8]: 18, 
   Re_8mkM8  = (0.6670863351223860848295e0 * e10 - 0.6852786622835112762612e-1 * e8) * v18 * y10 * q2; 
   Im_8mkM8  = 0.0e0; 

} else if (m == 0 && k == 9) { 

   // 18. Z_deg[8][0][9][-8]: 17, 
   Re_8mkM8  = (0.5961032335348564796171e-4 * e10 - 0.6286251570833158315266e-5 * e8) * y9 * v18 * q; 
   Im_8mkM8  = -0.4196241750985135775511e-6 * e10 * y9 * q * v17; 

} else if (m == -1 && k == 11) { 

   // 19. Z_deg[8][-1][11][-8]: 18, 
   Re_8mkM8  = 0.0e0; 
   Im_8mkM8  = (-0.6289350282692279598156e0 * e10 + 0.6460869188041557412107e-1 * e8) * v18 * y11 * q2 / Yp; 

} else if (m == -1 && k == 10) { 

   // 20. Z_deg[8][-1][10][-8]: 17, 
   Re_8mkM8  = -0.3956254663492935263337e-6 / Yp * y10 * q * e10 * v17; 
   Im_8mkM8  = (-0.8902466672539719833005e-4 * e10 + 0.9115042865010729331829e-5 * e8) * y10 * v18 * q / Yp; 

} else if (m == -2 && k == 12) { 

   // 21. Z_deg[8][-2][12][-8]: 18, 
   Re_8mkM8  = (-0.5262047974399418197249e0 * e10 + 0.5405550986300040565513e-1 * e8) * v18 * y12 / Yp2 * q2; 
   Im_8mkM8  = 0.0e0; 

} else if (m == -2 && k == 11) { 

   // 22. Z_deg[8][-2][11][-8]: 17, 
   Re_8mkM8  = (-0.7448338002465805248009e-4 * e10 + 0.7626192005299112806023e-5 * e8) * y11 * v18 * q / Yp2; 
   Im_8mkM8  = 0.3310040131733559345586e-6 * y11 * e10 * v17 / Yp2 * q; 

} else if (m == -3 && k == 13) { 

   // 23. Z_deg[8][-3][13][-8]: 18, 
   Re_8mkM8  = 0.0e0; 
   Im_8mkM8  = (0.3886279984641612381479e0 * e10 - 0.3992263982810847597702e-1 * e8) * v18 * y13 / Yp3 * q2; 

} else if (m == -3 && k == 12) { 

   // 24. Z_deg[8][-3][12][-8]: 17, 
   Re_8mkM8  = 0.2444626650099064503134e-6 * y12 * e10 * v17 / Yp3 * q; 
   Im_8mkM8  = (0.5500962180249245113864e-4 * e10 - 0.5632316066561610887263e-5 * e8) * y12 * v18 * q / Yp3; 

} else if (m == -4 && k == 14) { 

   // 25. Z_deg[8][-4][14][-8]: 18, 
   Re_8mkM8  = (0.2508582943202696770234e0 * e10 - 0.2576995319848350995695e-1 * e8) * v18 * y14 / Yp4 * q2; 
   Im_8mkM8  = 0.0e0; 

} else if (m == -4 && k == 13) { 

   // 26. Z_deg[8][-4][13][-8]: 17, 
   Re_8mkM8  = (0.3550855818703694815996e-4 * e10 - 0.3635644387728263984028e-5 * e8) * y13 * v18 * q / Yp4; 
   Im_8mkM8  = -0.1577999717254750486001e-6 * y13 * e10 * v17 / Yp4 * q; 

} else if (m == -5 && k == 15) { 

   // 27. Z_deg[8][-5][15][-8]: 18, 
   Re_8mkM8  = 0.0e0; 
   Im_8mkM8  = (-0.1391511450841798699985e0 * e10 + 0.1429459809591377380923e-1 * e8) * v18 * y15 / Yp5 * q2; 

} else if (m == -5 && k == 14) { 

   // 28. Z_deg[8][-5][14][-8]: 17, 
   Re_8mkM8  = -0.8753167528045661673781e-7 * y14 * e10 * v17 / Yp5 * q; 
   Im_8mkM8  = (-0.1969660419402436118817e-4 * e10 + 0.2016692655278021020098e-5 * e8) * y14 * v18 * q / Yp5; 

} else if (m == -6 && k == 16) { 

   // 29. Z_deg[8][-6][16][-8]: 18, 
   Re_8mkM8  = (-0.6441446351265009554462e-1 * e10 + 0.6617113117683707107517e-2 * e8) * v18 * y16 / Yp6 * q2; 
   Im_8mkM8  = 0.0e0; 

} else if (m == -6 && k == 15) { 

   // 30. Z_deg[8][-6][15][-8]: 17, 
   Re_8mkM8  = (-0.9117756030046044820820e-5 * e10 + 0.9335472976600345758589e-6 * e8) * y15 * v18 * q / Yp6; 
   Im_8mkM8  = 0.4051929217070546067236e-7 * y15 * e10 * v17 / Yp6 * q; 

} else if (m == -7 && k == 17) { 

   // 31. Z_deg[8][-7][17][-8]: 18, 
   Re_8mkM8  = 0.0e0; 
   Im_8mkM8  = (0.2352083646364794424073e-1 * e10 - 0.2416228080079135526756e-2 * e8) * v18 * y17 / Yp7 * q2; 

} else if (m == -7 && k == 16) { 

   // 32. Z_deg[8][-7][16][-8]: 17, 
   Re_8mkM8  = 0.1479555355735856583949e-7 * y16 * e10 * v17 / Yp7 * q; 
   Im_8mkM8  = (0.3329333767656646582764e-5 * e10 - 0.3408832756175938097902e-6 * e8) * y16 * v18 * q / Yp7; 

} else if (m == -8 && k == 18) { 

   // 33. Z_deg[8][-8][18][-8]: 18, 
   Re_8mkM8  = (0.5880209115911986060182e-2 * e10 - 0.6040570200197838816889e-3 * e8) * v18 * y18 / Yp8 * q2; 
   Im_8mkM8  = 0.0e0; 

} else if (m == -8 && k == 17) { 

   // 34. Z_deg[8][-8][17][-8]: 17, 
   Re_8mkM8  = (0.8323334419141616456905e-6 * e10 - 0.8522081890439845244749e-7 * e8) * y17 * v18 * q / Yp8; 
   Im_8mkM8  = -0.3698888389339641459872e-8 * y17 * e10 * v17 / Yp8 * q; 

 } 

 else {

        //perror("Parameter errors: hZ_8mkM8");
        //exit(1);

    }

    hZ_8mkM8 = cmplx(Re_8mkM8, Im_8mkM8);
    return hZ_8mkM8;

}

