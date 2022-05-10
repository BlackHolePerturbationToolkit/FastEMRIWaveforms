#ifndef hZ_5mkM4_5PNe10_H_ 
#define hZ_5mkM4_5PNe10_H_ 

/*
  Header of the PN Teukolsky amplitude Zlmkn8 (ell = 5, n = -4)

  25th May 2020; RF
  17th June. 2020; Sis

*/

//! \file hZ_5mkM4_5PNe10.h

#include "global.h"

// BHPC headers
#include "Zlmkn8_5PNe10.h"


// Define type


// Declare prototype 
CUDA_CALLABLE_MEMBER
cmplx hZ_5mkM4(const int m, const int k, inspiral_orb_PNvar* PN_orb);

#endif // hZ_5mkM4_5PNe10_H_ 