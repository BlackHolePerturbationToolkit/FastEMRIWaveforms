#ifndef hZ_2mkM4_5PNe10_H_ 
#define hZ_2mkM4_5PNe10_H_ 

/*
  Header of the PN Teukolsky amplitude Zlmkn8 (ell = 2, n = -4)

  25th May 2020; RF
  6th June. 2020; Sis

*/

//! \file Z2mkM4_5PNe10.h

#include "global.h"

// BHPC headers
#include "Zlmkn8_5PNe10.h"


// Define type


// Declare prototype 
CUDA_CALLABLE_MEMBER
cmplx hZ_2mkM4(const int m, const int k, inspiral_orb_PNvar* PN_orb);

#endif // hZ_2mkM4_5PNe10_H_ 