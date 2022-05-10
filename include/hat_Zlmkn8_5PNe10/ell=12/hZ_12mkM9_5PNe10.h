#ifndef hZ_12mkM9_5PNe10_H_ 
#define hZ_12mkM9_5PNe10_H_ 

/*
  Header of the PN Teukolsky amplitude Zlmkn8 (ell = 12, n = -9)

  25th May 2020; RF
  8th Nov 2020; Sis

*/

//! \file hZ_12mkM9_5PNe10.h

#include "global.h"

// BHPC headers
#include "Zlmkn8_5PNe10.h"


// Define type


// Declare prototype 
CUDA_CALLABLE_MEMBER
cmplx hZ_12mkM9(const int m, const int k, inspiral_orb_PNvar* PN_orb);

#endif // hZ_12mkM9_5PNe10_H_
