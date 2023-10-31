#ifndef hZ_7mkP9_5PNe10_H_ 
#define hZ_7mkP9_5PNe10_H_ 

/*
  Header of the PN Teukolsky amplitude Zlmkn8 (ell = 7, n = 9)

  25th May 2020; RF
  17th June. 2020; Sis

*/

//! \file hZ_7mkP9_5PNe10.h

#include "global.h"

// BHPC headers
#include "Zlmkn8_5PNe10.h"


// Define type


// Declare prototype 
CUDA_CALLABLE_MEMBER
cmplx hZ_7mkP9(const int m, const int k, inspiral_orb_PNvar* PN_orb);

#endif // hZ_7mkP9_5PNe10_H_ 