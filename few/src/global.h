#ifndef _GLOBAL_HEADER_
#define _GLOBAL_HEADER_

#include <stdlib.h>
#include <complex>

#define PI_2 1.57079632679
#define PI 3.141592653589793

typedef float fod;
typedef std::complex<float> cmplx;

#ifdef __CUDACC__
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif

#endif
