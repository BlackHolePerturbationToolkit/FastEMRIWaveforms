#ifndef _GLOBAL_HEADER_
#define _GLOBAL_HEADER_

#if defined(_MSC_VER)
    #define _USE_MATH_DEFINES
    #define FEW_INLINE __inline
#else
    #define FEW_INLINE __inline__
#endif

#include <stdlib.h>
#include <stdio.h>
#include <complex>
#include "cuda_complex.hpp"
#include <cmath>

// Definitions needed for Mathematicas CForm output
#define Power(x, y)     (pow((double)(x), (double)(y)))
#define Sqrt(x)         (sqrt((double)(x)))


// Constants from lisaconstants, via the generated lisaconstants_values.h
// (regenerate with python -m lisatools.utils.lisaconstants_header, or the
// generator in tests/test_lisaconstants_values.py) -- all in seconds.
#include "lisaconstants_values.h"
#define YRSID_SI LISACONSTANTS_ASTRONOMICAL_YEAR
#define MTSUN_SI LISACONSTANTS_MTSUN
#define GPCINSEC LISACONSTANTS_GPC_LIGHT_TIME
#define AUsec LISACONSTANTS_AU_LIGHT_TIME

typedef double fod;
typedef gcmplx::complex<double> cmplx;

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_KERNEL __global__
#define CUDA_SHARED __shared__
#define CUDA_SYNC_THREADS __syncthreads();
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_KERNEL
#define CUDA_SHARED
#define CUDA_SYNC_THREADS
#endif

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

typedef struct tagInterpContainer{
  fod *y;
  fod *c1;
  fod *c2;
  fod *c3;
  int length;

} InterpContainer;

typedef struct tagFilterContainer {

    int *d_mode_keep_inds;
    int *d_filter_modes_buffer;
    fod *working_modes_all;
    int *ind_working_modes_all;
    int *d_num_modes_kept;
    int num_modes_kept;
    fod tol;

} FilterContainer;

typedef struct tagModeReImContainer {
    fod *re_y;
    fod *re_c1;
    fod *re_c2;
    fod *re_c3;

    fod *im_y;
    fod *im_c1;
    fod *im_c2;
    fod *im_c3;

} ModeReImContainer;


typedef void (*fptr)(double *, double *, double *, double *, double *, double *, double, double, double, double, double);

#endif  // _GLOBAL_HEADER_
