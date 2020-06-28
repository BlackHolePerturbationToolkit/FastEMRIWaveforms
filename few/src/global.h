#ifndef _GLOBAL_HEADER_
#define _GLOBAL_HEADER_

#include <stdlib.h>
#include <complex>
#include "cuda_complex.hpp"

#define PI_2 1.57079632679
#define PI 3.141592653589793

typedef double fod;
typedef gcmplx::complex<double> cmplx;

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_KERNEL __global__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_KERNEL
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

#endif
