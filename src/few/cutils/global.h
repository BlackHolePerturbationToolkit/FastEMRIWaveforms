#ifndef _GLOBAL_HEADER_
#define _GLOBAL_HEADER_

#include <stdlib.h>
#include <stdio.h>
#include <complex>
#include "cuda_complex.hpp"

// Definitions needed for Mathematicas CForm output
#define Power(x, y)     (pow((double)(x), (double)(y)))
#define Sqrt(x)         (sqrt((double)(x)))

//#include "pdbParam.h" // LISA constants
#define MSUN_SI 1.98848e+30
#define YRSID_SI 31558149.763545603
#define AU_SI 149597870700.
#define C_SI 299792458.
#define G_SI 6.674080e-11
#define GMSUN 1.3271244210789466e+20
#define MTSUN_SI 4.925491025873693e-06
#define MRSUN_SI 1476.6250615036158
#define PC_SI 3.0856775814913674e+16

#define PI        3.141592653589793238462643383279502884
#define Pi        3.141592653589793238462643383279502884
#define TWOPI     6.283185307179586476925286766559005768
#define PI_2      1.570796326794896619231321691639751442
#define PI_4      0.785398163397448309615660845819875721
//#define MRSUN_SI  1.476625061404649406193430731479084713e+3
//#define MTSUN_SI 4.925491025543575903411922162094833998e-6
//#define MSUN_SI 1.988546954961461467461011951140572744e+30

#define GAMMA     0.577215664901532860606512090082402431
//#define PC_SI 3.085677581491367278913937957796471611e16 /**< Parsec, m */

#define  Gpc 3.0856775814913674e+25
#define  GPCINSEC 1.029271251e17

//#define YRSID_SI 31558149.763545600

#define F0 3.168753578687779e-08
#define Omega0 1.9909865927683788e-07

//#define ua 149597870700.
//#define R_SI 149597870700.
//#define AU_SI 149597870700.
//#define aorbit 149597870700.

//#define clight 299792458.0
#define sqrt3 1.7320508075688772
#define invsqrt3 0.5773502691896258
#define invsqrt6 0.4082482904638631
#define sqrt2 1.4142135623730951
#define L_SI 2.5e9
#define eorbit 0.004824185218078991
//#define C_SI 299792458.0
#define AUsec 499.004783836156412

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

#endif
