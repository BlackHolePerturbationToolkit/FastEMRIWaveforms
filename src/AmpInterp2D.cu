#include "stdio.h"
#include "global.h"
#include "AmpInterp2D.hh"

#define NUM_THREADS 256
__inline__ CUDA_CALLABLE_MEMBER
void fpbspl(const double* t, int k, double x, int l, double* h)
{
/*  subroutine fpbspl evaluates the (k+1) non-zero b-splines of
    degree k at t(l) <= x < t(l+1) using the stable recurrence
    relation of de boor and cox. */
  int i, j, li, lj;
  double f;
  double hh[5] = {0};
  h[0] = 1;
  for (j = 1; j <= k; j++) {
    for (i = 0; i < k; i++)
      hh[i] = h[i];
    h[0] = 0;
    for (i = 0; i < j; i++) {
      li = l + i;
      lj = li - j;
      f = hh[i] / (t[li] - t[lj]);
      h[i] += f * (t[li] - x);
      h[i + 1] = f * (x - t[lj]);
    }
  }
}

CUDA_CALLABLE_MEMBER
void fpbisp(
             double* z,
             const double* tx, int nx, const double* ty, int ny, double *c,
             int kx, int ky, const double x, int mx,
             const double y, int my)
{
  int i, i1, j, j1, kx1, ky1, l, l1, l2, m, nkx1, nky1;
  double arg, sp, tb, te;

    double wx[6] = {0.};
    double wy[6] = {0.};
    int lx, ly;

  //int* lx = new int[mx];
  //int* ly = new int[my];
  // mx * kx1 in size
  //wx = new double*[mx];
  kx1 = kx + 1;
  //wx[0] = new double[mx * kx1];
  //for (i = 1; i < mx; i++)
  //  wx[i] = &wx[0][i * kx1];
  nkx1 = nx - kx1;
  tb = tx[kx1 - 1];
  te = tx[nkx1];
  l = kx1;
  arg = x;
    if (arg < tb)
        arg = tb;
    if (arg > te)
        arg = te;
    while (!(arg < tx[l] || l == nkx1))
        l++;
    fpbspl(tx, kx, arg, l, wx);
    lx = l - kx1;
    //for (j = 0; j < kx1; ++j)
    //  wx[j] = h[j];

  ky1 = ky + 1;
  //wy = new double*[my];
  //wy[0] = new double[my * ky1];
  //for (i = 1; i < my; i++)
  //  wy[i] = &wy[0][i * ky1];
  nky1 = ny - ky1;
  tb = ty[ky1 - 1];
  te = ty[nky1];
  l = ky1;
  arg = y;
    if (arg < tb)
      arg = tb;
    if (arg > te)
      arg = te;
    while (!(arg < ty[l] || l == nky1))
      l++;
    fpbspl(ty, ky, arg, l, wy);
    //printf("%d %d %d %d %e %e %e %e %e %e %d\n", i, l, ky1, l - ky1, arg, y[i], tb, te, ty_shared[l], ty_shared[nky1], nky1);
    ly = l - ky1;
    //for (j = 0; j < ky1; ++j)
    //  wy[i * ky1 + j] = h[j];
    
  //m = 0;
  //for (i = 0; i < mx; i++) {
    l = lx * nky1;
    //for (i1 = 0; i1 < kx1; i1++)
      //h[i1] = wx[i][i1];
    //for (j = 0; j < my; j++) {
      l1 = l + ly;
      sp = 0;
      for (i1 = 0; i1 < kx1; i1++) {
        l2 = l1;
        for (j1 = 0; j1 < ky1; j1++)
          sp += c[l2++] * wx[i1] * wy[j1];
        l1 += nky1;
      }
      *z = sp;
    //}
  //}
  //delete [] wy[0];
  //delete [] wy;
  //delete [] wx[0];
  //delete [] wx;
  //delete [] ly;
  //delete [] lx;
}

CUDA_KERNEL 
void interp2D(double* z, const double* tx, int nx, const double* ty, int ny,
             double* c, int kx, int ky, const double* x, int mx,
             const double* y, int my, int num_indiv_c, int len_indiv_c)
{
    #ifdef __CUDACC__
    extern __shared__  unsigned char shared_mem[];
    double *shared_mem_in = (double*) shared_mem;
    #else
    double* shared_mem_in = new double[nx * ny * len_indiv_c];
    #endif

    double *tx_shared = &shared_mem_in[0];
    double *ty_shared = &shared_mem_in[nx];
    double *c_indiv = &ty_shared[ny];
    
    double z_temp;

    #ifdef __CUDACC__
    int start_block = blockIdx.y;
    int block_inc = gridDim.y;

    int start_thread = threadIdx.x;
    int thread_inc = blockDim.x;

    int full_loop_start = threadIdx.x + blockDim.x * blockIdx.x;
    int full_loop_inc = blockDim.x * gridDim.x;

    #else
    int start_block = 0;
    int block_inc = 1;

    int start_thread = 0;
    int thread_inc = 1;

    int full_loop_start = 0;
    int full_loop_inc = 1;
    #endif
    

    //double h[6] = {0.};
    // TODO: add omp?
    for (int c_i = start_block; c_i < num_indiv_c; c_i += block_inc)
    {
        for (int i = start_thread; i < nx; i += thread_inc)
        {
            tx_shared[i] = tx[i];
        }
        for (int i = start_thread; i < ny; i += thread_inc)
        {
            ty_shared[i] = ty[i];
        }
        for (int i = start_thread; i < len_indiv_c; i+= thread_inc)
        {
            c_indiv[i] = c[c_i * len_indiv_c + i];
        }
        CUDA_SYNC_THREADS;

        for (int i = full_loop_start; i < mx; i += full_loop_inc)
        {
            fpbisp(
                    &z_temp,
                    tx_shared, nx, ty_shared, ny, c_indiv,
                    kx, ky, x[i], 1,
                    y[i],1);

            z[c_i * mx + i] = z_temp;
            
        }
    }
    #ifdef __CUDACC__
    #else
    delete[] shared_mem_in;
    #endif
}

void interp2D_wrap(double* z, const double* tx, int nx, const double* ty, int ny, double* c,
             int kx, int ky, const double* x, int mx,
             const double* y, int my, int num_indiv_c, int len_indiv_c)
{

    if (mx != my)
    {
        throw std::invalid_argument("mx and my must be the same value.");
    }

    #ifdef __CUDACC__
    auto shared_memory_size = nx * sizeof(double) + ny * sizeof(double) + len_indiv_c * sizeof(double);

    gpuErrchk(cudaFuncSetAttribute(
        interp2D,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size));

    int num_blocks = std::ceil((mx + NUM_THREADS -1)/NUM_THREADS);
    dim3 grid(num_blocks, num_indiv_c);
    interp2D<<<grid, NUM_THREADS, shared_memory_size>>>(
        z,
        tx, nx, ty, ny, c,
        kx, ky, x, mx, y, my,
        num_indiv_c, len_indiv_c
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    #else
    interp2D(
        z,
        tx, nx, ty, ny, c,
        kx, ky, x, mx, y, my,
        num_indiv_c, len_indiv_c
    );

    #endif

}


