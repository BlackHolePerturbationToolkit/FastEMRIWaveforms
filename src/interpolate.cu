// Interpolate and sum modes for an EMRI waveform

// Copyright (C) 2020 Michael L. Katz, Alvin J.K. Chua, Niels Warburton, Scott A. Hughes
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "global.h"
#include "interpolate.hh"
#include <stdio.h>

// adjust imports based on CUDA or not
#ifdef __CUDACC__
#include "cusparse.h"
#else
#include "lapacke.h"
#endif
#ifdef __USE_OMP__
#include "omp.h"
#endif

#ifdef __CUDACC__
#define MAX_MODES_BLOCK 450
#else
#define MAX_MODES_BLOCK 5000
#endif

#define NUM_TERMS 4

// fills the coefficients of the cubic spline
// according to scipy Cubic Spline
CUDA_CALLABLE_MEMBER
void fill_coefficients(int i, int length, double *dydx, double dx, double *y, double *coeff1, double *coeff2, double *coeff3)
{
  double slope, t, dydx_i;

  slope = (y[i + 1] - y[i]) / dx;

  dydx_i = dydx[i];

  t = (dydx_i + dydx[i + 1] - 2 * slope) / dx;

  coeff1[i] = dydx_i;
  coeff2[i] = (slope - dydx_i) / dx - t;
  coeff3[i] = t / dx;
}

// fills the banded matrix that will be solved for spline coefficients
// according to scipy Cubic Spline
// this performs a not-a-knot spline
CUDA_CALLABLE_MEMBER
void prep_splines(int i, int length, double *b, double *ud, double *diag, double *ld, double *x, double *y)
{
  double dx1, dx2, d, slope1, slope2;

  // this performs a not-a-knot spline
  // need to adjust for ends of the splines
  if (i == length - 1)
  {
    dx1 = x[length - 2] - x[length - 3];
    dx2 = x[length - 1] - x[length - 2];
    d = x[length - 1] - x[length - 3];

    slope1 = (y[length - 2] - y[length - 3]) / dx1;
    slope2 = (y[length - 1] - y[length - 2]) / dx2;

    b[length - 1] = ((dx2 * dx2 * slope1 +
                      (2 * d + dx2) * dx1 * slope2) /
                     d);
    diag[length - 1] = dx1;
    ld[length - 1] = d;
    ud[length - 1] = 0.0;
  }

  else if (i == 0)
  {
    dx1 = x[1] - x[0];
    dx2 = x[2] - x[1];
    d = x[2] - x[0];

    slope1 = (y[1] - y[0]) / dx1;
    slope2 = (y[2] - y[1]) / dx2;

    b[0] = ((dx1 + 2 * d) * dx2 * slope1 +
            dx1 * dx1 * slope2) /
           d;
    diag[0] = dx2;
    ud[0] = d;
    ld[0] = 0.0;
  }

  else
  {
    dx1 = x[i] - x[i - 1];
    dx2 = x[i + 1] - x[i];

    slope1 = (y[i] - y[i - 1]) / dx1;
    slope2 = (y[i + 1] - y[i]) / dx2;

    b[i] = 3.0 * (dx2 * slope1 + dx1 * slope2);
    diag[i] = 2 * (dx1 + dx2);
    ud[i] = dx1;
    ld[i] = dx2;
  }
}

// wrapper to fill the banded matrix that will be solved for spline coefficients
// according to scipy Cubic Spline
CUDA_KERNEL
void fill_B(double *t_arr, double *y_all, double *B, double *upper_diag, double *diag, double *lower_diag,
            int ninterps, int length)
{

#ifdef __CUDACC__

  int start1 = blockIdx.y * blockDim.y + threadIdx.y;
  int end1 = ninterps;
  int diff1 = blockDim.y * gridDim.y;

  int start2 = blockIdx.x * blockDim.x + threadIdx.x;
  int end2 = length;
  int diff2 = blockDim.x * gridDim.x;
#else

  int start1 = 0;
  int end1 = ninterps;
  int diff1 = 1;

  int start2 = 0;
  int end2 = length;
  int diff2 = 1;

#ifdef __USE_OMP__
  #pragma omp parallel for
#endif  // __USE_OMP__
#endif // __CUDACC__
for (int interp_i = start1;
       interp_i < end1; // 2 for re and im
       interp_i += diff1)
  {

    for (int i = start2;
         i < end2;
         i += diff2)
    {

      int lead_ind = interp_i * length;
      prep_splines(i, length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], &t_arr[lead_ind], &y_all[interp_i * length]);
    }
  }
}

// wrapper to set spline coefficients
// according to scipy Cubic Spline
CUDA_KERNEL
void set_spline_constants(double *t_arr, double *interp_array, double *B,
                          int ninterps, int length)
{

  double dt;
  InterpContainer mode_vals;

#ifdef __CUDACC__
  int start1 = blockIdx.y * blockDim.y + threadIdx.y;
  int end1 = ninterps;
  int diff1 = blockDim.y * gridDim.y;

  int start2 = blockIdx.x * blockDim.x + threadIdx.x;
  int end2 = length - 1;
  int diff2 = blockDim.x * gridDim.x;
#else

  int start1 = 0;
  int end1 = ninterps;
  int diff1 = 1;

  int start2 = 0;
  int end2 = length - 1;
  int diff2 = 1;

#ifdef __USE_OMP__
  #pragma omp parallel for
#endif  // __USE_OMP__
#endif // __CUDACC__

  for (int interp_i = start1;
       interp_i < end1; // 2 for re and im
       interp_i += diff1)
  {

    for (int i = start2;
         i < end2;
         i += diff2)
    {

      dt = t_arr[interp_i * length + i + 1] - t_arr[interp_i * length + i];

      int lead_ind = interp_i * length;
      fill_coefficients(i, length, &B[lead_ind], dt,
                        &interp_array[0 * ninterps * length + lead_ind],
                        &interp_array[1 * ninterps * length + lead_ind],
                        &interp_array[2 * ninterps * length + lead_ind],
                        &interp_array[3 * ninterps * length + lead_ind]);
    }
  }
}

// wrapper for cusparse solution for coefficients from banded matrix
void fit_wrap(int m, int n, double *a, double *b, double *c, double *d_in)
{
#ifdef __CUDACC__
  size_t bufferSizeInBytes;

  cusparseHandle_t handle;
  void *pBuffer;

  CUSPARSE_CALL(cusparseCreate(&handle));
  CUSPARSE_CALL(cusparseDgtsv2StridedBatch_bufferSizeExt(handle, m, a, b, c, d_in, n, m, &bufferSizeInBytes));
  gpuErrchk(cudaMalloc(&pBuffer, bufferSizeInBytes));

  // solve banded matrix problem
  CUSPARSE_CALL(cusparseDgtsv2StridedBatch(handle,
                                           m,
                                           a, // dl
                                           b, // diag
                                           c, // du
                                           d_in,
                                           n,
                                           m,
                                           pBuffer));

  CUSPARSE_CALL(cusparseDestroy(handle));
  gpuErrchk(cudaFree(pBuffer));

#else

// use lapack on CPU
#ifdef __USE_OMP__
  #pragma omp parallel for
#endif  // __USE_OMP__
  for (int j = 0;
       j < n;
       j += 1)
  {
    int info = LAPACKE_dgtsv(LAPACK_COL_MAJOR, m, 1, &a[j * m + 1], &b[j * m], &c[j * m], &d_in[j * m], m);
  }

#endif // __CUDACC__
}

// interpolate many y arrays (interp_array) with a singular x array (t_arr)
// see python documentation for shape necessary for this to be done
void interpolate_arrays(double *t_arr, double *interp_array, int ninterps, int length, double *B, double *upper_diag, double *diag, double *lower_diag)
{

  // need to fill the banded matrix
  // solve it
  // fill the coefficient arrays
  // do that below on GPU or CPU

#ifdef __CUDACC__
  int NUM_THREADS = 64;
  int num_blocks = std::ceil((length + NUM_THREADS - 1) / NUM_THREADS);
  dim3 gridDim(num_blocks); //, num_teuk_modes);
  fill_B<<<gridDim, NUM_THREADS>>>(t_arr, interp_array, B, upper_diag, diag, lower_diag, ninterps, length);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  fit_wrap(length, ninterps, lower_diag, diag, upper_diag, B);

  set_spline_constants<<<gridDim, NUM_THREADS>>>(t_arr, interp_array, B,
                                                 ninterps, length);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

#else

  fill_B(t_arr, interp_array, B, upper_diag, diag, lower_diag, ninterps, length);

  fit_wrap(length, ninterps, lower_diag, diag, upper_diag, B);

  set_spline_constants(t_arr, interp_array, B,
                       ninterps, length);

#endif
}

/////////////////////////////////
/////////
/////////  MODE SUMMATION
/////////
/////////////////////////////////

// build mode value with specific phase and amplitude values; mode indexes; and spherical harmonics
CUDA_CALLABLE_MEMBER
cmplx get_mode_value(cmplx teuk_mode, fod Phi_phi, fod Phi_r, int m, int n, cmplx Ylm)
{
  cmplx minus_I(0.0, -1.0);
  fod phase = m * Phi_phi + n * Phi_r;
  cmplx out = (teuk_mode * Ylm) * gcmplx::exp(minus_I * phase);
  return out;
}

// Add functionality for proper summation in the kernel
#ifdef __CUDACC__
__device__ double atomicAddDouble(double *address, double val)
{
  unsigned long long *address_as_ull =
      (unsigned long long *)address;
  unsigned long long old = *address_as_ull, assumed;

  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

// Add functionality for proper summation in the kernel
__device__ void atomicAddComplex(cmplx *a, cmplx b)
{
  // transform the addresses of real and imag. parts to double pointers
  double *x = (double *)a;
  double *y = x + 1;
  // use atomicAdd for double variables
  atomicAddDouble(x, b.real());
  atomicAddDouble(y, b.imag());
}

#endif

// make a waveform in parallel
// this uses an efficient summation by loading mode information into shared memory
// shared memory is leveraged heavily
CUDA_KERNEL
void make_waveform(cmplx *waveform,
                   double *interp_array,
                   int *m_arr_in, int *n_arr_in, int num_teuk_modes, cmplx *Ylms_in,
                   double delta_t, double start_t, int old_ind, int start_ind, int end_ind, int init_length)
{

  int num_pars = 2;
  cmplx trans(0.0, 0.0);
  cmplx trans2(0.0, 0.0);

  cmplx complexI(0.0, 1.0);
  cmplx mode_val;
  cmplx trans_plus_m(0.0, 0.0), trans_minus_m(0.0, 0.0);
  double Phi_phi_i, Phi_r_i, t, x, x2, x3, mode_val_re, mode_val_im;
  int lm_i, num_teuk_here;
  double re_y, re_c1, re_c2, re_c3, im_y, im_c1, im_c2, im_c3;
  CUDA_SHARED double pp_y, pp_c1, pp_c2, pp_c3, pr_y, pr_c1, pr_c2, pr_c3;

  // declare all the shared memory
  // MAX_MODES_BLOCK is fixed based on shared memory
  CUDA_SHARED cmplx Ylms[2 * MAX_MODES_BLOCK];
  CUDA_SHARED double mode_re_y[MAX_MODES_BLOCK];
  CUDA_SHARED double mode_re_c1[MAX_MODES_BLOCK];
  CUDA_SHARED double mode_re_c2[MAX_MODES_BLOCK];
  CUDA_SHARED double mode_re_c3[MAX_MODES_BLOCK];

  CUDA_SHARED double mode_im_y[MAX_MODES_BLOCK];
  CUDA_SHARED double mode_im_c1[MAX_MODES_BLOCK];
  CUDA_SHARED double mode_im_c2[MAX_MODES_BLOCK];
  CUDA_SHARED double mode_im_c3[MAX_MODES_BLOCK];

  CUDA_SHARED int m_arr[MAX_MODES_BLOCK];
  CUDA_SHARED int n_arr[MAX_MODES_BLOCK];

  // number of splines
  int num_base = init_length * (2 * num_teuk_modes + num_pars);

  CUDA_SYNC_THREADS;

#ifdef __CUDACC__

  if ((threadIdx.x == 0))
  {
#else
  if (true)
  {
#endif

    // fill phase values. These will be same for all modes
    int ind_Phi_phi = old_ind * (2 * num_teuk_modes + num_pars) + num_teuk_modes * 2 + 0;
    int ind_Phi_r = old_ind * (2 * num_teuk_modes + num_pars) + num_teuk_modes * 2 + 1;

    pp_y = interp_array[0 * num_base + ind_Phi_phi];
    pp_c1 = interp_array[1 * num_base + ind_Phi_phi];
    pp_c2 = interp_array[2 * num_base + ind_Phi_phi];
    pp_c3 = interp_array[3 * num_base + ind_Phi_phi];

    pr_y = interp_array[0 * num_base + ind_Phi_r];
    pr_c1 = interp_array[1 * num_base + ind_Phi_r];
    pr_c2 = interp_array[2 * num_base + ind_Phi_r];
    pr_c3 = interp_array[3 * num_base + ind_Phi_r];
  }

  CUDA_SYNC_THREADS;

  int m, n, actual_mode_index;
  cmplx Ylm_plus_m, Ylm_minus_m;

  int num_breaks = (num_teuk_modes / MAX_MODES_BLOCK) + 1;

  // this does a special loop to fill mode information into shared memory in chunks
  for (int block_y = 0; block_y < num_breaks; block_y += 1)
  {
    num_teuk_here = (((block_y + 1) * MAX_MODES_BLOCK) <= num_teuk_modes) ? MAX_MODES_BLOCK : num_teuk_modes - (block_y * MAX_MODES_BLOCK);

    int init_ind = block_y * MAX_MODES_BLOCK;

#ifdef __CUDACC__

    int start = threadIdx.x;
    int end = num_teuk_here;
    int diff = blockDim.x;

#else

    int start = 0;
    int end = num_teuk_here;
    int diff = 1;
#ifdef __USE_OMP__
    #pragma omp parallel for
#endif  // __USE_OMP__
#endif // __CUDACC__
    for (int i = start; i < end; i += diff)
    {

      // fill mode values and Ylms
      int ind_re = old_ind * (2 * num_teuk_modes + num_pars) + (init_ind + i);
      int ind_im = old_ind * (2 * num_teuk_modes + num_pars) + num_teuk_modes + (init_ind + i);
      mode_re_y[i] = interp_array[0 * num_base + ind_re];
      mode_re_c1[i] = interp_array[1 * num_base + ind_re];
      mode_re_c2[i] = interp_array[2 * num_base + ind_re];
      mode_re_c3[i] = interp_array[3 * num_base + ind_re];

      mode_im_y[i] = interp_array[0 * num_base + ind_im];
      mode_im_c1[i] = interp_array[1 * num_base + ind_im];
      mode_im_c2[i] = interp_array[2 * num_base + ind_im];
      mode_im_c3[i] = interp_array[3 * num_base + ind_im];

      m_arr[i] = m_arr_in[init_ind + i];
      n_arr[i] = n_arr_in[init_ind + i];
      Ylms[2 * i] = Ylms_in[(init_ind + i)];
      Ylms[2 * i + 1] = Ylms_in[num_teuk_modes + (init_ind + i)];
    }

    CUDA_SYNC_THREADS;

#ifdef __CUDACC__

    start = start_ind + blockIdx.x * blockDim.x + threadIdx.x;
    end = end_ind;
    diff = blockDim.x * gridDim.x;

#else

    start = start_ind;
    end = end_ind;
    diff = 1;

#endif
#ifdef __CUDACC__
#else
#ifdef __USE_OMP__
    #pragma omp parallel for
#endif // __USE_OMP__
#endif // __CUDACC__

    // start and end is the start and end of points in this interpolation window
    for (int i = start;
         i < end;
         i += diff)
    {

      trans2 = 0.0 + 0.0 * complexI;

      trans = 0.0 + 0.0 * complexI;

      // determine interpolation information
      t = delta_t * i;
      x = t - start_t;
      x2 = x * x;
      x3 = x * x2;

      // get phases at this timestep
      Phi_phi_i = pp_y + pp_c1 * x + pp_c2 * x2 + pp_c3 * x3;
      Phi_r_i = pr_y + pr_c1 * x + pr_c2 * x2 + pr_c3 * x3;

      // calculate all modes at this timestep
      for (int j = 0; j < num_teuk_here; j += 1)
      {

        Ylm_plus_m = Ylms[2 * j];

        m = m_arr[j];
        n = n_arr[j];

        mode_val_re = mode_re_y[j] + mode_re_c1[j] * x + mode_re_c2[j] * x2 + mode_re_c3[j] * x3;
        mode_val_im = mode_im_y[j] + mode_im_c1[j] * x + mode_im_c2[j] * x2 + mode_im_c3[j] * x3;
        mode_val = mode_val_re + complexI * mode_val_im;

        trans_plus_m = get_mode_value(mode_val, Phi_phi_i, Phi_r_i, m, n, Ylm_plus_m);

        // minus m if m > 0
        // mode values for +/- m are taking care of when applying
        // specific mode selection by setting ylms to zero for the opposites
        if (m != 0)
        {

          Ylm_minus_m = Ylms[2 * j + 1];
          trans_minus_m = get_mode_value(gcmplx::conj(mode_val), Phi_phi_i, Phi_r_i, -m, -n, Ylm_minus_m);
        }
        else
          trans_minus_m = 0.0 + 0.0 * complexI;

        trans = trans + trans_minus_m + trans_plus_m;
      }

// fill waveform
#ifdef __CUDACC__
      atomicAddComplex(&waveform[i], trans);
#else
      waveform[i] += trans;
#endif
    }
    CUDA_SYNC_THREADS;
  }
}

// with uneven spacing in t in the sparse arrays, need to determine which timesteps the dense arrays fall into
// for interpolation
// effectively the boundaries and length of each interpolation segment of the dense array in the sparse array
void find_start_inds(int start_inds[], int unit_length[], double *t_arr, double delta_t, int *length, int new_length)
{

  double T = (new_length - 1) * delta_t;
  start_inds[0] = 0;
  int i = 1;
  for (i = 1;
       i < *length;
       i += 1)
  {

    double t = t_arr[i];

    // adjust for waveforms that hit the end of the trajectory
    if (t < T)
    {
      start_inds[i] = (int)std::ceil(t / delta_t);
      unit_length[i - 1] = start_inds[i] - start_inds[i - 1];
    }
    else
    {
      start_inds[i] = new_length;
      unit_length[i - 1] = new_length - start_inds[i - 1];
      break;
    }
  }

  // fixes for not using certain segments for the interpolation
  *length = i + 1;
}

// function for building interpolated EMRI waveform from python
void get_waveform(cmplx *d_waveform, double *interp_array,
                  int *d_m, int *d_n, int init_len, int out_len, int num_teuk_modes, cmplx *d_Ylms,
                  double delta_t, double *h_t, int dev)
{

  // arrays for determining spline windows for new arrays
  int start_inds[init_len];
  int unit_length[init_len - 1];

  int number_of_old_spline_points = init_len;

  // find the spline window information based on equally spaced new array
  find_start_inds(start_inds, unit_length, h_t, delta_t, &number_of_old_spline_points, out_len);

#ifdef __CUDACC__

  // prepare streams for CUDA
  int NUM_THREADS = 256;
  cudaStream_t streams[number_of_old_spline_points - 1];
  int num_breaks = num_teuk_modes / MAX_MODES_BLOCK;

#endif

#ifdef __USE_OMP__
  #pragma omp parallel for
#endif  // __USE_OMP__
  for (int i = 0; i < number_of_old_spline_points - 1; i++)
  {
#ifdef __CUDACC__
    cudaSetDevice(dev);
    // create and execute with streams
    cudaStreamCreate(&streams[i]);
    int num_blocks = std::ceil((unit_length[i] + NUM_THREADS - 1) / NUM_THREADS);

    // sometimes a spline interval will have zero points
    if (num_blocks <= 0)
      continue;

    dim3 gridDim(num_blocks, 1);
    // launch one worker kernel per stream
    make_waveform<<<gridDim, NUM_THREADS, 0, streams[i]>>>(d_waveform,
                                                           interp_array,
                                                           d_m, d_n, num_teuk_modes, d_Ylms,
                                                           delta_t, h_t[i], i, start_inds[i], start_inds[i + 1], init_len);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaDeviceSynchronize();
#else

    // CPU waveform generation
    make_waveform(d_waveform,
                  interp_array,
                  d_m, d_n, num_teuk_modes, d_Ylms,
                  delta_t, h_t[i], i, start_inds[i], start_inds[i + 1], init_len);
#endif
  }

// synchronize after all streams finish
#ifdef __CUDACC__
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

#ifdef __USE_OMP__
  #pragma omp parallel for
#endif  // __USE_OMP__
  for (int i = 0; i < number_of_old_spline_points - 1; i++)
  {
    // destroy the streams
    cudaStreamDestroy(streams[i]);
  }
#endif
}
