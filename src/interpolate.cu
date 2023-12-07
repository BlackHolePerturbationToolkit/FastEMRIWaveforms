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
#include "Utility.hh"
// adjust imports based on CUDA or not
#ifdef __CUDACC__
#include "cusparse.h"
#else
#include "lapacke.h"
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

  for (int j = 0;
       j < n;
       j += 1)
  {
    int info = LAPACKE_dgtsv(LAPACK_COL_MAJOR, m, 1, &a[j * m + 1], &b[j * m], &c[j * m], &d_in[j * m], m);
  }

#endif // __CUDACC__
}

CUDA_KERNEL
void fill_final_derivs(double *t_arr, double *interp_array,
                      int ninterps, int length)
{
    #ifdef __CUDACC__
    int start1 = blockIdx.x*blockDim.x + threadIdx.x;
    int end1 = ninterps;
    int diff1 = blockDim.x * gridDim.x;
    #else

    int start1 = 0;
    int end1 = ninterps;
    int diff1 = 1;

    #endif

    for (int interp_i= start1;
         interp_i<end1; // 2 for re and im
         interp_i+= diff1)
         {

             int lead_ind = interp_i*length;
            double c1 = interp_array[(1 * ninterps +  interp_i) * length + length - 2];
            double c2 = interp_array[(2 * ninterps +  interp_i) * length + length - 2];
            double c3 = interp_array[(3 * ninterps +  interp_i) * length + length - 2];

            double t_begin = t_arr[interp_i * length + length - 2];
            double t_end = t_arr[interp_i * length + length - 1];
            double x = t_end - t_begin;
            double x2 = x * x;
            double final_c1 = c1 + 2 * c2 * x + 3 * c3 * x2;
            double final_c2 = (2. * c2 + 6. * c3 * x)/2.;
            double final_c3 = c3;

            interp_array[(1 * ninterps +  interp_i) * length + length - 1] = c1;
            interp_array[(2 * ninterps +  interp_i) * length + length - 1] = c2;
            interp_array[(3 * ninterps +  interp_i) * length + length - 1] = c3;

        }
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

  int num_blocks_fill_derivs = std::ceil((ninterps + NUM_THREADS -1)/NUM_THREADS);

  fill_final_derivs<<<num_blocks_fill_derivs, NUM_THREADS>>>(t_arr, interp_array, ninterps, length);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

#else

  fill_B(t_arr, interp_array, B, upper_diag, diag, lower_diag, ninterps, length);

  fit_wrap(length, ninterps, lower_diag, diag, upper_diag, B);

  set_spline_constants(t_arr, interp_array, B,
                                 ninterps, length);

  fill_final_derivs(t_arr, interp_array, ninterps, length);

#endif
}

CUDA_KERNEL
void interp_time_for_fd(double* output, double *t_arr, double *tstar, int* ind_tstar, double *interp_array, int ninterps, int length, bool* run)
{

    int num_modes = int((ninterps - 4) / 2.);
    #ifdef __CUDACC__
    int start1 = blockIdx.x*blockDim.x + threadIdx.x;
    int end1 = num_modes;
    int diff1 = blockDim.x * gridDim.x;
    #else

    int start1 = 0;
    int end1 = num_modes;
    int diff1 = 1;

    
    #endif

    for (int interp_i= start1;
         interp_i<end1; // 2 for re and im
         interp_i+= diff1)
    {
             bool run_here = run[interp_i];
             if (!run_here) continue;

             int ind = ind_tstar[interp_i];
            double t_segment = t_arr[ind];
            double t = tstar[interp_i];
            double x = t - t_segment;
            double x2 = x * x;
            double x3 = x * x2;


             #ifdef __CUDACC__
             #else
             
             #endif
             for (int i = 0; i < 2; i += 1)
             {

                 int num_base = length * ninterps;
                // fill phase values. These will be same for all modes
                 int ind_f = ind * ninterps + (ninterps - 3 - i);

                 // access the frequencies
                double y0 = interp_array[0 * num_base + ind_f];
                double c1 = interp_array[1 * num_base + ind_f];
                double c2 = interp_array[2 * num_base + ind_f];
                double c3 = interp_array[3 * num_base + ind_f];

                double temp = y0 + c1 * x + c2 * x2 + c3 * x3;

                output[(1 - i) * num_modes + interp_i] = temp;

            }
    }
}

void interp_time_for_fd_wrap(double* output, double *t_arr, double *tstar, int* ind_tstar, double *interp_array, int ninterps, int length, bool* run)
{
    int num_modes = int((ninterps - 4) / 2.);
    #ifdef __CUDACC__
    int NUM_THREADS = 64;
    int num_blocks = std::ceil((num_modes + NUM_THREADS -1)/NUM_THREADS);
    interp_time_for_fd<<<num_blocks, NUM_THREADS>>>(output, t_arr, tstar, ind_tstar, interp_array, ninterps, length,run);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #else
    interp_time_for_fd(output, t_arr, tstar, ind_tstar, interp_array, ninterps, length,run);

    #endif
}

/////////////////////////////////
/////////
/////////  MODE SUMMATION
/////////
/////////////////////////////////

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
  cmplx minus_I(0.0, -1.0);

  cmplx mode_val;
  cmplx partial_mode;
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

        fod phase = m * Phi_phi_i + n * Phi_r_i;
        partial_mode = mode_val * gcmplx::exp(minus_I * phase);

        trans_plus_m = partial_mode * Ylm_plus_m;

        // minus m if m > 0
        // mode values for +/- m are taking care of when applying
        // specific mode selection by setting ylms to zero for the opposites
        if (m != 0)
        {

          Ylm_minus_m = Ylms[2 * j + 1];
          trans_minus_m = gcmplx::conj(partial_mode) * Ylm_minus_m;  // conjugate is distributive, so apply to the product
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


  for (int i = 0; i < number_of_old_spline_points - 1; i++)
  {
    // destroy the streams
    cudaStreamDestroy(streams[i]);
  }
#endif
}

// make a frequency domain waveform in parallel
// this uses an efficient summation by loading mode information into shared memory
// shared memory is leveraged heavily

//  Copyright (c) 2006 Xiaogang Zhang
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Modified Bessel functions of the first and second kind of fractional order

const double epsilon = 2.2204460492503131e-016;
const int max_series_iterations = 1;
const double log_max_value = 709.0;
const double max_value = 1.79769e+308;
const int max_factorial = 170;
const double euler = 0.577215664901532860606;


CUDA_CALLABLE_MEMBER
int iround(double x)
{
    double remain = fmod(x, 1.0);
    double temp;
    if (remain >= 0.5)
    {
        temp = ceil(x);
    }
    else
    {
        temp = floor(x);
    }
   return int(temp);
}

CUDA_CALLABLE_MEMBER
double tgamma1pm1(double dz)
{
  return tgamma(dz + 1.) - 1.;
}


// Calculate K(v, x) and K(v+1, x) by method analogous to
// Temme, Journal of Computational Physics, vol 21, 343 (1976)
CUDA_CALLABLE_MEMBER
int temme_ik(double v, cmplx x, cmplx* K, cmplx* K1)
{
    cmplx f, h, p, q, coef, sum, sum1;
    cmplx a, b, c, d, sigma, gamma1, gamma2;
    unsigned long kk;
    double k, tolerance;

    // |x| <= 2, Temme series converge rapidly
    // |x| > 2, the larger the |x|, the slower the convergence
    //BOOST_ASSERT(abs(x) <= 2);
    //BOOST_ASSERT(abs(v) <= 0.5f);

    double gp = tgamma1pm1(v);
    double gm = tgamma1pm1(-v);

    a = log(x / 2.);
    b = exp(v * a);
    sigma = -a * v;

    c = abs(v) < epsilon ?
       1.0 : sin(M_PI * v) / (v * M_PI);
    d = abs(sigma) < epsilon ?
        1.0 : sinh(sigma) / sigma;
    gamma1 = abs(cmplx(v, 0.0)) < epsilon ?
        -euler : (0.5 / v) * (gp - gm) * c;
    gamma2 = (2. + gp + gm) * c / 2.;

    // initial values
    p = (gp + 1.) / (2. * b);
    q = (1. + gm) * b / 2.;
    f = (cosh(sigma) * gamma1 + d * (-a) * gamma2) / c;
    h = p;
    coef = 1.;
    sum = coef * f;
    sum1 = coef * h;

    // series summation
    tolerance = epsilon;
    for (kk = 1; kk < max_series_iterations; kk++)
    {
        k = double(kk);

        f = (k * f + p + q) / (k*k - v*v);
        p /= k - v;
        q /= k + v;
        h = p - k * f;
        coef *= x * x / (4. * k);
        sum += coef * f;
        sum1 += coef * h;
        if (abs(coef * f) < abs(sum) * tolerance)
        {
           break;
        }
    }

    *K = sum;
    *K1 = 2. * sum1 / x;

    return 0;
}



// Calculate K(v, x) and K(v+1, x) by evaluating continued fraction
// z1 / z0 = U(v+1.5, 2v+1, 2x) / U(v+0.5, 2v+1, 2x), see
// Thompson and Barnett, Computer Physics Communications, vol 47, 245 (1987)
CUDA_CALLABLE_MEMBER
int CF2_ik(double v, cmplx x, cmplx* Kv, cmplx* Kv1)
{
    double tolerance;
    cmplx S, C, Q, D, f, a, b, q, delta, current, prev;
    unsigned long k;

    // |x| >= |v|, CF2_ik converges rapidly
    // |x| -> 0, CF2_ik fails to converge

    // TODO: deal with this line
    //assert(abs(x) > 1);

    // Steed's algorithm, see Thompson and Barnett,
    // Journal of Computational Physics, vol 64, 490 (1986)
    tolerance = epsilon;
    a = v * v - 0.25f;
    b = 2. * (x + 1.);                              // b1
    D = 1. / b;                                    // D1 = 1 / b1
    f = delta = D;                                // f1 = delta1 = D1, coincidence
    prev = 0;                                     // q0
    current = 1;                                  // q1
    Q = C = -a;                                   // Q1 = C1 because q1 = 1
    S = 1. + Q * delta;                            // S1

    for (k = 2; k < max_series_iterations; k++)     // starting from 2
    {
        // continued fraction f = z1 / z0
        a -= 2 * (k - 1);
        b += 2;
        D = 1. / (b + a * D);
        delta *= b * D - 1.;
        f += delta;

        // series summation S = 1 + \sum_{n=1}^{\infty} C_n * z_n / z_0
        q = (prev - (b - 2.) * current) / a;
        prev = current;
        current = q;                        // forward recurrence for q
        C *= -a / double(k);
        Q += C * q;
        S += Q * delta;
        //
        // Under some circumstances q can grow very small and C very
        // large, leading to under/overflow.  This is particularly an
        // issue for types which have many digits precision but a narrow
        // exponent range.  A typical example being a "double double" type.
        // To avoid this situation we can normalise q (and related prev/current)
        // and C.  All other variables remain unchanged in value.  A typical
        // test case occurs when x is close to 2, for example cyl_bessel_k(9.125, 2.125).
        //
        if(abs(q) < epsilon)
        {
           C *= q;
           prev /= q;
           current /= q;
           q = 1;
        }

        // S converges slower than f
        if (abs(Q * delta) < abs(S) * tolerance)
        {
           break;
        }
    }

    if(abs(x) >= log_max_value)
       *Kv = gcmplx::exp(0.5 * log(M_PI / (2. * x)) - x - log(S));
    else
      *Kv = sqrt(M_PI / (2. * x)) * gcmplx::exp(-x) / S;
    *Kv1 = *Kv * (0.5 + v + x + (v * v - 0.25) * f) / x;
    return 0;
}


// Compute I(v, x) and K(v, x) simultaneously by Temme's method, see
// Temme, Journal of Computational Physics, vol 19, 324 (1975)
CUDA_CALLABLE_MEMBER
int bessel_ik(double v, cmplx x, cmplx* K)
{
    // Kv1 = K_(v+1), fv = I_(v+1) / I_v
    // Ku1 = K_(u+1), fu = I_(u+1) / I_u
    double u;
    cmplx Iv, Kv, Kv1, Ku, Ku1, fv;
    cmplx W, current, prev, next;
    bool reflect = false;
    unsigned n, k;

    if (v < 0)
    {
        reflect = true;
        v = -v;                             // v is non-negative from here
    }
    n = iround(v);
    u = v - n;                              // -1/2 <= u < 1/2

    // x is positive until reflection
    //W = 1 / x;                                 // Wronskian
    if (abs(x) <= 2)                                // x in (0, 2]
    {
       temme_ik(u, x, &Ku, &Ku1);             // Temme series
    }
    else                                       // x in (2, \infty)
    {
        CF2_ik(u, x, &Ku, &Ku1);           // continued fraction CF2_ik
    }


    prev = Ku;
    current = Ku1;

    cmplx scale = 1.0;
    
    Kv = prev;
    Kv1 = current;

    *K = Kv / scale;

    return 0;
}

CUDA_CALLABLE_MEMBER
cmplx kve(double v, cmplx x)
{
    cmplx K;
    bessel_ik(v, x, &K);
    return K * gcmplx::exp(x);
}

#define MAX_SEGMENTS_BLOCK 400

CUDA_CALLABLE_MEMBER
cmplx SPAFunc(const double x)
{

    //$x = (2\pi/3)\dot f^3/\ddot f^2$ and spafunc is $i \sqrt{x} e^{-i x} K_{1/3}(-i x)$.
  cmplx II(0.0, 1.0);
  cmplx ans;
  const double Gamp13 = 2.67893853470774763;  // Gamma(1/3)
  const double Gamm13 = -4.06235381827920125; // Gamma(-1/3);
  if (abs(x) <= 7.) {
    const cmplx xx = ((cmplx)x);
    const cmplx pref1 = gcmplx::exp(-2.*M_PI*II/3.)*pow(xx, 5./6.)*Gamm13/pow(2., 1./3.);
    const cmplx pref2 = gcmplx::exp(-M_PI*II/3.)*pow(xx, 1./6.)*Gamp13/pow(2., 2./3.);
    const double x2 = x*x;

    const double c1_0 = 0.5, c1_2 = -0.09375, c1_4 = 0.0050223214285714285714;
    const double c1_6 = -0.00012555803571428571429, c1_8 = 1.8109332074175824176e-6;
    const double c1_10 = -1.6977498819539835165e-8, c1_12 = 1.1169407118118312608e-10;
    const double c1_14 = -5.4396463237589184781e-13, c1_16 = 2.0398673714095944293e-15;
    const double c1_18 = -6.0710338434809358015e-18, c1_20 = 1.4687985105195812423e-20;
    const double c1_22 = -2.9454515585285720100e-23, c1_24 = 4.9754249299469121790e-26;
    const double c1_26 = -7.1760936489618925658e-29;

    const double ser1 = c1_0 + x2*(c1_2 + x2*(c1_4 + x2*(c1_6 + x2*(c1_8 + x2*(c1_10 + x2*(c1_12 + x2*(c1_14 + x2*(c1_16 + x2*(c1_18 + x2*(c1_20 + x2*(c1_22 + x2*(c1_24 + x2*c1_26))))))))))));

    const double c2_0 = 1., c2_2 = -0.375, c2_4 = 0.028125, c2_6 = -0.00087890625;
    const double c2_8 = 0.000014981356534090909091, c2_10 = -1.6051453429383116883e-7;
    const double c2_12 = 1.1802539286311115355e-9, c2_14 = -6.3227889033809546546e-12;
    const double c2_16 = 2.5772237377911499951e-14, c2_18 = -8.2603324929203525483e-17;
    const double c2_20 = 2.1362928861000911763e-19, c2_22 = -4.5517604107246260858e-22;
    const double c2_24 = 8.1281435905796894390e-25, c2_26 = -1.2340298973552160079e-27;

    const double ser2 = c2_0 + x2*(c2_2 + x2*(c2_4 + x2*(c2_6 + x2*(c2_8 + x2*(c2_10 + x2*(c2_12 + x2*(c2_14 + x2*(c2_16 + x2*(c2_18 + x2*(c2_20 + x2*(c2_22 + x2*(c2_24 + x2*c2_26))))))))))));

    ans = gcmplx::exp(-II*x)*(pref1*ser1 + pref2*ser2);
  } else {
    const cmplx y = 1./x;
    const cmplx pref = gcmplx::exp(-0.75*II*M_PI)*sqrt(0.5*M_PI);

    const cmplx c_0 = II, c_1 = 0.069444444444444444444, c_2 = -0.037133487654320987654*II;
    const cmplx c_3 = -0.037993059127800640146, c_4 = 0.057649190412669721333*II;
    const cmplx c_5 = 0.11609906402551541102, c_6 = -0.29159139923075051147*II;
    const cmplx c_7 = -0.87766696951001691647, c_8 = 3.0794530301731669934*II;

    cmplx ser = c_0+y*(c_1+y*(c_2+y*(c_3+y*(c_4+y*(c_5+y*(c_6+y*(c_7+y*c_8)))))));

    ans = pref*ser;
  }
  return(ans);
}


// build mode value with specific phase and amplitude values; mode indexes; and spherical harmonics
// CUDA_CALLABLE_MEMBER
// cmplx get_mode_value_fd(double t, double f, double fdot, double fddot, cmplx amp_term1, double phase_term, cmplx Ylm){
//     cmplx I(0.0, 1.0);

//     // Waveform Amplitudes
//     // $x = (2\pi/3)\dot f^3/\ddot f^2$ and spafunc is $i \sqrt{x} e^{-i x} K_{1/3}(-i x)$.
//     double arg = 2.* PI * pow(fdot, 3) / (3.* pow(fddot, 2));
//     cmplx amp_term2 = -1.0 * fdot/abs(fddot) * 2./sqrt(3.) * SPAFunc(arg) / gcmplx::sqrt(cmplx(arg, 0.0));
//     //cmplx amp_term2 = 0.0;
//     cmplx out = amp_term1 * Ylm * amp_term2
//                 * gcmplx::exp(
//                     I* (2. * PI * f * t - phase_term)
//                 );

//     return out;
// }


CUDA_CALLABLE_MEMBER
void cube_roots(cmplx *r1o, cmplx *r2o, cmplx *r3o, double a, double b, double c, double d, bool check)
{
    double b2 = b * b;
    double b3 = b2 * b;
    double a2 = a * a;

    double Delta_0 = (b2 - 3 * a * c);
    double Delta_1 = (2 * b3 - 9 * a * b * c + 27 * a2 * d);

    cmplx C = pow((Delta_1 +  sqrt(cmplx(Delta_1 * Delta_1  - 4 * ( Delta_0 * Delta_0 * Delta_0), 0.0)))/ 2., 1./3.);

    cmplx xi = (-1. + sqrt(cmplx(-3., 0.0))) / 2.;

    cmplx r1 = - (1. / (3 * a)) * (b + C + Delta_0 / (C));
    cmplx r2 = - (1. / (3 * a)) * (b + xi * C + Delta_0 / (xi * C));
    cmplx r3 = - (1. / (3 * a)) * (b + xi * xi * C + Delta_0 / (xi * xi * C));
    
    *r1o = r1;
    *r2o = r2;
    *r3o = r3;

}

#define NUM_THREADS_FD 256
// make a waveform in parallel
// this uses an efficient summation by loading mode information into shared memory
// shared memory is leveraged heavily
#define MAX_SPLINE_POINTS 210
CUDA_KERNEL
void make_generic_kerr_waveform_fd(cmplx *waveform,
             double *interp_array,
              int *m_arr_in, int *k_arr_in, int *n_arr_in, int num_teuk_modes,
              double delta_t, double *old_time_arr, int init_length, int data_length,
              double *frequencies, int *seg_start_inds, int *seg_end_inds, int num_segments,
              cmplx *Ylm_all, int zero_index, bool include_minus_m, bool separate_modes
              )
{

    int num_pars = 4;

    cmplx complexI(0.0, 1.0);

    // declare all the shared memory
    // MAX_MODES_BLOCK is fixed based on shared memory
    double old_time_start, old_time_end;

    double R_mode_re_y;
    double R_mode_re_c1;
    double R_mode_re_c2;
    double R_mode_re_c3;

    double R_mode_im_y;
    double R_mode_im_c1;
    double R_mode_im_c2;
    double R_mode_im_c3;

    double Phi_phi_y;
    double Phi_phi_c1;
    double Phi_phi_c2;
    double Phi_phi_c3;

    double Phi_r_y;
    double Phi_r_c1;
    double Phi_r_c2;
    double Phi_r_c3;

    double f_phi_y;
    double f_phi_c1;
    double f_phi_c2;
    double f_phi_c3;

    double f_r_y;
    double f_r_c1;
    double f_r_c2;
    double f_r_c3;

    double roots[3];

    cmplx I(0.0, 1.0);

    // number of splines
    int num_base = (2 * num_teuk_modes + num_pars) * init_length;

    #ifdef __CUDACC__

    int start2 = blockIdx.y;
    int diff2 = gridDim.y;

    #else

    int start2 = 0;
    int diff2 = 1;
    #endif
    for (int mode_i = start2; mode_i < num_teuk_modes; mode_i += diff2) 
    {

        int m = m_arr_in[mode_i];
        int k = k_arr_in[mode_i];
        int n = n_arr_in[mode_i];
        cmplx Ylm_plus_m = Ylm_all[mode_i];
        cmplx Ylm_minus_m = Ylm_all[num_teuk_modes + mode_i];

        #ifdef __CUDACC__

        int start3 = blockIdx.x;
        int diff3 = gridDim.x;

        #else

        int start3 = 0;
        int diff3 = 1;
        #endif
        for (int seg_i = start3; seg_i < num_segments; seg_i += diff3) 
        {

            int seg_start_ind_here = seg_start_inds[mode_i * num_segments + seg_i];
            int seg_end_ind_here = seg_end_inds[mode_i * num_segments + seg_i];

            old_time_start = old_time_arr[seg_i];
            old_time_end = old_time_arr[seg_i + 1];

            int y_ind = 0 * num_base + mode_i * init_length + seg_i;
            int c1_ind = 1 * num_base + mode_i * init_length + seg_i;
            int c2_ind = 2 * num_base + mode_i * init_length + seg_i;
            int c3_ind = 3 * num_base + mode_i * init_length + seg_i;

            R_mode_re_y = interp_array[y_ind];
            R_mode_re_c1 = interp_array[c1_ind];
            R_mode_re_c2 = interp_array[c2_ind];
            R_mode_re_c3 = interp_array[c3_ind];

            y_ind = 0 * num_base + (num_teuk_modes + mode_i) * init_length + seg_i;
            c1_ind = 1 * num_base + (num_teuk_modes + mode_i) * init_length + seg_i;
            c2_ind = 2 * num_base + (num_teuk_modes + mode_i) * init_length + seg_i;
            c3_ind = 3 * num_base + (num_teuk_modes + mode_i) * init_length + seg_i;

            R_mode_im_y = interp_array[y_ind];
            R_mode_im_c1 = interp_array[c1_ind];
            R_mode_im_c2 = interp_array[c2_ind];
            R_mode_im_c3 = interp_array[c3_ind];

            y_ind = 0 * num_base + (2 + 2 * num_teuk_modes) * init_length + seg_i;
            c1_ind = 1 * num_base + (2 + 2 * num_teuk_modes) * init_length + seg_i;
            c2_ind = 2 * num_base + (2 + 2 * num_teuk_modes) * init_length + seg_i;
            c3_ind = 3 * num_base + (2 + 2 * num_teuk_modes) * init_length + seg_i;

            Phi_phi_y = interp_array[y_ind];
            Phi_phi_c1 = interp_array[c1_ind];
            Phi_phi_c2 = interp_array[c2_ind];
            Phi_phi_c3 = interp_array[c3_ind];

            y_ind = 0 * num_base + (3 + 2 * num_teuk_modes) * init_length + seg_i;
            c1_ind = 1 * num_base + (3 + 2 * num_teuk_modes) * init_length + seg_i;
            c2_ind = 2 * num_base + (3 + 2 * num_teuk_modes) * init_length + seg_i;
            c3_ind = 3 * num_base + (3 + 2 * num_teuk_modes) * init_length + seg_i;

            Phi_r_y = interp_array[y_ind];
            Phi_r_c1 = interp_array[c1_ind];
            Phi_r_c2 = interp_array[c2_ind];
            Phi_r_c3 = interp_array[c3_ind];

            y_ind = 0 * num_base + (0 + 2 * num_teuk_modes) * init_length + seg_i;
            c1_ind = 1 * num_base + (0 + 2 * num_teuk_modes) * init_length + seg_i;
            c2_ind = 2 * num_base + (0 + 2 * num_teuk_modes) * init_length + seg_i;
            c3_ind = 3 * num_base + (0 + 2 * num_teuk_modes) * init_length + seg_i;

            //if ((m == 2) && (n == 0) && (seg_i == 80))
            //    printf("seg_i: %d m: %d n: %d , %d %d %d %d %d \n", seg_i, m, n, y_ind, num_base, num_teuk_modes, mode_i, init_length);
                
            f_phi_y = interp_array[y_ind];
            double f_phi_y2 = interp_array[y_ind + 1];
            f_phi_c1 = interp_array[c1_ind];
            f_phi_c2 = interp_array[c2_ind];
            f_phi_c3 = interp_array[c3_ind];

            y_ind = 0 * num_base + (1 + 2 * num_teuk_modes) * init_length + seg_i;
            c1_ind = 1 * num_base + (1 + 2 * num_teuk_modes) * init_length + seg_i;
            c2_ind = 2 * num_base + (1 + 2 * num_teuk_modes) * init_length + seg_i;
            c3_ind = 3 * num_base + (1 + 2 * num_teuk_modes) * init_length + seg_i;

            f_r_y = interp_array[y_ind];
            double f_r_y2 = interp_array[y_ind + 1];
            f_r_c1 = interp_array[c1_ind];
            f_r_c2 = interp_array[c2_ind];
            f_r_c3 = interp_array[c3_ind];

            CUDA_SYNC_THREADS;
            // TODO: cleanup registers for CUDA without OMP
            #ifdef __CUDACC__

            int start = threadIdx.x;
            int diff = blockDim.x;

            #else

            int start = 0;
            int diff = 1;
            #endif
            for (int i = start + seg_start_ind_here; i <= seg_end_ind_here; i += diff)
            {
                int ind_f = i;
                double f = frequencies[ind_f];

                if ((i > data_length - 1) || (i < 0)) continue;

                int minus_m_freq_index;
                int diff = abs(zero_index - i);
                if (i < zero_index)
                {
                    minus_m_freq_index = zero_index + diff;
                }
                else
                {
                    minus_m_freq_index = zero_index - diff;
                }//= int((-f - start_freq) / df) + 1;

                //double f_y2 = m * f_phi_y[ind_i + 1] + n * f_r_y[ind_i + 1];
                double f_y = m * f_phi_y + n * f_r_y;
                double f_y2 = m * f_phi_y2 + n * f_r_y2;
                double f_c1 = m * f_phi_c1 + n * f_r_c1;
                double f_c2 = m * f_phi_c2 + n * f_r_c2;
                double f_c3 = m * f_phi_c3 + n * f_r_c3;

                cmplx root1 = -1e300;
                cmplx root2 = -1e300;
                cmplx root3 = -1e300;

                bool check = false;
                //if ((mode_i == 0) && ((i > 100) && (i < 150))) check = true;
                double factor;
                
                cube_roots(&root1, &root2, &root3, f_c3, f_c2, f_c1, (f_y - f), check);
                roots[0] = root1.real();
                roots[1] = root2.real();
                roots[2] = root3.real();
                //if ((f_y - f > 0.0)) printf("roots: %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %d %d %d\n", root1, root2, root3, f_c3, f_c2, f_c1, f_y - f, f_y, f_y2, f, start_t, end_t, ind_i, m, n);
                //if ((f_y - f > 0.0)) printf("roots: %d %d %d %d %.12e %.12e %.12e %d %d\n", ind_i, m, n, (mode_i * max_length + i) * 2 + which, f,f_y, f_y2, mode_start_ind_here, i);

                // if ((m == 2) && (seg_i == 80) && (i == seg_start_ind_here + 10)) printf("roots: i: %d seg_i: %d m: %d n: %d root1: %.12e\n t_s= %.12e\nt_f= %.12e \nf= %.12e \nf_phi_y= %.12e \nf_r_y= %.12e \nf_phi_c1= %.12e \nf_r_c1= %.12e \nf_phi_c2= %.12e \nf_r_c2= %.12e \nf_phi_c3= %.12e \nf_r_c3= %.12e\n", i, seg_i, m, n, roots[0], old_time_start, old_time_end, f, f_phi_y, f_r_y, f_phi_c1, f_r_c1, f_phi_c2, f_r_c2, f_phi_c3, f_r_c3);
                //if ((mode_i == 0) && (seg_i == 6)) printf("roots: %d %d %d %.12e %.12e %.12e\n", i, m, n, f_phi_y, f_r_y, f_y);

                //if ((i < 10) && (mode_i < 3)) printf("root check: %d %d %.18e %.18e %.18e %.18e %.18e \n", mode_i, i, root1, root2, root3, start_t, end_t);
                //if ((i < 10) && (mode_i < 3)) printf("root check: %d %d %.18e %.18e %.18e %.18e %.18e %.18e \n", mode_i, i, f_c3, f_c2, f_c1, (f_y - f), f_y, f);


                double t;
                double x, x2, x3;
                double root_here;
                for (int root_i = 0; root_i < 3; root_i += 1)
                {
                    // TODO: check imaginary part 
                    root_here = roots[root_i];
                    // if (f_y - f < 0.0) root_here *= -1;
                    t = old_time_start + root_here;
                    
                    if ((t < old_time_end) && (t >= old_time_start))
                    {
                        x = root_here;
                        x2 = x*x;
                        x3 = x*x2;
                    
                        // get mode values at this timestep
                        double R_mode_re = R_mode_re_y + R_mode_re_c1 * x + R_mode_re_c2 * x2  + R_mode_re_c3 * x3;
                        double R_mode_im = R_mode_im_y + R_mode_im_c1 * x + R_mode_im_c2 * x2  + R_mode_im_c3 * x3;
                        //double L_mode_re = L_mode_re_y + L_mode_re_c1 * x + L_mode_re_c2 * x2  + L_mode_re_c3 * x3;
                        //double L_mode_im = L_mode_im_y + L_mode_im_c1 * x + L_mode_im_c2 * x2  + L_mode_im_c3 * x3;

                        // get phases at this timestep
                        double Phi_phi_i =  Phi_phi_y +  Phi_phi_c1 * x +  Phi_phi_c2 * x2  +  Phi_phi_c3 * x3;
                        //double Phi_theta_i = pt_y + pt_c1 * x + pt_c2 * x2 + pt_c3 * x3;
                        double Phi_theta_i = 0.0;
                        double Phi_r_i =  Phi_r_y +  Phi_r_c1 * x +  Phi_r_c2 * x2  +  Phi_r_c3 * x3;

                        double fdot_phi_i = f_phi_c1 + 2 * f_phi_c2 * x  + 3 * f_phi_c3 * x2;
                        //double Phi_theta_i = pt_y + pt_c1 * x + pt_c2 * x2 + pt_c3 * x3;
                        double fdot_theta_i = 0.0;  // ft_c1 + 2 * ft_c2 * x  + 3 * ft_c3 * x2;
                        double fdot_r_i = f_r_c1 + 2 * f_r_c2 * x  + 3 * f_r_c3 * x2;

                        double fddot_phi_i = 2 * f_phi_c2 + 6 * f_phi_c3 * x;
                        //double Phi_theta_i = pt_y + pt_c1 * x + pt_c2 * x2 + pt_c3 * x3;
                        double fddot_theta_i = 0.0;  // 2 * ft_c2 * x  + 6 * ft_c3 * x;
                        double fddot_r_i = 2 * f_r_c2 + 6 * f_r_c3 * x;

                        double phase_term = m * Phi_phi_i + k * Phi_theta_i + n * Phi_r_i;
                        double fdot = m * fdot_phi_i + k * fdot_theta_i + n * fdot_r_i;
                        double fddot = m * fddot_phi_i + k * fddot_theta_i + n * fddot_r_i;

                        cmplx R_amp(R_mode_re, R_mode_im);
                        // if (i == 1653451) printf("%d %d %d %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", i, m, n, t, f, fdot, fddot, R_amp.real(), R_amp.imag(), phase_term, fddot_phi_i, fddot_r_i);
                        // if (i == 1653451) printf("%d %d %d %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", i, m, n, t, f, fdot, fddot, R_amp.real(), R_amp.imag(), phase_term);

                        // store some intermediate quantities to save some function evaluations
                        
                        // commutativity of the negative signs:
                        // SPAFUNC
                        // // bessel function: ix -> -ix is conjugate
                        // // sqrt: sqrt(-x) = i sqrt(x)
                        // // e^-ix: x -> -x is conjugate
                        // then i\sqrt{-x} e^{-i -x} K_{1/3}(-i -x) -> conj(i * (-i) * \sqrt{x} e^{-i x} K_{1/3}(-i x))

                        // 1 / sqrt(-x) -> 1 / i sqrt(x) under x -> -x, which cancels with the i from the first sqrt
                        // so:
                        // amp_term2(-x) = conj(-1.0 * -fdot/abs(fddot) * 2./sqrt(3.) * i\sqrt{x} e^{-i x} K_{1/3}(-i x)) / sqrt(-x)
                        //               = conj(-1.0 * -fdot/abs(fddot) * 2./sqrt(3.) * i * (-i) * \sqrt{x} e^{-i x} K_{1/3}(-i x)) / (i sqrt(x))
                        //               = conj(-1.0 * fdot/abs(fddot) * 2./sqrt(3.) * i\sqrt{x} e^{-i x} K_{1/3}(-i x) / sqrt(x))
                        //               = conj(amp_term2(x))

                        double temp_arg = 2.* PI * pow(fdot, 3) / (3.* pow(fddot, 2));
                        cmplx amp_term2 = -1.0 * fdot/abs(fddot) * 2./sqrt(3.) * SPAFunc(temp_arg) / gcmplx::sqrt(cmplx(temp_arg, 0.0)); 
                        
                        cmplx temp_exp = R_amp * amp_term2 * gcmplx::exp(I* (2. * PI * f * t - phase_term));

                        cmplx R_tmp = Ylm_plus_m * temp_exp;  // combine with spherical harmonic

                        // TODO: improve for generic
                        cmplx L_tmp(0.0, 0.0);
                        if ((m != 0.0) && (include_minus_m))
                        {

                            L_tmp = Ylm_minus_m * gcmplx::conj(temp_exp);  // as above, take the conjugate and we're done
                        }
                        
                        //if (m + k + n != 0)
                        //{
                        //cmplx L_tmp = get_mode_value_generic(L_amp, Phi_phi_i, Phi_r_i, Phi_theta_i, -m, -k, -n);
                        //}
                        //cmplx wave_mode_out = R_tmp + L_tmp;
                        if (!separate_modes)
                        {
                            // fill waveform
                            #ifdef __CUDACC__
                            atomicAddComplex(&waveform[i], R_tmp);
                            #else
                            waveform[i] += R_tmp;
                            #endif

                            if ((m != 0.0) && (include_minus_m))
                            {
                                #ifdef __CUDACC__
                                atomicAddComplex(&waveform[minus_m_freq_index], L_tmp);
                                #else
                                waveform[minus_m_freq_index] += L_tmp;
                                #endif
                            }
                        }
                        else
                        {
                            waveform[mode_i * data_length + i] = R_tmp;
                        }
                        //cmplx L_amp(L_mode_re, L_mode_im);

                        //cmplx R_tmp = get_mode_value_generic(R_amp, Phi_phi_i, Phi_r_i, Phi_theta_i, m, k, n);
                    }
                }
            }
        }
    }
}


// function for building interpolated EMRI waveform from python
void get_waveform_generic_fd(cmplx *waveform,
             double *interp_array,
              int *m_arr_in, int *k_arr_in, int *n_arr_in, int num_teuk_modes,
              double delta_t, double *old_time_arr, int init_length, int data_length,
              double *frequencies, int *seg_start_inds, int *seg_end_inds, int num_segments,
              cmplx *Ylm_all, int zero_index, bool include_minus_m, bool separate_modes)
{

     //int NUM_THREADS = 256;
     #ifdef __CUDACC__

      //int num_blocks = num_segments;
      
      dim3 gridDim(num_segments, num_teuk_modes);

      // launch one worker kernel per stream
      make_generic_kerr_waveform_fd<<<gridDim, NUM_THREADS_FD>>>(waveform,
             interp_array,
              m_arr_in, k_arr_in, n_arr_in, num_teuk_modes,
              delta_t, old_time_arr, init_length, data_length,
              frequencies, seg_start_inds, seg_end_inds, num_segments,
              Ylm_all, zero_index, include_minus_m, separate_modes);
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());
      
      #else

         // CPU waveform generation
         make_generic_kerr_waveform_fd(waveform,
             interp_array,
              m_arr_in, k_arr_in, n_arr_in, num_teuk_modes,
              delta_t, old_time_arr, init_length, data_length,
              frequencies, seg_start_inds, seg_end_inds, num_segments,
              Ylm_all, zero_index, include_minus_m, separate_modes);
         
        #endif

}