#include "global.h"
#include "interpolate.hh"
#include "cusparse.h"


#define MAX_MODES_BLOCK 450

__device__
void fill_coefficients(int i, int length, double *dydx, double dx, double *y, double *coeff1, double *coeff2, double *coeff3){
  double slope, t, dydx_i;

  slope = (y[i+1] - y[i])/dx;

  dydx_i = dydx[i];

  t = (dydx_i + dydx[i+1] - 2*slope)/dx;

  coeff1[i] = dydx_i;
  coeff2[i] = (slope - dydx_i) / dx - t;
  coeff3[i] = t/dx;
}


__device__
void prep_splines(int i, int length, double *b, double *ud, double *diag, double *ld, double *x, double *y){
  double dx1, dx2, d, slope1, slope2;
  if (i == length - 1){
    dx1 = x[length - 2] - x[length - 3];
    dx2 = x[length - 1] - x[length - 2];
    d = x[length - 1] - x[length - 3];

    slope1 = (y[length - 2] - y[length - 3])/dx1;
    slope2 = (y[length - 1] - y[length - 2])/dx2;

    b[length - 1] = ((dx2*dx2*slope1 +
                             (2*d + dx2)*dx1*slope2) / d);
    diag[length - 1] = dx1;
    ld[length - 1] = d;
    ud[length - 1] = 0.0;

  } else if (i == 0){
      dx1 = x[1] - x[0];
      dx2 = x[2] - x[1];
      d = x[2] - x[0];

      //amp
      slope1 = (y[1] - y[0])/dx1;
      slope2 = (y[2] - y[1])/dx2;

      b[0] = ((dx1 + 2*d) * dx2 * slope1 +
                          dx1*dx1 * slope2) / d;
      diag[0] = dx2;
      ud[0] = d;
      ld[0] = 0.0;

  } else{
    dx1 = x[i] - x[i-1];
    dx2 = x[i+1] - x[i];

    //amp
    slope1 = (y[i] - y[i-1])/dx1;
    slope2 = (y[i+1] - y[i])/dx2;

    b[i] = 3.0* (dx2*slope1 + dx1*slope2);
    diag[i] = 2*(dx1 + dx2);
    ud[i] = dx1;
    ld[i] = dx2;
  }
}



__global__
void fill_B(double *t_arr, double *y_all, double *B, double *upper_diag, double *diag, double *lower_diag,
                      int ninterps, int length){

    for (int interp_i= blockIdx.y*blockDim.y + threadIdx.y;
         interp_i<ninterps; // 2 for re and im
         interp_i+= blockDim.y*gridDim.y){

       for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < length;
            i += blockDim.x * gridDim.x){



            int lead_ind = interp_i*length;
            prep_splines(i, length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], t_arr, &y_all[interp_i*length]);

}
}
}



__global__
void set_spline_constants(double *t_arr, double *y_all, double *B,
                      double *c1, double *c2, double *c3,
                      int ninterps, int length){

    double dt;
    InterpContainer mode_vals;
    for (int interp_i= blockIdx.y*blockDim.y + threadIdx.y;
         interp_i<ninterps; // 2 for re and im
         interp_i+= blockDim.y*gridDim.y){

        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < length-1;
             i += blockDim.x * gridDim.x){

              dt = t_arr[i + 1] - t_arr[i];

              int lead_ind = interp_i*length;
              int lead_ind2 = interp_i*(length - 1);
              fill_coefficients(i, length, &B[lead_ind], dt, &y_all[lead_ind], &c1[lead_ind2], &c2[lead_ind2], &c3[lead_ind2]);


}
}
}



void fit_wrap(int m, int n, double *a, double *b, double *c, double *d_in){

    size_t bufferSizeInBytes;

    cusparseHandle_t handle;
    void *pBuffer;

    CUSPARSE_CALL(cusparseCreate(&handle));
    CUSPARSE_CALL( cusparseDgtsv2StridedBatch_bufferSizeExt(handle, m, a, b, c, d_in, n, m, &bufferSizeInBytes));
    gpuErrchk(cudaMalloc(&pBuffer, bufferSizeInBytes));

    CUSPARSE_CALL(cusparseDgtsv2StridedBatch(handle,
                                              m,
                                              a, // dl
                                              b, //diag
                                              c, // du
                                              d_in,
                                              n,
                                              m,
                                              pBuffer));

  CUSPARSE_CALL(cusparseDestroy(handle));
  gpuErrchk(cudaFree(pBuffer));

}

void interpolate_arrays(double *t_arr, double *y_all, double *c1, double *c2, double *c3, int ninterps, int length, double *B, double *upper_diag, double *diag, double *lower_diag)
{

  // TODO: Accelerate (?)

  int NUM_THREADS = 64;
  int num_blocks = std::ceil((length + NUM_THREADS -1)/NUM_THREADS);
  dim3 gridDim(num_blocks); //, num_teuk_modes);
  fill_B<<<gridDim, NUM_THREADS>>>(t_arr, y_all, B, upper_diag, diag, lower_diag, ninterps, length);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  fit_wrap(length, ninterps, lower_diag, diag, upper_diag, B);

  set_spline_constants<<<gridDim, NUM_THREADS>>>(t_arr, y_all, B, c1, c2, c3,
                                 ninterps, length);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

}








__device__
cmplx get_mode_value(cmplx teuk_mode, fod Phi_phi, fod Phi_r, int m, int n, cmplx Ylm){
    cmplx minus_I(0.0, -1.0);
    fod phase = m*Phi_phi + n*Phi_r;
    cmplx out = (teuk_mode*Ylm)*gcmplx::exp(minus_I*phase);
    return out;
}


__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long* address_as_ull =
                              (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}


__device__ void atomicAddComplex(cmplx* a, cmplx b){
  //transform the addresses of real and imag. parts to double pointers
  double *x = (double*)a;
  double *y = x+1;
  //use atomicAdd for double variables
  atomicAddDouble(x, b.real());
  atomicAddDouble(y, b.imag());
}



__global__
void make_waveform(cmplx *waveform,
             double *y_all, double *c1, double *c2, double *c3,
              int *m_arr_in, int *n_arr_in, int num_teuk_modes, cmplx *Ylms_in,
              double delta_t, double start_t, int old_ind, int start_ind, int end_ind, int init_length){

    int num_pars = 2;
    cmplx trans(0.0, 0.0);
    cmplx trans2(0.0, 0.0);
    cmplx I(0.0, 1.0);
    cmplx mode_val;
    cmplx trans_plus_m(0.0, 0.0), trans_minus_m(0.0, 0.0);
    double Phi_phi_i, Phi_r_i, t, x, x2, x3, mode_val_re, mode_val_im;
    int lm_i, num_teuk_here;
    double re_y, re_c1, re_c2, re_c3, im_y, im_c1, im_c2, im_c3;
     __shared__ double pp_y, pp_c1, pp_c2, pp_c3, pr_y, pr_c1, pr_c2, pr_c3;

     __shared__ cmplx Ylms[2*MAX_MODES_BLOCK];

     __shared__ double mode_re_y[MAX_MODES_BLOCK];
     __shared__ double mode_re_c1[MAX_MODES_BLOCK];
     __shared__ double mode_re_c2[MAX_MODES_BLOCK];
     __shared__ double mode_re_c3[MAX_MODES_BLOCK];

     __shared__ double mode_im_y[MAX_MODES_BLOCK];
     __shared__ double mode_im_c1[MAX_MODES_BLOCK];
     __shared__ double mode_im_c2[MAX_MODES_BLOCK];
     __shared__ double mode_im_c3[MAX_MODES_BLOCK];

     __shared__ int m_arr[MAX_MODES_BLOCK];
     __shared__ int n_arr[MAX_MODES_BLOCK];


     //cmplx *Ylms = (cmplx*) array;

     __syncthreads();

     if ((threadIdx.x == 0)){
         int ind_Phi_phi = num_teuk_modes*2 + 0;
         int ind_Phi_r = num_teuk_modes*2 + 1;
         pp_y = y_all[old_ind*(2*num_teuk_modes+num_pars) + ind_Phi_phi]; pp_c1 = c1[old_ind*(2*num_teuk_modes+num_pars) + ind_Phi_phi];
         pp_c2= c2[old_ind*(2*num_teuk_modes+num_pars) + ind_Phi_phi];  pp_c3 = c3[old_ind*(2*num_teuk_modes+num_pars) + ind_Phi_phi];

         pr_y = y_all[old_ind*(2*num_teuk_modes+num_pars) + ind_Phi_r]; pr_c1 = c1[old_ind*(2*num_teuk_modes+num_pars) + ind_Phi_r];
         pr_c2= c2[old_ind*(2*num_teuk_modes+num_pars) + ind_Phi_r];  pr_c3 = c3[old_ind*(2*num_teuk_modes+num_pars) + ind_Phi_r];
     }

     __syncthreads();



     int m, n, actual_mode_index;
     cmplx Ylm_plus_m, Ylm_minus_m;

     int num_breaks = (num_teuk_modes / MAX_MODES_BLOCK) + 1;

     for (int block_y=0; block_y<num_breaks; block_y+=1){
    num_teuk_here = (((block_y + 1)*MAX_MODES_BLOCK) <= num_teuk_modes) ? MAX_MODES_BLOCK : num_teuk_modes - (block_y*MAX_MODES_BLOCK);
    //if ((threadIdx.x == 0) && (blockIdx.x == 0)) printf("BLOCKY = %d %d\n", block_y, num_breaks);
    int init_ind = block_y*MAX_MODES_BLOCK;

        for (int i=threadIdx.x; i<num_teuk_here; i+=blockDim.x) {

        int ind_re = old_ind*(2*num_teuk_modes+num_pars) + (init_ind + i);
        int ind_im = old_ind*(2*num_teuk_modes+num_pars)  + num_teuk_modes + (init_ind + i);
        mode_re_y[i] = y_all[ind_re]; mode_re_c1[i] = c1[ind_re];
        mode_re_c2[i] = c2[ind_re]; mode_re_c3[i] = c3[ind_re];
        mode_im_y[i] = y_all[ind_im]; mode_im_c1[i] = c1[ind_im];
        mode_im_c2[i] = c2[ind_im]; mode_im_c3[i] = c3[ind_im];

        //printf("%d %d %d %d\n", init_ind, i, m_arr_in[init_ind + i], n_arr_in[init_ind + i]);
        m_arr[i] = m_arr_in[init_ind + i];
        n_arr[i] = n_arr_in[init_ind + i];
        Ylms[2*i] = Ylms_in[(init_ind + i)];
        Ylms[2*i + 1] = Ylms_in[num_teuk_modes + (init_ind + i)];
    }

    __syncthreads();


    //l'int j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = start_ind + blockIdx.x * blockDim.x + threadIdx.x;
         i < end_ind;
         i += blockDim.x * gridDim.x){

     trans2 = 0.0 + 0.0*I;

     trans = 0.0 + 0.0*I;
     t = delta_t*i;
      x = t - start_t;
      x2 = x*x;
      x3 = x*x2;

      Phi_phi_i = pp_y + pp_c1*x + pp_c2*x2  + pp_c3*x3;
      Phi_r_i = pr_y + pr_c1*x + pr_c2*x2  + pr_c3*x3;
        for (int j=0; j<num_teuk_here; j+=1){

            Ylm_plus_m = Ylms[2*j];

             m = m_arr[j];
             n = n_arr[j];

            mode_val_re =  mode_re_y[j] + mode_re_c1[j]*x + mode_re_c2[j]*x2  + mode_re_c3[j]*x3;
            mode_val_im = mode_im_y[j] + mode_im_c1[j]*x + mode_im_c2[j]*x2  + mode_im_c3[j]*x3;
            mode_val = mode_val_re + I*mode_val_im;

                trans_plus_m = get_mode_value(mode_val, Phi_phi_i, Phi_r_i, m, n, Ylm_plus_m);

                //trans = trans + trans_plus_m;
                //if ((i == 1) && (i < 10000)) printf("%d, %d: %.10e %.10e, %.10e; %.10e + 1j %.10e; %.10e + 1j %.10e; %.10e + 1j %.10e %d %d %.10e %.10e\n", i, j, t, mode_val_re, mode_val_im, trans_plus_m.real(), trans_plus_m.imag(), Ylm_plus_m.real(), Ylm_plus_m.imag(), angle.real(), angle.imag(), m, n, Phi_phi_i, Phi_r_i);

                //trans = (m == 0) ? trans + trans_plus_m : trans + trans_plus_m + gcmplx::conj(trans_plus_m);
                //trans = trans_plus_m + trans;
                // minus m
                if (m != 0){

                    Ylm_minus_m = Ylms[2*j + 1];
                    trans_minus_m = get_mode_value(gcmplx::conj(mode_val), Phi_phi_i, Phi_r_i, -m, -n, Ylm_minus_m);
                    //trans_minus_m = gcmplx::conj(trans_plus_m);

                    //trans = trans_minus_m + trans;
                } else trans_minus_m = 0.0 + 0.0*I;

                trans = trans + trans_minus_m + trans_plus_m;
                //if (i == 0) printf("%d %d %d: %.10e + %.10e 1j; %.10e + %.10e 1j;\n", i, m, n, trans_plus_m.real(), trans_plus_m.imag(), trans_minus_m.real(),trans_minus_m.imag());

                //atomicAddComplex(&waveform[i], mode_val);
        }

        atomicAddComplex(&waveform[i], trans);
    }
    __syncthreads();
}
}



void find_start_inds(int start_inds[], int unit_length[], double *t_arr, double delta_t, int length, int new_length)
{

  start_inds[0] = 0;
  for (int i = 1;
       i < length;
       i += 1){

          double t = t_arr[i];

          start_inds[i] = (int)std::ceil(t/delta_t);
          unit_length[i-1] = start_inds[i] - start_inds[i-1];

      }

  start_inds[length -1] = new_length;
  unit_length[length - 2] = start_inds[length -1] - start_inds[length -2];

  //for (int i=0; i < length; i++) printf("%d %d\n", start_inds[i], unit_length[i]);
}


void get_waveform(cmplx *d_waveform, double *y_vals, double *c1, double *c2, double *c3,
              int *d_m, int *d_n, int init_len, int out_len, int num_teuk_modes, cmplx *d_Ylms,
              double delta_t, double *h_t){

  /*cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
    cudaEventRecord(start);*/

    int NUM_THREADS = 256;
    int start_inds[init_len];
    int unit_length[init_len-1];

    find_start_inds(start_inds, unit_length, h_t, delta_t, init_len, out_len);

    //printf("Num modes: %d\n", num_teuk_modes);
    cudaStream_t streams[init_len-1];

    int num_breaks = num_teuk_modes/MAX_MODES_BLOCK;

    //printf("lm: %d, modes: %d %d\n", num_l_m, num_teuk_modes, num_breaks);

    //cudaEventRecord(start);
    #pragma omp parallel for
    for (int i = 0; i < init_len-1; i++) {
          cudaStreamCreate(&streams[i]);
          int num_blocks = std::ceil((unit_length[i] + NUM_THREADS -1)/NUM_THREADS);
          //printf("%d %d %d %d, %d %d\n", i, start_inds[i], unit_length[i], num_blocks, init_len, out_len);
          if (num_blocks <= 0) continue;

          //printf("%d %d %d %d\n", i, num_blocks, unit_length[i], init_len);
          dim3 gridDim(num_blocks, 1); //, num_teuk_modes);
          //dim3 threadDim(32, 32);
          // launch one worker kernel per stream

          make_waveform<<<gridDim, NUM_THREADS, 0, streams[i]>>>(d_waveform,
                        y_vals, c1, c2, c3,
                        d_m, d_n, num_teuk_modes, d_Ylms,
                        delta_t, h_t[i], i, start_inds[i], start_inds[i+1], init_len);

      }
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());

      #pragma omp parallel for
      for (int i = 0; i < init_len-1; i++) {
            cudaStreamDestroy(streams[i]);

        }

        //cudaEventRecord(stop);

        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&milliseconds, start, stop);
        //printf("after: %e\n", milliseconds);

      /*int num_blocks = std::ceil((input_len + NUM_THREADS -1)/NUM_THREADS);
      dim3 gridDim(num_blocks, num_teuk_modes); //, num_teuk_modes);
      make_waveform<<<gridDim, NUM_THREADS>>>(d_waveform, d_teuk_modes, d_Phi_phi, d_Phi_r, d_m, d_n, input_len, num_teuk_modes, d_Ylms, num_n);
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());*/
}
