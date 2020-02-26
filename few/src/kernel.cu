#include <math.h>
#include <random>
#include "global.h"
#include "cublas_v2.h"
#include "kernel.hh"
#include "interpolate.hh"

#define NUM_THREADS 256

__device__ __host__ fod LeakyReLU(fod x){
     fod out = 0.0;
     if (x>= 0.0) {out = x;}
     else {out = 0.2*x;}
     return out;
}

__global__
void add_bias_relu(fod *C, fod *bias, int input_len, int dim2){

 for (int j = blockIdx.y * blockDim.y + threadIdx.y;
      j < dim2;
      j += blockDim.y * gridDim.y){

for (int i = blockIdx.x * blockDim.x + threadIdx.x;
     i < input_len;
     i += blockDim.x * gridDim.x){

       C[input_len*j + i] = LeakyReLU(C[input_len*j + i] + bias[j]);
  }
}
}


void run_layer(fod *C, fod *layer_weight, fod *layer_bias, int dim1, int dim2, int input_len){
  int m=input_len, k=dim1, n=dim2;
  char * status;
  cublasHandle_t handle;
  cublasStatus_t stat;
  fod alpha = 1.0;
  fod beta = 0.0;
  stat = cublasCreate(&handle);
     if (stat != CUBLAS_STATUS_SUCCESS) {
             printf ("CUBLAS initialization failed\n");
             exit(0);
         }

  stat = cublasSgemm(handle,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         m, n, k,
                         &alpha,
                         C, m,
                         layer_weight, k,
                         &beta,
                         C, m);



   status = _cudaGetErrorEnum(stat);
    cudaDeviceSynchronize();

    if (stat != CUBLAS_STATUS_SUCCESS) {
            exit(0);
        }

  int num_blocks = std::ceil((input_len + NUM_THREADS -1)/NUM_THREADS);
  dim3 gridDim(num_blocks, dim2);
  add_bias_relu<<<gridDim, NUM_THREADS>>>(C, layer_bias, input_len, dim2);
  cudaDeviceSynchronize();
  gpuErrchk_here(cudaGetLastError());
}

__global__
void form_complex_output(cuComplex *complex_output, fod *nn_output, int input_len, int break_index,
                          cuComplex d_transform_factor_inv){
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < input_len;
       i += blockDim.x * gridDim.x){

   for (int ind = blockIdx.y * blockDim.y + threadIdx.y;
        ind < break_index;
        ind += blockDim.y * gridDim.y){
            complex_output[ind*input_len + i].x = nn_output[ind*input_len + i];
            complex_output[ind*input_len + i].y = nn_output[(break_index+ind)*input_len + i];
            complex_output[ind*input_len + i] = cuCmulf(complex_output[ind*input_len + i], d_transform_factor_inv);
         }
  }
}

void transform_output(cuComplex *d_teuk_modes, cuComplex *d_transform_matrix, cuComplex *d_nn_output_mat, fod *d_C,
                      int input_len, int break_index, cuComplex d_transform_factor_inv,
                      int num_teuk_modes){
  int num_blocks = std::ceil((input_len + NUM_THREADS -1)/NUM_THREADS);
  dim3 gridDim(num_blocks, break_index);
  form_complex_output<<<gridDim, NUM_THREADS>>>(d_nn_output_mat, d_C, input_len, break_index, d_transform_factor_inv);
  cudaDeviceSynchronize();
  gpuErrchk_here(cudaGetLastError());

  int m=input_len, k=break_index, n=num_teuk_modes;
  char * status;
  cublasHandle_t handle;
  cublasStatus_t stat;
  cuComplex alpha = make_cuComplex(1.0, 0.0);
  cuComplex beta = make_cuComplex(0.0, 0.0);
  stat = cublasCreate(&handle);
     if (stat != CUBLAS_STATUS_SUCCESS) {
             printf ("CUBLAS initialization failed\n");
             exit(0);
         }

  stat = cublasCgemm(handle,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         m, n, k,
                         &alpha,
                         d_nn_output_mat, m,
                         d_transform_matrix, k,
                         &beta,
                         d_teuk_modes, m);

   status = _cudaGetErrorEnum(stat);
    cudaDeviceSynchronize();

    if (stat != CUBLAS_STATUS_SUCCESS) {
            exit(0);
        }

}




__host__ __device__ cuComplex complex_exp(cuComplex arg){
  cuComplex res;
  fod s, c;
  fod e = exp(arg.x);
  sincos(arg.y, &s, &c);
  res.x = c * e;
  res.y = s * e;
  return res;
}

__device__
cuComplex get_mode_value(cuComplex teuk_mode, fod Phi_phi, fod Phi_r, int m, int n, cuComplex Ylm){
    cuComplex minus_I = make_cuComplex(0.0, -1.0);
    float phase = m*Phi_phi + n*Phi_r;
    cuComplex out = cuCmulf(cuCmulf(teuk_mode, Ylm), complex_exp(cuCmulf(minus_I, make_cuComplex(phase, 0.0))));
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

__device__ void atomicAddComplex(cuComplex* a, cuComplex b){
  //transform the addresses of real and imag. parts to double pointers
  double *x = (double*)a;
  double *y = x+1;
  //use atomicAdd for double variables
  atomicAddDouble(x, cuCrealf(b));
  atomicAddDouble(y, cuCimagf(b));
}

__global__
void make_waveform(cuComplex *waveform,
              InterpContainer *Phi_phi_, InterpContainer *Phi_r_, InterpContainer *modes,
              int *m_arr, int *n_arr, int num_teuk_modes, cuComplex *Ylms, int num_n,
              fod delta_t, fod start_t, int old_ind, int start_ind, int end_ind){

    cuComplex trans = make_cuComplex(0.0, 0.0);
    cuComplex mode_val;
    fod Phi_phi_i, Phi_r_i, t, x, x2, x3, mode_val_re, mode_val_im;
    int lm_i;
    __shared__ fod re_y, re_c1, re_c2, re_c3, im_y, im_c1, im_c2, im_c3;
    __shared__ fod pp_y, pp_c1, pp_c2, pp_c3, pr_y, pr_c1, pr_c2, pr_c3;
    __shared__ int m, n;
    __shared__ cuComplex Ylm;

    if (threadIdx.x == 0){

    pp_y = Phi_phi_->y[old_ind]; pp_c1 = Phi_phi_->c1[old_ind]; pp_c2 = Phi_phi_->c2[old_ind]; pp_c3 = Phi_phi_->c3[old_ind];
    pr_y = Phi_phi_->y[old_ind]; pr_c1 = Phi_phi_->c1[old_ind]; pr_c2 = Phi_phi_->c2[old_ind]; pr_c3 = Phi_phi_->c3[old_ind];

    int j = blockIdx.y * blockDim.y + threadIdx.y;

       lm_i = j / num_n;
       Ylm = Ylms[lm_i];

        re_y = modes[2*j].y[old_ind]; re_c1 = modes[2*j].c1[old_ind]; re_c2 = modes[2*j].c2[old_ind]; re_c3 = modes[2*j].c3[old_ind];
        im_y = modes[2*j].y[old_ind]; im_c1 = modes[2*j].c1[old_ind]; im_c2 = modes[2*j].c2[old_ind]; im_c3 = modes[2*j].c3[old_ind];

        m = m_arr[j];
        n = n_arr[j];

    }
    __syncthreads();

    for (int i = start_ind + blockIdx.x * blockDim.x + threadIdx.x;
         i < end_ind;
         i += blockDim.x * gridDim.x){

     t = delta_t*i;
      x = t - start_t;
      x2 = x*x;
      x3 = x*x2;

      Phi_phi_i = pp_y + pp_c1*x + pp_c2*x2  + pp_c3*x3;
      Phi_r_i = pr_y + pr_c1*x + pr_c2*x2  + pr_c3*x3;

        mode_val_re =  re_y + re_c1*x + re_c2*x2  + re_c3*x3;
        mode_val_im = im_y + im_c1*x + im_c2*x2  + im_c3*x3;
        mode_val = make_cuComplex(mode_val_re, mode_val_im);

        //if (i==0) printf("%d %d, %lf + %lfi\n", m[j], n[j], cuCrealf(Ylm), cuCimagf(Ylm));
            mode_val = get_mode_value(mode_val, Phi_phi_i, Phi_r_i, m, n, Ylm);
            atomicAddComplex(&waveform[i], mode_val);

  }
}

void find_start_inds(int start_inds[], int unit_length[], fod *t_arr, fod delta_t, int length, int new_length)
{

  start_inds[0] = 0;
  for (int i = 1;
       i < length;
       i += 1){

          fod t = t_arr[i];

          start_inds[i] = (int)std::ceil(t/delta_t);
          unit_length[i-1] = start_inds[i] - start_inds[i-1];

      }

  start_inds[length -1] = new_length;
  unit_length[length - 2] = start_inds[length -1] - start_inds[length -2];
}

void get_waveform(cuComplex *d_waveform,
              InterpContainer *d_interp_Phi_phi, InterpContainer *d_interp_Phi_r, InterpContainer *d_modes,
              int *d_m, int *d_n, int init_len, int out_len, int num_teuk_modes, cuComplex *d_Ylms, int num_n,
              fod delta_t, fod *h_t){

    int start_inds[init_len];
    int unit_length[init_len-1];

    find_start_inds(start_inds, unit_length, h_t, delta_t, init_len, out_len);

    cudaStream_t streams[init_len-1];

    for (int i = 0; i < init_len-2; i++) {
          cudaStreamCreate(&streams[i]);
          int num_blocks = std::ceil((unit_length[i] + NUM_THREADS -1)/NUM_THREADS);
          //printf("%d %d %d %d, %d %d\n", i, start_inds[i], unit_length[i], num_blocks, init_len, out_len);
          if (num_blocks == 0) continue;
          dim3 gridDim(num_blocks, num_teuk_modes); //, num_teuk_modes);
          // launch one worker kernel per stream
          make_waveform<<<num_blocks, NUM_THREADS, 0, streams[i]>>>(d_waveform,
                        d_interp_Phi_phi, d_interp_Phi_r, d_modes,
                        d_m, d_n, num_teuk_modes, d_Ylms, num_n,
                        delta_t, h_t[i], i, start_inds[i], start_inds[i+1]);

      }
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());
      for (int i = 0; i < init_len-2; i++) {
            cudaStreamDestroy(streams[i]);

        }

      /*int num_blocks = std::ceil((input_len + NUM_THREADS -1)/NUM_THREADS);
      dim3 gridDim(num_blocks, num_teuk_modes); //, num_teuk_modes);
      make_waveform<<<gridDim, NUM_THREADS>>>(d_waveform, d_teuk_modes, d_Phi_phi, d_Phi_r, d_m, d_n, input_len, num_teuk_modes, d_Ylms, num_n);
      cudaDeviceSynchronize();
      gpuErrchk_here(cudaGetLastError());*/
}
