#include <math.h>
#include <random>
#include "global.h"
#include "cublas_v2.h"
#include "kernel.hh"
//#include "interpolate.hh"
#include <thrust/sort.h>
#include "omp.h"

#define NUM_THREADS 256

typedef thrust::device_vector<double>::iterator Iterator;

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

__global__
void add_bias(fod *C, fod *bias, int input_len, int dim2){

 for (int j = blockIdx.y * blockDim.y + threadIdx.y;
      j < dim2;
      j += blockDim.y * gridDim.y){

for (int i = blockIdx.x * blockDim.x + threadIdx.x;
     i < input_len;
     i += blockDim.x * gridDim.x){

       C[input_len*j + i] = C[input_len*j + i] + bias[j];
  }
}
}

void test_layer(long ptr_mat_out, long ptr_mat_in, long ptr_layer_weight, long ptr_bias, int m, int n, int k, int run_bias, int run_activation)

{

fod *mat_out = (fod*)ptr_mat_out;
fod *mat_in = (fod*)ptr_mat_in;
fod *layer_weight = (fod*) ptr_layer_weight;
fod *layer_bias = (fod*) ptr_bias;

    cublasHandle_t handle;

    char * status;
    cublasStatus_t stat;
    fod alpha = 1.0;
    fod beta = 0.0;
    stat = cublasCreate(&handle);
       if (stat != CUBLAS_STATUS_SUCCESS) {
               printf ("CUBLAS initialization failed\n");
               exit(0);
           }

    stat = cublasDgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           m, n, k,
                           &alpha,
                           mat_in, m,
                           layer_weight, k,
                           &beta,
                           mat_out, m);

    if (stat != CUBLAS_STATUS_SUCCESS) {
           printf ("CUBLAS initialization failed\n");
           exit(0);
       }

int num_threads = 256;
int num_blocks = std::ceil((m + num_threads -1)/num_threads);
dim3 gridDim(num_blocks, n);


//if (run_bias)
  // add_bias<<<gridDim, num_threads>>>(mat_out, layer_bias, m, n);

//else if (run_activation)
  add_bias_relu<<<gridDim, num_threads>>>(mat_out, layer_bias, m, n);
   cudaDeviceSynchronize();
   gpuErrchk(cudaGetLastError());

   stat = cublasDestroy(handle);
      if (stat != CUBLAS_STATUS_SUCCESS) {
              printf ("CUBLAS initialization failed\n");
              exit(0);
          }

}


__global__
void printit3(fod *arr, int m, int n)
{
    for (int i=0; i<m; i++){
        printf("[");
        for (int j=0; j<n; j++){
            printf("%.18e", arr[j*m + i]);
            if (j != n-1) printf(", ");
        }
        printf("],\n");
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


  stat = cublasDgemm(handle,
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
  if (dim2 == 194) add_bias<<<gridDim, NUM_THREADS>>>(C, layer_bias, input_len, dim2);
  else add_bias_relu<<<gridDim, NUM_THREADS>>>(C, layer_bias, input_len, dim2);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  /*
  if (dim2 == 194){
    printit3<<<1,1>>>(C, m, n);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}*/
  stat = cublasDestroy(handle);
     if (stat != CUBLAS_STATUS_SUCCESS) {
             printf ("CUBLAS initialization failed\n");
             exit(0);
         }
}

__global__
void form_complex_output(cuDoubleComplex *complex_output, fod *nn_output, int input_len, int break_index,
                          cuDoubleComplex d_transform_factor_inv){
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < input_len;
       i += blockDim.x * gridDim.x){

   for (int ind = blockIdx.y * blockDim.y + threadIdx.y;
        ind < break_index;
        ind += blockDim.y * gridDim.y){
            complex_output[ind*input_len + i].x = nn_output[ind*input_len + i];
            complex_output[ind*input_len + i].y = nn_output[(break_index+ind)*input_len + i];
            complex_output[ind*input_len + i] = cuCmul(complex_output[ind*input_len + i], d_transform_factor_inv);
         }
  }
}

void transform_output(cuDoubleComplex *d_teuk_modes, cuDoubleComplex *d_transform_matrix, cuDoubleComplex *d_nn_output_mat, fod *d_C,
                      int input_len, int break_index, cuDoubleComplex d_transform_factor_inv,
                      int num_teuk_modes){
  int num_blocks = std::ceil((input_len + NUM_THREADS -1)/NUM_THREADS);
  dim3 gridDim(num_blocks, break_index);
  form_complex_output<<<gridDim, NUM_THREADS>>>(d_nn_output_mat, d_C, input_len, break_index, d_transform_factor_inv);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());


  int m=input_len, k=break_index, n=num_teuk_modes;
  char * status;
  cublasHandle_t handle;
  cublasStatus_t stat;
  cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
  cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
  stat = cublasCreate(&handle);
     if (stat != CUBLAS_STATUS_SUCCESS) {
             printf ("CUBLAS initialization failed\n");
             exit(0);
         }

  stat = cublasZgemm(handle,
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

__global__
void check_mode_power(cuDoubleComplex *input_mode_vals,  cuDoubleComplex *Ylms, int length, int num_teuk_modes, int *filter_modes_buffer, fod tol,
                      int num_n, int num_l_m, int *m_arr, fod *working_modes_all, int *ind_working_modes_all)
{

    int lm_i, m_fac;
    cuDoubleComplex Ylm;

    int offset = num_teuk_modes/ blockDim.x;
    int final_ind;
    for (int i=threadIdx.y+blockDim.y*blockIdx.y;
         i<length;
         i+=blockDim.y*gridDim.y){

             for (int mode_i=threadIdx.x+blockDim.x*blockIdx.x; mode_i<num_teuk_modes; mode_i+=blockDim.x*gridDim.x){
                 m_fac = m_arr[mode_i] > 0;
                 lm_i = mode_i / num_n;
                 Ylm = Ylms[lm_i];
                 working_modes_all[i*num_teuk_modes + mode_i] = (1+m_fac)*pow(cuCabs(cuCmul(input_mode_vals[mode_i*length + i], Ylm)), 2.0);
                 //if ((i==0) && (mode_i < 10)) printf("%d %d %e %e\n", length, mode_i, cuCrealf(input_mode_vals[mode_i*length + i]), cuCimagf(input_mode_vals[mode_i*length + i]));
                 ind_working_modes_all[i*num_teuk_modes + mode_i] = mode_i;
             }
         }
}

__global__
void reset_buffer(int *filter_modes_buffer, int num_teuk_modes)
{
    for (int i=threadIdx.x + blockDim.x*blockIdx.x;
         i<num_teuk_modes;
         i+=blockDim.x*gridDim.x){
             filter_modes_buffer[i] = 0;
    }
}

__global__
void printit2(fod *arr, int *ind_arr, int n)
{
    for (int i=0; i<n; i++) printf("%d %.18e\n", ind_arr[i], arr[i]);
}

__device__
void cumsum(fod arr[], int length)
{

    for (int i=length-2; i>=0; i-=1) arr[i] += arr[i+1];

}


__global__
void select_modes(int *filter_modes_buffer, fod *working_modes_all, int *ind_working_modes_all, int num_teuk_modes, int length, fod tol)
{

    for (int i=threadIdx.x+blockDim.x*blockIdx.x;
         i<length;
         i+=blockDim.x*gridDim.x){

             fod *working_modes = &working_modes_all[i*num_teuk_modes];
             int *ind_working_modes = &ind_working_modes_all[i*num_teuk_modes];

                cumsum(working_modes, num_teuk_modes); // could implement faster but really don't need to
                //if (i==0) for (int t=0; t<num_teuk_modes; t++) printf("after %d %d %e\n", t, ind_working_modes[t], working_modes[t]);

                 fod total_power = working_modes[0];
                 int j;
                 for (j=num_teuk_modes-1; j>0; j-=1){
                        atomicAdd(&filter_modes_buffer[ind_working_modes[j]], 1);
                        if (working_modes[j] > (1.0-tol)*total_power) break;
                 }
}
}

__global__
void produce_info_array(int *mode_keep_inds, int *filter_modes_buffer, int num_teuk_modes, int *num_modes_kept)
{

    int j = 0;
    for (int i=0; i<num_teuk_modes; i+=1){
        if (filter_modes_buffer[i] > 0){
            mode_keep_inds[j] = i;
            j+=1;
            //printf("INIT %d, %d\n", i, j);
        }

    }
    *num_modes_kept = j;
}


void filter_modes(FilterContainer *filter, cuDoubleComplex *d_teuk_modes, cuDoubleComplex *d_Ylms,
                  int *d_m_arr, int num_teuk_modes, int length, int num_n, int num_l_m)
{

    int number_of_threads = 256;
    int num_blocks_reset = std::ceil((num_teuk_modes + number_of_threads - 1)/number_of_threads);
    int num_blocks_filter = std::ceil((length + number_of_threads - 1)/number_of_threads);

    reset_buffer<<<num_blocks_reset, number_of_threads>>>(filter->d_filter_modes_buffer, num_teuk_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    dim3 check_dim(num_blocks_reset, length);
    check_mode_power<<<check_dim, number_of_threads>>>(d_teuk_modes, d_Ylms, length, num_teuk_modes, filter->d_filter_modes_buffer,
                                                                filter->tol, num_n, num_l_m, d_m_arr, filter->working_modes_all, filter->ind_working_modes_all);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    /*
    printit2<<<1,1>>>(working_modes_all, ind_working_modes_all, 40);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    */

    //printf("\n\n");
    #pragma omp parallel for
    for (int i=0; i<length; i++){
        //printf("%d sort\n", omp_get_thread_num());
        thrust::device_ptr<int> ind_t_a(&filter->ind_working_modes_all[i*num_teuk_modes]);
        thrust::device_ptr<fod> t_a(&filter->working_modes_all[i*num_teuk_modes]);  // add this line before the sort line
        thrust::sort_by_key(t_a, t_a + num_teuk_modes, ind_t_a);        // modify your sort line
    }


    select_modes<<<length, number_of_threads>>>(filter->d_filter_modes_buffer,
                                                filter->working_modes_all,
                                                filter->ind_working_modes_all,
                                                num_teuk_modes,
                                                length,
                                                filter->tol);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    /*
    printit2<<<1,1>>>(working_modes_all, d_filter_modes_buffer, num_teuk_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());*/


    produce_info_array<<<1,1>>>(filter->d_mode_keep_inds, filter->d_filter_modes_buffer,
                                num_teuk_modes, filter->d_num_modes_kept);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaMemcpy(&filter->num_modes_kept, filter->d_num_modes_kept, sizeof(int), cudaMemcpyDeviceToHost));

    /*printf("OUTOFIT: %d\n", num_modes_kept);
    printit2<<<1,1>>>(working_modes_all, d_mode_keep_inds, num_modes_kept+1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());*/

}




__host__ __device__ cuDoubleComplex complex_exp(cuDoubleComplex arg){
  cuDoubleComplex res;
  fod s, c;
  fod e = exp(arg.x);
  sincos(arg.y, &s, &c);
  res.x = c * e;
  res.y = s * e;
  return res;
}

__device__
cuDoubleComplex get_mode_value(cuDoubleComplex teuk_mode, fod Phi_phi, fod Phi_r, int m, int n, cuDoubleComplex Ylm){
    cuDoubleComplex minus_I = make_cuDoubleComplex(0.0, -1.0);
    fod phase = m*Phi_phi + n*Phi_r;
    cuDoubleComplex out = cuCmul(cuCmul(teuk_mode, Ylm), complex_exp(cuCmul(minus_I, make_cuDoubleComplex(phase, 0.0))));
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

__device__ void atomicAddComplex(cuDoubleComplex* a, cuDoubleComplex b){
  //transform the addresses of real and imag. parts to double pointers
  double *x = (double*)a;
  double *y = x+1;
  //use atomicAdd for double variables
  atomicAddDouble(x, cuCreal(b));
  atomicAddDouble(y, cuCimag(b));
}

__global__
void make_waveform(cuDoubleComplex *waveform,
              InterpContainer *Phi_phi_, InterpContainer *Phi_r_, InterpContainer *modes,
              int *m_arr, int *n_arr, int num_teuk_modes, cuDoubleComplex *Ylms, int num_n,
              fod delta_t, fod start_t, int old_ind, int start_ind, int end_ind, int num_l_m, int*mode_keep_inds){

    cuDoubleComplex trans = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex mode_val;
    cuDoubleComplex trans_plus_m, trans_minus_m;
    fod Phi_phi_i, Phi_r_i, t, x, x2, x3, mode_val_re, mode_val_im;
    int lm_i;
     fod re_y, re_c1, re_c2, re_c3, im_y, im_c1, im_c2, im_c3;
     fod pp_y, pp_c1, pp_c2, pp_c3, pr_y, pr_c1, pr_c2, pr_c3;
     int m, n, actual_mode_index;
     cuDoubleComplex Ylm_plus_m, Ylm_minus_m;

    //__shared__ int mode_keep_inds[2214];

    //for (int i=threadIdx.x; i<num_teuk_modes; i+=blockDim.x) {
    //    mode_keep_inds[i] = mode_keep_inds_trans[i];
    //}
    __syncthreads();

    pp_y = Phi_phi_->y[old_ind]; pp_c1 = Phi_phi_->c1[old_ind]; pp_c2 = Phi_phi_->c2[old_ind]; pp_c3 = Phi_phi_->c3[old_ind];
    pr_y = Phi_phi_->y[old_ind]; pr_c1 = Phi_phi_->c1[old_ind]; pr_c2 = Phi_phi_->c2[old_ind]; pr_c3 = Phi_phi_->c3[old_ind];

    //int j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = start_ind + blockIdx.x * blockDim.x + threadIdx.x;
         i < end_ind;
         i += blockDim.x * gridDim.x){

      trans = make_cuDoubleComplex(0.0, 0.0);
     t = delta_t*i;
      x = t - start_t;
      x2 = x*x;
      x3 = x*x2;

      Phi_phi_i = pp_y + pp_c1*x + pp_c2*x2  + pp_c3*x3;
      Phi_r_i = pr_y + pr_c1*x + pr_c2*x2  + pr_c3*x3;

        for (int j=0; j<num_teuk_modes; j++){

            actual_mode_index = mode_keep_inds[j];
            lm_i = actual_mode_index / num_n;
            Ylm_plus_m = Ylms[lm_i];
            Ylm_minus_m = Ylms[num_l_m + lm_i];

             re_y = modes[2*j].y[old_ind]; re_c1 = modes[2*j].c1[old_ind]; re_c2 = modes[2*j].c2[old_ind]; re_c3 = modes[2*j].c3[old_ind];
             im_y = modes[2*j].y[old_ind]; im_c1 = modes[2*j].c1[old_ind]; im_c2 = modes[2*j].c2[old_ind]; im_c3 = modes[2*j].c3[old_ind];

             m = m_arr[actual_mode_index];
             n = n_arr[actual_mode_index];

            mode_val_re =  re_y + re_c1*x + re_c2*x2  + re_c3*x3;
            mode_val_im = im_y + im_c1*x + im_c2*x2  + im_c3*x3;
            mode_val = make_cuDoubleComplex(mode_val_re, mode_val_im);

            //if (i==0) printf("%d %d, %lf + %lfi\n", m[j], n[j], cuCrealf(Ylm), cuCimagf(Ylm));
                trans_plus_m = get_mode_value(mode_val, Phi_phi_i, Phi_r_i, m, n, Ylm_plus_m);
                trans = cuCadd(trans_plus_m, trans);

                // minus m
                if (m != 0){
                    trans_minus_m = get_mode_value(cuConj(mode_val), Phi_phi_i, Phi_r_i, -m, -n, Ylm_minus_m);
                    trans = cuCadd(trans_minus_m, trans);
                }
                //atomicAddComplex(&waveform[i], mode_val);
        }
      waveform[i] = trans;
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

void get_waveform(cuDoubleComplex *d_waveform,
              InterpContainer *d_interp_Phi_phi, InterpContainer *d_interp_Phi_r, InterpContainer *d_modes,
              int *d_m, int *d_n, int init_len, int out_len, int num_teuk_modes, cuDoubleComplex *d_Ylms, int num_n,
              fod delta_t, fod *h_t, int num_l_m, FilterContainer *filter){


    int start_inds[init_len];
    int unit_length[init_len-1];

    find_start_inds(start_inds, unit_length, h_t, delta_t, init_len, out_len);

    //printf("Num modes: %d\n", num_teuk_modes);
    cudaStream_t streams[init_len-1];

    #pragma omp parallel for
    for (int i = 0; i < init_len-2; i++) {
          cudaStreamCreate(&streams[i]);
          int num_blocks = std::ceil((unit_length[i] + NUM_THREADS -1)/NUM_THREADS);
          //printf("%d %d %d %d, %d %d\n", i, start_inds[i], unit_length[i], num_blocks, init_len, out_len);
          if (num_blocks == 0) continue;
          dim3 gridDim(num_blocks, 1); //, num_teuk_modes);
          // launch one worker kernel per stream

          make_waveform<<<gridDim, NUM_THREADS, 0, streams[i]>>>(d_waveform,
                        d_interp_Phi_phi, d_interp_Phi_r, d_modes,
                        d_m, d_n, num_teuk_modes, d_Ylms, num_n,
                        delta_t, h_t[i], i, start_inds[i], start_inds[i+1], num_l_m, filter->d_mode_keep_inds);

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
      gpuErrchk(cudaGetLastError());*/
}
