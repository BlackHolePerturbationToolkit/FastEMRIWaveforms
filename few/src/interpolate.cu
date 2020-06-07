#include "global.h"
#include "interpolate.hh"
#include "cusparse.h"


__device__
void fill_coefficients(int i, int length, fod *dydx, fod dx, fod *y, fod *coeff1, fod *coeff2, fod *coeff3){
  fod slope, t, dydx_i;

  slope = (y[i+1] - y[i])/dx;

  dydx_i = dydx[i];

  t = (dydx_i + dydx[i+1] - 2*slope)/dx;

  coeff1[i] = dydx_i;
  coeff2[i] = (slope - dydx_i) / dx - t;
  coeff3[i] = t/dx;
}


__device__
void prep_splines(int i, int length, double *b, double *ud, double *diag, double *ld, double *x, double *y){
  fod dx1, dx2, d, slope1, slope2;
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

    fod dt;
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



void fit_wrap(int m, int n, fod *a, fod *b, fod *c, fod *d_in){

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
