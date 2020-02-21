#include "global.h"
#include "interpolate.hh"

void create_interp_containers(InterpContainer *d_interp, InterpContainer *h_interp, fod *y, int length)
{


  gpuErrchk(cudaMalloc(&h_interp->y, length*sizeof(fod)));
  gpuErrchk(cudaMalloc(&h_interp->c1, length*sizeof(fod)-1));
  gpuErrchk(cudaMalloc(&h_interp->c2, length*sizeof(fod)-1));
  gpuErrchk(cudaMalloc(&h_interp->c3, length*sizeof(fod)-1));

  gpuErrchk(cudaMemcpy(h_interp->y, y, length*sizeof(fod), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_interp, h_interp, sizeof(InterpContainer), cudaMemcpyHostToDevice));

}

void destroy_interp_containers(InterpContainer *d_interp, InterpContainer *h_interp)
{

  gpuErrchk(cudaFree(h_interp->y));
  gpuErrchk(cudaFree(h_interp->c1));
  gpuErrchk(cudaFree(h_interp->c2));
  gpuErrchk(cudaFree(h_interp->c3));

}

__device__
void prep_splines(int i, int length, fod *b, fod *ud, fod *diag, fod *ld, fod *x, fod *y){
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


__device__
void fill_B(fod *p, fod *e, fod *Phi_phi, fod *Phi_r,
                 fod *B, fod *t_arr, fod *upper_diag, fod *diag, fod *lower_diag, int length, int i){
    int lead_ind;

    // p
    lead_ind = 0*length;
    prep_splines(i, length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], t_arr, p);

    // e
    lead_ind = 1*length;
    prep_splines(i, length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], t_arr, e);

    // Phi_phi
    lead_ind = 2*length;
    prep_splines(i, length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], t_arr, Phi_phi);

    // Phi_r
    lead_ind = 3*length;
    prep_splines(i, length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], t_arr, Phi_r);

}

__global__
void fill_B_wrap(InterpContainer *p_, InterpContainer *e_, InterpContainer *Phi_phi_, InterpContainer *Phi_r_,
                      fod *t_arr, fod *B, fod *upper_diag, fod *diag, fod *lower_diag, int length){

        fod *p = p_->y;
        fod *e = e_->y;
        fod *Phi_phi = Phi_phi_->y;
        fod *Phi_r = Phi_r_->y;

       for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < length;
            i += blockDim.x * gridDim.x){

              fill_B(p, e, Phi_phi, Phi_r, B, t_arr, upper_diag, diag, lower_diag, length, i);

}
}

void fit_constants_serial_wrap(int m, int n, fod *a, fod *b, fod *c, fod *d_in){

  void *pBuffer;
  cusparseStatus_t stat;
  cusparseHandle_t handle;

  size_t bufferSizeInBytes;

  CUSPARSE_CALL(cusparseCreate(&handle));
  CUSPARSE_CALL( cusparseSgtsv2StridedBatch_bufferSizeExt(handle, m, a, b, c, d_in, n, m, &bufferSizeInBytes));
  gpuErrchk(cudaMalloc(&pBuffer, bufferSizeInBytes));

    CUSPARSE_CALL(cusparseSgtsv2StridedBatch(handle,
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
void set_spline_constants(InterpContainer *p_, InterpContainer *e_, InterpContainer *Phi_phi_, InterpContainer *Phi_r_,
                          fod *B, int length, int i, fod dt){

  int lead_ind;

  // p
  lead_ind = 0*length;
  fill_coefficients(i, length, &B[lead_ind], dt, p_->y, p_->c1, p_->c2, p_->c3);

  // e
  lead_ind = 1*length;
  fill_coefficients(i, length, &B[lead_ind], dt, e_->y, e_->c1, e_->c2, e_->c3);

  // Phi_phi
  lead_ind = 2*length;
  fill_coefficients(i, length, &B[lead_ind], dt, Phi_phi_->y, Phi_phi_->c1, Phi_phi_->c2, Phi_phi_->c3);

  // Phi_r
  lead_ind = 3*length;
  fill_coefficients(i, length, &B[lead_ind], dt, Phi_r_->y, Phi_r_->c1, Phi_r_->c2, Phi_r_->c3);

}

__global__
void set_spline_constants_wrap(InterpContainer *p_, InterpContainer *e_, InterpContainer *Phi_phi_, InterpContainer *Phi_r_,
                               fod *B, int length, fod *t_arr){

    fod dt;

        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < length-1;
             i += blockDim.x * gridDim.x){

              dt = t_arr[i + 1] - t_arr[i];

               set_spline_constants(p_, e_, Phi_phi_, Phi_r_, B, length, i, dt);
}
}


void setup_interpolate(InterpContainer *d_interp_p, InterpContainer *d_interp_e, InterpContainer *d_interp_Phi_phi, InterpContainer *d_interp_Phi_r,
                       fod *d_t, int length)
{

  int num_pars = 4;
  fod *upper_diag, *lower_diag, *diag, *B;

  gpuErrchk(cudaMalloc(&upper_diag, num_pars*length*sizeof(fod)));
  gpuErrchk(cudaMalloc(&lower_diag, num_pars*length*sizeof(fod)));
  gpuErrchk(cudaMalloc(&diag, num_pars*length*sizeof(fod)));
  gpuErrchk(cudaMalloc(&B, num_pars*length*sizeof(fod)));

  int NUM_THREADS = 256;
  int num_blocks = std::ceil((length + NUM_THREADS -1)/NUM_THREADS);
  dim3 gridDim(num_blocks); //, num_teuk_modes);
  fill_B_wrap<<<gridDim, NUM_THREADS>>>(d_interp_p, d_interp_e, d_interp_Phi_phi, d_interp_Phi_r,
                        d_t, B, upper_diag, diag, lower_diag, length);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  fit_constants_serial_wrap(length, num_pars, lower_diag, diag, upper_diag, B);

  set_spline_constants_wrap<<<gridDim, NUM_THREADS>>>(d_interp_p, d_interp_e, d_interp_Phi_phi, d_interp_Phi_r,
                                 B, length, d_t);
 cudaDeviceSynchronize();
 gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaFree(upper_diag));
  gpuErrchk(cudaFree(lower_diag));
  gpuErrchk(cudaFree(diag));
  gpuErrchk(cudaFree(B));

}
