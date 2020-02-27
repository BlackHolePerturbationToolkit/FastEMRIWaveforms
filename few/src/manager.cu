/*
This is the central piece of code. This file implements a class
that takes data in on the cpu side, copies
it to the gpu, and exposes functions that let
you perform actions with the GPU

This class will get translated into python via cython
*/

#include <kernel.hh>
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include "global.h"
#include <complex>
#include "cuComplex.h"
#include "elliptic.hh"
#include "ylm.hh"
#include "FluxInspiral.hh"
#include "interpolate.hh"

using namespace std;

#define NUM_THREADS 256

FastEMRIWaveforms::FastEMRIWaveforms (int time_batch_size_, int num_layers_, int *dim1_, int *dim2_,
    fod *flatten_weight_matrix, fod *flattened_bias_matrix,
    cmplx*transform_matrix, int trans_dim1_, int trans_dim2_, fod transform_factor_,
    int break_index_,
    int *l_, int *m_, int *n_,
    int max_input_len_, int num_l_m_, int num_n_, fod delta_t_,
    int max_init_len_)
{
    max_input_len = max_input_len_;
    time_batch_size = time_batch_size_;
    num_layers = num_layers_;
    dim1 = dim1_;
    dim2 = dim2_;
    break_index = break_index_;
    delta_t = delta_t_;

    trans_dim1 = trans_dim1_;
    trans_dim2 = trans_dim2_;
    transform_factor = transform_factor_;

    num_n = num_n_;
    num_l_m = num_l_m_;

    l_arr = l_;
    m_arr = m_;
    max_init_len = max_init_len_;

    num_teuk_modes = trans_dim2;

    d_transform_factor_inv = make_cuComplex(1./transform_factor, 0.0);

    d_layers_matrix = new fod*[num_layers];
    d_layers_bias = new fod*[num_layers];


    int start_int_weights = 0;
    int start_int_bias = 0;
    for (int i=0; i<num_layers; i++){
      gpuErrchk(cudaMalloc(&(d_layers_matrix[i]), dim1[i]*dim2[i]*sizeof(fod)));
      gpuErrchk(cudaMemcpy(d_layers_matrix[i], &flatten_weight_matrix[start_int_weights], dim1[i]*dim2[i]*sizeof(fod), cudaMemcpyHostToDevice));

      gpuErrchk(cudaMalloc(&d_layers_bias[i], dim2[i]*sizeof(fod)));
      gpuErrchk(cudaMemcpy(d_layers_bias[i], &flattened_bias_matrix[start_int_bias], dim2[i]*sizeof(fod), cudaMemcpyHostToDevice));

      start_int_weights += dim1[i]*dim2[i];
      start_int_bias += dim2[i];
    }

    gpuErrchk(cudaMalloc(&d_transform_matrix, trans_dim1*trans_dim2*sizeof(cuComplex)));
    gpuErrchk(cudaMemcpy(d_transform_matrix, transform_matrix, trans_dim1*trans_dim2*sizeof(cuComplex), cudaMemcpyHostToDevice));

    // allocate buffer matrix
    dim_max = 0;
    for (int i=0; i<num_layers; i++){
        if (dim2[i] > dim_max) dim_max = dim2[i];
    }
    if (dim1[0] > dim_max) dim_max = dim1[0];

    gpuErrchk(cudaMalloc(&d_l, num_teuk_modes*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_m, num_teuk_modes*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_n, num_teuk_modes*sizeof(int)));

    gpuErrchk(cudaMemcpy(d_l, l_, num_teuk_modes*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_m, m_, num_teuk_modes*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_n, n_, num_teuk_modes*sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_Ylms, 2*num_l_m*sizeof(cuComplex)));
    Ylms = new cmplx[2*num_l_m];

    gpuErrchk(cudaMalloc(&d_C, max_init_len*dim_max*sizeof(fod)));

    int complex_dim = (int)((float) dim2[num_layers - 1]/ 2.0);
    gpuErrchk(cudaMalloc(&d_nn_output_mat, complex_dim*max_init_len*sizeof(cuComplex)));
    gpuErrchk(cudaMalloc(&d_teuk_modes, trans_dim2*max_init_len*sizeof(cuComplex)));
    gpuErrchk(cudaMalloc(&d_waveform, max_input_len*sizeof(cuComplex)));

    //printf("length is %d\n", nit_vals.length);
    temp_t = new fod[max_init_len];
    temp_p = new fod[max_init_len];
    temp_e = new fod[max_init_len];
    temp_Phi_phi = new fod[max_init_len];
    temp_Phi_r = new fod[max_init_len];

    gpuErrchk(cudaMalloc(&d_init_t, max_init_len*sizeof(fod)));
    gpuErrchk(cudaMalloc(&d_init_p, max_init_len*sizeof(fod)));
    gpuErrchk(cudaMalloc(&d_init_e, max_init_len*sizeof(fod)));
    gpuErrchk(cudaMalloc(&d_init_Phi_phi, max_init_len*sizeof(fod)));
    gpuErrchk(cudaMalloc(&d_init_Phi_r, max_init_len*sizeof(fod)));

    gpuErrchk(cudaMalloc(&d_interp_p, sizeof(InterpContainer)));
    gpuErrchk(cudaMalloc(&d_interp_e, sizeof(InterpContainer)));
    gpuErrchk(cudaMalloc(&d_interp_Phi_phi, sizeof(InterpContainer)));
    gpuErrchk(cudaMalloc(&d_interp_Phi_r, sizeof(InterpContainer)));

    h_interp_p = new InterpContainer;
    h_interp_e = new InterpContainer;
    h_interp_Phi_phi = new InterpContainer;
    h_interp_Phi_r = new InterpContainer;

    h_interp_modes = new InterpContainer[num_teuk_modes*2];
    gpuErrchk(cudaMalloc(&d_interp_modes, num_teuk_modes*2*sizeof(InterpContainer)));

    create_interp_containers(d_interp_p, h_interp_p, max_init_len);
    create_interp_containers(d_interp_e, h_interp_e, max_init_len);
    create_interp_containers(d_interp_Phi_phi, h_interp_Phi_phi, max_init_len);
    create_interp_containers(d_interp_Phi_r, h_interp_Phi_r, max_init_len);

    create_mode_interp_containers(d_interp_modes, h_interp_modes, max_init_len, num_teuk_modes);

    interp = new InterpClass(num_teuk_modes, max_init_len);

}





void FastEMRIWaveforms::run_nn(cmplx *waveform, double p0, double e0, fod theta, fod phi, int* out_len){

    //gpuErrchk(cudaMemcpy(d_Phi_phi, Phi_phi, input_len*sizeof(fod), cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy(d_Phi_r, Phi_r, input_len*sizeof(fod), cudaMemcpyHostToDevice));

    double t0 = 0.0;

    NITHolder nit_vals = run_NIT(t0, p0, e0);

    for (int i=0; i<nit_vals.length; i++){
        //printf("%e %e %e, %e, %e\n", nit_vals.t_arr[i], nit_vals.p_arr[i], nit_vals.e_arr[i], nit_vals.Phi_phi_arr[i], nit_vals.Phi_r_arr[i]);
        temp_t[i] = nit_vals.t_arr[i];
        temp_p[i] = nit_vals.p_arr[i];
        temp_e[i] = nit_vals.e_arr[i];
        temp_Phi_phi[i] = nit_vals.Phi_phi_arr[i];
        temp_Phi_r[i] = nit_vals.Phi_r_arr[i];
    }

    gpuErrchk(cudaMemcpy(d_init_t, temp_t, nit_vals.length*sizeof(fod), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_init_p, temp_p, nit_vals.length*sizeof(fod), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_init_e, temp_e, nit_vals.length*sizeof(fod), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_init_Phi_phi, temp_Phi_phi, nit_vals.length*sizeof(fod), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_init_Phi_r, temp_Phi_r, nit_vals.length*sizeof(fod), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(h_interp_p->y, temp_p, nit_vals.length*sizeof(fod), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(h_interp_e->y, temp_e, nit_vals.length*sizeof(fod), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(h_interp_Phi_phi->y, temp_Phi_phi, nit_vals.length*sizeof(fod), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(h_interp_Phi_r->y, temp_Phi_r, nit_vals.length*sizeof(fod), cudaMemcpyHostToDevice));

    int num_points = std::floor(temp_t[nit_vals.length-1]/delta_t);
    *out_len = num_points;
    //printf("%d num_points, %d max\n", num_points, max_input_len);
    assert(num_points <= max_input_len);
    assert(nit_vals.length <= max_init_len);

    gpuErrchk(cudaMemcpy(d_C, d_init_p, nit_vals.length*sizeof(fod), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(&d_C[nit_vals.length], d_init_e, nit_vals.length*sizeof(fod), cudaMemcpyDeviceToDevice));

    //ellpe_test(d_C, num_points);

  int l,m;
  for (int i=0; i<num_l_m; i+=1){
        l = l_arr[i*num_n];
        m = m_arr[i*num_n];

        Ylms[i] = SpinWeightedSpheroidalHarmonic(l, m, theta, phi);
        Ylms[num_l_m + i] = SpinWeightedSpheroidalHarmonic(l, -m, theta, phi);
        //printf("%d %d, %lf, %lf\n", l , m, Ylms[i].real(), Ylms[i].imag());
  }

  gpuErrchk(cudaMemcpy(d_Ylms, Ylms, 2*num_l_m*sizeof(cuComplex), cudaMemcpyHostToDevice));


    for (int layer_i=0; layer_i<num_layers; layer_i++){
      run_layer(d_C, d_layers_matrix[layer_i], d_layers_bias[layer_i], dim1[layer_i], dim2[layer_i], nit_vals.length);
    }

    transform_output(d_teuk_modes, d_transform_matrix, d_nn_output_mat, d_C, nit_vals.length, break_index, d_transform_factor_inv, trans_dim2);

    fill_complex_y_vals(d_interp_modes, d_teuk_modes, nit_vals.length, num_teuk_modes);

    interp->setup_interpolate(d_interp_p, d_interp_e, d_interp_Phi_phi, d_interp_Phi_r,
                      d_interp_modes, num_teuk_modes,
                           d_init_t, nit_vals.length);

     get_waveform(d_waveform,
                  d_interp_Phi_phi, d_interp_Phi_r, d_interp_modes,
                  d_m, d_n, nit_vals.length, num_points, num_teuk_modes, d_Ylms, num_n,
                  delta_t, temp_t, num_l_m);

    //gpuErrchk(cudaMemcpy(waveform, d_waveform, num_points*sizeof(cuComplex), cudaMemcpyDeviceToHost));
}


FastEMRIWaveforms::~FastEMRIWaveforms()
{
    for (int i=0; i<num_layers; i++){
      gpuErrchk(cudaFree(d_layers_matrix[i]));
      gpuErrchk(cudaFree(d_layers_bias[i]));
    }
    gpuErrchk(cudaFree(d_transform_matrix));
    gpuErrchk(cudaFree(d_l));
    gpuErrchk(cudaFree(d_m));
    gpuErrchk(cudaFree(d_n));
    gpuErrchk(cudaFree(d_Ylms));
    gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaFree(d_nn_output_mat));
    gpuErrchk(cudaFree(d_teuk_modes));

    gpuErrchk(cudaFree(d_waveform));
    delete[] d_layers_matrix;
    delete[] d_layers_bias;
    delete[] Ylms;

    delete[] temp_t;
    delete[] temp_p;
    delete[] temp_e;
    delete[] temp_Phi_phi;
    delete[] temp_Phi_r;

    destroy_interp_containers(d_interp_p, h_interp_p);
    destroy_interp_containers(d_interp_e, h_interp_e);
    destroy_interp_containers(d_interp_Phi_phi, h_interp_Phi_phi);
    destroy_interp_containers(d_interp_Phi_r, h_interp_Phi_r);

    gpuErrchk(cudaFree(d_init_t));
    gpuErrchk(cudaFree(d_init_p));
    gpuErrchk(cudaFree(d_init_e));
    gpuErrchk(cudaFree(d_init_Phi_phi));
    gpuErrchk(cudaFree(d_init_Phi_r));

    gpuErrchk(cudaFree(d_interp_p));
    gpuErrchk(cudaFree(d_interp_e));
    gpuErrchk(cudaFree(d_interp_Phi_phi));
    gpuErrchk(cudaFree(d_interp_Phi_r));

    delete h_interp_p;
    delete h_interp_e;
    delete h_interp_Phi_phi;
    delete h_interp_Phi_r;

    destroy_mode_interp_containers(d_interp_modes, h_interp_modes, num_teuk_modes);

    gpuErrchk(cudaFree(d_interp_modes));
    delete[] h_interp_modes;

    delete interp;
}
