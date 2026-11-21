#pragma once

extern "C" {

const char *few_metal_sum_last_error();

void *few_metal_sum_context_create();

void few_metal_sum_context_destroy(void *opaque_context);

int few_metal_sum_evaluate(
    void *opaque_context, double *waveform, const double *interpolation,
    const double *phase_times, const double *phase_coefficients,
    const int *m_values, const int *k_values, const int *n_values,
    int init_length, int output_length, int mode_count, const double *ylms,
    double delta_t, const double *trajectory_times, double *gpu_seconds);

} // extern "C"
