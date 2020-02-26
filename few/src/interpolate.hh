#ifndef __INTERP_H__
#define __INTERP_H__

#include "global.h"
#include "cusparse.h"


typedef struct tagInterpContainer{
  fod *y;
  fod *c1;
  fod *c2;
  fod *c3;
  int length;

} InterpContainer;

class InterpClass {

public:
    fod *B, *upper_diag, *lower_diag, *diag;

    void *pBuffer;
    cusparseStatus_t stat;
    cusparseHandle_t handle;

    InterpClass(int num_modes, int length);
    ~InterpClass();

    void setup_interpolate(InterpContainer *d_interp_p, InterpContainer *d_interp_e, InterpContainer *d_interp_Phi_phi, InterpContainer *d_interp_Phi_r,
                           InterpContainer *d_modes, int num_modes,
                           fod *d_t, int length);

};


void create_interp_containers(InterpContainer *d_interp, InterpContainer *h_interp, int length);
void destroy_interp_containers(InterpContainer *d_interp, InterpContainer *h_interp);

void create_mode_interp_containers(InterpContainer *d_interp, InterpContainer *h_interp, int length, int num_modes);
void destroy_mode_interp_containers(InterpContainer *d_interp, InterpContainer *h_interp, int num_modes);
void fill_complex_y_vals(InterpContainer *d_interp, cuComplex *y, int length, int num_modes);

void setup_interpolate(InterpContainer *h_interp_p, InterpContainer *h_interp_e, InterpContainer *h_interp_Phi_phi, InterpContainer *h_interp_Phi_r,
                       InterpContainer *d_modes, int num_modes,
                       fod *d_t, int length);

void perform_interp(fod *p_out, fod *e_out, fod *Phi_phi_out, fod *Phi_r_out,
                   InterpContainer *d_interp_p, InterpContainer *d_interp_e, InterpContainer *d_interp_Phi_phi, InterpContainer *d_interp_Phi_r,
                      fod *d_t, fod *h_t, int length, int new_length, fod delta_t);

/*
CuSparse error checking
*/
#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
                            fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
                            exit(-1);}} while(0)

#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)


#endif // __INTERP_H__
