#ifndef __INTERP_H__
#define __INTERP_H__

#include "global.h"

typedef struct tagInterpContainer{
  fod *y;
  fod *c1;
  fod *c2;
  fod *c3;
  int length;

} InterpContainer;

void create_interp_containers(InterpContainer *d_interp, InterpContainer *h_interp, fod *y, int length);
void destroy_interp_containers(InterpContainer *d_interp, InterpContainer *h_interp);

void setup_interpolate(InterpContainer *h_interp_p, InterpContainer *h_interp_e, InterpContainer *h_interp_Phi_phi, InterpContainer *h_interp_Phi_r,
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
