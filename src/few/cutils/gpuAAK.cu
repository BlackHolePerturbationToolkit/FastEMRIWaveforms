#include "stdio.h"

#include "global.h"

#define  NUM_THREADS 32



#define ACC 40.0
#define BIGNO 1.0e10
#define BIGNI 1.0e-10



static double bessel_j0( double x ) {
   // Numerical Recipes, pp. 274-280.

   double ax,z;
   double xx,y,ans,ans1,ans2;

   if ((ax=fabs(x)) < 8.0) {
      // Padé approximation (composed of numerator ans1, denominator ans2)
      y=x*x;
      ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7
         +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
      ans2=57568490411.0+y*(1029532985.0+y*(9494680.718
         +y*(59272.64853+y*(267.8532712+y*1.0))));
      ans=ans1/ans2;
   } else {
      // Asymptotic expansion that generally improves as x grows
      z=8.0/ax;
      y=z*z;
      xx=ax-0.785398164;  // minus pi/4
      ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
         +y*(-0.2073370639e-5+y*0.2093887211e-6)));
      ans2 = -0.1562499995e-1+y*(0.1430488765e-3
         +y*(-0.6911147651e-5+y*(0.7621095161e-6
         -y*0.934935152e-7)));
      ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
   }
   return ans;
}

static double bessel_j1( double x ) {

   double ax,z;
   double xx,y,ans,ans1,ans2;

   if ((ax=fabs(x)) < 8.0) {
      // Padé approximation (composed of numerator ans1, denominator ans2)
      y=x*x;
      ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
         +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
      ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
         +y*(99447.43394+y*(376.9991397+y*1.0))));
      ans=ans1/ans2;
   } else {
      // Asymptotic expansion that generally improves as x grows
      z=8.0/ax;
      y=z*z;
      xx=ax-2.356194491;  // minus 3pi/4
      ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
         +y*(0.2457520174e-5+y*(-0.240337019e-6))));
      ans2=0.04687499995+y*(-0.2002690873e-3
         +y*(0.8449199096e-5+y*(-0.88228987e-6
         +y*0.105787412e-6)));
      ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
      if (x < 0.0) ans = -ans;
   }
   return ans;
}

double bessel_jn( int n, double x ) {
   // Numerical Recipes, pp. 274-280.
   // Constructs J_n for arbitrary n with a recurrence relation

   int    j, jsum, m;
   double ax, bj, bjm, bjp, sum, tox, ans;

   ax=fabs(x);
   // handle special cases
   if (n == 0)
      return( bessel_j0(ax) );
   if (n == 1)
      return( bessel_j1(ax) );

   if (ax == 0.0)
      return 0.0;
   else if (ax > (double) n) {
      // forward recurrence (assumed stable above n)
      tox=2.0/ax;
      bjm=bessel_j0(ax);
      bj=bessel_j1(ax);
      for (j=1;j<n;j++) {
         bjp=j*tox*bj-bjm;
         bjm=bj;
         bj=bjp;
      }
      ans=bj;
   } else {
      // backwards recurrence for small x
      tox=2.0/ax;
      m=2*((n+(int) sqrt(ACC*n))/2);
      jsum=0;
      bjp=ans=sum=0.0;
      bj=1.0;
      for (j=m;j>0;j--) {
         bjm=j*tox*bj-bjp;
         bjp=bj;
         bj=bjm;
         if (fabs(bj) > BIGNO) {
            bj *= BIGNI;
            bjp *= BIGNI;
            ans *= BIGNI;
            sum *= BIGNI;
         }
         if (jsum) sum += bj;
         jsum=!jsum;
         if (j == n) ans=bjp;
      }
      sum=2.0*sum-bj;
      ans /= sum;
   }

   // adjust sign for negative x if n odd
   return  x < 0.0 && n%2 == 1 ? -ans : ans;
}


CUDA_CALLABLE_MEMBER
double d_dot_product(const double *u,const double *v){
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}

CUDA_CALLABLE_MEMBER
void d_cross(const double *u,const double *v,double *w){
  w[0] = u[1]*v[2]-u[2]*v[1];
  w[1] = u[2]*v[0]-u[0]*v[2];
  w[2] = u[0]*v[1]-u[1]*v[0];
}

CUDA_CALLABLE_MEMBER
double d_vec_norm(const double *u){
    return sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
}

CUDA_CALLABLE_MEMBER
void d_RotCoeff(double* rot, double* n, double* L, double* S, double* nxL, double* nxS,
                double iota,double theta_S,double phi_S,double theta_K,double phi_K,double alpha){

  n[0] = sin(theta_S)*cos(phi_S);
  n[1] = sin(theta_S)*sin(phi_S);
  n[2] = cos(theta_S);
  S[0] = sin(theta_K)*cos(phi_K);
  S[1] = sin(theta_K)*sin(phi_K);
  S[2] = cos(theta_K);
  L[0] = cos(iota)*sin(theta_K)*cos(phi_K)+sin(iota)*(sin(alpha)*sin(phi_K)-cos(alpha)*cos(theta_K)*cos(phi_K));
  L[1] = cos(iota)*sin(theta_K)*sin(phi_K)-sin(iota)*(sin(alpha)*cos(phi_K)+cos(alpha)*cos(theta_K)*sin(phi_K));
  L[2] = cos(iota)*cos(theta_K)+sin(iota)*cos(alpha)*sin(theta_K);
  d_cross(n,L,nxL);
  d_cross(n,S,nxS);

  double norm=d_vec_norm(nxL)*d_vec_norm(nxS);
  double dot,cosrot,sinrot;
  //gsl_blas_ddot(nxL,nxS,&dot);
  dot = d_dot_product(nxL,nxS);

  if (norm < 1e-6) norm = 1e-6;

  cosrot=dot/norm;
  //gsl_blas_ddot(L,nxS,&dot);
  dot = d_dot_product(L,nxS);
  sinrot=dot;
  //gsl_blas_ddot(S,nxL,&dot);
  dot = d_dot_product(S,nxL);
  sinrot-=dot;
  sinrot/=norm;

  rot[0]=2.*cosrot*cosrot-1.;
  rot[1]=cosrot*sinrot;
  rot[2]=-rot[1];
  rot[3]=rot[0];
}

#define  NUM_PARS 6
CUDA_KERNEL
void make_waveform(cmplx *waveform,
              double* interp_array,
              double M_phys, double S_phys, double mu, double qS, double phiS, double qK, double phiK, double dist,
              int nmodes, bool mich,
              double delta_t, double start_t, int old_ind, int start_ind, int end_ind, int init_length, double segwidth)
{

      cmplx I(0.0, 1.0);

      #ifdef __CUDACC__

      __shared__ double rot_all[4 * NUM_THREADS];
      __shared__ double n_all[3 * NUM_THREADS];
      __shared__ double L_all[3 * NUM_THREADS];
      __shared__ double S_all[3 * NUM_THREADS];
      __shared__ double nxL_all[3 * NUM_THREADS];
      __shared__ double nxS_all[3 * NUM_THREADS];

      double* rot = &rot_all[threadIdx.x * 4];
      double* n_rot = &n_all[threadIdx.x * 3];
      double* L_rot = &L_all[threadIdx.x * 3];
      double* S_rot = &S_all[threadIdx.x * 3];
      double* nxL_rot = &nxL_all[threadIdx.x * 3];
      double* nxS_rot = &nxS_all[threadIdx.x * 3];

      #endif

      CUDA_SHARED double spline_coeffs[NUM_PARS * 8];

      int start, end, increment;
      #ifdef __CUDACC__
      start = threadIdx.x;
      end = 8 * NUM_PARS;
      increment = blockDim.x;
      #else
      start = 0;
      end = 8 * NUM_PARS;
      increment = 1;
      #endif // __CUDACC__

       // prepare interpolants
      // 6 parameters, 8 coefficient values for each parameter
      for (int i = start; i < end; i += increment)
      {
          int coeff_num = (int) (i / NUM_PARS);
          int par_num = i % NUM_PARS;

          int index = (coeff_num * (init_length-1) + old_ind) * NUM_PARS + par_num;

          spline_coeffs[par_num * 8 + coeff_num] = interp_array[index];

      }

      CUDA_SYNC_THREADS;

      // unroll coefficients

      CUDA_SHARED double e_y, e_c1, e_c2, e_c3, e_c4, e_c5, e_c6, e_c7;
      CUDA_SHARED double Y_y, Y_c1, Y_c2, Y_c3, Y_c4, Y_c5, Y_c6, Y_c7;
      CUDA_SHARED double Phi_phi_y, Phi_phi_c1, Phi_phi_c2, Phi_phi_c3, Phi_phi_c4, Phi_phi_c5, Phi_phi_c6, Phi_phi_c7;
      CUDA_SHARED double Phi_theta_y, Phi_theta_c1, Phi_theta_c2, Phi_theta_c3, Phi_theta_c4, Phi_theta_c5, Phi_theta_c6, Phi_theta_c7;
      CUDA_SHARED double Phi_r_y, Phi_r_c1, Phi_r_c2, Phi_r_c3, Phi_r_c4, Phi_r_c5, Phi_r_c6, Phi_r_c7;

      #ifdef __CUDACC__
      if (threadIdx.x == 0)
      #else
      if (true)
      #endif
      {
          e_y = spline_coeffs[1 * 8 + 0]; e_c1 = spline_coeffs[1 * 8 + 1]; e_c2 = spline_coeffs[1 * 8 + 2]; e_c3 = spline_coeffs[1 * 8 + 3]; e_c4 = spline_coeffs[1 * 8 + 4]; e_c5 = spline_coeffs[1 * 8 + 5]; e_c6 = spline_coeffs[1 * 8 + 6]; e_c7 = spline_coeffs[1 * 8 + 7];
          Y_y = spline_coeffs[2 * 8 + 0]; Y_c1 = spline_coeffs[2 * 8 + 1]; Y_c2 = spline_coeffs[2 * 8 + 2]; Y_c3 = spline_coeffs[2 * 8 + 3]; Y_c4 = spline_coeffs[2 * 8 + 4]; Y_c5 = spline_coeffs[2 * 8 + 5]; Y_c6 = spline_coeffs[2 * 8 + 6]; Y_c7 = spline_coeffs[2 * 8 + 7];
          Phi_phi_y = spline_coeffs[3 * 8 + 0]; Phi_phi_c1 = spline_coeffs[3 * 8 + 1]; Phi_phi_c2 = spline_coeffs[3 * 8 + 2]; Phi_phi_c3 = spline_coeffs[3 * 8 + 3]; Phi_phi_c4 = spline_coeffs[3 * 8 + 4]; Phi_phi_c5 = spline_coeffs[3 * 8 + 5]; Phi_phi_c6 = spline_coeffs[3 * 8 + 6]; Phi_phi_c7 = spline_coeffs[3 * 8 + 7];
          Phi_theta_y = spline_coeffs[4 * 8 + 0]; Phi_theta_c1 = spline_coeffs[4 * 8 + 1]; Phi_theta_c2 = spline_coeffs[4 * 8 + 2]; Phi_theta_c3 = spline_coeffs[4 * 8 + 3]; Phi_theta_c4 = spline_coeffs[4 * 8 + 4]; Phi_theta_c5 = spline_coeffs[4 * 8 + 5]; Phi_theta_c6 = spline_coeffs[4 * 8 + 6]; Phi_theta_c7 = spline_coeffs[4 * 8 + 7];
          Phi_r_y = spline_coeffs[5 * 8 + 0]; Phi_r_c1 = spline_coeffs[5 * 8 + 1]; Phi_r_c2 = spline_coeffs[5 * 8 + 2]; Phi_r_c3 = spline_coeffs[5 * 8 + 3]; Phi_r_c4 = spline_coeffs[5 * 8 + 4]; Phi_r_c5 = spline_coeffs[5 * 8 + 5]; Phi_r_c6 = spline_coeffs[5 * 8 + 6]; Phi_r_c7 = spline_coeffs[5 * 8 + 7];
      }

      CUDA_SYNC_THREADS;

      double fill_val = 1e-6;
      if (qS < fill_val) qS = fill_val;
      if (qK < fill_val) qK = fill_val;
      if (qS > M_PI - fill_val) qS = M_PI - fill_val;
      if (qK > M_PI - fill_val) qK = M_PI - fill_val;

      double cosqS=cos(qS);
      double sinqS=sin(qS);
      double cosqK=cos(qK);
      double sinqK=sin(qK);
      double cosphiK=cos(phiK);
      double sinphiK=sin(phiK);
      double halfsqrt3=sqrt(3.)/2.;
      double mu_sec = mu * MTSUN_SI;
      double zeta=mu_sec/dist/GPCINSEC; // M/D

      #ifdef __CUDACC__

      start = start_ind + threadIdx.x + blockIdx.x * blockDim.x;
      end = end_ind;
      increment = blockDim.x * gridDim.x;

      #else

      start = start_ind;
      end = end_ind;
      increment = 1;

      #endif
      for (int i = start; i < end; i += increment)
      {

          #ifdef __CUDACC__
          #else

          double rot_temp[4];
          double n_temp[3];
          double L_temp[3];
          double S_temp[3];
          double nxL_temp[3];
          double nxS_temp[3];

          double* rot = &rot_temp[0];
          double* n_rot = &n_temp[0];
          double* L_rot = &L_temp[0];
          double* S_rot = &S_temp[0];
          double* nxL_rot = &nxL_temp[0];
          double* nxS_rot = &nxS_temp[0];

          #endif

          waveform[i] = cmplx(0.0, 0.0);

          double t=delta_t * i;

          double s = (t - start_t) / segwidth;
          double s1 = (1.0 - s);
          double s2 = s * s;
          double s3 = s * s2;
          double s4 = s * s3;
          double s5 = s * s4;
          double s6 = s * s5;
          double s7 = s * s6;

          double e = e_y + s * (e_c1 + s1 * (e_c2 + s * (e_c3 + s1 * (e_c4 + s * (e_c5 + s1 * (e_c6 + s * e_c7))))));
          double Y = Y_y + s * (Y_c1 + s1 * (Y_c2 + s * (Y_c3 + s1 * (Y_c4 + s * (Y_c5 + s1 * (Y_c6 + s * Y_c7))))));

          double Phi_phi = Phi_phi_y + s * (Phi_phi_c1 + s1 * ( Phi_phi_c2 + s * (Phi_phi_c3 + s1 * (Phi_phi_c4 + s * (Phi_phi_c5 + s1 * (Phi_phi_c6 + s * Phi_phi_c7))))));
          double Phi_theta = Phi_theta_y + s * (Phi_theta_c1 + s1 * ( Phi_theta_c2 + s * (Phi_theta_c3 + s1 * (Phi_theta_c4 + s * (Phi_theta_c5 + s1 * (Phi_theta_c6 + s * Phi_theta_c7))))));
          double Phi_r = Phi_r_y + s * (Phi_r_c1 + s1 * ( Phi_r_c2 + s * (Phi_r_c3 + s1 * (Phi_r_c4 + s * (Phi_r_c5 + s1 * (Phi_r_c6 + s * Phi_r_c7))))));

          double OmegaPhi = (Phi_phi_c1 + Phi_phi_c2 * (1. - 2.*s) + Phi_phi_c3 * (2.*s - 3.*s2) + Phi_phi_c4 * (2.*s - 6.*s2 + 4.*s3) + Phi_phi_c5 * (3.*s2 - 8.*s3 + 5.*s4) + Phi_phi_c6 * (3.*s2 - 12.*s3 + 15*s4 - 6.*s5) + Phi_phi_c7 * (4.*s3 - 15.*s4 + 18.*s5 - 7.*s6)) / segwidth;
          double OmegaTheta = (Phi_theta_c1 + Phi_theta_c2 * (1. - 2.*s) + Phi_theta_c3 * (2.*s - 3.*s2) + Phi_theta_c4 * (2.*s - 6.*s2 + 4.*s3) + Phi_theta_c5 * (3.*s2 - 8.*s3 + 5.*s4) + Phi_theta_c6 * (3.*s2 - 12.*s3 + 15*s4 - 6.*s5) + Phi_theta_c7 * (4.*s3 - 15.*s4 + 18.*s5 - 7.*s6)) / segwidth;
          double OmegaR = (Phi_r_c1 + Phi_r_c2 * (1. - 2.*s) + Phi_r_c3 * (2.*s - 3.*s2) + Phi_r_c4 * (2.*s - 6.*s2 + 4.*s3) + Phi_r_c5 * (3.*s2 - 8.*s3 + 5.*s4) + Phi_r_c6 * (3.*s2 - 12.*s3 + 15*s4 - 6.*s5) + Phi_r_c7 * (4.*s3 - 15.*s4 + 18.*s5 - 7.*s6)) / segwidth;

          double Phi = Phi_r;
          double gim = Phi_theta - Phi_r;
          double alp = Phi_phi - Phi_theta;
          double nu = OmegaR / (2. * M_PI);
          double gimdot = OmegaTheta - OmegaR;
          double lam = acos(Y);

         //  if (lam > M_PI - fill_val) lam = M_PI - fill_val;
         //  if (lam < fill_val) lam = fill_val;

          double coslam=cos(lam);
          double sinlam=sin(lam);
          double cosalp=cos(alp);
          double sinalp=sin(alp);
          double cosqL=cosqK*coslam+sinqK*sinlam*cosalp;
          double sinqL=sqrt(1.-cosqL*cosqL);
          double phiLup=sinqK*sinphiK*coslam-cosphiK*sinlam*sinalp-cosqK*sinphiK*sinlam*cosalp;
          double phiLdown=sinqK*cosphiK*coslam+sinphiK*sinlam*sinalp-cosqK*cosphiK*sinlam*cosalp;
          double phiL=atan2(phiLup,phiLdown);
          double Ldotn=cosqL*cosqS+sinqL*sinqS*cos(phiL-phiS);
          double Ldotn2=Ldotn*Ldotn;
          double Sdotn=cosqK*cosqS+sinqK*sinqS*cos(phiK-phiS);
          double beta;
          if (S_phys == 0.0 || lam == 0.0)
          {
              beta = 0.0; // This seems to work nicely
          }
          else
          {
              // Be careful. I set beta = 0 due to division by zero in betadown.
              double betaup=-Sdotn+coslam*Ldotn;
              double betadown=sinqS*sin(phiK-phiS)*sinlam*cosalp+(cosqK*Sdotn-cosqS)/sinqK*sinlam*sinalp;
              beta=atan2(betaup,betadown);

              // beta=0.0;
          }
          double gam=2.*(gim+beta);
          double cos2gam=cos(gam);
          double sin2gam=sin(gam);

          double orbphs,cosorbphs,sinorbphs,FplusI,FcrosI,FplusII,FcrosII;
        if(mich){
          orbphs=2.*M_PI*t/YRSID_SI;
          cosorbphs=cos(orbphs-phiS);
          sinorbphs=sin(orbphs-phiS);
          double cosq=.5*cosqS-halfsqrt3*sinqS*cosorbphs;
          double phiw=orbphs+atan2(halfsqrt3*cosqS+.5*sinqS*cosorbphs,sinqS*sinorbphs);
          double psiup=.5*cosqK-halfsqrt3*sinqK*cos(orbphs-phiK)-cosq*(cosqK*cosqS+sinqK*sinqS*cos(phiK-phiS));
          double psidown=.5*sinqK*sinqS*sin(phiK-phiS)-halfsqrt3*cos(orbphs)*(cosqK*sinqS*sin(phiS)-cosqS*sinqK*sin(phiK))-halfsqrt3*sin(orbphs)*(cosqS*sinqK*cos(phiK)-cosqK*sinqS*cos(phiS));
          double psi=atan2(psiup,psidown);
          double cosq1=.5*(1.+cosq*cosq);
          double cos2phi=cos(2.*phiw);
          double sin2phi=sin(2.*phiw);
          double cos2psi=cos(2.*psi);
          double sin2psi=sin(2.*psi);
          FplusI=cosq1*cos2phi*cos2psi-cosq*sin2phi*sin2psi;
          FcrosI=cosq1*cos2phi*sin2psi+cosq*sin2phi*cos2psi;
          FplusII=cosq1*sin2phi*cos2psi+cosq*cos2phi*sin2psi;
          FcrosII=cosq1*sin2phi*sin2psi-cosq*cos2phi*cos2psi;
        }
        else
        {
            /*
            double up_ldc = (cosqS*sinqK*cos(phiS-phiK) - cosqK*sinqS);
              double dw_ldc = (sinqK*sin(phiS-phiK));
              double psi_ldc;
              if (dw_ldc != 0.0) {
                psi_ldc = atan2(up_ldc, dw_ldc);
              }
              else {
            psi_ldc = 0.5*M_PI;
              }
              double c2psi_ldc=cos(2.*psi_ldc);
              double s2psi_ldc=sin(2.*psi_ldc);

            FplusI=c2psi_ldc;
            FcrosI=-s2psi_ldc;
            FplusII=s2psi_ldc;
            FcrosII=c2psi_ldc;*/

            FplusI = 1.0;
            FcrosI = 0.0;
            FplusII = 0.0;
            FcrosII = 1.0;
        }

          double Amp=pow(abs(OmegaPhi)*M_phys*MTSUN_SI,2./3.)*zeta;

          d_RotCoeff(rot, n_rot, L_rot, S_rot, nxL_rot, nxS_rot,
                   lam,qS,phiS,qK,phiK,alp);

          double hItemp = 0.0;
          double hIItemp = 0.0;
          for(int n=1;n<=nmodes;n++)
          {

              double fn,Doppler,nPhi;
              if(mich){
                fn=n*nu+gimdot/M_PI;
                Doppler=2.*M_PI*fn*AUsec*sinqS*cosorbphs;
                nPhi=n*Phi+Doppler;
              }
              else nPhi=n*Phi;

               double ne=n*e;
               double J0, J1, J2, J3, J4;
              #ifdef __CUDACC__

              if(n==1){ J0=-1.0*j1(ne); }
              else { J0 = jn(n-2, ne); }

              J1=jn(n-1, ne);
              J2=jn(n, ne);
              J3=jn(n+1,ne);
              J4=jn(n+2,ne);

              #else

              if(n==1){ J0=-1.0*bessel_j1(ne); }
              else { J0 = bessel_jn(n-2, ne); }

              J1=bessel_jn(n-1, ne);
              J2=bessel_jn(n, ne);
              J3=bessel_jn(n+1,ne);
              J4=bessel_jn(n+2,ne);

              #endif

      double a=-n*Amp*(J0-2.*e*J1+2./n*J2+2.*e*J3-J4)*cos(nPhi);
      double b=-n*Amp*sqrt(1-e*e)*(J0-2.*J2+J4)*sin(nPhi);
      double c=2.*Amp*J2*cos(nPhi);
      double Aplus=-(1.+Ldotn2)*(a*cos2gam-b*sin2gam)+c*(1-Ldotn2);
      double Acros=2.*Ldotn*(b*cos2gam+a*sin2gam);


      // ----- rotate to NK wave frame -----
      double Aplusold=Aplus;
      double Acrosold=Acros;

      Aplus=Aplusold*rot[0]+Acrosold*rot[1];
      Acros=Aplusold*rot[2]+Acrosold*rot[3];
      // ----------

      double hnI, hnII;
      if(mich){
      	hnI=halfsqrt3*(FplusI*Aplus+FcrosI*Acros);
        hnII=halfsqrt3*(FplusII*Aplus+FcrosII*Acros);
      }
      else{
      	hnI=FplusI*Aplus+FcrosI*Acros;
        hnII=FplusII*Aplus+FcrosII*Acros;
      }

      hItemp+=hnI;
      hIItemp+=hnII;
  }

    waveform[i] = cmplx(hItemp, -hIItemp);
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
       i += 1){

          double t = t_arr[i];

          // adjust for waveforms that hit the end of the trajectory
          if (t < T){
              start_inds[i] = (int)std::ceil(t/delta_t);
              unit_length[i-1] = start_inds[i] - start_inds[i-1];
          } else {
            start_inds[i] = new_length;
            unit_length[i-1] = new_length - start_inds[i-1];
            break;
        }

      }

  // fixes for not using certain segments for the interpolation
  *length = i + 1;
}

// function for building interpolated EMRI waveform from python
void get_waveform(cmplx *waveform, double* interp_array,
              double M_phys, double S_phys, double mu, double qS, double phiS, double qK, double phiK, double dist,
              int nmodes, bool mich,
              int init_len, int out_len,
              double delta_t, double *h_t){

    // arrays for determining spline windows for new arrays
    int start_inds[init_len];
    int unit_length[init_len-1];

    int number_of_old_spline_points = init_len;

    // find the spline window information based on equally spaced new array
    find_start_inds(start_inds, unit_length, h_t, delta_t, &number_of_old_spline_points, out_len);

    #ifdef __CUDACC__

    // prepare streams for CUDA
    cudaStream_t streams[number_of_old_spline_points-1];

    #endif

    for (int i = 0; i < number_of_old_spline_points-1; i++) {
          #ifdef __CUDACC__

          // create and execute with streams
          cudaStreamCreate(&streams[i]);
          int num_blocks = std::ceil((unit_length[i] + NUM_THREADS -1)/NUM_THREADS);

          // sometimes a spline interval will have zero points
          if (num_blocks <= 0) continue;

          dim3 gridDim(num_blocks, 1);

          // launch one worker kernel per stream
          make_waveform<<<gridDim, NUM_THREADS, 0, streams[i]>>>(waveform,
                        interp_array,
                        M_phys, S_phys, mu, qS, phiS, qK, phiK, dist,
                        nmodes, mich,
                        delta_t, h_t[i], i, start_inds[i], start_inds[i+1], init_len,
                        h_t[i+1] - h_t[i]);
         #else

         // CPU waveform generation
         make_waveform(waveform,
                       interp_array,
                       M_phys, S_phys, mu, qS, phiS, qK, phiK, dist,
                       nmodes, mich,
                       delta_t, h_t[i], i, start_inds[i], start_inds[i+1], init_len,
                       h_t[i+1] - h_t[i]);
         #endif

      }

      //synchronize after all streams finish
      #ifdef __CUDACC__
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());

      for (int i = 0; i < number_of_old_spline_points-1; i++) {
            //destroy the streams
            cudaStreamDestroy(streams[i]);
        }
      #endif
}
