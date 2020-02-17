#include "global.h"
#include <cfloat>
#include <assert.h>

#define NUM_THREADS 256

#define PI_2 1.57079632679
#define PI 3.141592653589793

#define MACHEP 1.38777878078144567553E-17
#define MAXNUM 1.701411834604692317316873e38

/*
__device__
static fod P[] = {
  1.53552577301013293365E-4,
  2.50888492163602060990E-3,
  8.68786816565889628429E-3,
  1.07350949056076193403E-2,
  7.77395492516787092951E-3,
  7.58395289413514708519E-3,
  1.15688436810574127319E-2,
  2.18317996015557253103E-2,
  5.68051945617860553470E-2,
  4.43147180560990850618E-1,
  1.00000000000000000299E0
};
__device__
static fod Q[] = {
  3.27954898576485872656E-5,
  1.00962792679356715133E-3,
  6.50609489976927491433E-3,
  1.68862163993311317300E-2,
  2.61769742454493659583E-2,
  3.34833904888224918614E-2,
  4.27180926518931511717E-2,
  5.85936634471101055642E-2,
  9.37499997197644278445E-2,
  2.49999999999888314361E-1
};

__device__ __host__ fod polevl(fod x, fod *coeffs, int N)
{
    fod ans;
    int i;
    const fod *p;

    p = coeffs;
    ans = *p++;
    i = N;

    do
	ans = ans * x + *p++;
    while (--i);

    return (ans);
}

__device__ __host__ fod ellpe(fod x)
{
    x = 1.0 - x;
    if (x <= 0.0) {
	if (x == 0.0)
	    return (1.0);
	assert(0);
    }
    if (x > 1.0) {
        return ellpe(1.0 - 1/x) * sqrt(x);
    }
    return (polevl(x, P, 10) - log(x) * (x * polevl(x, Q, 9)));
}

__device__ static fod PK[] =
{
 1.37982864606273237150E-4,
 2.28025724005875567385E-3,
 7.97404013220415179367E-3,
 9.85821379021226008714E-3,
 6.87489687449949877925E-3,
 6.18901033637687613229E-3,
 8.79078273952743772254E-3,
 1.49380448916805252718E-2,
 3.08851465246711995998E-2,
 9.65735902811690126535E-2,
 1.38629436111989062502E0
};

__device__ static fod QK[] =
{
 2.94078955048598507511E-5,
 9.14184723865917226571E-4,
 5.94058303753167793257E-3,
 1.54850516649762399335E-2,
 2.39089602715924892727E-2,
 3.01204715227604046988E-2,
 3.73774314173823228969E-2,
 4.88280347570998239232E-2,
 7.03124996963957469739E-2,
 1.24999999999870820058E-1,
 4.99999999999999999821E-1
};
*/

// TODO: change fod back to template
__device__
fod

    __ellint_rd(fod __x, fod __y, fod __z)
    {
      const fod __eps = FLT_EPSILON;
      const fod __errtol = pow(__eps / fod(8), fod(1) / fod(6));
          const fod __c0 = fod(1) / fod(4);
          const fod __c1 = fod(3) / fod(14);
          const fod __c2 = fod(1) / fod(6);
          const fod __c3 = fod(9) / fod(22);
          const fod __c4 = fod(3) / fod(26);
          fod __xn = __x;
          fod __yn = __y;
          fod __zn = __z;
          fod __sigma = fod(0);
          fod __power4 = fod(1);
          fod __mu;
          fod __xndev, __yndev, __zndev;
          const unsigned int __max_iter = 100;
          for (unsigned int __iter = 0; __iter < __max_iter; ++__iter)
            {
              __mu = (__xn + __yn + fod(3) * __zn) / fod(5);
              __xndev = (__mu - __xn) / __mu;
              __yndev = (__mu - __yn) / __mu;
              __zndev = (__mu - __zn) / __mu;
              fod __epsilon = max(abs(__xndev), abs(__yndev));
              __epsilon = max(__epsilon, abs(__zndev));
              if (__epsilon < __errtol)
                break;
              fod __xnroot = sqrt(__xn);
              fod __ynroot = sqrt(__yn);
              fod __znroot = sqrt(__zn);
              fod __lambda = __xnroot * (__ynroot + __znroot)
                           + __ynroot * __znroot;
              __sigma += __power4 / (__znroot * (__zn + __lambda));
              __power4 *= __c0;
              __xn = __c0 * (__xn + __lambda);
              __yn = __c0 * (__yn + __lambda);
              __zn = __c0 * (__zn + __lambda);
            }
          // Note: __ea is an SPU badname.
          fod __eaa = __xndev * __yndev;
          fod __eb = __zndev * __zndev;
          fod __ec = __eaa - __eb;
          fod __ed = __eaa - fod(6) * __eb;
          fod __ef = __ed + __ec + __ec;
          fod __s1 = __ed * (-__c1 + __c3 * __ed
                                   / fod(3) - fod(3) * __c4 * __zndev * __ef
                                   / fod(2));
          fod __s2 = __zndev
                   * (__c2 * __ef
                    + __zndev * (-__c3 * __ec - __zndev * __c4 - __eaa));
          return fod(3) * __sigma + __power4 * (fod(1) + __s1 + __s2)
                                        / (__mu * sqrt(__mu));
        }

__device__ fod
    __ellint_rc(fod __x, fod __y)
    {
          const fod __c0 = fod(1) / fod(4);
          const fod __c1 = fod(1) / fod(7);
          const fod __c2 = fod(9) / fod(22);
          const fod __c3 = fod(3) / fod(10);
          const fod __c4 = fod(3) / fod(8);
          fod __xn = __x;
          fod __yn = __y;
          const fod __eps = FLT_EPSILON;
          const fod __errtol = pow(__eps / fod(30), fod(1) / fod(6));
          fod __mu;
          fod __sn;
          const unsigned int __max_iter = 100;
          for (unsigned int __iter = 0; __iter < __max_iter; ++__iter)
            {
              __mu = (__xn + fod(2) * __yn) / fod(3);
              __sn = (__yn + __mu) / __mu - fod(2);
              if (abs(__sn) < __errtol)
                break;
              const fod __lambda = fod(2) * sqrt(__xn) * sqrt(__yn)
                             + __yn;
              __xn = __c0 * (__xn + __lambda);
              __yn = __c0 * (__yn + __lambda);
            }
          fod __s = __sn * __sn
                  * (__c3 + __sn*(__c1 + __sn * (__c4 + __sn * __c2)));
          return (fod(1) + __s) / sqrt(__mu);
        }

        __device__ fod
        __ellint_rf(fod __x, fod __y, fod __z)
            {
              const fod __c0 = fod(1) / fod(4);
              const fod __c1 = fod(1) / fod(24);
              const fod __c2 = fod(1) / fod(10);
              const fod __c3 = fod(3) / fod(44);
              const fod __c4 = fod(1) / fod(14);
              fod __xn = __x;
              fod __yn = __y;
              fod __zn = __z;
              const fod __eps = FLT_EPSILON;
              const fod __errtol = pow(__eps, fod(1) / fod(6));
              fod __mu;
              fod __xndev, __yndev, __zndev;
              const unsigned int __max_iter = 100;
              for (unsigned int __iter = 0; __iter < __max_iter; ++__iter)
                {
                  __mu = (__xn + __yn + __zn) / fod(3);
                  __xndev = 2 - (__mu + __xn) / __mu;
                  __yndev = 2 - (__mu + __yn) / __mu;
                  __zndev = 2 - (__mu + __zn) / __mu;
                  fod __epsilon = max(abs(__xndev), abs(__yndev));
                  __epsilon = max(__epsilon, abs(__zndev));
                  if (__epsilon < __errtol)
                    break;
                  const fod __xnroot = sqrt(__xn);
                  const fod __ynroot = sqrt(__yn);
                  const fod __znroot = sqrt(__zn);
                  const fod __lambda = __xnroot * (__ynroot + __znroot)
                                     + __ynroot * __znroot;
                  __xn = __c0 * (__xn + __lambda);
                  __yn = __c0 * (__yn + __lambda);
                  __zn = __c0 * (__zn + __lambda);
                }
              const fod __e2 = __xndev * __yndev - __zndev * __zndev;
              const fod __e3 = __xndev * __yndev * __zndev;
              const fod __s  = fod(1) + (__c1 * __e2 - __c2 - __c3 * __e3) * __e2
                       + __c4 * __e3;
              return __s / sqrt(__mu);
            }

__device__
fod
__comp_ellint_2(fod __k)
{
      const fod __kk = __k * __k;
      return __ellint_rf(fod(0), fod(1) - __kk, fod(1))
           - __kk * __ellint_rd(fod(0), fod(1) - __kk, fod(1)) / fod(3);
}



__device__
    fod
    __ellint_rj(fod __x, fod __y, fod __z, fod __p)
    {
          const fod __c0 = fod(1) / fod(4);
          const fod __c1 = fod(3) / fod(14);
          const fod __c2 = fod(1) / fod(3);
          const fod __c3 = fod(3) / fod(22);
          const fod __c4 = fod(3) / fod(26);
          fod __xn = __x;
          fod __yn = __y;
          fod __zn = __z;
          fod __pn = __p;
          fod __sigma = fod(0);
          fod __power4 = fod(1);
          const fod __eps = FLT_EPSILON;
          const fod __errtol = pow(__eps / fod(8), fod(1) / fod(6));
          fod __lambda, __mu;
          fod __xndev, __yndev, __zndev, __pndev;
          const unsigned int __max_iter = 100;
          for (unsigned int __iter = 0; __iter < __max_iter; ++__iter)
            {
              __mu = (__xn + __yn + __zn + fod(2) * __pn) / fod(5);
              __xndev = (__mu - __xn) / __mu;
              __yndev = (__mu - __yn) / __mu;
              __zndev = (__mu - __zn) / __mu;
              __pndev = (__mu - __pn) / __mu;
              fod __epsilon = max(abs(__xndev), abs(__yndev));
              __epsilon = max(__epsilon, abs(__zndev));
              __epsilon = max(__epsilon, abs(__pndev));
              if (__epsilon < __errtol)
                break;
              const fod __xnroot = sqrt(__xn);
              const fod __ynroot = sqrt(__yn);
              const fod __znroot = sqrt(__zn);
              const fod __lambda = __xnroot * (__ynroot + __znroot)
                                 + __ynroot * __znroot;
              const fod __alpha1 = __pn * (__xnroot + __ynroot + __znroot)
                                + __xnroot * __ynroot * __znroot;
              const fod __alpha2 = __alpha1 * __alpha1;
              const fod __beta = __pn * (__pn + __lambda)
                                      * (__pn + __lambda);
              __sigma += __power4 * __ellint_rc(__alpha2, __beta);
              __power4 *= __c0;
              __xn = __c0 * (__xn + __lambda);
              __yn = __c0 * (__yn + __lambda);
              __zn = __c0 * (__zn + __lambda);
              __pn = __c0 * (__pn + __lambda);
            }
          // Note: __ea is an SPU badname.
          fod __eaa = __xndev * (__yndev + __zndev) + __yndev * __zndev;
          fod __eb = __xndev * __yndev * __zndev;
          fod __ec = __pndev * __pndev;
          fod __e2 = __eaa - fod(3) * __ec;
          fod __e3 = __eb + fod(2) * __pndev * (__eaa - __ec);
          fod __s1 = fod(1) + __e2 * (-__c1 + fod(3) * __c3 * __e2 / fod(4)
                            - fod(3) * __c4 * __e3 / fod(2));
          fod __s2 = __eb * (__c2 / fod(2)
                   + __pndev * (-__c3 - __c3 + __pndev * __c4));
          fod __s3 = __pndev * __eaa * (__c2 - __pndev * __c3)
                   - __c2 * __pndev * __ec;
          return fod(3) * __sigma + __power4 * (__s1 + __s2 + __s3)
                                             / (__mu * sqrt(__mu));
        }


__device__ static fod C1 = 1.3862943611198906188E0;

__device__ fod
__comp_ellint_3(fod __k, fod __nu)
{
      const fod __kk = __k * __k;
      return __ellint_rf(fod(0), fod(1) - __kk, fod(1))
           + __nu
           * __ellint_rj(fod(0), fod(1) - __kk, fod(1), fod(1) - __nu)
           / fod(3);
}

  __device__  fod
    __ellint_3(fod __k, fod __nu, fod __phi)
    {
          //  Reduce phi to -pi/2 < phi < +pi/2.
          const int __n = floorf(__phi / PI
                                   + fod(0.5L));
          const fod __phi_red = __phi
                              - __n * PI;
          const fod __kk = __k * __k;
          const fod __s = sin(__phi_red);
          const fod __ss = __s * __s;
          const fod __sss = __ss * __s;
          const fod __c = cos(__phi_red);
          const fod __cc = __c * __c;
          const fod __Pi = __s
                         * __ellint_rf(__cc, fod(1) - __kk * __ss, fod(1))
                         + __nu * __sss
                         * __ellint_rj(__cc, fod(1) - __kk * __ss, fod(1),
                                       fod(1) - __nu * __ss) / fod(3);
          if (__n == 0)
            return __Pi;
          else
            return __Pi + fod(2) * __n * __comp_ellint_3(__k, __nu);
        }

__device__
fod
    __ellint_2(fod __k, fod __phi)
    {
          //  Reduce phi to -pi/2 < phi < +pi/2.
          const int __n = floor(__phi / PI
                                   + fod(0.5L));
          const fod __phi_red = __phi
                              - __n * PI;
          const fod __kk = __k * __k;
          const fod __s = sin(__phi_red);
          const fod __ss = __s * __s;
          const fod __sss = __ss * __s;
          const fod __c = cos(__phi_red);
          const fod __cc = __c * __c;
          const fod __E = __s
                        * __ellint_rf(__cc, fod(1) - __kk * __ss, fod(1))
                        - __kk * __sss
                        * __ellint_rd(__cc, fod(1) - __kk * __ss, fod(1))
                        / fod(3);
          if (__n == 0)
            return __E;
          else
            return __E + fod(2) * __n * __comp_ellint_2(__k);
        }

/*

__device__ __host__ fod ellpk(fod x)
{

if( (x < 0.0) || (x > 1.0) )
	{
	//mtherr( "ellpk", DOMAIN );
	return( 0.0 );
	}

if( x > MACHEP )
	{
	return( polevl(x,PK,10) - log(x) * polevl(x,QK,10) );
	}
else
	{
	if( x == 0.0 )
		{
		//mtherr( "ellpk", SING );
		return( MAXNUM );
		}
	else
		{
		return( C1 - 0.5 * log(x) );
		}
	}
}
*/
template<typename _Tp>
__device__   _Tp
     __comp_ellint_1(const _Tp __k)
     {
         return __ellint_rf(_Tp(0), _Tp(1) - __k * __k, _Tp(1));
     }

template<typename _Tp>
__device__ _Tp
     __ellint_1(const _Tp __k, const _Tp __phi)
     {
           //  Reduce phi to -pi/2 < phi < +pi/2.
           const int __n = floor(__phi / PI
                                    + _Tp(0.5L));
           const _Tp __phi_red = __phi
                               - __n * PI;

           const _Tp __s = sin(__phi_red);
           const _Tp __c = cos(__phi_red);

           const _Tp __F = __s
                         * __ellint_rf(__c * __c,
                                 _Tp(1) - __k * __k * __s * __s, _Tp(1));

          if (__n == 0)
             return __F;
          else
            return __F + _Tp(2) * __n * __comp_ellint_1(__k);
}

__global__
void ellpe_kernel(fod *out, fod *in, int num){
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < num;
       i += blockDim.x * gridDim.x){
        //out[i] = ellpe(0.5);
        //out[i] = ellpk(0.5);
        out[i] = __ellint_3(sqrt(0.4), 0.6, 3.1415926535897932384626433832795028841971/4.);
        //out[i] = __ellint_2(sqrt(0.5), 3.1415926535897932384626433832795028841971/4.);
        //out[i] = __ellint_1(sqrt(0.5), 3.1415926535897932384626433832795028841971/4.);
    }
}



void ellpe_test(){
  int num = 10000000;
  fod *out, *in;
  fod *outn = new fod[num];
  fod *inn = new fod[num];
  cudaMalloc(&out, num*sizeof(fod));
  cudaMalloc(&in, num*sizeof(fod));

  for (int i=0; i<num; i++){
    inn[i] = 0.2389239;
  }
  cudaMemcpy(in, inn, num*sizeof(fod), cudaMemcpyHostToDevice);
  int num_blocks = ceil((num + NUM_THREADS -1)/NUM_THREADS);
  dim3 gridDim(num_blocks); //, num_teuk_modes);
  ellpe_kernel<<<gridDim, NUM_THREADS>>>(out, in, num);
  cudaDeviceSynchronize();
  gpuErrchk_here(cudaGetLastError());

  cudaMemcpy(outn, out, num*sizeof(fod), cudaMemcpyDeviceToHost);
  printf("%e %e\n", outn[0], outn[1]);

  delete[] outn;
  delete[] inn;
  cudaFree(out);
  cudaFree(in);
}
