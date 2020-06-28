#ifndef __ELLIPTIC_H__
#define __ELLIPTIC_H__

#define MACHEP 1.38777878078144567553E-17
#define MAXNUM 1.701411834604692317316873e38


/*
* Complete Elliptic Integral of the first kind
*
*/
template<typename _Tp>
__device__ _Tp
EllipticK(const _Tp m);

/*
* Incomplete Elliptic Integral of the first kind
*
*/
template<typename _Tp>
__device__ _Tp
EllipticF(const _Tp phi, const _Tp m);

/*
* Incomplete/Complete Elliptic Integral of the second kind
*
*/
template<typename _Tp>
__device__ _Tp
EllipticE(const _Tp phi, const _Tp m);

/*
* Incomplete/Complete Elliptic Integral of the third kind
*
*/
template<typename _Tp>
__device__ _Tp
EllipticPi(const _Tp n, const _Tp phi, const _Tp m);

void ellpe_test(fod *input_mat, int num_p_e);

#endif // __ELLIPTIC_H__
