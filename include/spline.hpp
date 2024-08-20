#ifndef SPLINE_HPP
#define SPLINE_HPP

#include <vector>
#include <algorithm>
#include "omp.h"
#include <chrono>

class StopWatch{
public:
	StopWatch();

	void start();
	void stop();
	void reset();
	void print();
	void print(int cycles);
	double time();

private:
	double time_elapsed;
	std::chrono::high_resolution_clock::time_point t1;
	std::chrono::high_resolution_clock::time_point t2;
};

typedef std::vector<double> Vector;
class Matrix{
public:
	Matrix();
	Matrix(int n);
	Matrix(int n, int m);
	Matrix(int n, int m, Vector A);
	Matrix(int n, int m, double val);

	int rows() const;
	int cols() const;
	int size() const;

	void row_replace(int i, Vector row);
	void col_replace(int j, Vector col);

	Vector row(int i);
	Vector col(int j);

	void reshape(int n, int m);
	Matrix reshaped(int n, int m) const;
	Matrix transpose() const;
	void transposeInPlace();

	void set_value(int i, int j, double val);

	double& operator()(int i, int j);
	const double& operator()(int i, int j) const;

private:
	int _n;
	int _m;
	Vector _A;
};

class ThreeTensor{
public:
	ThreeTensor();
	ThreeTensor(int nx);
	ThreeTensor(int nx, int ny, int nz);
	ThreeTensor(int nx, int ny, int nz, Vector A);
	ThreeTensor(int nx, int ny, int nz, double *A);
	ThreeTensor(int nx, int ny, int nz, double val);

	int rows() const;
	int cols() const;
	int slcs() const; // slices
	int size() const;

	void row_replace(int i, Matrix row);
	void col_replace(int j, Matrix col);
	void slc_replace(int k, Matrix slc);

	Matrix row(int i);
	Vector rowcol(int i, int j);
	Vector rowslc(int i, int k);
	Matrix col(int j);
	Vector colslc(int j, int k);
	Matrix slc(int k);

	void reshape(int nx, int ny, int nz);
	ThreeTensor reshaped(int nx, int ny, int nz) const;

	void set_value(int i, int j, int k, double val);

	double& operator()(int i, int j, int k);
	const double& operator()(int i, int j, int k) const;

private:
	int _nx;
	int _ny;
	int _nz;
	Vector _A;
};

/////////////////////////////////////////////////////////
////               Basic Interpolators               ////
/////////////////////////////////////////////////////////

class CubicSpline{
public:
	CubicSpline(double x0, double dx, int nx, const Vector &y, int method = 1);
	CubicSpline(double x0, double dx, const Vector &y, int method = 1);
	CubicSpline(const Vector &x, const Vector &y, int method = 1);

    CubicSpline(double x0, double dx, int nintervals, Matrix cij);
	
	double evaluate(const double x);
    double derivative(const double x);
    double derivative2(const double x);

	double getSplineCoefficient(int i, int j);

private:
	double evaluateInterval(int i, const double x);
    double evaluateDerivativeInterval(int i, const double x);
    double evaluateSecondDerivativeInterval(int i, const double x);
	void computeSplineCoefficients(double dx, const Vector &y);
	void computeSplineCoefficientsNaturalFirst(double dx, const Vector &y);
	void computeSplineCoefficientsNotAKnot(double dx, const Vector &y);
	void computeSplineCoefficientsZeroClamped(double dx, const Vector &y);
	void computeSplineCoefficientsE3(double dx, const Vector &y);
	int findInterval(const double x);

	double dx;
	int nintervals;
	double x0;
	Matrix cij;
};

class BicubicSpline{
public:
	BicubicSpline(const Vector &x, const Vector &y, Matrix &z, int method = 3);
	BicubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, Matrix &z, int method = 3);
	BicubicSpline(const Vector &x, const Vector &y, const Vector &z, int method = 3);
	BicubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, const Vector &z_vec, int method = 3);
	double evaluate(const double x, const double y);
    double derivative_x(const double x, const double y);
    double derivative_y(const double x, const double y);
    double derivative_xy(const double x, const double y);
    double derivative_xx(const double x, const double y);
    double derivative_yy(const double x, const double y);
    CubicSpline reduce_x(const double x);
    CubicSpline reduce_y(const double y);

	double getSplineCoefficient(int i, int j, int nx, int ny);

private:
	double evaluateInterval(int i, int j, const double x, const double y);
    double evaluateDerivativeXInterval(int i, int j, const double x, const double y);
    double evaluateDerivativeYInterval(int i, int j, const double x, const double y);
    double evaluateDerivativeXYInterval(int i, int j, const double x, const double y);
    double evaluateDerivativeXXInterval(int i, int j, const double x, const double y);
    double evaluateDerivativeYYInterval(int i, int j, const double x, const double y);
	Matrix computeSplineCoefficientsDX(Matrix &m_z, int method = 3);
	Matrix computeSplineCoefficientsDY(Matrix &m_z, int method = 3);
	void computeSplineCoefficients(Matrix &z, int method = 3);
	int findXInterval(const double x);
	int findYInterval(const double y);

	double dx;
	double dy;
	int nx;
	int ny;
	double x0;
	double y0;
	Matrix cij;
};

class TricubicSpline{
public:
	TricubicSpline(const Vector &x, const Vector &y, const Vector &z, ThreeTensor &f, int method = 3);
	TricubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, ThreeTensor &f, int method = 3);
	TricubicSpline(const Vector &x, const Vector &y, const Vector &z, const Vector &f, int method = 3);
	TricubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, const Vector &f_vec, int method = 3);
	double evaluate(const double x, const double y, const double z);
    double derivative_x(const double x, const double y, const double z);
    double derivative_y(const double x, const double y, const double z);
	double derivative_z(const double x, const double y, const double z);
    double derivative_xy(const double x, const double y, const double z);
	double derivative_xz(const double x, const double y, const double z);
	double derivative_yz(const double x, const double y, const double z);
    double derivative_xx(const double x, const double y, const double z);
    double derivative_yy(const double x, const double y, const double z);
	double derivative_zz(const double x, const double y, const double z);
    // BicubicSpline reduce_x(const double x);
    // BicubicSpline reduce_y(const double y);
	// BicubicSpline reduce_z(const double z);

	double getSplineCoefficient(int i, int j, int k, int nx, int ny, int nz);

private:
	double evaluateInterval(int i, int j, int k, const double x, const double y, const double z);
    double evaluateDerivativeXInterval(int i, int j, int k, const double x, const double y, const double z);
    double evaluateDerivativeYInterval(int i, int j, int k, const double x, const double y, const double z);
	double evaluateDerivativeZInterval(int i, int j, int k, const double x, const double y, const double z);
    double evaluateDerivativeXYInterval(int i, int j, int k, const double x, const double y, const double z);
	double evaluateDerivativeXZInterval(int i, int j, int k, const double x, const double y, const double z);
	double evaluateDerivativeYZInterval(int i, int j, int k, const double x, const double y, const double z);
    double evaluateDerivativeXXInterval(int i, int j, int k, const double x, const double y, const double z);
    double evaluateDerivativeYYInterval(int i, int j, int k, const double x, const double y, const double z);
	double evaluateDerivativeZZInterval(int i, int j, int k, const double x, const double y, const double z);
	void computeSplineCoefficients(ThreeTensor &z, int method = 3);
	int findXInterval(const double x);
	int findYInterval(const double y);
	int findZInterval(const double z);

	void setSplineCoefficient(int i, int j, int k, int nx, int ny, int nz, double coeff);

	double dx;
	double dy;
	double dz;
	int nx;
	int ny;
	int nz;
	double x0;
	double y0;
	double z0;
	ThreeTensor cijk;
};

#endif