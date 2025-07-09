// build_grad_hess_r_QC: assembly the gradient and Hessian
// for Iso-P triangular elemetns with linear, quadratic,
// or cubic interpolations and compressible hyperelastic material model

#include <algorithm>
#include <math.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "matrix.h"
#include "mex.h"
#include "myfem.h"
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#define EIGEN_DONT_PARALLELIZE

typedef Eigen::Triplet<double> T;

/************************ Main program ************************/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
				 const mxArray *prhs[])
{

	// Test for six input arguments
	if (nrhs != 6)
		mexErrMsgTxt("Six input arguments required.");

	// Get the input data
	double *prp = mxGetPr(prhs[0]);
	double *prt = mxGetPr(prhs[1]);
	double *prmat = mxGetPr(prhs[2]);
	int ngauss = (int)mxGetScalar(prhs[3]);
	double *pru = mxGetPr(prhs[4]);
	int maxNumThreads = (int)mxGetScalar(prhs[5]);
	int nnodes = (int)mxGetN(prhs[0]);
	int nelem = (int)mxGetN(prhs[1]);
	int mt = (int)mxGetM(prhs[1]);
	int np = mt - 1; // number of points in an element
	int mmat = (int)mxGetM(prhs[2]);
	int ndof = 2 * nnodes;

	// Test the number of dofs
	if (ndof != (int)mxGetM(prhs[4]))
		mexErrMsgTxt("Wrong number of dofs.");

	// Compute the gradient and Hessian, populate prG, prI, prJ, prS
	Eigen::SparseMatrix<double> M(ndof, ndof);
	int tr;				   // triangle ID
	int errCountGauss = 0; // integration rule: error control inside parallel region
	int errCountEl = 0;	// constitutive law: error control inside parallel region
#pragma omp parallel num_threads(maxNumThreads)
	{
		int _i, _j, _k, _ig;
		double _Jdet, _thickness;
		double _tcoord[3], _matconst[8]; // use total Lagrangian formulation
		Eigen::SparseMatrix<double> _M(ndof, ndof);
		auto _tripletList = std::vector<T>{};
		_tripletList.reserve(4 * np * np * nelem);
		int *_idel = new int[np];		   // ids of nodes of given element
		int *_cod = new int[2 * np];	   // code numbers of given element
		double *_uel = new double[2 * np]; // displacements associated with given element
		double *_x = new double[np];	   // x-coordinates of all nodes of given element
		double *_y = new double[np];	   // y-coordinates of all nodes of given element
		double *_N = new double[np];	   // shape function evaluations
		double **_Ne = new double *[2];	// element's N matrix
		for (_i = 0; _i < 2; _i++)
			_Ne[_i] = new double[2 * np];
		double **_Me = new double *[2 * np]; // element's mass matrix
		for (_i = 0; _i < 2 * np; _i++)
			_Me[_i] = new double[2 * np];
		double **_Be = new double *[4]; // element's matrix of derivatives of basis functions
		for (_i = 0; _i < 4; _i++)
			_Be[_i] = new double[2 * np];
		double **_IP = new double *[abs(ngauss)];
		for (_i = 0; _i < abs(ngauss); _i++)
			_IP[_i] = new double[3];
		double *_alpha = new double[abs(ngauss)];
		int (*_getShapeFun)(const int np, const double tcoord[3], double *N) = NULL; // pointer to shape functions function
		int (*_getBe)(const int np, const double tcoord[3],
					  const double *x, const double *y, double &Jdet, double **Be) = NULL; // pointer to Be functions
		if (np == 3 || np == 6 || np == 10)
		{

			// Choose Gauss integration rule
			if (ngauss == 1 || ngauss == 3 || ngauss == -3 || ngauss == 4 ||
				ngauss == 6 || ngauss == 7)
				getTriagGaussInfo(ngauss, _alpha, _IP); // triangles: ngauss = 1, 3 (interior), -3 (midedge), 4, 6, 7
			else
			{
#pragma omp atomic
				errCountGauss++; // throw error if unrecognized integration rule used
			}

			// Choose getBe function
			_getBe = &getTriagBeTLF;

			// Choose _getShapeFun function
			_getShapeFun = &getTriagShapeFun;
		}
		else if (np == 4 || np == 8 || np == 9 || np == 16)
		{
			// Choose Gauss integration rule
			if (ngauss == 1 || ngauss == 4 || ngauss == 9 || ngauss == 16 ||
				ngauss == 25)
				getQuadGaussInfo(ngauss, _alpha, _IP); // quadrangles: ngauss = 1, 4, 9, 16, 25
			else
			{
#pragma omp atomic
				errCountGauss++; // throw error if unrecognized integration rule used
			}

			// Choose getBe function
			_getBe = &getQuadBeTLF;

			// Choose _getShapeFun function
			_getShapeFun = &getQuadShapeFun;
		}
		else
		{
#pragma omp atomic
			errCountEl++; // throw error if unrecognized element type used
		}

#pragma omp for
		for (tr = 0; tr < nelem; tr++)
		{ // loop over all elements

			// Get code numbers of i-th element and material ID
			for (_i = 0; _i < np; _i++)
				_idel[_i] = (int)prt[_i + mt * tr];
			for (_i = 0; _i < np; _i++)
			{
				_cod[2 * _i] = 2 * _idel[_i] - 1;
				_cod[2 * _i + 1] = 2 * _idel[_i];
			}

			// Get element's displacements
			for (_i = 0; _i < 2 * np; _i++)
				_uel[_i] = pru[_cod[_i] - 1];

			// Get element's coordinates
			for (_i = 0; _i < np; _i++)
			{
				_x[_i] = prp[0 + (_idel[_i] - 1) * 2];
				_y[_i] = prp[1 + (_idel[_i] - 1) * 2];
			}

			// Get material constants
			for (_i = 0; _i < 8; _i++)
				_matconst[_i] = prmat[int(prt[np + mt * tr] - 1) + _i * mmat];
			_thickness = _matconst[7]; // element's thinckness

			// Gauss integration rule
			for (_i = 0; _i < 2 * np; _i++)
			{
				for (_j = 0; _j < 2 * np; _j++)
				{
					_Me[_i][_j] = 0.0;
				}
			}
			for (_ig = 0; _ig < abs(ngauss); _ig++)
			{

				// Get shape function information evaluated at given Gauss point: jacobian _Jdet, matrix of basis functions' derivatives _Be
				for (_i = 0; _i < 3; _i++)
					_tcoord[_i] = _IP[_ig][_i]; // get natural coordinates of Gauss integration rule
				_getBe(np, _tcoord, _x, _y, _Jdet, _Be);

				// Get shape function evaluations
				_getShapeFun(np, _tcoord, _N);

				// Matrix Ne of bais functions
				for (_i = 0; _i < np; _i++)
				{
					_Ne[0][2 * _i + 0] = _N[_i];
					_Ne[0][2 * _i + 1] = 0.0;
					_Ne[1][2 * _i + 0] = 0.0;
					_Ne[1][2 * _i + 1] = _N[_i];
				}

				// Integrate element's mass matrix _Me = _Me + alpha(ig)*(Ne')*Ne*Jdet*h
				for (_i = 0; _i < 2 * np; _i++)
				{
					for (_j = 0; _j < 2 * np; _j++)
					{
						for (_k = 0; _k < 2; _k++)
						{
							_Me[_i][_j] += _alpha[_ig] * _thickness * _Jdet * _Ne[_k][_i] * _Ne[_k][_j];
						}
					}
				}
			}

			// Allocate mass matrix
			for (_i = 0; _i < 2 * np; _i++)
			{
				for (_j = 0; _j < 2 * np; _j++)
				{
					_tripletList.push_back(T(_cod[_i] - 1, _cod[_j] - 1, _Me[_i][_j]));
				}
			}
		}

		// Assemble _K from triplets
		_M.setFromTriplets(_tripletList.begin(), _tripletList.end());

		// Collect _K from all threads
#pragma omp critical(ALLOCATE)
		{
			M += _M;
		}

		// Delete dynamically allocated memory
		delete[] _idel;
		delete[] _cod;
		delete[] _uel;
		delete[] _x;
		delete[] _y;
		for (_i = 0; _i < 2; _i++)
			delete[] _Ne[_i];
		delete[] _Ne;
		for (_i = 0; _i < 2 * np; _i++)
			delete[] _Me[_i];
		delete[] _Me;
		for (_i = 0; _i < 4; _i++)
			delete[] _Be[_i];
		delete[] _Be;
		for (_i = 0; _i < abs(ngauss); _i++)
			delete[] _IP[_i];
		delete[] _IP;
		delete[] _alpha;
	}

	// Catch errors
	if (errCountGauss != 0)
		mexErrMsgTxt("Unrecognized Gauss integration rule chosen.");
	if (errCountEl != 0)
		mexErrMsgTxt("Unrecognized element type chosen.");

	// Send out data back to matlab
	nlhs = 1;
	plhs[0] = mxCreateSparse(M.rows(), M.cols(), M.nonZeros(), mxREAL);
	int *ir = M.innerIndexPtr();
	int *jc = M.outerIndexPtr();
	double *pr = M.valuePtr();
	mwIndex *ir2 = mxGetIr(plhs[0]);
	mwIndex *jc2 = mxGetJc(plhs[0]);
	double *pr2 = mxGetPr(plhs[0]);
	for (auto i = 0; i < M.nonZeros(); ++i)
	{
		pr2[i] = pr[i];
		ir2[i] = ir[i];
	}
	for (auto i = 0; i < M.cols() + 1; ++i)
	{
		jc2[i] = jc[i];
	}
}
