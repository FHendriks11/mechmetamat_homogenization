// build_kinematic_averages
// computes FE integrals on a mesh represented by the point p and connectivity t
// matrices of a vector mode phi, rank 2 tensor grad(phi), and rank 3 tensor X
// grad(phi)

#include <algorithm>
#include <math.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "matrix.h"
#include "mex.h"
#include "myfem.h"

/************************ Main program ************************/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	// Test for six input arguments
	if (nrhs != 6)
		mexErrMsgTxt("Six input arguments required.");

	// Get the input data
	double *prp = mxGetPr(prhs[0]);
	double *prt = mxGetPr(prhs[1]);
	int ngauss = (int)mxGetScalar(prhs[2]);
	double *prgradvi = mxGetPr(prhs[3]);
	double *prphi = mxGetPr(prhs[4]);
	int maxNumThreads = (int)mxGetScalar(prhs[5]);
	int nnodes = (int)mxGetN(prhs[0]);
	int nelem = (int)mxGetN(prhs[1]);
	int mt = (int)mxGetM(prhs[1]);
	int np = mt - 1; // number of points in an element
	int ndof = 2 * nnodes;

	// Test dimensionality of gradvi
	if (2 != (int)std::max(mxGetM(prhs[3]), mxGetN(prhs[3])))
		mexErrMsgTxt("Two components of gradvi expected.");

	// Test the number of dofs
	if (ndof != (int)mxGetM(prhs[4]))
		mexErrMsgTxt("Wrong number of dofs.");

	// Allocate outputs
	nlhs = 4;
	// Averaged geometric quantities
	plhs[0] = mxCreateDoubleMatrix(2, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(4, 1, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(4, 1, mxREAL);
	plhs[3] = mxCreateDoubleMatrix(4, 1, mxREAL);
	// Get pointers
	double volume = 0.0;
	double *prMeanPhi = mxGetPr(plhs[0]);
	double *prMeanGradPhi = mxGetPr(plhs[1]);
	double *prMeanXGradPhi = mxGetPr(plhs[2]);
	double *prMeanXPhi = mxGetPr(plhs[3]);

	// Compute the gradient and Hessian, populate prG, prI, prJ, prS
	int tr; // triangle ID
	int errCountGauss =
		0;				// integration rule: error control inside parallel region
	int errCountEl = 0; // constitutive law: error control inside parallel region
#pragma omp parallel num_threads(maxNumThreads)
	{
		int _i, _j, _k, _ig;
		double _Jdet, _XdotGradPhi;
		double _tcoord[3], _phi[2], _gradphi[4], _X[2];
		double _elvolume, _meanPhi[2], _meanGradPhi[4], _meanXPhi[4],
			_meanGradXPhi[4];
		int *_idel = new int[np];	// ids of nodes of given element
		int *_cod = new int[2 * np]; // code numbers of given element
		double *_phiel =
			new double[2 * np];		 // displacements associated with given element
		double *_x = new double[np]; // x-coordinates of all nodes of given element
		double *_y = new double[np]; // y-coordinates of all nodes of given element
		double *_N = new double[np]; // shape function evaluations
		double **_Be =
			new double *[4]; // element's matrix of derivatives of basis functions
		for (_i = 0; _i < 4; _i++)
			_Be[_i] = new double[2 * np];
		double **_IP = new double *[abs(ngauss)];
		for (_i = 0; _i < abs(ngauss); _i++)
			_IP[_i] = new double[3];
		double *_alpha = new double[abs(ngauss)];
		int (*_getShapeFun)(const int np, const double tcoord[3], double *N) =
			NULL; // pointer to shape functions function
		int (*_getBe)(const int np, const double tcoord[3], const double *x,
					  const double *y, double &Jdet, double **Be) =
			NULL; // pointer to Be functions
		if (np == 3 || np == 6 || np == 10)
		{

			// Choose Gauss integration rule
			if (ngauss == 1 || ngauss == 3 || ngauss == -3 || ngauss == 4 ||
				ngauss == 6 || ngauss == 7)
				getTriagGaussInfo(
					ngauss, _alpha,
					_IP); // triangles: ngauss = 1, 3 (interior), -3 (midedge), 4, 6, 7
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
				getQuadGaussInfo(ngauss, _alpha,
								 _IP); // quadrangles: ngauss = 1, 4, 9, 16, 25
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
			{
				_phiel[_i] = prphi[_cod[_i] - 1];
			}

			// Get element's coordinates
			for (_i = 0; _i < np; _i++)
			{
				_x[_i] = prp[0 + (_idel[_i] - 1) * 2];
				_y[_i] = prp[1 + (_idel[_i] - 1) * 2];
			}

			// Gauss integration rule
			// Initialize all kinematic average quantities
			_elvolume = 0.0;
			for (_i = 0; _i < 2; _i++)
			{
				_meanPhi[_i] = 0.0;
				for (_j = 0; _j < 2; _j++)
				{
					_meanGradPhi[_i + 2 * _j] = 0.0;
					_meanXPhi[_i + 2 * _j] = 0.0;
					_meanGradXPhi[_i + 2 * _j] = 0.0;
				}
			}

			// Loop over all Gauss integration points
			for (_ig = 0; _ig < abs(ngauss); _ig++)
			{

				// Get shape function information evaluated at given Gauss point:
				// jacobian _Jdet, matrix of basis functions' derivatives _Be
				for (_i = 0; _i < 3; _i++)
					_tcoord[_i] =
						_IP[_ig][_i]; // get natural coordinates of Gauss integration rule

				// Get matrix of shape functions derivatives
				_getBe(np, _tcoord, _x, _y, _Jdet, _Be);

				// Get shape function evaluations
				_getShapeFun(np, _tcoord, _N);

				// Get phi and X_m
				_phi[0] = 0.0; // initialize phi
				_phi[1] = 0.0;
				_X[0] = 0.0; // initialize X_m
				_X[1] = 0.0;
				for (_i = 0; _i < np; _i++)
				{
					_phi[0] += _phiel[2 * _i + 0] * _N[_i];
					_phi[1] += _phiel[2 * _i + 1] * _N[_i];
					_X[0] += _x[_i] * _N[_i];
					_X[1] += _y[_i] * _N[_i];
				}

				// Get grad(phi)
				for (_i = 0; _i < 2; _i++)
					for (int _j = 0; _j < 2; _j++)
					{
						_gradphi[_i + 2 * _j] = 0.0;
						for (_k = 0; _k < 2 * np; _k++)
							_gradphi[_i + 2 * _j] += _Be[_i + 2 * _j][_k] * _phiel[_k];
					}

				// Integrate element's volume _elvolume = _elvolume + alpha(ig)*Jdet
				_elvolume += _alpha[_ig] * _Jdet;

				// Integrate average mode _meanPhi = _meanPhi + alpha(ig)*_phi(:)*Jdet
				for (_i = 0; _i < 2; _i++)
					_meanPhi[_i] += _alpha[_ig] * _Jdet * _phi[_i];

				// Integrate average gradient of the mode _meanGradPhi = _meanGradPhi +
				// alpha(ig)*_gradphi(:)*Jdet
				for (_i = 0; _i < 2; _i++)
					for (_j = 0; _j < 2; _j++)
						_meanGradPhi[_i + 2 * _j] +=
							_alpha[_ig] * _Jdet * _gradphi[_i + 2 * _j];

				// Integrate average of X times gradient of the mode _meanGradXPhi =
				// _meanGradXPhi + alpha(ig)*X(:)*_gradphi(:)'*Jdet
				_XdotGradPhi = _X[0] * prgradvi[0] + _X[1] * prgradvi[1];
				for (_i = 0; _i < 2; _i++)
					for (_j = 0; _j < 2; _j++)
						_meanGradXPhi[_i + 2 * _j] +=
							_alpha[_ig] * _Jdet * _XdotGradPhi * _gradphi[_i + 2 * _j];

				// Integrate the first moments of the mode _meanXPhi = _meanXPhi +
				// alpha(ig)*X*_phi(:)*Jdet
				for (_i = 0; _i < 2; _i++)
					for (_j = 0; _j < 2; _j++)
						_meanXPhi[_i + 2 * _j] += _alpha[_ig] * _Jdet * _X[_i] * _phi[_j];
			}

			// Allocate volume
#pragma omp atomic
			volume += _elvolume;

			// Allocate _meanPhi
#pragma omp atomic
			prMeanPhi[0] += _meanPhi[0];
#pragma omp atomic
			prMeanPhi[1] += _meanPhi[1];

			// Allocate prMeanGradPhi
			for (_i = 0; _i < 4; _i++)
			{
#pragma omp atomic
				prMeanGradPhi[_i] += _meanGradPhi[_i];
			}

			// Allocate _meanGradXPhi
			for (_i = 0; _i < 4; _i++)
			{
#pragma omp atomic
				prMeanXGradPhi[_i] += _meanGradXPhi[_i];
			}

			// Allocate prMeanXPhi
			for (_i = 0; _i < 4; _i++)
			{
#pragma omp atomic
				prMeanXPhi[_i] += _meanXPhi[_i];
			}
		}

		// Delete dynamically allocated memory
		delete[] _idel;
		delete[] _cod;
		delete[] _phiel;
		delete[] _x;
		delete[] _y;
		delete[] _N;
		for (_i = 0; _i < 4; _i++)
			delete[] _Be[_i];
		delete[] _Be;
		for (_i = 0; _i < abs(ngauss); _i++)
			delete[] _IP[_i];
		delete[] _IP;
		delete[] _alpha;
	}

	// Normalize integrated quantities by the volume
	for (int i = 0; i < 2; i++)
	{
		prMeanPhi[i] /= volume;
		for (int j = 0; j < 2; j++)
		{
			prMeanGradPhi[i + 2 * j] /= volume;
			prMeanXPhi[i + 2 * j] /= volume;
			prMeanXGradPhi[i + 2 * j] /= volume;
		}
	}

	// Catch errors
	if (errCountGauss != 0)
		mexErrMsgTxt("Unrecognized Gauss integration rule chosen.");
	if (errCountEl != 0)
		mexErrMsgTxt("Unrecognized element type chosen.");
}
