// build_grad_hess_r_QC: assembly the gradient and Hessian
// for Iso-P triangular elemetns with linear, quadratic,
// or cubic interpolations and compressible hyperelastic material model

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <mex.h>
#include "matrix.h"
#include "myfem.h"
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#define EIGEN_DONT_PARALLELIZE

typedef Eigen::Triplet<double> T;

/************************ Main program ************************/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	// Test for nine input arguments
	if (nrhs != 9)
		mexErrMsgTxt("Nine input arguments required.");

	// Get the input data
	double *prp = mxGetPr(prhs[0]);
	double *prt = mxGetPr(prhs[1]);
	double *prmat = mxGetPr(prhs[2]);
	double *prNgauss = mxGetPr(prhs[3]);
	double *pru = mxGetPr(prhs[4]);
	double *prmaxNumThreads = mxGetPr(prhs[5]);
	double stabilizationConstant = mxGetScalar(prhs[6]); // stabilization constant
	int swmethod = (int)mxGetScalar(prhs[7]);
	int swstab = (int)mxGetScalar(prhs[8]);
	int nnodes = (int)mxGetN(prhs[0]);
	int nelem = (int)mxGetN(prhs[1]);
	int mt = (int)mxGetM(prhs[1]);
	int np = mt - 1; // number of points in an element
	int mmat = (int)mxGetM(prhs[2]);
	int ngauss = (int)prNgauss[0];
	int ndof = 2 * nnodes;
	int maxNumThreads = (int)prmaxNumThreads[0];

	// Test the number of dofs
	if (ndof != (int)mxGetM(prhs[4]))
		mexErrMsgTxt("Wrong number of dofs.");

	// Allocate outputs
	nlhs = 4;
	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(ndof, 1, mxREAL);
	double *prW = mxGetPr(plhs[0]);
	double *prWhourglass = mxGetPr(plhs[1]);
	double *prG = mxGetPr(plhs[2]);

	// Compute the gradient and Hessian, populate prG, prI, prJ, prS
	Eigen::SparseMatrix<double> K(ndof, ndof);
	int tr;				   // triangle ID
	int errCountGauss = 0; // integration rule: error control inside parallel region
	int errCountMat = 0;   // constitutive law: error control inside parallel region
	int errCountEl = 0;	   // constitutive law: error control inside parallel region
#pragma omp parallel num_threads(maxNumThreads)
	{
		int _i, _j, _k, _l, _ig;
		double _w, _W, _Whourglass, _Jdet, _thickness;
		double _F[4], _C[3], _S[3], _S4[4][4], _CD[3][3], _tcoord[3], _matconst[8]; // use total Lagrangian formulation
		Eigen::SparseMatrix<double> _K(ndof, ndof);
		auto _tripletList = std::vector<T>{};
		_tripletList.reserve(4 * np * np * nelem);
		int *_idel = new int[np];			 // ids of nodes of given element
		int *_cod = new int[2 * np];		 // code numbers of given element
		double *_uel = new double[2 * np];	 // displacements associated with given element
		double *_x = new double[np];		 // x-coordinates of all nodes of given element
		double *_y = new double[np];		 // y-coordinates of all nodes of given element
		double *_fe = new double[2 * np];	 // internal force vector
		double **_Ke = new double *[2 * np]; // element's stiffness matrix
		for (_i = 0; _i < 2 * np; _i++)
			_Ke[_i] = new double[2 * np];

		// Stabilization
		double **_Kestab = new double *[2 * np]; // element's stabilization matrix
		for (_i = 0; _i < 2 * np; _i++)
			_Kestab[_i] = new double[2 * np];
		double *_festab = new double[2 * np]; // internal force vector
		double *_gamma = new double[np];
		double *_h = new double[np];
		double *_bx = new double[np];
		double *_by = new double[np];
		double *_tbx = new double[np];
		double *_tby = new double[np];
		double *_tx = new double[np];
		double *_ty = new double[np];

		double **_Be = new double *[3]; // element's matrix of derivatives of basis functions (Green-Lagrange strain)
		for (_i = 0; _i < 3; _i++)
			_Be[_i] = new double[2 * np];
		double **_Bf = new double *[4]; // element's matrix of derivatives of basis functions (deformation gradient)
		for (_i = 0; _i < 4; _i++)
			_Bf[_i] = new double[2 * np];
		double **_IP = new double *[abs(ngauss)];
		for (_i = 0; _i < abs(ngauss); _i++)
			_IP[_i] = new double[3];
		double *_alpha = new double[abs(ngauss)];
		for (_i = 0; _i < 4; _i++)
			for (_j = 0; _j < 4; _j++)
				_S4[_i][_j] = 0.0;
		int (*_getBe)(const int np, const double tcoord[3], const double *x, const double *y, const double *uel, double &Jdet, double **Be, double **Bf, double F[4]) = NULL; // pointer to Be functions
		int (*_getWSC)(const double matconst[8], const double C[3], double &w, double S[3], double D[3][3]) = NULL;															  // pointer to constitutive law functions
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
			_getBe = &getTriagBeTLE;
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
			_getBe = &getQuadBeTLE;
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

			// Choose constitutive law
			switch ((int)_matconst[0])
			{
			case 1: // OOFEM
				_getWSC = &material_oofemE2d;
				break;
			case 2: // Bertoldi, Boyce
				_getWSC = &material_bbE2d;
				break;
			case 3: // Jamus, Green, Simpson
				_getWSC = &material_jgsE2d;
				break;
			case 4: // five-term Mooney-Rivlin
				_getWSC = &material_5mrE2d;
				break;
			case 5: // elastic material
				_getWSC = &material_leE2d;
				break;
			case 6: // Ogden material
				_getWSC = &material_ogdenE2d;
				break;
			case 7: // Ogden nematic material
				_getWSC = &material_ogdennematicE2d;
				break;
			default: // if unrecognized material used, use OOFEM in order not to
					 // crash, but throw an error
				_getWSC = &material_oofemE2d;
#pragma omp atomic
				errCountMat++;
				break;
			}

			// Gauss integration rule
			_W = 0.0;
			_Whourglass = 0.0;
			for (_i = 0; _i < 2 * np; _i++)
			{
				_fe[_i] = 0.0;
				_festab[_i] = 0.0;
				for (_j = 0; _j < 2 * np; _j++)
				{
					_Ke[_i][_j] = 0.0;
					_Kestab[_i][_j] = 0.0;
				}
			}
			for (_ig = 0; _ig < abs(ngauss); _ig++)
			{

				// Get shape function information evaluated at given Gauss point:
				// jacobian _Jdet, matrix of basis functions' derivatives _Be
				for (_i = 0; _i < 3; _i++)
					_tcoord[_i] = _IP[_ig][_i]; // get natural coordinates of Gauss integration rule

				_getBe(np, _tcoord, _x, _y, _uel, // input
					   _Jdet, _Be, _Bf, _F);	  // output
				getC2d(_F, _C);

				// Get energy density, stress, and material stiffness
				_getWSC(_matconst, _C, // inputs
						_w, _S, _CD);  // outputs

				// Integrate element's energy W = W + alpha(ig)*Wd*_Jdet*h
				_W += _alpha[_ig] * _thickness * _Jdet * _w;

				// Integrate element's gradient f = f + alpha(ig)*(B')*S(:)*Jdet*h
				for (_i = 0; _i < 3; _i++)
					for (_k = 0; _k < 2 * np; _k++)
						_fe[_k] += _alpha[_ig] * _thickness * _Jdet * _Be[_i][_k] * _S[_i];

				// Integrate element's Hessian K = K + alpha(ig)*( (B')*D*B + (G')*S*G)*Jdet*h
				for (_i = 0; _i < 2 * np; _i++)
					for (_j = 0; _j < 2 * np; _j++)
						for (_k = 0; _k < 3; _k++)
							for (_l = 0; _l < 3; _l++)
								_Ke[_i][_j] += _alpha[_ig] * _Jdet * _thickness * _Be[_k][_i] * _CD[_k][_l] * _Be[_l][_j]; // small and large displacement matrix

				// Geometric stiffness
				_S4[0][0] = _S[0];
				_S4[1][1] = _S[0];
				_S4[2][0] = _S[2];
				_S4[3][1] = _S[2];
				_S4[0][2] = _S[2];
				_S4[1][3] = _S[2];
				_S4[2][2] = _S[1];
				_S4[3][3] = _S[1];
				for (_i = 0; _i < 2 * np; _i++)
					for (_j = 0; _j < 2 * np; _j++)
						for (_k = 0; _k < 4; _k++)
							for (_l = 0; _l < 4; _l++)
								_Ke[_i][_j] += _alpha[_ig] * _Jdet * _thickness * _Bf[_k][_i] * _S4[_k][_l] * _Bf[_l][_j]; // geometric stiffness matrix
			}

			// Construct stabilization matrix _Kestab
			if ((np == 4) && (ngauss == 1))
			{

				// Get gradients
				double _A = 0.5 * ((_x[2] - _x[0]) * (_y[3] - _y[1]) + (_x[1] - _x[3]) * (_y[2] - _y[0])); // A = 4*Jdet
				_bx[0] = 1.0 / (2.0 * _A) * (_y[1] - _y[3]);
				_bx[1] = 1.0 / (2.0 * _A) * (_y[2] - _y[0]);
				_bx[2] = 1.0 / (2.0 * _A) * (_y[3] - _y[1]);
				_bx[3] = 1.0 / (2.0 * _A) * (_y[0] - _y[2]);

				_by[0] = 1.0 / (2.0 * _A) * (_x[3] - _x[1]);
				_by[1] = 1.0 / (2.0 * _A) * (_x[0] - _x[2]);
				_by[2] = 1.0 / (2.0 * _A) * (_x[1] - _x[3]);
				_by[3] = 1.0 / (2.0 * _A) * (_x[2] - _x[0]);

				// Get stabilization vector gamma
				_h[0] = 1.0;
				_h[1] = -1.0;
				_h[2] = 1.0;
				_h[3] = -1.0;

				double _htx = 0.0;
				double _hty = 0.0;
				double _normb = 0.0;
				for (_i = 0; _i < np; _i++)
				{
					_htx += _h[_i] * _x[_i];
					_hty += _h[_i] * _y[_i];
					_normb += _bx[_i] * _bx[_i] + _by[_i] * _by[_i];
				}

				for (_i = 0; _i < np; _i++)
					_gamma[_i] = 1.0 / 4.0 * (_h[_i] - _htx * _bx[_i] - _hty * _by[_i]);

				// Use perturbation hourglass stabilization
				if (swmethod == 1)
				{
					// Stabilization matrix
					// double _coeff = 4.0 * _Jdet * _normb * stabilizationConstant;
					// double _coeff = (1.0 / 12.0) * _A * _A * _normb * stabilizationConstant;
					// double _coeff = _normb * stabilizationConstant;
					double _coeff = stabilizationConstant;
					for (_i = 0; _i < np; _i++)
						for (_j = 0; _j < np; _j++)
						{
							_Kestab[2 * _i + 0][2 * _j + 0] += _coeff * _A * _thickness * _gamma[_i] * _gamma[_j];
							_Kestab[2 * _i + 1][2 * _j + 1] += _coeff * _A * _thickness * _gamma[_i] * _gamma[_j];
							_Kestab[2 * _i + 0][2 * _j + 1] += _coeff * _A * _thickness * _gamma[_i] * _gamma[_j];
							_Kestab[2 * _i + 1][2 * _j + 0] += _coeff * _A * _thickness * _gamma[_i] * _gamma[_j];
						}
					for (_i = 0; _i < 2 * np; _i++)
						for (_j = 0; _j < 2 * np; _j++)
							_festab[_i] += _Kestab[_i][_j] * _uel[_j];
				}

				// Use assumed strain
				else if (swmethod == 2)
				{

					// Get integrals of hourglassing shape function h = \xi*\eta (four-point Gauss integration rule for quadrangles)
					const int _tngauss = 4;
					double _talpha[_tngauss];
					double **_tIP = new double *[_tngauss];
					for (int _i = 0; _i < _tngauss; _i++)
						_tIP[_i] = new double[3];
					double *_ttcoord = new double[3];
					double *_tDxiN = new double[np];
					double *_tDetaN = new double[np];
					double *_tDxN = new double[np];
					double *_tDyN = new double[np];
					getQuadGaussInfo(_tngauss, _talpha, _tIP);
					double _Hxx = 0.0; // use numerical integration (for isoparametric elements, otherwise could be done analytically)
					double _Hyy = 0.0;
					double _Hxy = 0.0;
					for (int _tig = 0; _tig < _tngauss; _tig++)
					{

						// Get natural coordinates
						for (int _i = 0; _i < 3; _i++)
							_ttcoord[_i] = _tIP[_tig][_i]; // get natural coordinates of Gauss integration rule

						// Derivatives of the basis functions with respect to natural coordinates
						getQuadShapeFunGradNat(np, _ttcoord, _tDxiN, _tDetaN);
						double _Dxih = _ttcoord[1];	 // h = \xi*\eta, Dxih = \eta;
						double _Detah = _ttcoord[0]; // h = \xi*\eta, Detah = \eta;

						// Jacobian
						double _tJ11 = 0.0;
						double _tJ21 = 0.0;
						double _tJ12 = 0.0;
						double _tJ22 = 0.0;
						for (int _i = 0; _i < np; _i++)
						{
							_tJ11 += _tDxiN[_i] * _x[_i];
							_tJ12 += _tDxiN[_i] * _y[_i];
							_tJ21 += _tDetaN[_i] * _x[_i];
							_tJ22 += _tDetaN[_i] * _y[_i];
						}
						double _tJdet = _tJ11 * _tJ22 - _tJ12 * _tJ21;

						// Derivatives of basis functions (DxN - derivative w.r.t. x, DyN - derivatives w.r.t. y)
						for (int _i = 0; _i < np; _i++)
						{
							_tDxN[_i] = (_tJ22 * _tDxiN[_i] - _tJ12 * _tDetaN[_i]) / _tJdet;
							_tDyN[_i] = (-_tJ21 * _tDxiN[_i] + _tJ11 * _tDetaN[_i]) / _tJdet;
						}
						double _Dxh = (_tJ22 * _Dxih - _tJ12 * _Detah) / _tJdet;
						double _Dyh = (-_tJ21 * _Dxih + _tJ11 * _Detah) / _tJdet;

						// Integrate
						_Hxx += _talpha[_tig] * _tJdet * _Dxh * _Dxh;
						_Hyy += _talpha[_tig] * _tJdet * _Dyh * _Dyh;
						_Hxy += _talpha[_tig] * _tJdet * _Dxh * _Dyh;
					}

					// Release allocated memory
					for (_i = 0; _i < _tngauss; _i++)
						delete[] _tIP[_i];
					delete[] _tIP;
					delete[] _ttcoord;
					delete[] _tDxiN;
					delete[] _tDetaN;
					delete[] _tDxN;
					delete[] _tDyN;

					// Get stabilization constants C1 and C2
					double _stabConst1 = 0.0;
					double _stabConst2 = 0.0;
					switch (swstab)
					{
					case 1:
						// Optimal bending element (OB)
						_stabConst1 = _CD[0][0]; // \lambda+2*\mu
						_stabConst2 = _CD[0][1]; // \lambda
						break;
					case 2:
					{
						// Quintessential bending/incompressible element (QBI)
						double _lambda = _CD[0][1];
						double _mu = _CD[2][2];
						double _E = _mu * (3.0 * _lambda + 2.0 * _mu) / (_lambda + _mu);
						double _nu = _lambda / (2.0 * (_lambda + _mu));
						double _Ebar = _E / (1.0 - _nu * _nu);
						double _nubar = _nu / (1.0 - _nu);
						_stabConst1 = _Ebar * _Hxx * _Hyy / (_Hxx * _Hyy - _nubar * _nubar * _Hxy * _Hxy);
						_stabConst2 = _nubar * _stabConst1;
						break;
					}
					case 3:
						// Frame-invariant bending element (FIB)
						_stabConst1 = _CD[0][0]; // \lambda+2*\mu
						_stabConst2 = _stabConst1;
						break;
					case 4:
						// Optimal incompressible element (OI)
						_stabConst1 = 4.0 * _CD[2][2] * _Hxx * _Hyy / (_Hxx * _Hyy - _Hxy * _Hxy);
						_stabConst2 = _stabConst1;
						break;
					default:
#pragma omp critical(ERROR)
					{
						mexErrMsgTxt("Wrong stabilization chosen.");
					}
					break;
					}

					// Contruct stabilization matrix Kestab

					for (_i = 0; _i < np; _i++)
						for (_j = 0; _j < np; _j++)
						{
							_Kestab[2 * _i + 0][2 * _j + 0] += _stabConst1 * _Hxx * _thickness * _gamma[_i] * _gamma[_j];
							_Kestab[2 * _i + 1][2 * _j + 1] += _stabConst1 * _Hyy * _thickness * _gamma[_i] * _gamma[_j];
							_Kestab[2 * _i + 0][2 * _j + 1] += _stabConst2 * _Hxy * _thickness * _gamma[_i] * _gamma[_j];
							_Kestab[2 * _i + 1][2 * _j + 0] += _stabConst2 * _Hxy * _thickness * _gamma[_i] * _gamma[_j];
						}
					for (_i = 0; _i < 2 * np; _i++)
						for (_j = 0; _j < 2 * np; _j++)
							_festab[_i] += _Kestab[_i][_j] * _uel[_j];
				}
				else
				{
#pragma omp critical(WARNING)
					{
						mexWarnMsgTxt("Wrong stabilization type chosen for Q4G1 element. Skipping.");
					}
				}

				// Add stabilization force _festab and matrix _Kestab to _fe and _Ke, and compute hourglassing energy
				for (_i = 0; _i < 2 * np; _i++)
				{
					_fe[_i] += _festab[_i];
					_Whourglass += 0.5 * _festab[_i] * _uel[_i];
					for (_j = 0; _j < 2 * np; _j++)
					{
						_Ke[_i][_j] += _Kestab[_i][_j];
					}
				}
			}

			// Allocate energy
#pragma omp atomic
			prW[0] += _W;

			// Allocate hourglassing energy
#pragma omp atomic
			prWhourglass[0] += _Whourglass;

			// Allocate gradient
			for (_i = 0; _i < 2 * np; _i++)
			{
#pragma omp atomic
				prG[_cod[_i] - 1] += _fe[_i];
			}

			// Allocate Hessian
			for (_i = 0; _i < 2 * np; _i++)
			{
				for (_j = 0; _j < 2 * np; _j++)
				{
					_tripletList.push_back(T(_cod[_i] - 1, _cod[_j] - 1, _Ke[_i][_j]));
				}
			}
		}

		// Assemble _K from triplets
		_K.setFromTriplets(_tripletList.begin(), _tripletList.end());

		// Collect _K from all threads
#pragma omp critical(ALLOCATE)
		{
			K += _K;
		}

		// Delete dynamically allocated memory
		delete[] _idel;
		delete[] _cod;
		delete[] _uel;
		delete[] _x;
		delete[] _y;
		delete[] _fe;
		for (_i = 0; _i < 2 * np; _i++)
			delete[] _Ke[_i];
		delete[] _Ke;

		// Stabilization
		for (_i = 0; _i < 2 * np; _i++)
			delete[] _Kestab[_i];
		delete[] _Kestab;
		delete[] _festab;
		delete[] _gamma;
		delete[] _h;
		delete[] _bx;
		delete[] _by;
		delete[] _tbx;
		delete[] _tby;
		delete[] _tx;
		delete[] _ty;

		for (_i = 0; _i < 3; _i++)
			delete[] _Be[_i];
		delete[] _Be;
		for (_i = 0; _i < 4; _i++)
			delete[] _Bf[_i];
		delete[] _Bf;
		for (_i = 0; _i < abs(ngauss); _i++)
			delete[] _IP[_i];
		delete[] _IP;
		delete[] _alpha;
	}

	// Catch errors
	if (errCountMat != 0)
		mexErrMsgTxt("Unrecognized material chosen.");
	if (errCountGauss != 0)
		mexErrMsgTxt("Unrecognized Gauss integration rule chosen.");
	if (errCountEl != 0)
		mexErrMsgTxt("Unrecognized element type chosen.");

	// Send out data back to matlab
	plhs[3] = mxCreateSparse(K.rows(), K.cols(), K.nonZeros(), mxREAL);
	int *ir = K.innerIndexPtr();
	int *jc = K.outerIndexPtr();
	double *pr = K.valuePtr();
	mwIndex *ir2 = mxGetIr(plhs[3]);
	mwIndex *jc2 = mxGetJc(plhs[3]);
	double *pr2 = mxGetPr(plhs[3]);
	for (auto i = 0; i < K.nonZeros(); ++i)
	{
		pr2[i] = pr[i];
		ir2[i] = ir[i];
	}
	for (auto i = 0; i < K.cols() + 1; ++i)
	{
		jc2[i] = jc[i];
	}
}
