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
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	// Test for six input arguments
	if (nrhs != 6)
		mexErrMsgTxt("Six input arguments required.");

	// Get the input data
	double *prp = mxGetPr(prhs[0]);
	double *prt = mxGetPr(prhs[1]);
	double *prmat = mxGetPr(prhs[2]);
	double *prNgauss = mxGetPr(prhs[3]);
	double *pru = mxGetPr(prhs[4]);
	double *prmaxNumThreads = mxGetPr(prhs[5]);
	int nnodes = (int)mxGetN(prhs[0]);
	int nelem = (int)mxGetN(prhs[1]);
	int mt = (int)mxGetM(prhs[1]);
	int np = mt - 1; // number of nodes in an element
	int mmat = (int)mxGetM(prhs[2]);
	int ngauss = (int)prNgauss[0];
	int ndof = 3 * nnodes;
	int maxNumThreads = (int)prmaxNumThreads[0];

	// Test the number of dofs
	if (ndof != (int)mxGetM(prhs[4]))
		mexErrMsgTxt("Wrong number of dofs.");

	// Allocate outputs
	nlhs = 3;
	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(ndof, 1, mxREAL);
	double *prW = mxGetPr(plhs[0]);
	double *prG = mxGetPr(plhs[1]);

	// Compute the gradient and Hessian, populate prG, prI, prJ, prS
	Eigen::SparseMatrix<double> K(ndof, ndof);
	int tr;				   // triangle ID
	int errCountGauss = 0; // integration rule: error control inside parallel region
	int errCountMat = 0;   // constitutive law: error control inside parallel region
	int errCountEl = 0;	// constitutive law: error control inside parallel region
#pragma omp parallel num_threads(maxNumThreads)
	{
		int _i, _j, _k, _l, _ig;
		double _w, _W, _Jdet;
		double _F[9], _C[6], _S[6], _CD[6][6], _S4[9][9], _tcoord[4], _matconst[8]; // use total Lagrangian formulation
		Eigen::SparseMatrix<double> _K(ndof, ndof);
		auto _tripletList = std::vector<T>{};
		_tripletList.reserve(4 * np * np * nelem);
		int *_idel = new int[np];			 // ids of nodes of given element
		int *_cod = new int[3 * np];		 // code numbers of given element
		double *_uel = new double[3 * np];   // displacements associated with given element
		double *_x = new double[np];		 // x-coordinates of all nodes of given element
		double *_y = new double[np];		 // y-coordinates of all nodes of given element
		double *_z = new double[np];		 // z-coordinates of all nodes of given element
		double *_fe = new double[3 * np];	// internal force vector
		double **_Ke = new double *[3 * np]; // element's stiffness matrix
		for (_i = 0; _i < 3 * np; _i++)
			_Ke[_i] = new double[3 * np];
		double **_Be = new double *[6]; // element's matrix of derivatives of basis										// functions (Green-Lagrange strain)
		for (_i = 0; _i < 6; _i++)
			_Be[_i] = new double[3 * np];
		double **_Bf = new double *[9]; // element's matrix of derivatives of basis										// functions (deformation gradient)
		for (_i = 0; _i < 9; _i++)
			_Bf[_i] = new double[3 * np];
		double **_IP = new double *[abs(ngauss)];
		for (_i = 0; _i < abs(ngauss); _i++)
			_IP[_i] = new double[4];
		double *_alpha = new double[abs(ngauss)];
		for (_i = 0; _i < 9; _i++)
			for (_j = 0; _j < 9; _j++)
				_S4[_i][_j] = 0.0;
		int (*_getBe)(const int np, const double tcoord[4], const double *x, const double *y, const double *z, const double *uel, double &Jdet, double **Be, double **Bf, double F[9]) = NULL; // pointer to Be functions
		int (*_getWSC)(const double matconst[8], const double C[6], double &w, double S[6], double D[6][6]) = NULL;																			   // pointer to constitutive law functions
		if (np == 4 || np == 10)
		{

			// Choose Gauss integration rule
			if (ngauss == 1 || ngauss == 4 || ngauss == 5 || ngauss == 11 ||
				ngauss == 15)
				getTetraGaussInfo(ngauss, _alpha, _IP); // tetrahendra: ngauss = 1, 4, 5, 11, 15
			else
			{
#pragma omp atomic
				errCountGauss++; // throw error if unrecognized integration rule used
			}

			// Choose getBe function
			_getBe = &getTetraBeTLE;
		}
		else if (np == 8 || np == 20 || np == 27)
		{
			// Choose Gauss integration rule
			if (ngauss == 1 || ngauss == 8 || ngauss == 27 || ngauss == 64)
				getHexaGaussInfo(ngauss, _alpha, _IP); // hexahedra: ngauss = 1, 8, 27, 64
			else
			{
#pragma omp atomic
				errCountGauss++; // throw error if unrecognized integration rule used
			}

			// Choose getBe function
			_getBe = &getHexaBeTLE;
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
				_cod[3 * _i] = 3 * _idel[_i] - 2;
				_cod[3 * _i + 1] = 3 * _idel[_i] - 1;
				_cod[3 * _i + 2] = 3 * _idel[_i];
			}

			// Get element's displacements
			for (_i = 0; _i < 3 * np; _i++)
				_uel[_i] = pru[_cod[_i] - 1];

			// Get element's coordinates
			for (_i = 0; _i < np; _i++)
			{
				_x[_i] = prp[0 + (_idel[_i] - 1) * 3];
				_y[_i] = prp[1 + (_idel[_i] - 1) * 3];
				_z[_i] = prp[2 + (_idel[_i] - 1) * 3];
			}

			// Get material constants
			for (_i = 0; _i < 8; _i++)
				_matconst[_i] = prmat[int(prt[np + mt * tr] - 1) + _i * mmat];

			// Choose constitutive law
			switch ((int)_matconst[0])
			{
			case 1: // W(F) = m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+1/2*kappa(ln(J))^2
					// (OOFEM)
				_getWSC = &material_oofemE3d;
				break;
			case 2: // W(F) = m1(I1-3)+m2(I1-3)^2-2m1*ln(J)+1/2*kappa(J-1)^2
					// (Bertoldi, Boyce)
				_getWSC = &material_bbE3d;
				break;
			case 3: // W(F) =
					// m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I1/J^{2/3}-3)^3+9/2*kappa(J^{1/3}-1)^2
					// (Jamus, Green, Simpson)
				_getWSC = &material_jgsE3d;
				break;
			case 4: // W(F) =
					// m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I2/J^{2/3}-3)^2+kappa*(J-1)^2
					// (five-term Mooney-Rivlin)
				_getWSC = &material_5mrE3d;
				break;
			case 5: // W(F) = 0.5*(0.5*(C-I)*(m1*IxI+2*kappa*I)*0.5*(C-I)) (linear
					// elastic material)
				_getWSC = &material_leE3d;
				break;
			case 6: // Ogden material
				_getWSC = &material_ogdenE3d;
				break;
			case 7: // Ogden nematic material
				_getWSC = &material_ogdennematicE3d;
				break;
			default: // if unrecognized material used, use OOFEM in order not to
					 // crash, but throw an error
				_getWSC = &material_oofemE3d;
#pragma omp atomic
				errCountMat++;
				break;
			}

			// Gauss integration rule
			_W = 0.0;
			for (_i = 0; _i < 3 * np; _i++)
			{
				_fe[_i] = 0.0;
				for (_j = 0; _j < 3 * np; _j++)
					_Ke[_i][_j] = 0.0;
			}
			for (_ig = 0; _ig < abs(ngauss); _ig++)
			{

				// Get natural coordinates of Gauss integration rule
				for (_i = 0; _i < 4; _i++)
					_tcoord[_i] = _IP[_ig][_i];

				// Get Be matrix
				_getBe(np, _tcoord, _x, _y, _z, _uel, // input
					   _Jdet, _Be, _Bf, _F);		  // output
				getC3d(_F, _C);

				// Get energy density, stress, and material stiffness evaluated at
				// current Gauss point
				_getWSC(_matconst, _C, // inputs
						_w, _S, _CD);  // outputs

				// Integrate element's energy W = W + alpha(ig)*Wd*_Jdet*h
				_W += _alpha[_ig] * _Jdet * _w;

				// Integrate element's gradient f = f + alpha(ig)*(B')*S(:)*Jdet*h
				for (_i = 0; _i < 6; _i++)
					for (_k = 0; _k < 3 * np; _k++)
						_fe[_k] += _alpha[_ig] * _Jdet * _Be[_i][_k] * _S[_i];

				// Integrate element's Hessian K = K + alpha(ig)*( (B')*D*B + (G')*S*G
				// )*Jdet*h
				for (_i = 0; _i < 3 * np; _i++)
					for (_j = 0; _j < 3 * np; _j++)
						for (_k = 0; _k < 6; _k++)
							for (_l = 0; _l < 6; _l++)
								_Ke[_i][_j] +=
									_alpha[_ig] * _Jdet * _Be[_k][_i] * _CD[_k][_l] *
									_Be[_l][_j]; // small and large displacement matrix

				// Geometric stiffness
				_S4[0][0] = _S[0];
				_S4[3][3] = _S[1];
				_S4[6][6] = _S[2];
				_S4[0][3] = _S[3];
				_S4[3][6] = _S[4];
				_S4[0][6] = _S[5];
				_S4[3][0] = _S[3];
				_S4[6][3] = _S[4];
				_S4[6][0] = _S[5];

				_S4[1][1] = _S[0];
				_S4[4][4] = _S[1];
				_S4[7][7] = _S[2];
				_S4[1][4] = _S[3];
				_S4[4][7] = _S[4];
				_S4[1][7] = _S[5];
				_S4[4][1] = _S[3];
				_S4[7][4] = _S[4];
				_S4[7][1] = _S[5];

				_S4[2][2] = _S[0];
				_S4[5][5] = _S[1];
				_S4[8][8] = _S[2];
				_S4[2][5] = _S[3];
				_S4[5][8] = _S[4];
				_S4[2][8] = _S[5];
				_S4[5][2] = _S[3];
				_S4[8][5] = _S[4];
				_S4[8][2] = _S[5];

				for (_i = 0; _i < 3 * np; _i++)
					for (_j = 0; _j < 3 * np; _j++)
						for (_k = 0; _k < 9; _k++)
							for (_l = 0; _l < 9; _l++)
								_Ke[_i][_j] += _alpha[_ig] * _Jdet * _Bf[_k][_i] * _S4[_k][_l] *
											   _Bf[_l][_j]; // geometric stiffness matrix
			}

			// Allocate energy
#pragma omp atomic
			prW[0] += _W;

			// Allocate gradient
			for (_i = 0; _i < 3 * np; _i++)
			{
#pragma omp atomic
				prG[_cod[_i] - 1] += _fe[_i];
			}

			// Allocate Hessian
			for (_i = 0; _i < 3 * np; _i++)
			{
				for (_j = 0; _j < 3 * np; _j++)
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
		for (_i = 0; _i < 3 * np; _i++)
			delete[] _Ke[_i];
		delete[] _Ke;
		for (_i = 0; _i < 6; _i++)
			delete[] _Be[_i];
		delete[] _Be;
		for (_i = 0; _i < 9; _i++)
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
	plhs[2] = mxCreateSparse(K.rows(), K.cols(), K.nonZeros(), mxREAL);
	int *ir = K.innerIndexPtr();
	int *jc = K.outerIndexPtr();
	double *pr = K.valuePtr();
	mwIndex *ir2 = mxGetIr(plhs[2]);
	mwIndex *jc2 = mxGetJc(plhs[2]);
	double *pr2 = mxGetPr(plhs[2]);
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
