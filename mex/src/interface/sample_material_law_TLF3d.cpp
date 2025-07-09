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

/************************ Main program ************************/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	// Test for two input arguments
	if (nrhs != 2)
		mexErrMsgTxt("Two input arguments required.");

	// Test correct size of material
	if ((mxGetM(prhs[0]) == 0) || (mxGetN(prhs[0]) != 8))
		mexErrMsgTxt("Wrong dimensionality of material.");

	// Test correct size of F
	if (mxGetM(prhs[1]) * mxGetN(prhs[1]) != 9)
		mexErrMsgTxt("Wrong dimensionality of F.");

	// Get the input data
	double *prmat = mxGetPr(prhs[0]);
	double *prF = mxGetPr(prhs[1]);

	// Allocate outputs
	nlhs = 3;
	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(9, 1, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(9, 9, mxREAL);
	double *prW = mxGetPr(plhs[0]);
	double *prP = mxGetPr(plhs[1]);
	double *prD = mxGetPr(plhs[2]);

	// Compute the gradient and Hessian, populate prG, prI, prJ, prS
	int errCountMat = 0; // constitutive law: error control inside parallel region

	int i, j;
	double W, F[9], P[9], D[9][9], matconst[8];																	// use total Lagrangian formulation casted in F and P
	int (*getWPD)(const double matconst[8], const double F[9], double &Wd, double P[9], double D[9][9]) = NULL; // pointer to constitutive law functions

	// Get deformation gradient
	for (i = 0; i < 9; i++)
		F[i] = prF[i];

	// Get material constants
	for (i = 0; i < 8; i++)
		matconst[i] = prmat[i];

	// Choose constitutive law
	switch ((int)matconst[0])
	{
	case 1: // W(F) = m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+1/2*kappa(ln(J))^2
			// (OOFEM)
		getWPD = &material_oofemF3d;
		break;
	case 2: // W(F) = m1(I1-3)+m2(I1-3)^2-2m1*ln(J)+1/2*kappa(J-1)^2
			// (Bertoldi, Boyce)
		getWPD = &material_bbF3d;
		break;
	case 3: // W(F) =
			// m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I1/J^{2/3}-3)^3+9/2*kappa(J^{1/3}-1)^2
			// (Jamus, Green, Simpson)
		getWPD = &material_jgsF3d;
		break;
	case 4: // W(F) =
			// m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I2/J^{2/3}-3)^2+kappa*(J-1)^2
			// (five-term Mooney-Rivlin)
		getWPD = &material_5mrF3d;
		break;
	case 5: // W(F) = 0.5*(0.5*(C-I)*(m1*IxI+2*kappa*I)*0.5*(C-I)) (linear
			// elastic material)
		getWPD = &material_leF3d;
		break;
	default: // if unrecognized material used, use OOFEM in order not to crash, but throw an error
		getWPD = &material_oofemF3d;
#pragma omp atomic
		errCountMat++;
		break;
	}

	// Get energy density, stress, and material stiffness
	getWPD(matconst, F, // inputs
		   W, P, D);	// outputs

	// Allocate energy
	prW[0] += W;

	// Allocate stress
	for (i = 0; i < 9; i++)
		prP[i] = P[i];

	// Allocate constitutive tangent
	for (i = 0; i < 9; i++)
		for (j = 0; j < 9; j++)
			prD[9 * i + j] = D[i][j];
}
