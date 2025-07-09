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

	// Test correct size of C
	if (mxGetM(prhs[1]) * mxGetN(prhs[1]) != 6)
		mexErrMsgTxt("Wrong dimensionality of C.");

	// Get the input data
	double *prmat = mxGetPr(prhs[0]);
	double *prC = mxGetPr(prhs[1]);

	// Allocate outputs
	nlhs = 3;
	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(6, 1, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(6, 6, mxREAL);
	double *prW = mxGetPr(plhs[0]);
	double *prS = mxGetPr(plhs[1]);
	double *prCD = mxGetPr(plhs[2]);

	// Compute the gradient and Hessian, populate prG, prI, prJ, prS
	int errCountMat = 0; // constitutive law: error control inside parallel region

	int i, j;
	double W, C[6], S[6], CD[6][6], matconst[8];															   // use total Lagrangian formulation casted in F and P
	int (*getWSC)(const double matconst[8], const double C[6], double &w, double S[6], double D[6][6]) = NULL; // pointer to constitutive law functions

	// Get deformation gradient and the right Cauchy-Green strain tensor
	for (i = 0; i < 6; i++)
		C[i] = prC[i];

	// Get material constants
	for (i = 0; i < 8; i++)
		matconst[i] = prmat[i];

	// Choose constitutive law
	switch ((int)matconst[0])
	{
	case 1: // OOFEM
		getWSC = &material_oofemE3d;
		break;
	case 2: // Bertoldi, Boyce
		getWSC = &material_bbE3d;
		break;
	case 3: // Jamus, Green, Simpson
		getWSC = &material_jgsE3d;
		break;
	case 4: // five-term Mooney-Rivlin
		getWSC = &material_5mrE3d;
		break;
	case 5: // elastic material
		getWSC = &material_leE3d;
		break;
	case 6: // Ogden material
		getWSC = &material_ogdenE3d;
		break;
	case 7: // Ogden nematic material
		getWSC = &material_ogdennematicE3d;
		break;
	default: // if unrecognized material used, use OOFEM in order not to
			 // crash, but throw an error
		getWSC = &material_oofemE3d;
#pragma omp atomic
		errCountMat++;
		break;
	}

	// Get energy density, stress, and material stiffness
	getWSC(matconst, C, // inputs
		   W, S, CD);   // outputs

	// Allocate energy
	prW[0] += W;

	// Allocate stress
	for (i = 0; i < 6; i++)
		prS[i] = S[i];

	// Allocate constitutive tangent
	for (i = 0; i < 6; i++)
		for (j = 0; j < 6; j++)
			prCD[6 * i + j] = CD[i][j];
}
