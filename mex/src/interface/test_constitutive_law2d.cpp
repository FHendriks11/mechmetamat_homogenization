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

	// Test for three input arguments
	if (nrhs != 3)
		mexErrMsgTxt("Three input arguments required.");

	// Get the input data
	double *C = mxGetPr(prhs[0]);
	double *matconst = mxGetPr(prhs[1]);
	double perturbation = mxGetScalar(prhs[2]);

	// Get material constants
	int idmat = 0;

	// Choose constitutive law
	int (*getWSC)(const double matconst[8], const double C[3], double &w, double S[3], double D[3][3]) = NULL; // pointer to constitutive law functions
	switch ((int)matconst[0])
	{
	case 1: // OOFEM
		getWSC = &material_oofemE2d;
		idmat = 1;
		break;
	case 2: // Bertoldi, Boyce
		getWSC = &material_bbE2d;
		idmat = 2;
		break;
	case 3: // Jamus, Green, Simpson
		getWSC = &material_jgsE2d;
		idmat = 3;
		break;
	case 4: // five-term Mooney-Rivlin
		getWSC = &material_5mrE2d;
		idmat = 4;
		break;
	case 5: // elastic material
		getWSC = &material_leE2d;
		idmat = 5;
		break;
	case 6: // Ogden material
		getWSC = &material_ogdenE2d;
		idmat = 6;
		break;
	case 7: // Ogden nematic material
		getWSC = &material_ogdennematicE2d;
		idmat = 7;
		break;
	default: // if unrecognized material used, use OOFEM in order not to
			 // crash, but throw an error
		getWSC = &material_oofemE2d;
		idmat = 1;
		break;
	}
	printf("\n=====================================================\n");
	printf("Testing constitutive law no. %d\n\n", idmat);

	// Test constitutive law by finite differencing the elastic energy density
	double w, S[3], CD[3][3];
	getWSC(matconst, C, w, S, CD);

	// Get stress numerically by finite differences
	int i, j, k;
	double tS[3], C1[3], C2[3], C3[3], C4[3], ttS[3];
	double tCD[3][3], ttCD[3][3], w1, w2, w3, w4;

	// Piola-Kirchhoff stress
	for (i = 0; i < 3; i++)
	{

		for (j = 0; j < 3; j++)
		{
			C1[j] = C[j];
			C2[j] = C[j];
		}

		C1[i] -= perturbation;
		C2[i] += perturbation;

		getWSC(matconst, C1, w1, ttS, ttCD);
		getWSC(matconst, C2, w2, ttS, ttCD);

		tS[i] = 2.0 * (-w1 + w2) / (2.0 * perturbation);
	}

	// Perturbing C perturbs 2*E on off-diagonal terms; correct for this
	tS[2] /= 2.0;

	// Error in stress
	printf("tS = [ %f %f %f ]\n S = [ %f %f %f ]\n", tS[0], tS[1], tS[2], S[0], S[1], S[2]);
	double errS = 0.0;
	for (i = 0; i < 3; i++)
		errS += fabs(S[i] - tS[i]);
	printf("Error in second PK stress errS = %e\n\n", errS);

	// Constitutive tangent
	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{

			for (k = 0; k < 3; k++)
			{
				C1[k] = C[k];
				C2[k] = C[k];
				C3[k] = C[k];
				C4[k] = C[k];
			}

			C1[i] += perturbation;
			C1[j] += perturbation;
			C2[i] += perturbation;
			C2[j] -= perturbation;
			C3[i] -= perturbation;
			C3[j] += perturbation;
			C4[i] -= perturbation;
			C4[j] -= perturbation;

			getWSC(matconst, C1, w1, ttS, ttCD);
			getWSC(matconst, C2, w2, ttS, ttCD);
			getWSC(matconst, C3, w3, ttS, ttCD);
			getWSC(matconst, C4, w4, ttS, ttCD);

			tCD[i][j] = 4.0 * (w1 - w2 - w3 + w4) / (4.0 * perturbation * perturbation);
		}
	}

	// Perturbing C perturs 2*E on off-diagonal terms; correct for this
	for (i = 2; i < 3; i++)
		for (j = 0; j < 3; j++)
		{
			tCD[i][j] /= 2.0;
			tCD[j][i] /= 2.0;
		}

	// Error in tangent
	printf("tCD = [ ");
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			printf("%f ", tCD[i][j]);
	printf("]\n CD = [ ");
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			printf("%f ", CD[i][j]);
	printf("]\n");
	double errD = 0.0;
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			errD += fabs(CD[i][j] - tCD[i][j]);
	printf("Error in constitutive tangent errC = %e\n", errD);
}
