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
	int (*getWSC)(const double matconst[8], const double C[6], double &w, double S[6],
				  double D[6][6]) = NULL; // pointer to constitutive law functions
	switch ((int)matconst[0])
	{
	case 1: // OOFEM
		getWSC = &material_oofemE3d;
		idmat = 1;
		break;
	case 2: // Bertoldi, Boyce
		getWSC = &material_bbE3d;
		idmat = 2;
		break;
	case 3: // Jamus, Green, Simpson
		getWSC = &material_jgsE3d;
		idmat = 3;
		break;
	case 4: // five-term Mooney-Rivlin
		getWSC = &material_5mrE3d;
		idmat = 4;
		break;
	case 5: // elastic material
		getWSC = &material_leE3d;
		idmat = 5;
		break;
	case 6: // Ogedn material
		getWSC = &material_ogdenE3d;
		idmat = 6;
		break;
	case 7: // Ogedn nematic material
		getWSC = &material_ogdennematicE3d;
		idmat = 7;
		break;
	default: // if unrecognized material used, use OOFEM in order not to
			 // crash, but throw an error
		getWSC = &material_oofemE3d;
		idmat = 1;
		break;
	}
	printf("\n=====================================================\n");
	printf("Testing constitutive law no. %d\n\n", idmat);

	// Test constitutive law by finite differencing the elastic energy density
	double w, S[6], CD[6][6];
	getWSC(matconst, C, w, S, CD);

	// Get stress numerically by finite differences
	int i, j, k;
	double tS[6], C1[6], C2[6], C3[6], C4[6], ttS[6];
	double tCD[6][6], ttCD[6][6], w1, w2, w3, w4;

	// Piola-Kirchhoff stress
	for (i = 0; i < 6; i++)
	{

		for (j = 0; j < 6; j++)
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

	// Perturbing C perturs 2*E on off-diagonal terms; correct for this
	for (i = 3; i < 6; i++)
		tS[i] /= 2.0;

	// Error in stress
	printf("tS = [ ");
	for (i = 0; i < 6; i++)
		printf("%f ", tS[i]);
	printf("]\n S = [ ");
	for (i = 0; i < 6; i++)
		printf("%f ", S[i]);
	printf("]\n");
	double errS = 0.0;
	for (i = 0; i < 6; i++)
		errS += fabs(S[i] - tS[i]);
	printf("Error in second PK stress errS = %e\n\n", errS);

	// Constitutive tangent
	for (i = 0; i < 6; i++)
	{
		for (j = 0; j < 6; j++)
		{

			for (k = 0; k < 6; k++)
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

	// Perturbing C perturs 2*E on off-diagonal terms; correct for
	// this
	for (i = 3; i < 6; i++)
		for (j = 0; j < 6; j++)
		{
			tCD[i][j] /= 2.0;
			tCD[j][i] /= 2.0;
		}

	// Error in tangent
	printf("tCD = [ ");
	for (i = 0; i < 6; i++)
		for (j = 0; j < 6; j++)
			printf("%f ", tCD[i][j]);
	printf("]\n CD = [ ");
	for (i = 0; i < 6; i++)
		for (j = 0; j < 6; j++)
			printf("%f ", CD[i][j]);
	printf("]\n");
	double errD = 0.0;
	for (i = 0; i < 6; i++)
		for (j = 0; j < 6; j++)
			errD += fabs(CD[i][j] - tCD[i][j]);
	printf("Error in constitutive tangent errC = %e\n", errD);
}
