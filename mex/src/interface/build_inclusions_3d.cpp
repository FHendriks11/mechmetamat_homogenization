// build_inclusions: builds a database of inclusions

#include "matrix.h"
#include "mex.h"
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/******************* Definition of constants ******************/
const double PI = 3.14159265358979323846; // Ludolph's number
const int MAXITER = 1000;				  // max number of trials during placing an aggregate

/******************** Structure declarations ******************/
struct GRAIN
{
	double p[3]; // centre of circular grain
	double r;	// radius of circular grain
};

/************************ Main program ************************/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	/*<<<<<<<<<<<<<<<<<<<<<<<< Get inputs >>>>>>>>>>>>>>>>>>>>>>>>*/
	// Test for one input argument
	if (nrhs != 1)
		mexErrMsgTxt("One input argument required.");

	// Copy input variables
	double *prIn = mxGetPr(prhs[0]);
	if ((int)std::max(mxGetM(prhs[0]), mxGetN(prhs[0])) != 10)
		mexErrMsgTxt("Required length of the input vector is 14.");
	double SizeX = prIn[0];
	double SizeY = prIn[1];
	double SizeZ = prIn[2];
	double Dmin = prIn[3];
	double Dmax = prIn[4];
	double minDist = prIn[5];
	double Content = prIn[6];
	double Step = prIn[7];
	// double TOL_g = prIn[8]; // a positive distance < TOL_g is treated as zero
	int SW = (int)prIn[9];

	/*<<<<<<<<<<<<<<<<<<<<<<<< Build random inclusions inside/outside
   * >>>>>>>>>>>>>>>>>>>>>>>>*/
	int totalNumGrains = 0, i, j, k;
	double diam;
	GRAIN *grainsDatabase = NULL;
	if (SW == 0)
	{

		// Allocate database of grains
		int maxGrains = 0, numGrains, accepted;
		i = 0;
		while (Dmax - (i + 1) * Step >= Dmin)
		{
			diam = (Dmax - i * Step + Dmax - (i + 1) * Step) / 2;
			maxGrains += (int)floor(2 * 2 * 2 * SizeX * SizeY * SizeZ /
									(4.0 / 3.0 * PI * pow(diam / 2.0, 3)));
			i++;
		}
		grainsDatabase = new GRAIN[maxGrains]; // grains database

		// Populate grainsDatabase
		srand((unsigned)time(NULL)); // initialize random seed
		i = 0;
		while (Dmax - (i + 1) * Step >= Dmin)
		{

			// Get number of aggregates for given fraction (radius) from the Fuller
			// distribution
			diam = (Dmax - i * Step + Dmax - (i + 1) * Step) / 2;
			numGrains =
				(int)round(1 / sqrt(Dmax) *
						   (sqrt(Dmax - i * Step) - sqrt(Dmax - (i + 1) * Step)) *
						   Content * 2 * 2 * 2 * SizeX * SizeY * SizeZ /
						   (4.0 / 3.0 * PI * pow(diam / 2.0, 3)));

			// Distribute numGrains over the domain
			double trialp[3];
			for (j = 0; j < numGrains; j++)
			{
				accepted =
					0; // flag: 1 if randomly placed grain is accepted, 0 otherwise
				int iter = 0;
				while (!accepted)
				{
					iter++;

					// Generate trial position of the aggregate
					trialp[0] = 2 * SizeX * ((double)rand() / (double)RAND_MAX) - SizeX;
					trialp[1] = 2 * SizeY * ((double)rand() / (double)RAND_MAX) - SizeY;
					trialp[2] = 2 * SizeZ * ((double)rand() / (double)RAND_MAX) - SizeZ;

					// Test for distance w.r.t. other aggregates
					int intersect = 0;
					k = 0;
					while (k < totalNumGrains && !intersect)
					{
						if (sqrt(pow(trialp[0] - grainsDatabase[k].p[0], 2) +
								 pow(trialp[1] - grainsDatabase[k].p[1], 2) +
								 pow(trialp[2] - grainsDatabase[k].p[2], 2)) <
							std::max(minDist, double(1)) *
								(diam + 2 * fabs(grainsDatabase[k].r)) / 2)
							intersect = 1;
						k++;
					}

					// Test for distance w.r.t. RVE boundaries
					if (!intersect)
						accepted = 1;

					// Test number of iterations
					if (iter >= MAXITER)
					{
						// printf("%i iter. exceeded for r = %g.\n", MAXITER, diam / 2);
						break;
					}
				}

				// Assign placed grain, if accepted
				if (accepted)
				{
					grainsDatabase[totalNumGrains].p[0] = trialp[0];
					grainsDatabase[totalNumGrains].p[1] = trialp[1];
					grainsDatabase[totalNumGrains].p[2] = trialp[2];
					grainsDatabase[totalNumGrains].r = diam / 2;
					totalNumGrains++;
				}

				// Test fullness of the database
				if (totalNumGrains > 0.95 * maxGrains)
					mexErrMsgTxt(
						"grainsDatabase is full, enlarge maxGrains and recompile.");
			}
			i++;
		}
	}

	/*<<<<<<<<<<<<<<<<<<<<<<<< Build random inclusions only inside
   * >>>>>>>>>>>>>>>>>>>>>>>>*/
	if (SW == 1)
	{

		// Allocate database of grains
		int maxGrains = 0, numGrains, accepted;
		i = 0;
		while (Dmax - (i + 1) * Step >= Dmin)
		{
			diam = (Dmax - i * Step + Dmax - (i + 1) * Step) / 2;
			maxGrains += (int)floor(2 * 2 * 2 * SizeX * SizeY * SizeZ /
									(4.0 / 3.0 * PI * pow(diam / 2.0, 3)));
			i++;
		}
		grainsDatabase = new GRAIN[maxGrains]; // grains database

		// Populate grainsDatabase
		srand((unsigned)time(NULL)); // initialize random seed
		i = 0;
		while (Dmax - (i + 1) * Step >= Dmin)
		{

			// Get number of aggregates for given fraction (radius) from the Fuller
			// distribution
			diam = (Dmax - i * Step + Dmax - (i + 1) * Step) / 2;
			numGrains =
				(int)round(1 / sqrt(Dmax) *
						   (sqrt(Dmax - i * Step) - sqrt(Dmax - (i + 1) * Step)) *
						   Content * 2 * 2 * 2 * SizeX * SizeY * SizeZ /
						   (4.0 / 3.0 * PI * pow(diam / 2.0, 3)));

			// Distribute numGrains over the domain
			double trialp[3];
			for (j = 0; j < numGrains; j++)
			{
				accepted =
					0; // flag: 1 if randomly placed grain is accepted, 0 otherwise

				int iter = 0;
				while (!accepted)
				{
					iter++;

					// Generate trial position of the aggregate
					trialp[0] = 2 * SizeX * ((double)rand() / (double)RAND_MAX) - SizeX;
					trialp[1] = 2 * SizeY * ((double)rand() / (double)RAND_MAX) - SizeY;
					trialp[2] = 2 * SizeZ * ((double)rand() / (double)RAND_MAX) - SizeZ;

					// Test for distance w.r.t. other aggregates
					int intersect = 0;
					k = 0;
					while (k < totalNumGrains && !intersect)
					{
						if (sqrt(pow(trialp[0] - grainsDatabase[k].p[0], 2) +
								 pow(trialp[1] - grainsDatabase[k].p[1], 2) +
								 pow(trialp[2] - grainsDatabase[k].p[2], 2)) <
							std::max(minDist, double(1)) *
								(diam + 2 * fabs(grainsDatabase[k].r)) / 2)
							intersect = 1;
						k++;
					}

					// Test for distance w.r.t. RVE boundaries
					if (fabs(trialp[0]) <
							SizeX - std::max(minDist, double(1)) * diam / 2 &&
						fabs(trialp[1]) <
							SizeY - std::max(minDist, double(1)) * diam / 2 &&
						fabs(trialp[2]) <
							SizeZ - std::max(minDist, double(1)) * diam / 2 &&
						!intersect)
						accepted = 1;

					// Test number of iterations
					if (iter >= MAXITER)
					{
						// printf("%i iter. exceeded for r = %g.\n", MAXITER, diam / 2);
						break;
					}
				}

				// Assign placed grain, if accepted
				if (accepted)
				{
					grainsDatabase[totalNumGrains].p[0] = trialp[0];
					grainsDatabase[totalNumGrains].p[1] = trialp[1];
					grainsDatabase[totalNumGrains].p[2] = trialp[2];
					grainsDatabase[totalNumGrains].r = diam / 2;
					totalNumGrains++;
				}
			}

			// Test fullness of the database
			if (totalNumGrains > 0.95 * maxGrains)
				mexErrMsgTxt(
					"grainsDatabase is full, enlarge maxGrains and recompile.");

			i++;
		}
	}

	/*<<<<<<<<<<<<<<<<<<<<<<<< Send out the data back to MATLAB
   * >>>>>>>>>>>>>>>>>>>>>>>>*/
	// Send out bonds database back to MATLAB
	nlhs = 1;

	// Send out grainsDatabase to MATLAB
	mwSize dimsG[] = {1, (mwSize)totalNumGrains};
	const char *field_namesG[] = {"p", "r"};
	plhs[0] = mxCreateStructArray(2, dimsG, 2, field_namesG);

	// Populate output with computed data
	int name_p = mxGetFieldNumber(plhs[0], "p");
	int name_r = mxGetFieldNumber(plhs[0], "r");
	mxArray *array_p, *array_r;
	double *prp, *prr;
	for (i = 0; i < totalNumGrains; i++)
	{

		// Create arrays
		array_p = mxCreateDoubleMatrix(3, 1, mxREAL);
		array_r = mxCreateDoubleMatrix(1, 1, mxREAL);

		// Get data arrays
		prp = mxGetPr(array_p);
		prr = mxGetPr(array_r);

		// And populate them
		prp[0] = grainsDatabase[i].p[0];
		prp[1] = grainsDatabase[i].p[1];
		prp[2] = grainsDatabase[i].p[2];
		prr[0] = grainsDatabase[i].r;

		// Assign output arrays
		mxSetFieldByNumber(plhs[0], i, name_p, array_p);
		mxSetFieldByNumber(plhs[0], i, name_r, array_r);
	}

	// Delete inclusions' database
	delete[] grainsDatabase;
}
