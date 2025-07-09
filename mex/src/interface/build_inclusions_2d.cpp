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
const int MAXITER = 1000; // max number of trials during placing an aggregate

/******************** Structure declarations ******************/
struct GRAIN {
  double p[2]; // centre of circular grain
  double r;    // radius of circular grain
  int n;       // length of the polygon
  double *x;   // polygon x-coordinates
  double *y;   // polygon y-coordinates
};

/************************ Main program ************************/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  /*<<<<<<<<<<<<<<<<<<<<<<<< Get inputs >>>>>>>>>>>>>>>>>>>>>>>>*/
  // Test for one input argument
  if (nrhs != 1) {
    mexErrMsgTxt("One input argument required.");
  }

  // Copy input variables
  double *prIn = mxGetPr(prhs[0]);
  if ((int)std::max(mxGetM(prhs[0]), mxGetN(prhs[0])) != 14) {
    mexErrMsgTxt("Required length of the input vector is 14.");
  }
  double SizeX = prIn[0];
  double SizeY = prIn[1];
  double Dmin = prIn[2];
  double Dmax = prIn[3];
  double minDist = prIn[4];
  double Content = prIn[5];
  double Step = prIn[6];
  double TOL_g = prIn[7]; // a positive distance < TOL_g is treated as zero
  int sw1 = (int)prIn[8];
  int sw2 = (int)prIn[9];
  double hmin = prIn[10];
  double Perdiam = prIn[11];
  double Perangle = prIn[12];

  /*<<<<<<<<<<<<<<<<<<<<<<<< Build inclusions >>>>>>>>>>>>>>>>>>>>>>>>*/
  // Allocate database of aggregates
  int totalNumGrains = 0, i, j, k;
  double diam;
  GRAIN *grainsDatabase = NULL;

  // REGULAR INCLUSIONS
  if (sw1 == 0) {
    diam = (Dmin + Dmax) / 2;
    double dist = minDist * diam;
    int numx = (int)ceil(2 * SizeX / dist) - 1;
    int numy = (int)ceil(2 * SizeY / dist) - 1;
    grainsDatabase = new GRAIN[numx * numy];
    for (i = 0; i < numx; i++) {
      for (j = 0; j < numy; j++) {
        grainsDatabase[totalNumGrains].p[0] =
            -numx * dist / 2 + dist / 2 + dist * i;
        grainsDatabase[totalNumGrains].p[1] =
            -numy * dist / 2 + dist / 2 + dist * j;
        grainsDatabase[totalNumGrains].r = diam / 2;
        totalNumGrains++;
      }
    }
  }

  // RANDOM INCLUSIONS INSIDE/OUTSIDE
  if (sw1 == 1) {

    // Allocate database of grains
    int maxGrains = 0, numGrains, accepted;
    i = 0;
    while (Dmax - (i + 1) * Step >= Dmin) {
      diam = (Dmax - i * Step + Dmax - (i + 1) * Step) / 2;
      maxGrains += (int)floor(
          4 * 4 * SizeX * SizeY /
          (0.25 * PI *
           pow(diam, 2))); // may be for copies of each due to periodicity
      i++;
    }
    maxGrains += 2 * ceil((SizeY / sin(Perangle)) / Perdiam);
    grainsDatabase = new GRAIN[maxGrains]; // grains database

    // Populate grainsDatabase
    srand((unsigned)time(NULL)); // initialize random seed
    i = 0;
    while (Dmax - (i + 1) * Step >= Dmin) {

      // Get number of aggregates for given fraction (radius) from the Fuller
      // distribution
      diam = (Dmax - i * Step + Dmax - (i + 1) * Step) / 2;
      numGrains =
          (int)round(1 / sqrt(Dmax) *
                     (sqrt(Dmax - i * Step) - sqrt(Dmax - (i + 1) * Step)) *
                     Content * 4 * SizeX * SizeY / (0.25 * PI * pow(diam, 2)));

      // Distribute numGrains over the domain
      int flagBottom, flagRight, flagTop, flagLeft, flagBottomLeft,
          flagBottomRight, flagTopRight, flagTopLeft;
      double trialp[2], tempTrialp[2];
      for (j = 0; j < numGrains; j++) {
        accepted =
            0; // flag: 1 if randomly placed grain is accepted, 0 otherwise
        int iter = 0;
        while (!accepted) {
          iter++;

          // Generate trial position of the aggregate
          trialp[0] = 2 * SizeX * ((double)rand() / (double)RAND_MAX) - SizeX;
          trialp[1] = 2 * SizeY * ((double)rand() / (double)RAND_MAX) - SizeY;

          // Test for distance w.r.t. other aggregates
          int intersect = 0;
          k = 0;
          while (k < totalNumGrains && !intersect) {
            if (sqrt(pow(trialp[0] - grainsDatabase[k].p[0], 2) +
                     pow(trialp[1] - grainsDatabase[k].p[1], 2)) <
                std::max(minDist, double(1)) *
                    (diam + 2 * fabs(grainsDatabase[k].r)) / 2)
              intersect = 1;
            k++;
          }

          // Check periodicity
          flagBottom = 0;
          flagRight = 0;
          flagTop = 0;
          flagLeft = 0;
          flagBottomLeft = 0;
          flagBottomRight = 0;
          flagTopRight = 0;
          flagTopLeft = 0;
          if (sw2 == 1) {
            if (!intersect) {

              // EDGES
              // Bottom part of the boundary
              if (trialp[1] - diam / 2 < -SizeY + TOL_g) {

                // Shift the inclusion
                tempTrialp[0] = trialp[0];
                tempTrialp[1] = trialp[1] + 2 * SizeY;

                // Test for distance w.r.t. other aggregates
                k = 0;
                while (k < totalNumGrains && !intersect) {
                  if (sqrt(pow(tempTrialp[0] - grainsDatabase[k].p[0], 2) +
                           pow(tempTrialp[1] - grainsDatabase[k].p[1], 2)) <
                      std::max(minDist, double(1)) *
                          (diam + 2 * fabs(grainsDatabase[k].r)) / 2)
                    intersect = 1;
                  k++;
                }
                flagBottom = 1;
              }

              // Right of the boundary
              if (trialp[0] + diam / 2 > SizeX - TOL_g) {

                // Shift the inclusion
                tempTrialp[0] = trialp[0] - 2 * SizeX;
                tempTrialp[1] = trialp[1];

                // Test for distance w.r.t. other aggregates
                k = 0;
                while (k < totalNumGrains && !intersect) {
                  if (sqrt(pow(tempTrialp[0] - grainsDatabase[k].p[0], 2) +
                           pow(tempTrialp[1] - grainsDatabase[k].p[1], 2)) <
                      std::max(minDist, double(1)) *
                          (diam + 2 * fabs(grainsDatabase[k].r)) / 2)
                    intersect = 1;
                  k++;
                }
                flagRight = 1;
              }

              // Top part of the boundary
              if (trialp[1] + diam / 2 > SizeY - TOL_g) {

                // Shift the inclusion
                tempTrialp[0] = trialp[0];
                tempTrialp[1] = trialp[1] - 2 * SizeY;

                // Test for distance w.r.t. other aggregates
                k = 0;
                while (k < totalNumGrains && !intersect) {
                  if (sqrt(pow(tempTrialp[0] - grainsDatabase[k].p[0], 2) +
                           pow(tempTrialp[1] - grainsDatabase[k].p[1], 2)) <
                      std::max(minDist, double(1)) *
                          (diam + 2 * fabs(grainsDatabase[k].r)) / 2)
                    intersect = 1;
                  k++;
                }
                flagTop = 1;
              }

              // The left part of the boundary
              if (trialp[0] - diam / 2 < -SizeX + TOL_g) {

                // Shift the inclusion
                tempTrialp[0] = trialp[0] + 2 * SizeX;
                tempTrialp[1] = trialp[1];

                // Test for distance w.r.t. other aggregates
                k = 0;
                while (k < totalNumGrains && !intersect) {
                  if (sqrt(pow(tempTrialp[0] - grainsDatabase[k].p[0], 2) +
                           pow(tempTrialp[1] - grainsDatabase[k].p[1], 2)) <
                      std::max(minDist, double(1)) *
                          (diam + 2 * fabs(grainsDatabase[k].r)) / 2)
                    intersect = 1;
                  k++;
                }
                flagLeft = 1;
              }

              // CORNERS
              // Left bottom corner
              if (trialp[1] - diam / 2 < -SizeY + TOL_g &&
                  trialp[0] - diam / 2 < -SizeX + TOL_g) {

                // Shift the inclusion
                tempTrialp[0] = trialp[0] + 2 * SizeX;
                tempTrialp[1] = trialp[1] + 2 * SizeY;

                // Test for distance w.r.t. other aggregates
                k = 0;
                while (k < totalNumGrains && !intersect) {
                  if (sqrt(pow(tempTrialp[0] - grainsDatabase[k].p[0], 2) +
                           pow(tempTrialp[1] - grainsDatabase[k].p[1], 2)) <
                      std::max(minDist, double(1)) *
                          (diam + 2 * fabs(grainsDatabase[k].r)) / 2)
                    intersect = 1;
                  k++;
                }
                flagBottomLeft = 1;
              }

              // Right bottom corner
              if (trialp[1] - diam / 2 < -SizeY + TOL_g &&
                  trialp[0] + diam / 2 > SizeX - TOL_g) {

                // Shift the inclusion
                tempTrialp[0] = trialp[0] - 2 * SizeX;
                tempTrialp[1] = trialp[1] + 2 * SizeY;

                // Test for distance w.r.t. other aggregates
                k = 0;
                while (k < totalNumGrains && !intersect) {
                  if (sqrt(pow(tempTrialp[0] - grainsDatabase[k].p[0], 2) +
                           pow(tempTrialp[1] - grainsDatabase[k].p[1], 2)) <
                      std::max(minDist, double(1)) *
                          (diam + 2 * fabs(grainsDatabase[k].r)) / 2)
                    intersect = 1;
                  k++;
                }
                flagBottomRight = 1;
              }

              // Right top corner
              if (trialp[1] + diam / 2 > SizeY - TOL_g &&
                  trialp[0] + diam / 2 > SizeX - TOL_g) {

                // Shift the inclusion
                tempTrialp[0] = trialp[0] - 2 * SizeX;
                tempTrialp[1] = trialp[1] - 2 * SizeY;

                // Test for distance w.r.t. other aggregates
                k = 0;
                while (k < totalNumGrains && !intersect) {
                  if (sqrt(pow(tempTrialp[0] - grainsDatabase[k].p[0], 2) +
                           pow(tempTrialp[1] - grainsDatabase[k].p[1], 2)) <
                      std::max(minDist, double(1)) *
                          (diam + 2 * fabs(grainsDatabase[k].r)) / 2)
                    intersect = 1;
                  k++;
                }
                flagTopRight = 1;
              }

              // Left top corner
              if (trialp[0] - diam / 2 < -SizeY + TOL_g &&
                  trialp[1] + diam / 2 > SizeX - TOL_g) {

                // Shift the inclusion
                tempTrialp[0] = trialp[0] + 2 * SizeX;
                tempTrialp[1] = trialp[1] - 2 * SizeY;

                // Test for distance w.r.t. other aggregates
                k = 0;
                while (k < totalNumGrains && !intersect) {
                  if (sqrt(pow(tempTrialp[0] - grainsDatabase[k].p[0], 2) +
                           pow(tempTrialp[1] - grainsDatabase[k].p[1], 2)) <
                      std::max(minDist, double(1)) *
                          (diam + 2 * fabs(grainsDatabase[k].r)) / 2)
                    intersect = 1;
                  k++;
                }
                flagTopLeft = 1;
              }
            }
          }

          // Test for distance w.r.t. RVE boundaries
          if (!intersect)
            accepted = 1;

          // Test number of iterations
          if (iter >= MAXITER) {
            printf("%i iter. exceeded for r = %g.\n", MAXITER, diam / 2);
            break;
          }
        }

        // Assign placed grain, if accepted
        if (accepted) {
          grainsDatabase[totalNumGrains].p[0] = trialp[0];
          grainsDatabase[totalNumGrains].p[1] = trialp[1];
          grainsDatabase[totalNumGrains].r = diam / 2;
          totalNumGrains++;

          // Assign periodic copies
          // EDGES
          // Bottom part of the boundary
          if (flagBottom) {
            tempTrialp[0] = trialp[0];
            tempTrialp[1] = trialp[1] + 2 * SizeY;
            grainsDatabase[totalNumGrains].p[0] = tempTrialp[0];
            grainsDatabase[totalNumGrains].p[1] = tempTrialp[1];
            grainsDatabase[totalNumGrains].r = diam / 2;
            totalNumGrains++;
          }

          // Right of the boundary
          if (flagRight) {
            tempTrialp[0] = trialp[0] - 2 * SizeX;
            tempTrialp[1] = trialp[1];
            grainsDatabase[totalNumGrains].p[0] = tempTrialp[0];
            grainsDatabase[totalNumGrains].p[1] = tempTrialp[1];
            grainsDatabase[totalNumGrains].r = diam / 2;
            totalNumGrains++;
          }

          // Top part of the boundary
          if (flagTop) {
            tempTrialp[0] = trialp[0];
            tempTrialp[1] = trialp[1] - 2 * SizeY;
            grainsDatabase[totalNumGrains].p[0] = tempTrialp[0];
            grainsDatabase[totalNumGrains].p[1] = tempTrialp[1];
            grainsDatabase[totalNumGrains].r = diam / 2;
            totalNumGrains++;
          }

          // The left part of the boundary
          if (flagLeft) {
            tempTrialp[0] = trialp[0] + 2 * SizeX;
            tempTrialp[1] = trialp[1];
            grainsDatabase[totalNumGrains].p[0] = tempTrialp[0];
            grainsDatabase[totalNumGrains].p[1] = tempTrialp[1];
            grainsDatabase[totalNumGrains].r = diam / 2;
            totalNumGrains++;
          }

          // CORNERS
          // Left bottom corner
          if (flagBottomLeft) {
            tempTrialp[0] = trialp[0] + 2 * SizeX;
            tempTrialp[1] = trialp[1] + 2 * SizeY;
            grainsDatabase[totalNumGrains].p[0] = tempTrialp[0];
            grainsDatabase[totalNumGrains].p[1] = tempTrialp[1];
            grainsDatabase[totalNumGrains].r = diam / 2;
            totalNumGrains++;
          }

          // Right bottom corner
          if (flagBottomRight) {
            tempTrialp[0] = trialp[0] - 2 * SizeX;
            tempTrialp[1] = trialp[1] + 2 * SizeY;
            grainsDatabase[totalNumGrains].p[0] = tempTrialp[0];
            grainsDatabase[totalNumGrains].p[1] = tempTrialp[1];
            grainsDatabase[totalNumGrains].r = diam / 2;
            totalNumGrains++;
          }

          // Right top corner
          if (flagTopRight) {
            tempTrialp[0] = trialp[0] - 2 * SizeX;
            tempTrialp[1] = trialp[1] - 2 * SizeY;
            grainsDatabase[totalNumGrains].p[0] = tempTrialp[0];
            grainsDatabase[totalNumGrains].p[1] = tempTrialp[1];
            grainsDatabase[totalNumGrains].r = diam / 2;
            totalNumGrains++;
          }

          // Left top corner
          if (flagTopLeft) {
            tempTrialp[0] = trialp[0] + 2 * SizeX;
            tempTrialp[1] = trialp[1] - 2 * SizeY;
            grainsDatabase[totalNumGrains].p[0] = tempTrialp[0];
            grainsDatabase[totalNumGrains].p[1] = tempTrialp[1];
            grainsDatabase[totalNumGrains].r = diam / 2;
            totalNumGrains++;
          }
        }

        // Test fullness of the database
        if (totalNumGrains > 0.95 * maxGrains)
          mexErrMsgTxt(
              "grainsDatabase is full, enlarge maxGrains and recompile.");
      }
      i++;
    }
  }

  // RANDOM INCLUSIONS ONLY INSIDE THE DOMAIN
  if (sw1 == 2) {

    // Allocate database of grains
    int maxGrains = 0, numGrains, accepted;
    i = 0;
    while (Dmax - (i + 1) * Step >= Dmin) {
      diam = (Dmax - i * Step + Dmax - (i + 1) * Step) / 2;
      maxGrains += (int)floor(
          4 * 4 * SizeX * SizeY /
          (0.25 * PI *
           pow(diam, 2))); // may be for copies of each due to periodicity
      i++;
    }
    maxGrains += 2 * ceil((SizeY / sin(Perangle)) / Perdiam);
    grainsDatabase = new GRAIN[maxGrains]; // grains database

    // Populate grainsDatabase
    srand((unsigned)time(NULL)); // initialize random seed
    i = 0;
    while (Dmax - (i + 1) * Step >= Dmin) {

      // Get number of aggregates for given fraction (radius) from the Fuller
      // distribution
      diam = (Dmax - i * Step + Dmax - (i + 1) * Step) / 2;
      numGrains =
          (int)round(1 / sqrt(Dmax) *
                     (sqrt(Dmax - i * Step) - sqrt(Dmax - (i + 1) * Step)) *
                     Content * 4 * SizeX * SizeY / (0.25 * PI * pow(diam, 2)));

      // Distribute numGrains over the domain
      double trialp[2];
      for (j = 0; j < numGrains; j++) {
        accepted =
            0; // flag: 1 if randomly placed grain is accepted, 0 otherwise

        int iter = 0;
        while (!accepted) {
          iter++;

          // Generate trial position of the aggregate
          trialp[0] = 2 * SizeX * ((double)rand() / (double)RAND_MAX) - SizeX;
          trialp[1] = 2 * SizeY * ((double)rand() / (double)RAND_MAX) - SizeY;

          // Test for distance w.r.t. other aggregates
          int intersect = 0;
          k = 0;
          while (k < totalNumGrains && !intersect) {
            if (sqrt(pow(trialp[0] - grainsDatabase[k].p[0], 2) +
                     pow(trialp[1] - grainsDatabase[k].p[1], 2)) <
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
              !intersect)
            accepted = 1;

          // Test number of iterations
          if (iter >= MAXITER) {
            printf("%i iter. exceeded for r = %g.\n", MAXITER, diam / 2);
            break;
          }
        }

        // Assign placed grain, if accepted
        if (accepted) {
          grainsDatabase[totalNumGrains].p[0] = trialp[0];
          grainsDatabase[totalNumGrains].p[1] = trialp[1];
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

  /*<<<<<<<<<<<<<<<<<<<<<<<< Postprocess inclusions >>>>>>>>>>>>>>>>>>>>>>>>*/
  int nPhi;
  double phi, point[2];
  for (i = 0; i < totalNumGrains; i++) {
    nPhi = (int)ceil(100 * 2 * PI * fabs(grainsDatabase[i].r) / hmin);
    grainsDatabase[i].n = 0;
    grainsDatabase[i].x = new double[nPhi];
    grainsDatabase[i].y = new double[nPhi];
    grainsDatabase[i].x[0] =
        round((grainsDatabase[i].p[0] + fabs(grainsDatabase[i].r) * cos(0)) /
              hmin) *
        hmin;
    grainsDatabase[i].y[0] =
        round((grainsDatabase[i].p[1] + fabs(grainsDatabase[i].r) * sin(0)) /
              hmin) *
        hmin;
    phi = 0;
    while (phi <= 2 * PI) {
      phi += 2 * PI / nPhi;
      point[0] = round((grainsDatabase[i].p[0] +
                        fabs(grainsDatabase[i].r) * cos(phi)) /
                       hmin) *
                 hmin;
      point[1] = round((grainsDatabase[i].p[1] +
                        fabs(grainsDatabase[i].r) * sin(phi)) /
                       hmin) *
                 hmin;

      if (fabs(point[0] - grainsDatabase[i].x[grainsDatabase[i].n]) +
              fabs(point[1] - grainsDatabase[i].y[grainsDatabase[i].n]) >
          hmin / 2) {
        grainsDatabase[i].n++;
        grainsDatabase[i].x[grainsDatabase[i].n] = point[0];
        grainsDatabase[i].y[grainsDatabase[i].n] = point[1];
      }
    }
    grainsDatabase[i].n++;
  }

  /*<<<<<<<<<<<<<<<<<<<<<<<< Send out the data back to MATLAB
   * >>>>>>>>>>>>>>>>>>>>>>>>*/
  // Send out bonds database back to MATLAB
  nlhs = 1;

  // Send out grainsDatabase to MATLAB
  mwSize dimsG[] = {1, (mwSize)totalNumGrains};
  const char *field_namesG[] = {"p", "r", "x", "y"};
  plhs[0] = mxCreateStructArray(2, dimsG, 4, field_namesG);

  // Populate output with computed data
  int name_p = mxGetFieldNumber(plhs[0], "p");
  int name_r = mxGetFieldNumber(plhs[0], "r");
  int name_x = mxGetFieldNumber(plhs[0], "x");
  int name_y = mxGetFieldNumber(plhs[0], "y");
  mxArray *array_p, *array_r, *array_x, *array_y;
  ;
  double *prp, *prr, *prx, *pry;
  ;
  for (i = 0; i < totalNumGrains; i++) {

    // Create arrays
    array_p = mxCreateDoubleMatrix(1, 2, mxREAL);
    array_r = mxCreateDoubleMatrix(1, 1, mxREAL);
    array_x = mxCreateDoubleMatrix(1, grainsDatabase[i].n, mxREAL);
    array_y = mxCreateDoubleMatrix(1, grainsDatabase[i].n, mxREAL);

    // Get data arrays
    prp = mxGetPr(array_p);
    prr = mxGetPr(array_r);
    prx = mxGetPr(array_x);
    pry = mxGetPr(array_y);

    // And populate them
    prp[0] = grainsDatabase[i].p[0];
    prp[1] = grainsDatabase[i].p[1];
    prr[0] = grainsDatabase[i].r;
    for (j = 0; j < grainsDatabase[i].n; j++) {
      prx[j] = grainsDatabase[i].x[j];
      pry[j] = grainsDatabase[i].y[j];
    }

    // Assign output arrays
    mxSetFieldByNumber(plhs[0], i, name_p, array_p);
    mxSetFieldByNumber(plhs[0], i, name_r, array_r);
    mxSetFieldByNumber(plhs[0], i, name_x, array_x);
    mxSetFieldByNumber(plhs[0], i, name_y, array_y);
  }
  for (i = 0; i < totalNumGrains; i++) {
    delete[] grainsDatabase[i].x;
    delete[] grainsDatabase[i].y;
  }
  delete[] grainsDatabase;
}
