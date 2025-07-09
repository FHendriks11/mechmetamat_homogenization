// myfem: functions for integration rules, shape function evaluations, etc.
#include <stdio.h>
#include <math.h>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "mex.h"
#include "matrix.h"
#include "myfem.h"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#define EIGEN_DONT_PARALLELIZE

/******************** Constant ********************/
const double EPS = 1.0e-12; // defines margin for equality of two doubles (determinants, eigenvalues, etc.)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/******************** Deformation ********************/

// Get deformation gradient (F = [F11,F21,F12,F22], F = [F11 F21 F31 F12 F22 F32 F13 F23 F33])
int getF2d(double **Be, const double *uel, const int np,
           double F[4])
{

    // Deformation gradient F = I + B*uel
    for (int i = 0; i < 4; i++)
    {
        F[i] = 0.0;
        for (int j = 0; j < 2 * np; j++)
            F[i] += Be[i][j] * uel[j];
    }
    F[0] += 1.0;
    F[3] += 1.0;

    return 1;
}

int getF3d(double **Be, const double *uel, const int np,
           double F[9])
{

    // Deformation gradient F = I + B*uel
    for (int i = 0; i < 9; i++)
    {
        F[i] = 0.0;
        for (int j = 0; j < 3 * np; j++)
            F[i] += Be[i][j] * uel[j];
    }
    F[0] += 1.0;
    F[4] += 1.0;
    F[8] += 1.0;

    return 1;
}

// Get the right Cauchy-Green strain tensor C = F'*F, Voigt notation (i.e C = [C11, C22, C12], C = [C11, C22, C33, C12, C23, C31])
int getC2d(const double F[4], double C[3])
{
    C[0] = F[0] * F[0] + F[1] * F[1];
    C[1] = F[2] * F[2] + F[3] * F[3];
    C[2] = F[0] * F[2] + F[1] * F[3];

    return 1;
}

int getC3d(const double F[9], double C[9])
{
    C[0] = F[0] * F[0] + F[1] * F[1] + F[2] * F[2];
    C[1] = F[3] * F[3] + F[4] * F[4] + F[5] * F[5];
    C[2] = F[6] * F[6] + F[7] * F[7] + F[8] * F[8];
    C[3] = F[0] * F[3] + F[1] * F[4] + F[2] * F[5];
    C[4] = F[3] * F[6] + F[4] * F[7] + F[5] * F[8];
    C[5] = F[0] * F[6] + F[1] * F[7] + F[2] * F[8];

    return 1;
}

// Get the Green-Lagrange strain tensor E = 0.5*(F'*F-I), Voigt notation (i.e E = [E11, E22, 2*E12], E = [E11, E22, E33, 2*E12, 2*E23, 2*E31])
int getE2d(const double F[4], double E[3])
{
    double C[3];
    getC2d(F, C);
    E[0] = 0.5 * (C[0] - 1.0);
    E[1] = 0.5 * (C[1] - 1.0);
    E[2] = C[2];

    return 1;
}

int getE3d(const double F[9], double E[6])
{
    double C[6];
    getC3d(F, C);
    E[0] = 0.5 * (C[0] - 1.0);
    E[1] = 0.5 * (C[1] - 1.0);
    E[2] = 0.5 * (C[2] - 1.0);
    E[3] = C[3];
    E[4] = C[4];
    E[5] = C[5];

    return 1;
}

/******************** Tensor operations ********************/

// Get second-order identity tensor tensor in Voigt notation
int getId2d(double Id[3])
{
    Id[0] = 1.0;
    Id[1] = 1.0;
    Id[2] = 0.0;

    return 1;
}
int getId3d(double Id[6])
{
    Id[0] = 1.0;
    Id[1] = 1.0;
    Id[2] = 1.0;
    Id[3] = 0.0;
    Id[4] = 0.0;
    Id[5] = 0.0;

    return 1;
}

// Take symmetric producdt of two second-order tensors in Voigt notation ( Cijkl = 0.5*(Aik*Bjl+Ail*Bjk) )
int getSymmetricProduct2d(double C[3][3], const double A[3], const double B[3])
{
    C[0][0] = 0.5 * (A[0] * B[0] + A[0] * B[0]);
    C[1][0] = 0.5 * (A[2] * B[2] + A[2] * B[2]);
    C[2][0] = 0.5 * (A[0] * B[2] + A[0] * B[2]);
    C[0][1] = 0.5 * (A[2] * B[2] + A[2] * B[2]);
    C[1][1] = 0.5 * (A[1] * B[1] + A[1] * B[1]);
    C[2][1] = 0.5 * (A[2] * B[1] + A[2] * B[1]);
    C[0][2] = 0.5 * (A[0] * B[2] + A[2] * B[0]);
    C[1][2] = 0.5 * (A[2] * B[1] + A[1] * B[2]);
    C[2][2] = 0.5 * (A[0] * B[1] + A[2] * B[2]);

    return 1;
}

int getSymmetricProduct3d(double C[6][6], double A[6], double B[6])
{
    C[0][0] = 0.5 * (A[0] * B[0] + A[0] * B[0]);
    C[1][0] = 0.5 * (A[3] * B[3] + A[3] * B[3]);
    C[2][0] = 0.5 * (A[5] * B[5] + A[5] * B[5]);
    C[3][0] = 0.5 * (A[0] * B[3] + A[0] * B[3]);
    C[4][0] = 0.5 * (A[3] * B[5] + A[3] * B[5]);
    C[5][0] = 0.5 * (A[5] * B[0] + A[5] * B[0]);

    C[0][1] = 0.5 * (A[3] * B[3] + A[3] * B[3]);
    C[1][1] = 0.5 * (A[1] * B[1] + A[1] * B[1]);
    C[2][1] = 0.5 * (A[4] * B[4] + A[4] * B[4]);
    C[3][1] = 0.5 * (A[3] * B[1] + A[3] * B[1]);
    C[4][1] = 0.5 * (A[1] * B[4] + A[1] * B[4]);
    C[5][1] = 0.5 * (A[4] * B[3] + A[4] * B[3]);

    C[0][2] = 0.5 * (A[5] * B[5] + A[5] * B[5]);
    C[1][2] = 0.5 * (A[4] * B[4] + A[4] * B[4]);
    C[2][2] = 0.5 * (A[2] * B[2] + A[2] * B[2]);
    C[3][2] = 0.5 * (A[5] * B[4] + A[5] * B[4]);
    C[4][2] = 0.5 * (A[4] * B[2] + A[4] * B[2]);
    C[5][2] = 0.5 * (A[2] * B[5] + A[2] * B[5]);

    C[0][3] = 0.5 * (A[0] * B[3] + A[3] * B[0]);
    C[1][3] = 0.5 * (A[3] * B[1] + A[1] * B[3]);
    C[2][3] = 0.5 * (A[5] * B[4] + A[4] * B[5]);
    C[3][3] = 0.5 * (A[0] * B[1] + A[3] * B[3]);
    C[4][3] = 0.5 * (A[3] * B[4] + A[1] * B[5]);
    C[5][3] = 0.5 * (A[5] * B[3] + A[4] * B[0]);

    C[0][4] = 0.5 * (A[3] * B[5] + A[5] * B[3]);
    C[1][4] = 0.5 * (A[1] * B[4] + A[4] * B[1]);
    C[2][4] = 0.5 * (A[4] * B[2] + A[2] * B[4]);
    C[3][4] = 0.5 * (A[3] * B[4] + A[5] * B[1]);
    C[4][4] = 0.5 * (A[1] * B[2] + A[4] * B[4]);
    C[5][4] = 0.5 * (A[4] * B[5] + A[2] * B[3]);

    C[0][5] = 0.5 * (A[5] * B[0] + A[0] * B[5]);
    C[1][5] = 0.5 * (A[4] * B[3] + A[3] * B[4]);
    C[2][5] = 0.5 * (A[2] * B[5] + A[5] * B[2]);
    C[3][5] = 0.5 * (A[5] * B[3] + A[0] * B[4]);
    C[4][5] = 0.5 * (A[4] * B[5] + A[3] * B[2]);
    C[5][5] = 0.5 * (A[2] * B[0] + A[5] * B[5]);

    return 1;
}

/******************** Triangles ********************/

// Gauss integration rule for triangular elements
int getTriagGaussInfo(const int ngauss, double *alpha, double **IP)
{
    int i;

    // Choose Gauss integration rule
    switch (ngauss)
    {
    case 1: // 1 Gauss point in the middle
    {
        for (i = 0; i < 3; i++)
            IP[0][i] = 1.0 / 3.0;
        alpha[0] = 1.0;
        break;
    }

    case 3: // 3 Gauss points inside
    {
        IP[0][0] = 2.0 / 3.0;
        IP[0][1] = 1.0 / 6.0;
        IP[0][2] = 1.0 / 6.0;
        IP[1][0] = 1.0 / 6.0;
        IP[1][1] = 2.0 / 3.0;
        IP[1][2] = 1.0 / 6.0;
        IP[2][0] = 1.0 / 6.0;
        IP[2][1] = 1.0 / 6.0;
        IP[2][2] = 2.0 / 3.0;
        alpha[0] = 1.0 / 3.0;
        alpha[1] = 1.0 / 3.0;
        alpha[2] = 1.0 / 3.0;
        break;
    }

    case -3: // 3 Gauss points at mid edges
    {
        IP[0][0] = 1.0 / 2.0;
        IP[0][1] = 1.0 / 2.0;
        IP[0][2] = 0.0;
        IP[1][0] = 0.0;
        IP[1][1] = 1.0 / 2.0;
        IP[1][2] = 1.0 / 2.0;
        IP[2][0] = 1.0 / 2.0;
        IP[2][1] = 0.0;
        IP[2][2] = 1.0 / 2.0;
        alpha[0] = 1.0 / 3.0;
        alpha[1] = 1.0 / 3.0;
        alpha[2] = 1.0 / 3.0;
        break;
    }

    case 4: // 4 Gauss points inside
    {
        IP[0][0] = 1.0 / 3.0;
        IP[0][1] = 1.0 / 3.0;
        IP[0][2] = 1.0 / 3.0;
        IP[1][0] = 0.6;
        IP[1][1] = 0.2;
        IP[1][2] = 0.2;
        IP[2][0] = 0.2;
        IP[2][1] = 0.6;
        IP[2][2] = 0.2;
        IP[3][0] = 0.2;
        IP[3][1] = 0.2;
        IP[3][2] = 0.6;
        alpha[0] = -27.0 / 48.0;
        alpha[1] = 25.0 / 48.0;
        alpha[2] = 25.0 / 48.0;
        alpha[3] = 25.0 / 48.0;
        break;
    }

    case 6: // 6 Gauss points inside
    {
        IP[0][0] = 0.44594849091597;
        IP[0][1] = 0.44594849091597;
        IP[0][2] = 0.10810301816807;
        IP[1][0] = 0.44594849091597;
        IP[1][1] = 0.10810301816807;
        IP[1][2] = 0.44594849091597;
        IP[2][0] = 0.10810301816807;
        IP[2][1] = 0.44594849091597;
        IP[2][2] = 0.44594849091597;
        IP[3][0] = 0.09157621350977;
        IP[3][1] = 0.09157621350977;
        IP[3][2] = 0.81684757298046;
        IP[4][0] = 0.09157621350977;
        IP[4][1] = 0.81684757298046;
        IP[4][2] = 0.09157621350977;
        IP[5][0] = 0.81684757298046;
        IP[5][1] = 0.09157621350977;
        IP[5][2] = 0.09157621350977;
        alpha[0] = 0.22338158967801;
        alpha[1] = 0.22338158967801;
        alpha[2] = 0.22338158967801;
        alpha[3] = 0.10995174365532;
        alpha[4] = 0.10995174365532;
        alpha[5] = 0.10995174365532;
        break;
    }

    case 7: // 7 Gauss points inside
    {
        double alpha1 = 0.05971587178977;
        double beta1 = 0.47014206410511;
        double alpha2 = 0.79742698535309;
        double beta2 = 0.10128650732346;
        IP[0][0] = 1.0 / 3.0;
        IP[0][1] = 1.0 / 3.0;
        IP[0][2] = 1.0 / 3.0;
        IP[1][0] = alpha1;
        IP[1][1] = beta1;
        IP[1][2] = beta1;
        IP[2][0] = beta1;
        IP[2][1] = alpha1;
        IP[2][2] = beta1;
        IP[3][0] = beta1;
        IP[3][1] = beta1;
        IP[3][2] = alpha1;
        IP[4][0] = alpha2;
        IP[4][1] = beta2;
        IP[4][2] = beta2;
        IP[5][0] = beta2;
        IP[5][1] = alpha2;
        IP[5][2] = beta2;
        IP[6][0] = beta2;
        IP[6][1] = beta2;
        IP[6][2] = alpha2;
        alpha[0] = 0.225;
        alpha[1] = 0.13239415278851;
        alpha[2] = 0.13239415278851;
        alpha[3] = 0.13239415278851;
        alpha[4] = 0.12593918054483;
        alpha[5] = 0.12593918054483;
        alpha[6] = 0.12593918054483;
        break;
    }

    default:
        return -1;
    }
    return 1;
}

// Returns shape functions evaluated at tcoord for triangular elements
int getTriagShapeFun(const int np, const double tcoord[3], double *N)
{

    // Choose element type
    switch (np)
    {
    case 3: // linear triangle

        // Basis functions expressed in terms of planar coordinates
        N[0] = tcoord[0];
        N[1] = tcoord[1];
        N[2] = tcoord[2];

        break;

    case 6: // quadratic triangle

        // Basis functions expressed in terms of planar coordinates
        N[0] = tcoord[0] * (2.0 * tcoord[0] - 1.0);
        N[1] = tcoord[1] * (2.0 * tcoord[1] - 1.0);
        N[2] = tcoord[2] * (2.0 * tcoord[2] - 1.0);
        N[3] = 4.0 * tcoord[0] * tcoord[1];
        N[4] = 4.0 * tcoord[1] * tcoord[2];
        N[5] = 4.0 * tcoord[2] * tcoord[0];

        break;

    case 10: // cubic triangle

        // Basis functions expressed in terms of planar coordinates
        N[0] = 1.0 / 2.0 * tcoord[0] * (3.0 * tcoord[0] - 1.0) * (3.0 * tcoord[0] - 2.0);
        N[1] = 1.0 / 2.0 * tcoord[1] * (3.0 * tcoord[1] - 1.0) * (3.0 * tcoord[1] - 2.0);
        N[2] = 1.0 / 2.0 * tcoord[2] * (3.0 * tcoord[2] - 1.0) * (3.0 * tcoord[2] - 2.0);
        N[3] = 9.0 / 2.0 * tcoord[0] * tcoord[1] * (3.0 * tcoord[0] - 1.0);
        N[4] = 9.0 / 2.0 * tcoord[0] * tcoord[1] * (3.0 * tcoord[1] - 1.0);
        N[5] = 9.0 / 2.0 * tcoord[1] * tcoord[2] * (3.0 * tcoord[1] - 1.0);
        N[6] = 9.0 / 2.0 * tcoord[1] * tcoord[2] * (3.0 * tcoord[2] - 1.0);
        N[7] = 9.0 / 2.0 * tcoord[2] * tcoord[0] * (3.0 * tcoord[2] - 1.0);
        N[8] = 9.0 / 2.0 * tcoord[2] * tcoord[0] * (3.0 * tcoord[0] - 1.0);
        N[9] = 27.0 * tcoord[0] * tcoord[1] * tcoord[2];

        break;

    default:
        return -2;
    }

    return 1;
}

// Matrix of derivatives with respect to physical coordinates of basis functions for triangles
int getTriagBeTLF(const int np, const double tcoord[3],
                  const double *x, const double *y, double &Jdet, double **Be)
{

    // Initialize some variables
    int i, errorcode;
    double *DxN = new double[np];
    double *DyN = new double[np];

    // Get derivatives of shape functions with respect to physical coordinates
    errorcode = getTriagShapeFunGradPhys(np, tcoord, x, y, DxN, DyN, Jdet);
    if (Jdet < 0)
        return -3;

    // Matrix of derivatives of basis functions
    for (i = 0; i < np; i++)
    {
        Be[0][2 * i + 0] = DxN[i];
        Be[0][2 * i + 1] = 0.0;
        Be[1][2 * i + 0] = 0.0;
        Be[1][2 * i + 1] = DxN[i];
        Be[2][2 * i + 0] = DyN[i];
        Be[2][2 * i + 1] = 0.0;
        Be[3][2 * i + 0] = 0.0;
        Be[3][2 * i + 1] = DyN[i];
    }

    // Clear memory
    delete[] DxN;
    delete[] DyN;

    return errorcode;
}

// Matrix of derivatives with respect to physical coordinates of basis functions for triangles
int getTriagBeTLE(const int np, const double tcoord[3],
                  const double *x, const double *y, const double *uel, double &Jdet,
                  double **Be, double **Bf, double F[4])
{

    // Initialize some variables
    int i, errorcode;
    double *DxN = new double[np];
    double *DyN = new double[np];

    // Get derivatives of shape functions with respect to physical coordinates
    errorcode = getTriagShapeFunGradPhys(np, tcoord, x, y, DxN, DyN, Jdet);
    if (Jdet < 0)
        return -3;

    // Bf matrix
    for (i = 0; i < np; i++)
    {
        Bf[0][2 * i + 0] = DxN[i];
        Bf[0][2 * i + 1] = 0.0;
        Bf[1][2 * i + 0] = 0.0;
        Bf[1][2 * i + 1] = DxN[i];
        Bf[2][2 * i + 0] = DyN[i];
        Bf[2][2 * i + 1] = 0.0;
        Bf[3][2 * i + 0] = 0.0;
        Bf[3][2 * i + 1] = DyN[i];
    }

    // Get F
    getF2d(Bf, uel, np, F);

    // Be matrix
    for (i = 0; i < np; i++)
    {
        Be[0][2 * i + 0] = F[0] * DxN[i];
        Be[1][2 * i + 0] = F[2] * DyN[i];
        Be[2][2 * i + 0] = F[0] * DyN[i] + F[2] * DxN[i];
        Be[0][2 * i + 1] = F[1] * DxN[i];
        Be[1][2 * i + 1] = F[3] * DyN[i];
        Be[2][2 * i + 1] = F[3] * DxN[i] + F[1] * DyN[i];
    }

    // Clear memory
    delete[] DxN;
    delete[] DyN;

    return errorcode;
}

// Convert coordinates from natural to physical for triangles
int ConvertTriagNat2Phys(const int np, const double tcoord[3],
                         const double *x, const double *y, double &X, double &Y)
{
    int errorcode;
    double *N = new double[np];

    // Get basis functions evaluations
    errorcode = getTriagShapeFun(np, tcoord, N);

    // Physical coordinates
    X = 0.0;
    Y = 0.0;
    for (int i = 0; i < np; i++)
    {
        X += N[i] * x[i];
        Y += N[i] * y[i];
    }

    // Clear memory
    delete[] N;

    return errorcode;
}

// Convert coordinates from physical to natural for triangles
int ConvertTriagPhys2Nat(const int np, const double X, const double Y,
                         const double *x, const double *y, const double TOL_g, double tcoord[3])
{

    // Initialize variables
    int i, j, errorcode;
    double f[3], J[3][3], IJ[3][3], dtcoord[3], detJ;
    double *N = new double[np];
    double *DalphaN = new double[np];
    double *DbetaN = new double[np];
    double *DgammaN = new double[np];

    // Get an initial guess (barycentric coordinates)
    tcoord[0] = ((y[1] - y[2]) * (X - x[2]) + (x[2] - x[1]) * (Y - y[2])) /
                ((y[1] - y[2]) * (x[0] - x[2]) +
                 (x[2] - x[1]) * (y[0] - y[2]));
    tcoord[1] = ((y[2] - y[0]) * (X - x[2]) + (x[0] - x[2]) * (Y - y[2])) /
                ((y[1] - y[2]) * (x[0] - x[2]) +
                 (x[2] - x[1]) * (y[0] - y[2]));
    tcoord[2] = 1.0 - tcoord[0] - tcoord[1];

    // Get shape function evaluations at current tcoord
    errorcode = getTriagShapeFun(np, tcoord, N);

    // Update the residual vector
    for (i = 0; i < 3; i++)
        f[i] = 0.0; // initialize f
    for (i = 0; i < np; i++)
    { // compute f
        f[0] += 1.0 * N[i];
        f[1] += x[i] * N[i];
        f[2] += y[i] * N[i];
    }
    f[0] -= 1.0;
    f[1] -= X;
    f[2] -= Y;

    // Update error
    double eps = 0.0;
    for (i = 0; i < 3; i++)
        eps += fabs(f[i]);

    // Newton loop
    int Niter = 0;
    int Maxiter = 100;
    while (eps > TOL_g && Niter < Maxiter)
    {
        Niter++;

        // Get derivatives of basis functions with respect to natural coordinates evaluated at current tcoord
        errorcode = getTriagShapeFunGradNat(np, tcoord, DalphaN, DbetaN, DgammaN);

        // Jacobian of the nonlinear system of equations f = 0
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                J[i][j] = 0.0;
        for (i = 0; i < np; i++)
        {
            J[0][0] += 1.0 * DalphaN[i];
            J[0][1] += 1.0 * DbetaN[i];
            J[0][2] += 1.0 * DgammaN[i];

            J[1][0] += x[i] * DalphaN[i];
            J[1][1] += x[i] * DbetaN[i];
            J[1][2] += x[i] * DgammaN[i];

            J[2][0] += y[i] * DalphaN[i];
            J[2][1] += y[i] * DbetaN[i];
            J[2][2] += y[i] * DgammaN[i];
        }

        // Inverse Jacobian
        if (!getInverse33(J, detJ, IJ))
            break;

        // Solve the system tcoord = tcoord - J\f
        for (i = 0; i < 3; i++)
        {
            dtcoord[i] = 0.0;
            for (j = 0; j < 3; j++)
                dtcoord[i] -= IJ[i][j] * f[j];
        }

        // Update tcoord
        for (i = 0; i < 3; i++)
            tcoord[i] += dtcoord[i];

        // Get shape function evaluations at current tcoord
        errorcode = getTriagShapeFun(np, tcoord, N);

        // Update the residual vector
        for (i = 0; i < 3; i++)
            f[i] = 0.0; // initialize f
        for (i = 0; i < np; i++)
        { // compute f
            f[0] += 1.0 * N[i];
            f[1] += x[i] * N[i];
            f[2] += y[i] * N[i];
        }
        f[0] -= 1.0;
        f[1] -= X;
        f[2] -= Y;

        // Update error
        eps = 0.0;
        for (i = 0; i < 3; i++)
            eps += fabs(f[i]);
    }

    // Print warning message if maximum number of iterations was exceeded
    if (Niter >= Maxiter)
    {
        for (i = 0; i < 3; i++)
            tcoord[i] = NAN; // return NANs
        errorcode = -5;
    }
    else
    { // check if given point is inside element or not
        if ((tcoord[0] > -TOL_g) && (tcoord[1] > -TOL_g) &&
            (tcoord[2] > -TOL_g) &&
            (tcoord[0] < 1.0 + TOL_g) && (tcoord[1] < 1.0 + TOL_g) &&
            (tcoord[2] < 1.0 + TOL_g))
        {
            // If it is inside, do nothing
        }
        else
        {
            for (i = 0; i < 3; i++)
                tcoord[i] = NAN; // return NANs
        }
    }

    // Clear memory
    delete[] N;
    delete[] DalphaN;
    delete[] DbetaN;
    delete[] DgammaN;

    return errorcode;
}

// Returns gradient of shape functions for triangles with respect to natural coordinates
int getTriagShapeFunGradNat(const int np, const double tcoord[3],
                            double *DalphaN, double *DbetaN, double *DgammaN)
{

    // Choose element type
    switch (np)
    {
    case 3: // linear triangle

        // Derivatives of basis functions with respect to natural coordinates
        DalphaN[0] = 1.0;
        DalphaN[1] = 0.0;
        DalphaN[2] = 0.0;

        DbetaN[0] = 0.0;
        DbetaN[1] = 1.0;
        DbetaN[2] = 0.0;

        DgammaN[0] = 0.0;
        DgammaN[1] = 0.0;
        DgammaN[2] = 1.0;

        break;

    case 6: // quadratic triangle

        // Derivatives of basis functions with respect to natural coordinates
        DalphaN[0] = 4.0 * tcoord[0] - 1.0;
        DalphaN[1] = 0.0;
        DalphaN[2] = 0.0;
        DalphaN[3] = 4.0 * tcoord[1];
        DalphaN[4] = 0.0;
        DalphaN[5] = 4.0 * tcoord[2];

        DbetaN[0] = 0.0;
        DbetaN[1] = 4.0 * tcoord[1] - 1.0;
        DbetaN[2] = 0.0;
        DbetaN[3] = 4.0 * tcoord[0];
        DbetaN[4] = 4.0 * tcoord[2];
        DbetaN[5] = 0.0;

        DgammaN[0] = 0.0;
        DgammaN[1] = 0.0;
        DgammaN[2] = 4.0 * tcoord[2] - 1.0;
        DgammaN[3] = 0.0;
        DgammaN[4] = 4.0 * tcoord[1];
        DgammaN[5] = 4.0 * tcoord[0];

        break;

    case 10: // cubic triangle

        // Derivatives of basis functions with respect to natural coordinates
        DalphaN[0] = 9.0 / 2.0 * (2.0 / 9.0 - 2.0 * tcoord[0] + 3.0 * tcoord[0] * tcoord[0]);
        DalphaN[1] = 0.0;
        DalphaN[2] = 0.0;
        DalphaN[3] = 9.0 / 2.0 * (tcoord[1] * (6.0 * tcoord[0] - 1.0));
        DalphaN[4] = 9.0 / 2.0 * (tcoord[1] * (3.0 * tcoord[1] - 1.0));
        DalphaN[5] = 0.0;
        DalphaN[6] = 0.0;
        DalphaN[7] = 9.0 / 2.0 * (tcoord[2] * (3.0 * tcoord[2] - 1.0));
        DalphaN[8] = 9.0 / 2.0 * (tcoord[2] * (6.0 * tcoord[0] - 1.0));
        DalphaN[9] = 9.0 / 2.0 * (6.0 * tcoord[1] * tcoord[2]);

        DbetaN[0] = 0.0;
        DbetaN[1] = 9.0 / 2.0 * (2.0 / 9.0 - 2.0 * tcoord[1] + 3.0 * tcoord[1] * tcoord[1]);
        DbetaN[2] = 0.0;
        DbetaN[3] = 9.0 / 2.0 * (tcoord[0] * (3.0 * tcoord[0] - 1.0));
        DbetaN[4] = 9.0 / 2.0 * (tcoord[0] * (6.0 * tcoord[1] - 1.0));
        DbetaN[5] = 9.0 / 2.0 * (tcoord[2] * (6.0 * tcoord[1] - 1.0));
        DbetaN[6] = 9.0 / 2.0 * (tcoord[2] * (3.0 * tcoord[2] - 1.0));
        DbetaN[7] = 0.0;
        DbetaN[8] = 0.0;
        DbetaN[9] = 9.0 / 2.0 * (6.0 * tcoord[0] * tcoord[2]);

        DgammaN[0] = 0.0;
        DgammaN[1] = 0.0;
        DgammaN[2] = 9.0 / 2.0 * (2.0 / 9.0 - 2.0 * tcoord[2] + 3.0 * tcoord[2] * tcoord[2]);
        DgammaN[3] = 0.0;
        DgammaN[4] = 0.0;
        DgammaN[5] = 9.0 / 2.0 * (tcoord[1] * (3.0 * tcoord[1] - 1.0));
        DgammaN[6] = 9.0 / 2.0 * (tcoord[1] * (6.0 * tcoord[2] - 1.0));
        DgammaN[7] = 9.0 / 2.0 * (tcoord[0] * (6.0 * tcoord[2] - 1.0));
        DgammaN[8] = 9.0 / 2.0 * (tcoord[0] * (3.0 * tcoord[0] - 1.0));
        DgammaN[9] = 9.0 / 2.0 * (6.0 * tcoord[0] * tcoord[1]);

        break;
    default:
        return -2;
    }

    return 1;
}

// Returns gradient of shape functions for triangles with respect to physical coordinates
int getTriagShapeFunGradPhys(const int np, const double tcoord[3],
                             const double *x, const double *y, double *DxN, double *DyN, double &Jdet)
{

    // Choose element type
    switch (np)
    {
    case 3: // linear triangle
    {
        // Jacobian corresponds to triangle's area
        Jdet = 1.0 / 2.0 * fabs((x[1] * y[2] - x[2] * y[1]) - (x[0] * y[2] - x[2] * y[0]) - (x[1] * y[0] - x[0] * y[1]));

        // Coefficients of basis functions
        // double a1 = x(2)*y(3)-x(3)*y(2);
        // double a2 = x(3)*y(1)-x(1)*y(3);
        // double a3 = x(1)*y(2)-x(2)*y(1);
        double b1 = y[1] - y[2];
        double b2 = y[2] - y[0];
        double b3 = y[0] - y[1];
        double c1 = x[2] - x[1];
        double c2 = x[0] - x[2];
        double c3 = x[1] - x[0];

        // Derivatives of the basis functions(DxN1 - derivative of N1 w.r.t. x, etc.)
        DxN[0] = b1 / (2.0 * Jdet);
        DxN[1] = b2 / (2.0 * Jdet);
        DxN[2] = b3 / (2.0 * Jdet);
        DyN[0] = c1 / (2.0 * Jdet);
        DyN[1] = c2 / (2.0 * Jdet);
        DyN[2] = c3 / (2.0 * Jdet);

        break;
    }

    case 6: // quadratic triangle
    {
        // Jacobian
        double dx4 = x[3] - (x[0] + x[1]) / 2.0;
        double dx5 = x[4] - (x[1] + x[2]) / 2.0;
        double dx6 = x[5] - (x[2] + x[0]) / 2.0;
        double dy4 = y[3] - (y[0] + y[1]) / 2.0;
        double dy5 = y[4] - (y[1] + y[2]) / 2.0;
        double dy6 = y[5] - (y[2] + y[0]) / 2.0;
        double Jx21 = x[1] - x[0] + 4.0 * (dx4 * (tcoord[0] - tcoord[1]) + (dx5 - dx6) * tcoord[2]);
        double Jx32 = x[2] - x[1] + 4.0 * (dx5 * (tcoord[1] - tcoord[2]) + (dx6 - dx4) * tcoord[0]);
        double Jx13 = x[0] - x[2] + 4.0 * (dx6 * (tcoord[2] - tcoord[0]) + (dx4 - dx5) * tcoord[1]);
        double Jy12 = y[0] - y[1] + 4.0 * (dy4 * (tcoord[1] - tcoord[0]) + (dy6 - dy5) * tcoord[2]);
        double Jy23 = y[1] - y[2] + 4.0 * (dy5 * (tcoord[2] - tcoord[1]) + (dy4 - dy6) * tcoord[0]);
        double Jy31 = y[2] - y[0] + 4.0 * (dy6 * (tcoord[0] - tcoord[2]) + (dy5 - dy4) * tcoord[1]);
        Jdet = 1.0 / 2.0 * (Jx21 * Jy31 - Jy12 * Jx13);

        // Derivatives of the basis functions(DxN1 - derivative of N1 w.r.t. x)
        DxN[0] = (4.0 * tcoord[0] - 1.0) * Jy23 / (2.0 * Jdet);
        DxN[1] = (4.0 * tcoord[1] - 1.0) * Jy31 / (2.0 * Jdet);
        DxN[2] = (4.0 * tcoord[2] - 1.0) * Jy12 / (2.0 * Jdet);
        DxN[3] = 4.0 * (tcoord[1] * Jy23 + tcoord[0] * Jy31) / (2.0 * Jdet);
        DxN[4] = 4.0 * (tcoord[2] * Jy31 + tcoord[1] * Jy12) / (2.0 * Jdet);
        DxN[5] = 4.0 * (tcoord[0] * Jy12 + tcoord[2] * Jy23) / (2.0 * Jdet);
        DyN[0] = (4.0 * tcoord[0] - 1.0) * Jx32 / (2.0 * Jdet);
        DyN[1] = (4.0 * tcoord[1] - 1.0) * Jx13 / (2.0 * Jdet);
        DyN[2] = (4.0 * tcoord[2] - 1.0) * Jx21 / (2.0 * Jdet);
        DyN[3] = 4.0 * (tcoord[1] * Jx32 + tcoord[0] * Jx13) / (2.0 * Jdet);
        DyN[4] = 4.0 * (tcoord[2] * Jx13 + tcoord[1] * Jx21) / (2.0 * Jdet);
        DyN[5] = 4.0 * (tcoord[0] * Jx21 + tcoord[2] * Jx32) / (2.0 * Jdet);

        break;
    }

    case 10: // cubic triangle
    {
        // Jacobian
        double dx4 = x[3] - (2.0 * x[0] + x[1]) / 3.0;
        double dx5 = x[4] - (x[0] + 2.0 * x[1]) / 3.0;
        double dx6 = x[5] - (2.0 * x[1] + x[2]) / 3.0;
        double dx7 = x[6] - (x[1] + 2.0 * x[2]) / 3.0;
        double dx8 = x[7] - (2.0 * x[2] + x[0]) / 3.0;
        double dx9 = x[8] - (x[2] + 2.0 * x[0]) / 3.0;
        double dx10 = x[9] - (x[0] + x[1] + x[2]) / 3.0;
        double dy4 = y[3] - (2.0 * y[0] + y[1]) / 3.0;
        double dy5 = y[4] - (y[0] + 2.0 * y[1]) / 3.0;
        double dy6 = y[5] - (2.0 * y[1] + y[2]) / 3.0;
        double dy7 = y[6] - (y[1] + 2.0 * y[2]) / 3.0;
        double dy8 = y[7] - (2.0 * y[2] + y[0]) / 3.0;
        double dy9 = y[8] - (y[2] + 2.0 * y[0]) / 3.0;
        double dy10 = y[9] - (y[0] + y[1] + y[2]) / 3.0;
        double Jx21 = x[1] - x[0] + 9.0 / 2.0 * (dx4 * (tcoord[0] * (3.0 * tcoord[0] - 6.0 * tcoord[1] - 1.0) + tcoord[1]) + dx5 * (tcoord[1] * (1.0 - 3.0 * tcoord[1] + 6.0 * tcoord[0]) - tcoord[0]) + dx6 * tcoord[2] * (6.0 * tcoord[1] - 1.0) + dx7 * tcoord[2] * (3.0 * tcoord[2] - 1.0) + dx8 * tcoord[2] * (1.0 - 3.0 * tcoord[2]) + dx9 * tcoord[2] * (1.0 - 6.0 * tcoord[0]) + 6.0 * dx10 * tcoord[2] * (tcoord[0] - tcoord[1]));
        double Jx32 = x[2] - x[1] + 9.0 / 2.0 * (dx4 * tcoord[0] * (1.0 - 3.0 * tcoord[0]) + dx5 * tcoord[0] * (1.0 - 6.0 * tcoord[1]) + dx6 * (tcoord[1] * (3.0 * tcoord[1] - 6.0 * tcoord[2] - 1.0) + tcoord[2]) + dx7 * (tcoord[2] * (1.0 - 3.0 * tcoord[2] + 6.0 * tcoord[1]) - tcoord[1]) + dx8 * tcoord[0] * (6.0 * tcoord[2] - 1.0) + dx9 * tcoord[0] * (3.0 * tcoord[0] - 1.0) + 6.0 * dx10 * tcoord[0] * (tcoord[1] - tcoord[2]));
        double Jx13 = x[0] - x[2] + 9.0 / 2.0 * (dx4 * (6.0 * tcoord[0] - 1.0) * tcoord[1] + dx5 * tcoord[1] * (3.0 * tcoord[1] - 1.0) + dx6 * tcoord[1] * (1.0 - 3.0 * tcoord[1]) + dx7 * tcoord[1] * (1.0 - 6.0 * tcoord[2]) + dx8 * (tcoord[2] * (3.0 * tcoord[2] - 6.0 * tcoord[0] - 1.0) + tcoord[0]) + dx9 * (tcoord[0] * (1.0 - 3.0 * tcoord[0] + 6.0 * tcoord[2]) - tcoord[2]) + 6.0 * dx10 * tcoord[1] * (tcoord[2] - tcoord[0]));
        double Jy12 = y[0] - y[1] - 9.0 / 2.0 * (dy4 * (tcoord[0] * (3.0 * tcoord[0] - 6.0 * tcoord[1] - 1.0) + tcoord[1]) + dy5 * (tcoord[1] * (1.0 - 3.0 * tcoord[1] + 6.0 * tcoord[0]) - tcoord[0]) + dy6 * tcoord[2] * (6.0 * tcoord[1] - 1.0) + dy7 * tcoord[2] * (3.0 * tcoord[2] - 1.0) + dy8 * tcoord[2] * (1.0 - 3.0 * tcoord[2]) + dy9 * tcoord[2] * (1.0 - 6.0 * tcoord[0]) + 6.0 * dy10 * tcoord[2] * (tcoord[0] - tcoord[1]));
        double Jy23 = y[1] - y[2] - 9.0 / 2.0 * (dy4 * tcoord[0] * (1.0 - 3.0 * tcoord[0]) + dy5 * tcoord[0] * (1.0 - 6.0 * tcoord[1]) + dy6 * (tcoord[1] * (3.0 * tcoord[1] - 6.0 * tcoord[2] - 1.0) + tcoord[2]) + dy7 * (tcoord[2] * (1.0 - 3.0 * tcoord[2] + 6.0 * tcoord[1]) - tcoord[1]) + dy8 * tcoord[0] * (6.0 * tcoord[2] - 1.0) + dy9 * tcoord[0] * (3.0 * tcoord[0] - 1.0) + 6.0 * dy10 * tcoord[0] * (tcoord[1] - tcoord[2]));
        double Jy31 = y[2] - y[0] - 9.0 / 2.0 * (dy4 * (6.0 * tcoord[0] - 1.0) * tcoord[1] + dy5 * tcoord[1] * (3.0 * tcoord[1] - 1.0) + dy6 * tcoord[1] * (1.0 - 3.0 * tcoord[1]) + dy7 * tcoord[1] * (1.0 - 6.0 * tcoord[2]) + dy8 * (tcoord[2] * (3.0 * tcoord[2] - 6.0 * tcoord[0] - 1.0) + tcoord[0]) + dy9 * (tcoord[0] * (1.0 - 3.0 * tcoord[0] + 6.0 * tcoord[2]) - tcoord[2]) + 6.0 * dy10 * tcoord[1] * (tcoord[2] - tcoord[0]));
        Jdet = 1.0 / 2.0 * (Jx21 * Jy31 - Jy12 * Jx13);

        // Derivatives of the basis functions(DxN1 - derivative of N1 w.r.t. x)
        DxN[0] = (9.0 / (4.0 * Jdet)) * Jy23 * (2.0 / 9.0 - 2.0 * tcoord[0] + 3.0 * tcoord[0] * tcoord[0]);
        DxN[1] = (9.0 / (4.0 * Jdet)) * Jy31 * (2.0 / 9.0 - 2.0 * tcoord[1] + 3.0 * tcoord[1] * tcoord[1]);
        DxN[2] = (9.0 / (4.0 * Jdet)) * Jy12 * (2.0 / 9.0 - 2.0 * tcoord[2] + 3.0 * tcoord[2] * tcoord[2]);
        DxN[3] = (9.0 / (4.0 * Jdet)) * (Jy31 * tcoord[0] * (3.0 * tcoord[0] - 1.0) + Jy23 * tcoord[1] * (6.0 * tcoord[0] - 1.0));
        DxN[4] = (9.0 / (4.0 * Jdet)) * (Jy23 * tcoord[1] * (3.0 * tcoord[1] - 1.0) + Jy31 * tcoord[0] * (6.0 * tcoord[1] - 1.0));
        DxN[5] = (9.0 / (4.0 * Jdet)) * (Jy12 * tcoord[1] * (3.0 * tcoord[1] - 1.0) + Jy31 * tcoord[2] * (6.0 * tcoord[1] - 1.0));
        DxN[6] = (9.0 / (4.0 * Jdet)) * (Jy31 * tcoord[2] * (3.0 * tcoord[2] - 1.0) + Jy12 * tcoord[1] * (6.0 * tcoord[2] - 1.0));
        DxN[7] = (9.0 / (4.0 * Jdet)) * (Jy23 * tcoord[2] * (3.0 * tcoord[2] - 1.0) + Jy12 * tcoord[0] * (6.0 * tcoord[2] - 1.0));
        DxN[8] = (9.0 / (4.0 * Jdet)) * (Jy12 * tcoord[0] * (3.0 * tcoord[0] - 1.0) + Jy23 * tcoord[2] * (6.0 * tcoord[0] - 1.0));
        DxN[9] = (9.0 / (4.0 * Jdet)) * 6.0 * (Jy12 * tcoord[0] * tcoord[1] + Jy31 * tcoord[0] * tcoord[2] + Jy23 * tcoord[1] * tcoord[2]);
        DyN[0] = (9.0 / (4.0 * Jdet)) * Jx32 * (2.0 / 9.0 - 2.0 * tcoord[0] + 3.0 * tcoord[0] * tcoord[0]);
        DyN[1] = (9.0 / (4.0 * Jdet)) * Jx13 * (2.0 / 9.0 - 2.0 * tcoord[1] + 3.0 * tcoord[1] * tcoord[1]);
        DyN[2] = (9.0 / (4.0 * Jdet)) * Jx21 * (2.0 / 9.0 - 2.0 * tcoord[2] + 3.0 * tcoord[2] * tcoord[2]);
        DyN[3] = (9.0 / (4.0 * Jdet)) * (Jx13 * tcoord[0] * (3.0 * tcoord[0] - 1.0) + Jx32 * tcoord[1] * (6.0 * tcoord[0] - 1.0));
        DyN[4] = (9.0 / (4.0 * Jdet)) * (Jx32 * tcoord[1] * (3.0 * tcoord[1] - 1.0) + Jx13 * tcoord[0] * (6.0 * tcoord[1] - 1.0));
        DyN[5] = (9.0 / (4.0 * Jdet)) * (Jx21 * tcoord[1] * (3.0 * tcoord[1] - 1.0) + Jx13 * tcoord[2] * (6.0 * tcoord[1] - 1.0));
        DyN[6] = (9.0 / (4.0 * Jdet)) * (Jx13 * tcoord[2] * (3.0 * tcoord[2] - 1.0) + Jx21 * tcoord[1] * (6.0 * tcoord[2] - 1.0));
        DyN[7] = (9.0 / (4.0 * Jdet)) * (Jx32 * tcoord[2] * (3.0 * tcoord[2] - 1.0) + Jx21 * tcoord[0] * (6.0 * tcoord[2] - 1.0));
        DyN[8] = (9.0 / (4.0 * Jdet)) * (Jx21 * tcoord[0] * (3.0 * tcoord[0] - 1.0) + Jx32 * tcoord[2] * (6.0 * tcoord[0] - 1.0));
        DyN[9] = (9.0 / (4.0 * Jdet)) * 6.0 * (Jx21 * tcoord[0] * tcoord[1] + Jx13 * tcoord[0] * tcoord[2] + Jx32 * tcoord[1] * tcoord[2]);

        break;
    }

    default:
        return -2;
    }

    return 1;
}

/******************** Quadrangles ********************/

// Gauss integration rule for quadrangles
int getQuadGaussInfo(const int ngauss, double *alpha, double **IP)
{
    int i, j, n = 0;
    double *coord = NULL, *weight = NULL;

    // Choose Gauss integration rule
    switch (ngauss)
    {
    case 1: // 1 Gauss point
    {
        // 1-D integration rule
        n = 1;
        coord = new double[n];
        weight = new double[n];
        coord[0] = 0.0;
        weight[0] = 2.0;
        break;
    }

    case 4: // 4 Gauss points
    {
        // 1-D integration rule
        n = 2;
        coord = new double[n];
        weight = new double[n];
        coord[0] = -1.0 / sqrt(3.0);
        coord[1] = 1.0 / sqrt(3.0);
        weight[0] = 1.0;
        weight[1] = 1.0;
        break;
    }

    case 9: // 9 Gauss points
    {
        // 1-D integration rule
        n = 3;
        coord = new double[n];
        weight = new double[n];
        coord[0] = -sqrt(0.6);
        coord[1] = 0.0;
        coord[2] = sqrt(0.6);
        weight[0] = 5.0 / 9.0;
        weight[1] = 8.0 / 9.0;
        weight[2] = 5.0 / 9.0;
        break;
    }

    case 16: // 16 Gauss points
    {
        // 1-D integration rule
        n = 4;
        coord = new double[n];
        weight = new double[n];
        coord[0] = -0.3399810435848563;
        coord[1] = 0.3399810435848563;
        coord[2] = -0.8611363115940526;
        coord[3] = 0.8611363115940526;
        weight[0] = 0.6521451548625461;
        weight[1] = 0.6521451548625461;
        weight[2] = 0.3478548451374538;
        weight[3] = 0.3478548451374538;
        break;
    }

    case 25: // 25 Gauss points
    {
        // 1-D integration rule
        n = 5;
        coord = new double[n];
        weight = new double[n];
        coord[0] = 0.0;
        coord[1] = -0.5384693101056831;
        coord[2] = 0.5384693101056831;
        coord[3] = -0.9061798459386640;
        coord[4] = 0.9061798459386640;
        weight[0] = 0.5688888888888889;
        weight[1] = 0.4786286704993665;
        weight[2] = 0.4786286704993665;
        weight[3] = 0.2369268850561891;
        weight[4] = 0.2369268850561891;
        break;
    }

    default:
        return -1;
    }

    // 2-D integration rule created from 1-D integration rule
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            IP[n * i + j][0] = coord[i];
            IP[n * i + j][1] = coord[j];
            IP[n * i + j][2] = 0;
            alpha[n * i + j] = weight[i] * weight[j];
        }

    // Clear memory
    delete[] coord;
    delete[] weight;

    return 1;
}

// Returns shape functions evaluated at tcoord for triangular elements
int getQuadShapeFun(const int np, const double tcoord[3], double *N)
{

    // Choose element type
    switch (np)
    {
    case 4: // four node quadrangle
    {
        // Basis functions expressed in terms of natural coordinates
        N[0] = 1.0 / 4.0 * (1.0 - tcoord[0]) * (1.0 - tcoord[1]);
        N[1] = 1.0 / 4.0 * (1.0 + tcoord[0]) * (1.0 - tcoord[1]);
        N[2] = 1.0 / 4.0 * (1.0 + tcoord[0]) * (1.0 + tcoord[1]);
        N[3] = 1.0 / 4.0 * (1.0 - tcoord[0]) * (1.0 + tcoord[1]);

        break;
    }
    case 8: // eight node quadrangle
    {
        // Basis functions expressed in terms of natural coordinates
        N[0] = 1.0 / 4.0 * (1.0 - tcoord[0]) * (1.0 - tcoord[1]) * (-tcoord[0] - tcoord[1] - 1.0);
        N[1] = 1.0 / 4.0 * (1.0 + tcoord[0]) * (1.0 - tcoord[1]) * (tcoord[0] - tcoord[1] - 1.0);
        N[2] = 1.0 / 4.0 * (1.0 + tcoord[0]) * (1.0 + tcoord[1]) * (tcoord[0] + tcoord[1] - 1.0);
        N[3] = 1.0 / 4.0 * (1.0 - tcoord[0]) * (1.0 + tcoord[1]) * (-tcoord[0] + tcoord[1] - 1.0);
        N[4] = 1.0 / 2.0 * (1.0 - tcoord[0] * tcoord[0]) * (1.0 - tcoord[1]);
        N[5] = 1.0 / 2.0 * (1.0 + tcoord[0]) * (1.0 - tcoord[1] * tcoord[1]);
        N[6] = 1.0 / 2.0 * (1.0 - tcoord[0] * tcoord[0]) * (1.0 + tcoord[1]);
        N[7] = 1.0 / 2.0 * (1.0 - tcoord[0]) * (1.0 - tcoord[1] * tcoord[1]);

        break;
    }
    case 9: // nine node quadrangle
    {
        // GMSH: mapping from node numbering to xi- and eta-coordinates
        double node2xi[9] = {-1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0};
        double node2eta[9] = {-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0};

        // Construct shape function evaluations as products of 1D Lagrange polynomials
        for (int i = 0; i < 9; i++)
        {
            N[i] = getLagrange1DPoly(2, node2xi[i], 0, tcoord[0]) *
                   getLagrange1DPoly(2, node2eta[i], 0, tcoord[1]);
        }

        break;
    }
    case 16: // sixteen node quadrangle
    {
        // GMSH: mapping from node numbering to xi- and eta-coordinates
        double node2xi[16] = {-1.0, 1.0, 1.0, -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0, -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0};
        double node2eta[16] = {-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};

        // Construct shape function evaluations as products of 1D Lagrange polynomials
        for (int i = 0; i < 16; i++)
        {
            N[i] = getLagrange1DPoly(3, node2xi[i], 0, tcoord[0]) *
                   getLagrange1DPoly(3, node2eta[i], 0, tcoord[1]);
        }

        break;
    }
    default:
        return -2;
    }
    return 1;
}

// Matrix of derivatives with respect to physical coordinates of basis functions for quadrangles
int getQuadBeTLF(const int np, const double tcoord[3],
                 const double *x, const double *y, double &Jdet, double **Be)
{

    // Initialize some variables
    int i, errorcode;
    double *DxiN = new double[np];
    double *DetaN = new double[np];
    double *DxN = new double[np];
    double *DyN = new double[np];

    // Derivatives of the basis functions with respect to natural coordinates
    errorcode = getQuadShapeFunGradNat(np, tcoord, DxiN, DetaN);

    // Jacobian
    double J11 = 0.0;
    double J21 = 0.0;
    double J12 = 0.0;
    double J22 = 0.0;
    for (i = 0; i < np; i++)
    {
        J11 += DxiN[i] * x[i];
        J12 += DxiN[i] * y[i];
        J21 += DetaN[i] * x[i];
        J22 += DetaN[i] * y[i];
    }
    Jdet = J11 * J22 - J12 * J21;
    if (Jdet < 0)
        return -3;

    // Derivatives of basis functions (DxN - derivative w.r.t. x, DyN - derivatives w.r.t. y)
    for (i = 0; i < np; i++)
    {
        DxN[i] = (J22 * DxiN[i] - J12 * DetaN[i]) / Jdet;
        DyN[i] = (-J21 * DxiN[i] + J11 * DetaN[i]) / Jdet;
    }

    // Matrix of derivatives of basis functions
    for (i = 0; i < np; i++)
    {
        Be[0][2 * i + 0] = DxN[i];
        Be[0][2 * i + 1] = 0.0;
        Be[1][2 * i + 0] = 0.0;
        Be[1][2 * i + 1] = DxN[i];
        Be[2][2 * i + 0] = DyN[i];
        Be[2][2 * i + 1] = 0.0;
        Be[3][2 * i + 0] = 0.0;
        Be[3][2 * i + 1] = DyN[i];
    }

    // Clear memory
    delete[] DxiN;
    delete[] DetaN;
    delete[] DxN;
    delete[] DyN;

    return errorcode;
}

// Matrix of derivatives with respect to physical coordinates of basis functions for quadrangles
int getQuadBeTLE(const int np, const double tcoord[3],
                 const double *x, const double *y, const double *uel, double &Jdet,
                 double **Be, double **Bf, double F[4])
{

    // Initialize some variables
    int i, errorcode;
    double *DxiN = new double[np];
    double *DetaN = new double[np];
    double *DxN = new double[np];
    double *DyN = new double[np];

    // Derivatives of the basis functions with respect to natural coordinates
    errorcode = getQuadShapeFunGradNat(np, tcoord, DxiN, DetaN);

    // Jacobian
    double J11 = 0.0;
    double J21 = 0.0;
    double J12 = 0.0;
    double J22 = 0.0;
    for (i = 0; i < np; i++)
    {
        J11 += DxiN[i] * x[i];
        J12 += DxiN[i] * y[i];
        J21 += DetaN[i] * x[i];
        J22 += DetaN[i] * y[i];
    }
    Jdet = J11 * J22 - J12 * J21;
    if (Jdet < 0)
        return -3;

    // Derivatives of basis functions (DxN - derivative w.r.t. x, DyN - derivatives w.r.t. y)
    for (i = 0; i < np; i++)
    {
        DxN[i] = (J22 * DxiN[i] - J12 * DetaN[i]) / Jdet;
        DyN[i] = (-J21 * DxiN[i] + J11 * DetaN[i]) / Jdet;
    }

    // Bf matrix
    for (i = 0; i < np; i++)
    {
        Bf[0][2 * i + 0] = DxN[i];
        Bf[0][2 * i + 1] = 0.0;
        Bf[1][2 * i + 0] = 0.0;
        Bf[1][2 * i + 1] = DxN[i];
        Bf[2][2 * i + 0] = DyN[i];
        Bf[2][2 * i + 1] = 0.0;
        Bf[3][2 * i + 0] = 0.0;
        Bf[3][2 * i + 1] = DyN[i];
    }

    // Get F
    getF2d(Bf, uel, np, F);

    // Be matrix
    for (i = 0; i < np; i++)
    {
        Be[0][2 * i + 0] = F[0] * DxN[i];
        Be[1][2 * i + 0] = F[2] * DyN[i];
        Be[2][2 * i + 0] = F[0] * DyN[i] + F[2] * DxN[i];
        Be[0][2 * i + 1] = F[1] * DxN[i];
        Be[1][2 * i + 1] = F[3] * DyN[i];
        Be[2][2 * i + 1] = F[3] * DxN[i] + F[1] * DyN[i];
    }

    // Clear memory
    delete[] DxN;
    delete[] DyN;

    return errorcode;
}

// Convert coordinates from natural to physical for quadrangles
int ConvertQuadNat2Phys(const int np, const double tcoord[3],
                        const double *x, const double *y, double &X, double &Y)
{
    int errorcode;
    double *N = new double[np];

    // Get basis functions evaluations
    errorcode = getQuadShapeFun(np, tcoord, N);

    // Physical coordinates
    X = 0.0;
    Y = 0.0;
    for (int i = 0; i < np; i++)
    {
        X += N[i] * x[i];
        Y += N[i] * y[i];
    }

    // Clear memory
    delete[] N;

    return errorcode;
}

// Convert coordinates from physical to natural for quadrangles
int ConvertQuadPhys2Nat(const int np, const double X, const double Y,
                        const double *x, const double *y, const double TOL_g, double tcoord[3])
{

    // Initialize variables
    int i, j, errorcode;
    double f[2], J[2][2], IJ[2][2], dtcoord[2], detJ;
    double *N = new double[np];
    double *DxiN = new double[np];
    double *DetaN = new double[np];

    // Get an initial guess (explicit solution to 4-noded quadrangle):
    // based on paper Ch. Hua, An inverse transformation for quadrilateral isoparametric elements: Analysis and application
    double d1 = 4.0 * X - (x[0] + x[1] + x[2] + x[3]);
    double d2 = 4.0 * Y - (y[0] + y[1] + y[2] + y[3]);
    double a1 = +x[0] - x[1] + x[2] - x[3];
    double a2 = +y[0] - y[1] + y[2] - y[3];
    double b1 = -x[0] + x[1] + x[2] - x[3];
    double b2 = -y[0] + y[1] + y[2] - y[3];
    double c1 = -x[0] - x[1] + x[2] + x[3];
    double c2 = -y[0] - y[1] + y[2] + y[3];

    // Find the inverse mapping, distinguish several cases
    double ab = a1 * b2 - b1 * a2;
    double bc = b1 * c2 - c1 * b2;
    double bd = b1 * d2 - d1 * b2;
    double ad = a1 * d2 - d1 * a2;
    double ac = a1 * c2 - c1 * a2;
    double dc = d1 * c2 - c1 * d2;

    // Distinguish 6 cases
    // Cases 1, 2, 3
    if ((fabs(a1 * a2 * ab * ac) > 0.0) ||
        (fabs(a1) < TOL_g && fabs(a2 * c1) > TOL_g) ||
        (fabs(a2) < TOL_g && fabs(a1 * b2) > TOL_g))
    {

        // Solve quadratic equation in tcoord[0]
        double a = ab;
        double b = -bc - ad;
        double c = dc;
        double D = sqrt(pow(b, 2) - 4.0 * a * c);
        double t1 = (-b + D) / (2.0 * a);
        double t2 = (-b - D) / (2.0 * a);
        if (fabs(t1) < 1.0 + TOL_g)
            tcoord[0] = t1;
        else
            tcoord[0] = t2;
        tcoord[1] = (ad - ab * tcoord[0]) / ac;
    }

    // Case 4
    else if ((fabs(a1 * a2) > 0.0) && (fabs(ab) < TOL_g))
    {
        tcoord[0] = a1 * dc / (b1 * ac + a1 * ad);
        tcoord[1] = ad / ac;
    }

    // Case 5
    else if (fabs(a1 * a2) > 0.0 && fabs(ac) < TOL_g)
    {
        tcoord[0] = ad / ab;
        tcoord[1] = -a1 * bd / (c1 * ab + a1 * ad);
    }

    // Case 6
    else
    {
        tcoord[0] = dc / (a1 * d2 + bc);
        tcoord[1] = bd / (a2 * d1 + bc);
    }
    tcoord[2] = 0.0;

    // Get shape function evaluations at current tcoord
    errorcode = getQuadShapeFun(np, tcoord, N);

    // Update the residual vector
    for (i = 0; i < 2; i++)
        f[i] = 0.0; // initialize f
    for (i = 0; i < np; i++)
    { // compute f
        f[0] += x[i] * N[i];
        f[1] += y[i] * N[i];
    }
    f[0] -= X;
    f[1] -= Y;

    // Update error
    double eps = 0.0;
    for (i = 0; i < 2; i++)
        eps += fabs(f[i]);

    // Newton loop
    int Niter = 0;
    int Maxiter = 100;
    while (eps > TOL_g && Niter < Maxiter)
    {
        Niter++;

        // Get derivatives of basis functions with respect to natural coordinates evaluated at current tcoord
        errorcode = getQuadShapeFunGradNat(np, tcoord, DxiN, DetaN);

        // Jacobian of the nonlinear system of equations f = 0
        for (i = 0; i < 2; i++)
            for (j = 0; j < 2; j++)
                J[i][j] = 0.0;
        for (i = 0; i < np; i++)
        {
            J[0][0] += x[i] * DxiN[i];
            J[0][1] += x[i] * DetaN[i];

            J[1][0] += y[i] * DxiN[i];
            J[1][1] += y[i] * DetaN[i];
        }

        // Inverse Jacobian
        if (!getInverse22(J, detJ, IJ))
            break;

        // Solve the system tcoord = tcoord - J\f
        for (i = 0; i < 2; i++)
        {
            dtcoord[i] = 0.0;
            for (j = 0; j < 2; j++)
                dtcoord[i] -= IJ[i][j] * f[j];
        }

        // Update tcoord
        for (i = 0; i < 2; i++)
            tcoord[i] += dtcoord[i];

        // Get shape function evaluations at current tcoord
        errorcode = getQuadShapeFun(np, tcoord, N);

        // Update the residual vector
        for (i = 0; i < 2; i++)
            f[i] = 0.0; // initialize f
        for (i = 0; i < np; i++)
        { // compute f
            f[0] += x[i] * N[i];
            f[1] += y[i] * N[i];
        }
        f[0] -= X;
        f[1] -= Y;

        // Update error
        eps = 0.0;
        for (i = 0; i < 2; i++)
            eps += fabs(f[i]);
    }

    // Print warning message if maximum number of iterations was exceeded
    if (Niter >= Maxiter)
    {
        // printf("ConvertTriagCoordinates: %i Nwtn. iter. exceeded\n", Maxiter);
        for (i = 0; i < 3; i++)
            tcoord[i] = NAN; // return NANs
        errorcode = -5;
    }
    else
    { // check if given point is inside element or not
        if ((fabs(tcoord[0]) < 1.0 + TOL_g) && (fabs(tcoord[1]) < 1.0 + TOL_g))
        {
            // If it is inside, do nothing
        }
        else
        {
            for (i = 0; i < 3; i++)
                tcoord[i] = NAN; // return NANs
        }
    }

    // Clear memory
    delete[] N;
    delete[] DxiN;
    delete[] DetaN;

    return errorcode;
}

// Returns gradient of shape functions for quadrangles with respect to natural coordinates
int getQuadShapeFunGradNat(const int np, const double tcoord[3], double *DxiN, double *DetaN)
{

    // Choose element type
    switch (np)
    {
    case 4: // four node quadrangle
    {
        // Derivatives of the basis functions (DxiN - derivatives w.r.t. xi, DetaN - derivatives w.r.t eta)
        DxiN[0] = -(1.0 - tcoord[1]) / 4.0;
        DxiN[1] = (1.0 - tcoord[1]) / 4.0;
        DxiN[2] = (1.0 + tcoord[1]) / 4.0;
        DxiN[3] = -(1.0 + tcoord[1]) / 4.0;

        DetaN[0] = -(1.0 - tcoord[0]) / 4.0;
        DetaN[1] = -(1.0 + tcoord[0]) / 4.0;
        DetaN[2] = (1.0 + tcoord[0]) / 4.0;
        DetaN[3] = (1.0 - tcoord[0]) / 4.0;

        break;
    }
    case 8: // eight node quadrangle
    {
        // Derivatives of the basis functions (DxiN - derivatives w.r.t. xi, DetaN - derivatives w.r.t eta)
        DxiN[0] = (1.0 - tcoord[1]) * (2.0 * tcoord[0] + tcoord[1]) / 4.0;
        DxiN[1] = (1.0 - tcoord[1]) * (2.0 * tcoord[0] - tcoord[1]) / 4.0;
        DxiN[2] = (1.0 + tcoord[1]) * (2.0 * tcoord[0] + tcoord[1]) / 4.0;
        DxiN[3] = (1.0 + tcoord[1]) * (2.0 * tcoord[0] - tcoord[1]) / 4.0;
        DxiN[4] = tcoord[0] * (tcoord[1] - 1.0);
        DxiN[5] = (1.0 - tcoord[1] * tcoord[1]) / 2.0;
        DxiN[6] = -tcoord[0] * (1.0 + tcoord[1]);
        DxiN[7] = -(1.0 - tcoord[1] * tcoord[1]) / 2.0;

        DetaN[0] = (1.0 - tcoord[0]) * (tcoord[0] + 2.0 * tcoord[1]) / 4.0;
        DetaN[1] = -(1.0 + tcoord[0]) * (tcoord[0] - 2.0 * tcoord[1]) / 4.0;
        DetaN[2] = (1.0 + tcoord[0]) * (tcoord[0] + 2.0 * tcoord[1]) / 4.0;
        DetaN[3] = -(1.0 - tcoord[0]) * (tcoord[0] - 2.0 * tcoord[1]) / 4.0;
        DetaN[4] = -(1.0 - tcoord[0] * tcoord[0]) / 2.0;
        DetaN[5] = -(1.0 + tcoord[0]) * tcoord[1];
        DetaN[6] = (1.0 - tcoord[0] * tcoord[0]) / 2.0;
        DetaN[7] = -(1.0 - tcoord[0]) * tcoord[1];

        break;
    }
    case 9: // nine node quadrangle
    {
        // GMSH: mapping from node numbering to xi- and eta-coordinates
        double node2xi[9] = {-1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0};
        double node2eta[9] = {-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0};

        // Construct derivatives as products of 1D Lagrange polynomials
        for (int i = 0; i < 9; i++)
        {
            DxiN[i] = getLagrange1DPoly(2, node2xi[i], 1, tcoord[0]) *
                      getLagrange1DPoly(2, node2eta[i], 0, tcoord[1]);
            DetaN[i] = getLagrange1DPoly(2, node2xi[i], 0, tcoord[0]) *
                       getLagrange1DPoly(2, node2eta[i], 1, tcoord[1]);
        }

        break;
    }
    case 16: // sixteen node quadrangle
    {
        // GMSH: mapping from node numbering to xi- and eta-coordinates
        double node2xi[16] = {-1.0, 1.0, 1.0, -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0, -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0};
        double node2eta[16] = {-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};

        // Construct derivatives as products of 1D Lagrange polynomials
        for (int i = 0; i < 16; i++)
        {
            DxiN[i] = getLagrange1DPoly(3, node2xi[i], 1, tcoord[0]) *
                      getLagrange1DPoly(3, node2eta[i], 0, tcoord[1]);
            DetaN[i] = getLagrange1DPoly(3, node2xi[i], 0, tcoord[0]) *
                       getLagrange1DPoly(3, node2eta[i], 1, tcoord[1]);
        }

        break;
    }
    default:
        return -2;
    }
    return 1;
}

/******************** Tetrahedrons ********************/

// Gauss integration rule for tetrahedra elements
int getTetraGaussInfo(const int ngauss, double *alpha, double **IP)
{
    int i;

    // Choose Gauss integration rule
    switch (ngauss)
    {
    case 1: // 1 Gauss point in the middle
    {
        for (i = 0; i < 4; i++)
            IP[0][i] = 1.0 / 4.0;
        alpha[0] = 1.0;
        break;
    }

    case 4: // 4 Gauss points inside
    {
        IP[0][0] = 0.5854101966249685;
        IP[0][1] = 0.1381966011250105;
        IP[0][2] = 0.1381966011250105;
        IP[0][3] = 0.1381966011250105;
        IP[1][0] = 0.1381966011250105;
        IP[1][1] = 0.5854101966249685;
        IP[1][2] = 0.1381966011250105;
        IP[1][3] = 0.1381966011250105;
        IP[2][0] = 0.1381966011250105;
        IP[2][1] = 0.1381966011250105;
        IP[2][2] = 0.5854101966249685;
        IP[2][3] = 0.1381966011250105;
        IP[3][0] = 0.1381966011250105;
        IP[3][1] = 0.1381966011250105;
        IP[3][2] = 0.1381966011250105;
        IP[3][3] = 0.5854101966249685;
        alpha[0] = 1.0 / 4.0;
        alpha[1] = 1.0 / 4.0;
        alpha[2] = 1.0 / 4.0;
        alpha[3] = 1.0 / 4.0;
        break;
    }

    case 5: // 5 Gauss points inside
    {
        IP[0][0] = 1.0 / 4.0;
        IP[0][1] = 1.0 / 4.0;
        IP[0][2] = 1.0 / 4.0;
        IP[0][3] = 1.0 / 4.0;
        IP[1][0] = 2.0 / 6.0;
        IP[1][1] = 1.0 / 6.0;
        IP[1][2] = 1.0 / 6.0;
        IP[1][3] = 1.0 / 6.0;
        IP[2][0] = 1.0 / 6.0;
        IP[2][1] = 2.0 / 6.0;
        IP[2][2] = 1.0 / 6.0;
        IP[2][3] = 1.0 / 6.0;
        IP[3][0] = 1.0 / 6.0;
        IP[3][1] = 1.0 / 6.0;
        IP[3][2] = 2.0 / 6.0;
        IP[3][3] = 1.0 / 6.0;
        IP[4][0] = 1.0 / 6.0;
        IP[4][1] = 1.0 / 6.0;
        IP[4][2] = 1.0 / 6.0;
        IP[4][3] = 2.0 / 6.0;
        alpha[0] = -4.0 / 5.0;
        alpha[1] = 9.0 / 20.0;
        alpha[2] = 9.0 / 20.0;
        alpha[3] = 9.0 / 20.0;
        alpha[4] = 9.0 / 20.0;
        break;
    }

    case 11: // 11 Gauss point in the middle
    {
        IP[0][0] = 0.2500000000000000;
        IP[0][1] = 0.2500000000000000;
        IP[0][2] = 0.2500000000000000;
        IP[0][3] = 0.2500000000000000;
        IP[1][0] = 0.7857142857142857;
        IP[1][1] = 0.0714285714285714;
        IP[1][2] = 0.0714285714285714;
        IP[1][3] = 0.0714285714285714;
        IP[2][0] = 0.0714285714285714;
        IP[2][1] = 0.0714285714285714;
        IP[2][2] = 0.0714285714285714;
        IP[2][3] = 0.7857142857142857;
        IP[3][0] = 0.0714285714285714;
        IP[3][1] = 0.0714285714285714;
        IP[3][2] = 0.7857142857142857;
        IP[3][3] = 0.0714285714285714;
        IP[4][0] = 0.0714285714285714;
        IP[4][1] = 0.7857142857142857;
        IP[4][2] = 0.0714285714285714;
        IP[4][3] = 0.0714285714285714;
        IP[5][0] = 0.1005964238332008;
        IP[5][1] = 0.3994035761667992;
        IP[5][2] = 0.3994035761667992;
        IP[5][3] = 0.1005964238332008;
        IP[6][0] = 0.3994035761667992;
        IP[6][1] = 0.1005964238332008;
        IP[6][2] = 0.3994035761667992;
        IP[6][3] = 0.1005964238332008;
        IP[7][0] = 0.3994035761667992;
        IP[7][1] = 0.3994035761667992;
        IP[7][2] = 0.1005964238332008;
        IP[7][3] = 0.1005964238332008;
        IP[8][0] = 0.3994035761667992;
        IP[8][1] = 0.1005964238332008;
        IP[8][2] = 0.1005964238332008;
        IP[8][3] = 0.3994035761667992;
        IP[9][0] = 0.1005964238332008;
        IP[9][1] = 0.3994035761667992;
        IP[9][2] = 0.1005964238332008;
        IP[9][3] = 0.3994035761667992;
        IP[10][0] = 0.1005964238332008;
        IP[10][1] = 0.1005964238332008;
        IP[10][2] = 0.3994035761667992;
        IP[10][3] = 0.3994035761667992;
        alpha[0] = -0.0789333333333333;
        alpha[1] = 0.0457333333333333;
        alpha[2] = 0.0457333333333333;
        alpha[3] = 0.0457333333333333;
        alpha[4] = 0.0457333333333333;
        alpha[5] = 0.1493333333333333;
        alpha[6] = 0.1493333333333333;
        alpha[7] = 0.1493333333333333;
        alpha[8] = 0.1493333333333333;
        alpha[9] = 0.1493333333333333;
        alpha[10] = 0.1493333333333333;

        break;
    }

    case 15: // 15 Gauss point in the middle
    {
        IP[0][0] = 0.2500000000000000;
        IP[0][1] = 0.2500000000000000;
        IP[0][2] = 0.2500000000000000;
        IP[0][3] = 0.2500000000000000;
        IP[1][0] = 0.0000000000000000;
        IP[1][1] = 0.3333333333333333;
        IP[1][2] = 0.3333333333333333;
        IP[1][3] = 0.3333333333333333;
        IP[2][0] = 0.3333333333333333;
        IP[2][1] = 0.3333333333333333;
        IP[2][2] = 0.3333333333333333;
        IP[2][3] = 0.0000000000000000;
        IP[3][0] = 0.3333333333333333;
        IP[3][1] = 0.3333333333333333;
        IP[3][2] = 0.0000000000000000;
        IP[3][3] = 0.3333333333333333;
        IP[4][0] = 0.3333333333333333;
        IP[4][1] = 0.0000000000000000;
        IP[4][2] = 0.3333333333333333;
        IP[4][3] = 0.3333333333333333;
        IP[5][0] = 0.7272727272727273;
        IP[5][1] = 0.0909090909090909;
        IP[5][2] = 0.0909090909090909;
        IP[5][3] = 0.0909090909090909;
        IP[6][0] = 0.0909090909090909;
        IP[6][1] = 0.0909090909090909;
        IP[6][2] = 0.0909090909090909;
        IP[6][3] = 0.7272727272727273;
        IP[7][0] = 0.0909090909090909;
        IP[7][1] = 0.0909090909090909;
        IP[7][2] = 0.7272727272727273;
        IP[7][3] = 0.0909090909090909;
        IP[8][0] = 0.0909090909090909;
        IP[8][1] = 0.7272727272727273;
        IP[8][2] = 0.0909090909090909;
        IP[8][3] = 0.0909090909090909;
        IP[9][0] = 0.4334498464263357;
        IP[9][1] = 0.0665501535736643;
        IP[9][2] = 0.0665501535736643;
        IP[9][3] = 0.4334498464263357;
        IP[10][0] = 0.0665501535736643;
        IP[10][1] = 0.4334498464263357;
        IP[10][2] = 0.0665501535736643;
        IP[10][3] = 0.4334498464263357;
        IP[11][0] = 0.0665501535736643;
        IP[11][1] = 0.0665501535736643;
        IP[11][2] = 0.4334498464263357;
        IP[11][3] = 0.4334498464263357;
        IP[12][0] = 0.0665501535736643;
        IP[12][1] = 0.4334498464263357;
        IP[12][2] = 0.4334498464263357;
        IP[12][3] = 0.0665501535736643;
        IP[13][0] = 0.4334498464263357;
        IP[13][1] = 0.0665501535736643;
        IP[13][2] = 0.4334498464263357;
        IP[13][3] = 0.0665501535736643;
        IP[14][0] = 0.4334498464263357;
        IP[14][1] = 0.4334498464263357;
        IP[14][2] = 0.0665501535736643;
        IP[14][3] = 0.0665501535736643;
        alpha[0] = 0.1817020685825351;
        alpha[1] = 0.0361607142857143;
        alpha[2] = 0.0361607142857143;
        alpha[3] = 0.0361607142857143;
        alpha[4] = 0.0361607142857143;
        alpha[5] = 0.0698714945161738;
        alpha[6] = 0.0698714945161738;
        alpha[7] = 0.0698714945161738;
        alpha[8] = 0.0698714945161738;
        alpha[9] = 0.0656948493683187;
        alpha[10] = 0.0656948493683187;
        alpha[11] = 0.0656948493683187;
        alpha[12] = 0.0656948493683187;
        alpha[13] = 0.0656948493683187;
        alpha[14] = 0.0656948493683187;

        break;
    }

    default:
        return -1;
    }
    return 1;
}

// Returns shape functions evaluated at tcoord for tetrahedrons
int getTetraShapeFun(const int np, const double tcoord[4], double *N)
{

    // Choose element type
    switch (np)
    {
    case 4: // linear tetrahedron

        // Basis functions expressed in terms of volumetric coordinates
        N[0] = tcoord[0];
        N[1] = tcoord[1];
        N[2] = tcoord[2];
        N[3] = tcoord[3];

        break;

    case 10: // quadratic tetrahedron

        // Basis functions expressed in terms of volumetric coordinates
        N[0] = tcoord[0] * (2.0 * tcoord[0] - 1.0);
        N[1] = tcoord[1] * (2.0 * tcoord[1] - 1.0);
        N[2] = tcoord[2] * (2.0 * tcoord[2] - 1.0);
        N[3] = tcoord[3] * (2.0 * tcoord[3] - 1.0);
        N[4] = 4.0 * tcoord[0] * tcoord[1];
        N[5] = 4.0 * tcoord[1] * tcoord[2];
        N[6] = 4.0 * tcoord[2] * tcoord[0];
        N[7] = 4.0 * tcoord[0] * tcoord[3];
        N[8] = 4.0 * tcoord[2] * tcoord[3];
        N[9] = 4.0 * tcoord[1] * tcoord[3];

        break;
    default:
        return -2;
    }
    return 1;
}

// Matrix of derivatives with respect to physical coordinates of basis functions for tetrahedrons
int getTetraBeTLF(const int np, const double tcoord[4],
                  const double *x, const double *y, const double *z, double &Jdet,
                  double **Be)
{
    int i, j, k, errorcode;
    double *DxiN = new double[np];
    double *DetaN = new double[np];
    double *DzetaN = new double[np];
    double *DgammaN = new double[np];
    double *DxN = new double[np];
    double *DyN = new double[np];
    double *DzN = new double[np];
    double J[4][4];
    double IJ[4][4];
    double detJ;

    // Derivatives of the basis functions with respect to natural coordinates
    errorcode = getTetraShapeFunGradNat(np, tcoord, DxiN, DetaN, DzetaN, DgammaN);

    // Jacobian
    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            J[i][j] = 0.0;
        }
    }
    for (i = 0; i < np; i++)
    {
        J[0][0] = 1.0;
        J[0][1] = 1.0;
        J[0][2] = 1.0;
        J[0][3] = 1.0;
        J[1][0] += DxiN[i] * x[i];
        J[1][1] += DetaN[i] * x[i];
        J[1][2] += DzetaN[i] * x[i];
        J[1][3] += DgammaN[i] * x[i];
        J[2][0] += DxiN[i] * y[i];
        J[2][1] += DetaN[i] * y[i];
        J[2][2] += DzetaN[i] * y[i];
        J[2][3] += DgammaN[i] * y[i];
        J[3][0] += DxiN[i] * z[i];
        J[3][1] += DetaN[i] * z[i];
        J[3][2] += DzetaN[i] * z[i];
        J[3][3] += DgammaN[i] * z[i];
    }
    if (!getInverse44(J, detJ, IJ))
        return -4;
    if (detJ < 0)
        return -3;
    Jdet = detJ / 6.0;

    // Solve for tetrahedron coordinate partials
    double P[4][3];
    double Iaug[4][3];
    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 3; j++)
        {
            P[i][j] = 0.0;
            Iaug[i][j] = 0.0;
        }
    }
    Iaug[1][0] = 1.0;
    Iaug[2][1] = 1.0;
    Iaug[3][2] = 1.0;
    for (i = 0; i < 4; i++)
        for (j = 0; j < 3; j++)
            for (k = 0; k < 4; k++)
                P[i][j] += IJ[i][k] * Iaug[k][j];

    // Derivatives of basis functions (DxN - derivative w.r.t. x, DyN - derivatives w.r.t. y, DzN - derivatives w.r.t. z)
    for (i = 0; i < np; i++)
    {
        DxN[i] = P[0][0] * DxiN[i] + P[1][0] * DetaN[i] + P[2][0] * DzetaN[i] + P[3][0] * DgammaN[i];
        DyN[i] = P[0][1] * DxiN[i] + P[1][1] * DetaN[i] + P[2][1] * DzetaN[i] + P[3][1] * DgammaN[i];
        DzN[i] = P[0][2] * DxiN[i] + P[1][2] * DetaN[i] + P[2][2] * DzetaN[i] + P[3][2] * DgammaN[i];
    }

    // Matrix of derivatives of basis functions
    for (i = 0; i < np; i++)
    {
        Be[0][3 * i + 0] = DxN[i];
        Be[0][3 * i + 1] = 0.0;
        Be[0][3 * i + 2] = 0.0;

        Be[1][3 * i + 0] = 0.0;
        Be[1][3 * i + 1] = DxN[i];
        Be[1][3 * i + 2] = 0.0;

        Be[2][3 * i + 0] = 0.0;
        Be[2][3 * i + 1] = 0.0;
        Be[2][3 * i + 2] = DxN[i];

        Be[3][3 * i + 0] = DyN[i];
        Be[3][3 * i + 1] = 0.0;
        Be[3][3 * i + 2] = 0.0;

        Be[4][3 * i + 0] = 0.0;
        Be[4][3 * i + 1] = DyN[i];
        Be[4][3 * i + 2] = 0.0;

        Be[5][3 * i + 0] = 0.0;
        Be[5][3 * i + 1] = 0.0;
        Be[5][3 * i + 2] = DyN[i];

        Be[6][3 * i + 0] = DzN[i];
        Be[6][3 * i + 1] = 0.0;
        Be[6][3 * i + 2] = 0.0;

        Be[7][3 * i + 0] = 0.0;
        Be[7][3 * i + 1] = DzN[i];
        Be[7][3 * i + 2] = 0.0;

        Be[8][3 * i + 0] = 0.0;
        Be[8][3 * i + 1] = 0.0;
        Be[8][3 * i + 2] = DzN[i];
    }

    // Clear memory
    delete[] DxiN;
    delete[] DetaN;
    delete[] DzetaN;
    delete[] DgammaN;
    delete[] DxN;
    delete[] DyN;
    delete[] DzN;

    return errorcode;
}

// Matrix of derivatives with respect to physical coordinates of basis functions for tetrahedrons
int getTetraBeTLE(const int np, const double tcoord[4], const double *x,
                  const double *y, const double *z, const double *uel, double &Jdet,
                  double **Be, double **Bf, double F[9])
{

    int i, j, k, errorcode;
    double *DxiN = new double[np];
    double *DetaN = new double[np];
    double *DzetaN = new double[np];
    double *DgammaN = new double[np];
    double *DxN = new double[np];
    double *DyN = new double[np];
    double *DzN = new double[np];
    double J[4][4];
    double IJ[4][4];
    double detJ;

    // Derivatives of the basis functions with respect to natural coordinates
    errorcode = getTetraShapeFunGradNat(np, tcoord, DxiN, DetaN, DzetaN, DgammaN);

    // Jacobian
    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++)
            J[i][j] = 0.0;
    for (i = 0; i < np; i++)
    {
        J[0][0] = 1.0;
        J[0][1] = 1.0;
        J[0][2] = 1.0;
        J[0][3] = 1.0;
        J[1][0] += DxiN[i] * x[i];
        J[1][1] += DetaN[i] * x[i];
        J[1][2] += DzetaN[i] * x[i];
        J[1][3] += DgammaN[i] * x[i];
        J[2][0] += DxiN[i] * y[i];
        J[2][1] += DetaN[i] * y[i];
        J[2][2] += DzetaN[i] * y[i];
        J[2][3] += DgammaN[i] * y[i];
        J[3][0] += DxiN[i] * z[i];
        J[3][1] += DetaN[i] * z[i];
        J[3][2] += DzetaN[i] * z[i];
        J[3][3] += DgammaN[i] * z[i];
    }
    if (!getInverse44(J, detJ, IJ))
        return -4;
    if (detJ < 0)
        return -3;
    Jdet = detJ / 6.0;

    // Solve for tetrahedron coordinate partials
    double P[4][3];
    double Iaug[4][3];
    for (i = 0; i < 4; i++)
        for (j = 0; j < 3; j++)
        {
            P[i][j] = 0.0;
            Iaug[i][j] = 0.0;
        }
    Iaug[1][0] = 1.0;
    Iaug[2][1] = 1.0;
    Iaug[3][2] = 1.0;
    for (i = 0; i < 4; i++)
        for (j = 0; j < 3; j++)
            for (k = 0; k < 4; k++)
                P[i][j] += IJ[i][k] * Iaug[k][j];

    // Derivatives of basis functions (DxN - derivative w.r.t. x, DyN - derivatives w.r.t. y, DzN - derivatives w.r.t. z)
    for (i = 0; i < np; i++)
    {
        DxN[i] = P[0][0] * DxiN[i] + P[1][0] * DetaN[i] + P[2][0] * DzetaN[i] + P[3][0] * DgammaN[i];
        DyN[i] = P[0][1] * DxiN[i] + P[1][1] * DetaN[i] + P[2][1] * DzetaN[i] + P[3][1] * DgammaN[i];
        DzN[i] = P[0][2] * DxiN[i] + P[1][2] * DetaN[i] + P[2][2] * DzetaN[i] + P[3][2] * DgammaN[i];
    }

    // Bf matrix
    for (i = 0; i < np; i++)
    {
        Bf[0][3 * i + 0] = DxN[i];
        Bf[0][3 * i + 1] = 0.0;
        Bf[0][3 * i + 2] = 0.0;

        Bf[1][3 * i + 0] = 0.0;
        Bf[1][3 * i + 1] = DxN[i];
        Bf[1][3 * i + 2] = 0.0;

        Bf[2][3 * i + 0] = 0.0;
        Bf[2][3 * i + 1] = 0.0;
        Bf[2][3 * i + 2] = DxN[i];

        Bf[3][3 * i + 0] = DyN[i];
        Bf[3][3 * i + 1] = 0.0;
        Bf[3][3 * i + 2] = 0.0;

        Bf[4][3 * i + 0] = 0.0;
        Bf[4][3 * i + 1] = DyN[i];
        Bf[4][3 * i + 2] = 0.0;

        Bf[5][3 * i + 0] = 0.0;
        Bf[5][3 * i + 1] = 0.0;
        Bf[5][3 * i + 2] = DyN[i];

        Bf[6][3 * i + 0] = DzN[i];
        Bf[6][3 * i + 1] = 0.0;
        Bf[6][3 * i + 2] = 0.0;

        Bf[7][3 * i + 0] = 0.0;
        Bf[7][3 * i + 1] = DzN[i];
        Bf[7][3 * i + 2] = 0.0;

        Bf[8][3 * i + 0] = 0.0;
        Bf[8][3 * i + 1] = 0.0;
        Bf[8][3 * i + 2] = DzN[i];
    }

    // Get F
    getF3d(Bf, uel, np, F);

    // Be matrix
    for (i = 0; i < np; i++)
    {
        Be[0][3 * i + 0] = F[0] * DxN[i];
        Be[0][3 * i + 1] = F[1] * DxN[i];
        Be[0][3 * i + 2] = F[2] * DxN[i];

        Be[1][3 * i + 0] = F[3] * DyN[i];
        Be[1][3 * i + 1] = F[4] * DyN[i];
        Be[1][3 * i + 2] = F[5] * DyN[i];

        Be[2][3 * i + 0] = F[6] * DzN[i];
        Be[2][3 * i + 1] = F[7] * DzN[i];
        Be[2][3 * i + 2] = F[8] * DzN[i];

        Be[3][3 * i + 0] = F[0] * DyN[i] + F[3] * DxN[i];
        Be[3][3 * i + 1] = F[1] * DyN[i] + F[4] * DxN[i];
        Be[3][3 * i + 2] = F[2] * DyN[i] + F[5] * DxN[i];

        Be[4][3 * i + 0] = F[3] * DzN[i] + F[6] * DyN[i];
        Be[4][3 * i + 1] = F[4] * DzN[i] + F[7] * DyN[i];
        Be[4][3 * i + 2] = F[5] * DzN[i] + F[8] * DyN[i];

        Be[5][3 * i + 0] = F[6] * DxN[i] + F[0] * DzN[i];
        Be[5][3 * i + 1] = F[7] * DxN[i] + F[1] * DzN[i];
        Be[5][3 * i + 2] = F[8] * DxN[i] + F[2] * DzN[i];
    }

    // Clear memory
    delete[] DxiN;
    delete[] DetaN;
    delete[] DzetaN;
    delete[] DgammaN;
    delete[] DxN;
    delete[] DyN;
    delete[] DzN;

    return errorcode;
}

// Convert coordinates from natural to physical for tetrahedrons
int ConvertTetraNat2Phys(const int np, const double tcoord[4],
                         const double *x, const double *y, const double *z, double &X,
                         double &Y, double &Z)
{
    int errorcode;
    double *N = new double[np];

    // Get basis functions evaluations
    errorcode = getTetraShapeFun(np, tcoord, N);

    // Physical coordinates
    X = 0.0;
    Y = 0.0;
    Z = 0.0;
    for (int i = 0; i < np; i++)
    {
        X += N[i] * x[i];
        Y += N[i] * y[i];
        Z += N[i] * z[i];
    }

    // Clear memory
    delete[] N;

    return errorcode;
}

// Convert coordinates from physical to natural for triangles
int ConvertTetraPhys2Nat(const int np, const double X, const double Y,
                         const double Z, const double *x, const double *y, const double *z,
                         const double TOL_g, double tcoord[4])
{

    // Initialize variables
    int i, j, errorcode;
    double f[4], J[4][4], IJ[4][4], dtcoord[4], detT, detJ;
    double *N = new double[np];
    double *DalphaN = new double[np];
    double *DbetaN = new double[np];
    double *DgammaN = new double[np];
    double *DdeltaN = new double[np];

    // Get an initial guess (barycentric coordinates)
    double T[3][3], IT[3][3], r[3];
    r[0] = X - x[3];
    r[1] = Y - y[3];
    r[2] = Z - z[3];
    for (i = 0; i < 3; i++)
    {
        T[0][i] = x[i] - x[3];
        T[1][i] = y[i] - y[3];
        T[2][i] = z[i] - z[3];
    }
    if (!getInverse33(T, detT, IT))
    {
        for (i = 0; i < 4; i++)
            tcoord[i] = NAN;

        // Clean up
        delete[] N;
        delete[] DalphaN;
        delete[] DbetaN;
        delete[] DgammaN;
        delete[] DdeltaN;

        return -4;
    }
    // Solve the system tcoord = T\r
    for (i = 0; i < 3; i++)
    {
        tcoord[i] = 0.0;
        for (j = 0; j < 3; j++)
            tcoord[i] += IT[i][j] * r[j];
    }
    tcoord[3] = 1.0 - tcoord[0] - tcoord[1] - tcoord[2];

    // Get shape function evaluations at current tcoord
    errorcode = getTetraShapeFun(np, tcoord, N);

    // Update the residual vector
    for (i = 0; i < 4; i++)
        f[i] = 0.0; // initialize f
    for (i = 0; i < np; i++)
    { // compute f
        f[0] += 1.0 * N[i];
        f[1] += x[i] * N[i];
        f[2] += y[i] * N[i];
        f[3] += z[i] * N[i];
    }
    f[0] -= 1.0;
    f[1] -= X;
    f[2] -= Y;
    f[3] -= Z;

    // Update error
    double eps = 0.0;
    for (i = 0; i < 4; i++)
        eps += fabs(f[i]);

    // Newton loop
    int Niter = 0;
    int Maxiter = 100;
    while (eps > TOL_g && Niter < Maxiter)
    {
        Niter++;

        // Get derivatives of basis functions with respect to natural coordinates evaluated at current tcoord
        errorcode = getTetraShapeFunGradNat(np, tcoord, DalphaN, DbetaN, DgammaN, DdeltaN);

        // Jacobian of the nonlinear system of equations f = 0
        for (i = 0; i < 4; i++)
            for (j = 0; j < 4; j++)
                J[i][j] = 0.0;
        for (i = 0; i < np; i++)
        {
            J[0][0] += 1.0 * DalphaN[i];
            J[0][1] += 1.0 * DbetaN[i];
            J[0][2] += 1.0 * DgammaN[i];
            J[0][3] += 1.0 * DdeltaN[i];

            J[1][0] += x[i] * DalphaN[i];
            J[1][1] += x[i] * DbetaN[i];
            J[1][2] += x[i] * DgammaN[i];
            J[1][3] += x[i] * DdeltaN[i];

            J[2][0] += y[i] * DalphaN[i];
            J[2][1] += y[i] * DbetaN[i];
            J[2][2] += y[i] * DgammaN[i];
            J[2][3] += y[i] * DdeltaN[i];

            J[3][0] += z[i] * DalphaN[i];
            J[3][1] += z[i] * DbetaN[i];
            J[3][2] += z[i] * DgammaN[i];
            J[3][3] += z[i] * DdeltaN[i];
        }

        // Inverse Jacobian
        if (!getInverse44(J, detJ, IJ))
            break;

        // Solve the system tcoord = tcoord - J\f
        for (i = 0; i < 4; i++)
        {
            dtcoord[i] = 0.0;
            for (j = 0; j < 4; j++)
                dtcoord[i] -= IJ[i][j] * f[j];
        }

        // Update tcoord
        for (i = 0; i < 4; i++)
            tcoord[i] += dtcoord[i];

        // Get shape function evaluations at current tcoord
        errorcode = getTetraShapeFun(np, tcoord, N);

        // Update the residual vector
        for (i = 0; i < 4; i++)
            f[i] = 0.0; // initialize f
        for (i = 0; i < np; i++)
        { // compute f
            f[0] += 1.0 * N[i];
            f[1] += x[i] * N[i];
            f[2] += y[i] * N[i];
            f[3] += z[i] * N[i];
        }
        f[0] -= 1.0;
        f[1] -= X;
        f[2] -= Y;
        f[3] -= Z;

        // Update error
        eps = 0.0;
        for (i = 0; i < 4; i++)
            eps += fabs(f[i]);
    }

    // Print warning message if maximum number of iterations was exceeded
    if (Niter >= Maxiter)
    {
        // printf("ConvertTriagCoordinates: %i Nwtn. iter. exceeded\n", Maxiter);
        for (i = 0; i < 4; i++)
            tcoord[i] = NAN; // return NANs
        errorcode = -5;
    }
    else
    { // check if given point is inside element or not
        if ((tcoord[0] > -TOL_g) && (tcoord[1] > -TOL_g) &&
            (tcoord[2] > -TOL_g) && (tcoord[3] > -TOL_g) &&
            (tcoord[0] < 1.0 + TOL_g) && (tcoord[1] < 1.0 + TOL_g) &&
            (tcoord[2] < 1.0 + TOL_g) && (tcoord[3] < 1.0 + TOL_g))
        {
            // If it is inside, do nothing
        }
        else
        {
            for (i = 0; i < 4; i++)
                tcoord[i] = NAN; // return NANs
        }
    }

    // Clear memory
    delete[] N;
    delete[] DalphaN;
    delete[] DbetaN;
    delete[] DgammaN;
    delete[] DdeltaN;

    return errorcode;
}

// Returns gradient of shape functions for triangles with respect to natural coordinates
int getTetraShapeFunGradNat(const int np, const double tcoord[4],
                            double *DalphaN, double *DbetaN, double *DgammaN, double *DdeltaN)
{

    // Choose element type
    switch (np)
    {
    case 4: // linear tetrahedron

        // Derivatives of basis functions with respect to natural coordinates
        DalphaN[0] = 1.0;
        DalphaN[1] = 0.0;
        DalphaN[2] = 0.0;
        DalphaN[3] = 0.0;

        DbetaN[0] = 0.0;
        DbetaN[1] = 1.0;
        DbetaN[2] = 0.0;
        DbetaN[3] = 0.0;

        DgammaN[0] = 0.0;
        DgammaN[1] = 0.0;
        DgammaN[2] = 1.0;
        DgammaN[3] = 0.0;

        DdeltaN[0] = 0.0;
        DdeltaN[1] = 0.0;
        DdeltaN[2] = 0.0;
        DdeltaN[3] = 1.0;

        break;

    case 10: // quadratic tetrahedron

        // Derivatives of basis functions with respect to natural coordinates
        DalphaN[0] = 4.0 * tcoord[0] - 1.0;
        DalphaN[1] = 0.0;
        DalphaN[2] = 0.0;
        DalphaN[3] = 0.0;
        DalphaN[4] = 4.0 * tcoord[1];
        DalphaN[5] = 0.0;
        DalphaN[6] = 4.0 * tcoord[2];
        DalphaN[7] = 4.0 * tcoord[3];
        DalphaN[8] = 0.0;
        DalphaN[9] = 0.0;

        DbetaN[0] = 0.0;
        DbetaN[1] = 4.0 * tcoord[1] - 1.0;
        DbetaN[2] = 0.0;
        DbetaN[3] = 0.0;
        DbetaN[4] = 4.0 * tcoord[0];
        DbetaN[5] = 4.0 * tcoord[2];
        DbetaN[6] = 0.0;
        DbetaN[7] = 0.0;
        DbetaN[8] = 0.0;
        DbetaN[9] = 4.0 * tcoord[3];

        DgammaN[0] = 0.0;
        DgammaN[1] = 0.0;
        DgammaN[2] = 4.0 * tcoord[2] - 1.0;
        DgammaN[3] = 0.0;
        DgammaN[4] = 0.0;
        DgammaN[5] = 4.0 * tcoord[1];
        DgammaN[6] = 4.0 * tcoord[0];
        DgammaN[7] = 0.0;
        DgammaN[8] = 4.0 * tcoord[3];
        DgammaN[9] = 0.0;

        DdeltaN[0] = 0.0;
        DdeltaN[1] = 0.0;
        DdeltaN[2] = 0.0;
        DdeltaN[3] = 4.0 * tcoord[3] - 1.0;
        DdeltaN[4] = 0.0;
        DdeltaN[5] = 0.0;
        DdeltaN[6] = 0.0;
        DdeltaN[7] = 4.0 * tcoord[0];
        DdeltaN[8] = 4.0 * tcoord[2];
        DdeltaN[9] = 4.0 * tcoord[1];

        break;

    default:
        return -2;
    }
    return 1;
}

/******************** Hexahedrons ********************/

// Gauss integration rule for hexahedra
int getHexaGaussInfo(const int ngauss, double *alpha, double **IP)
{
    int i, j, k, n = 0;
    double *coord = NULL, *weight = NULL;

    // Choose Gauss integration rule
    switch (ngauss)
    {
    case 1: // 1 Gauss point
    {
        // 1-D integration rule
        n = 1;
        coord = new double[n];
        weight = new double[n];
        coord[0] = 0.0;
        weight[0] = 2.0;
        break;
    }

    case 8: // 8 Gauss points
    {
        // 1-D integration rule
        n = 2;
        coord = new double[n];
        weight = new double[n];
        coord[0] = -1.0 / sqrt(3.0);
        coord[1] = 1.0 / sqrt(3.0);
        weight[0] = 1.0;
        weight[1] = 1.0;
        break;
    }

    case 27: // 27 Gauss points
    {
        // 1-D integration rule
        n = 3;
        coord = new double[n];
        weight = new double[n];
        coord[0] = -sqrt(0.6);
        coord[1] = 0.0;
        coord[2] = sqrt(0.6);
        weight[0] = 5.0 / 9.0;
        weight[1] = 8.0 / 9.0;
        weight[2] = 5.0 / 9.0;
        break;
    }

    case 64: // 64 Gauss points
    {
        // 1-D integration rule
        n = 4;
        coord = new double[n];
        weight = new double[n];
        coord[0] = -0.3399810435848563;
        coord[1] = 0.3399810435848563;
        coord[2] = -0.8611363115940526;
        coord[3] = 0.8611363115940526;
        weight[0] = 0.6521451548625461;
        weight[1] = 0.6521451548625461;
        weight[2] = 0.3478548451374538;
        weight[3] = 0.3478548451374538;
        break;
    }

    default:
        return -1;
    }

    // 3-D integration rule created from 1-D integration rule
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n; k++)
            {
                IP[n * n * i + n * j + k][0] = coord[i];
                IP[n * n * i + n * j + k][1] = coord[j];
                IP[n * n * i + n * j + k][2] = coord[k];
                IP[n * n * i + n * j + k][3] = 0;
                alpha[n * n * i + n * j + k] = weight[i] * weight[j] * weight[k];
            }
        }
    }

    // Clear memory
    delete[] coord;
    delete[] weight;

    return 1;
}

// Returns shape functions evaluated at tcoord for hexahedra
int getHexaShapeFun(const int np, const double tcoord[4], double *N)
{

    // Choose element type
    switch (np)
    {
    case 8: // eight node hexahedron
    {
        // Basis functions expressed in terms of natural coordinates
        N[0] = 1.0 / 8.0 * (1.0 - tcoord[0]) * (1.0 - tcoord[1]) * (1.0 - tcoord[2]);
        N[1] = 1.0 / 8.0 * (1.0 + tcoord[0]) * (1.0 - tcoord[1]) * (1.0 - tcoord[2]);
        N[2] = 1.0 / 8.0 * (1.0 + tcoord[0]) * (1.0 + tcoord[1]) * (1.0 - tcoord[2]);
        N[3] = 1.0 / 8.0 * (1.0 - tcoord[0]) * (1.0 + tcoord[1]) * (1.0 - tcoord[2]);
        N[4] = 1.0 / 8.0 * (1.0 - tcoord[0]) * (1.0 - tcoord[1]) * (1.0 + tcoord[2]);
        N[5] = 1.0 / 8.0 * (1.0 + tcoord[0]) * (1.0 - tcoord[1]) * (1.0 + tcoord[2]);
        N[6] = 1.0 / 8.0 * (1.0 + tcoord[0]) * (1.0 + tcoord[1]) * (1.0 + tcoord[2]);
        N[7] = 1.0 / 8.0 * (1.0 - tcoord[0]) * (1.0 + tcoord[1]) * (1.0 + tcoord[2]);

        break;
    }
    case 20: // twenty node hexahedron
    {
        // Basis functions expressed in terms of natural coordinates
        N[0] = 1.0 / 8.0 * (1.0 - tcoord[0]) * (1.0 - tcoord[1]) * (1.0 - tcoord[2]) * (-tcoord[0] - tcoord[1] - tcoord[2] - 2.0);
        N[1] = 1.0 / 8.0 * (1.0 + tcoord[0]) * (1.0 - tcoord[1]) * (1.0 - tcoord[2]) * (tcoord[0] - tcoord[1] - tcoord[2] - 2.0);
        N[2] = 1.0 / 8.0 * (1.0 + tcoord[0]) * (1.0 + tcoord[1]) * (1.0 - tcoord[2]) * (tcoord[0] + tcoord[1] - tcoord[2] - 2.0);
        N[3] = 1.0 / 8.0 * (1.0 - tcoord[0]) * (1.0 + tcoord[1]) * (1.0 - tcoord[2]) * (-tcoord[0] + tcoord[1] - tcoord[2] - 2.0);
        N[4] = 1.0 / 8.0 * (1.0 - tcoord[0]) * (1.0 - tcoord[1]) * (1.0 + tcoord[2]) * (-tcoord[0] - tcoord[1] + tcoord[2] - 2.0);
        N[5] = 1.0 / 8.0 * (1.0 + tcoord[0]) * (1.0 - tcoord[1]) * (1.0 + tcoord[2]) * (tcoord[0] - tcoord[1] + tcoord[2] - 2.0);
        N[6] = 1.0 / 8.0 * (1.0 + tcoord[0]) * (1.0 + tcoord[1]) * (1.0 + tcoord[2]) * (tcoord[0] + tcoord[1] + tcoord[2] - 2.0);
        N[7] = 1.0 / 8.0 * (1.0 - tcoord[0]) * (1.0 + tcoord[1]) * (1.0 + tcoord[2]) * (-tcoord[0] + tcoord[1] + tcoord[2] - 2.0);
        N[8] = 1.0 / 4.0 * (1.0 - tcoord[0] * tcoord[0]) * (1.0 - tcoord[1]) * (1.0 - tcoord[2]);
        N[9] = 1.0 / 4.0 * (1.0 - tcoord[1] * tcoord[1]) * (1.0 - tcoord[0]) * (1.0 - tcoord[2]);
        N[10] = 1.0 / 4.0 * (1.0 - tcoord[2] * tcoord[2]) * (1.0 - tcoord[0]) * (1.0 - tcoord[1]);
        N[11] = 1.0 / 4.0 * (1.0 - tcoord[1] * tcoord[1]) * (1.0 + tcoord[0]) * (1.0 - tcoord[2]);
        N[12] = 1.0 / 4.0 * (1.0 - tcoord[2] * tcoord[2]) * (1.0 + tcoord[0]) * (1.0 - tcoord[1]);
        N[13] = 1.0 / 4.0 * (1.0 - tcoord[0] * tcoord[0]) * (1.0 + tcoord[1]) * (1.0 - tcoord[2]);
        N[14] = 1.0 / 4.0 * (1.0 - tcoord[2] * tcoord[2]) * (1.0 + tcoord[0]) * (1.0 + tcoord[1]);
        N[15] = 1.0 / 4.0 * (1.0 - tcoord[2] * tcoord[2]) * (1.0 - tcoord[0]) * (1.0 + tcoord[1]);
        N[16] = 1.0 / 4.0 * (1.0 - tcoord[0] * tcoord[0]) * (1.0 - tcoord[1]) * (1.0 + tcoord[2]);
        N[17] = 1.0 / 4.0 * (1.0 - tcoord[1] * tcoord[1]) * (1.0 - tcoord[0]) * (1.0 + tcoord[2]);
        N[18] = 1.0 / 4.0 * (1.0 - tcoord[1] * tcoord[1]) * (1.0 + tcoord[0]) * (1.0 + tcoord[2]);
        N[19] = 1.0 / 4.0 * (1.0 - tcoord[0] * tcoord[0]) * (1.0 + tcoord[1]) * (1.0 + tcoord[2]);

        break;
    }
    case 27: // twenty seven node hexahedron
    {
        // GMSH: mapping from node numbering to xi-, eta-, and mu-coordinates
        double node2xi[27] = {-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 0.0, -1.0, -1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0};
        double node2eta[27] = {-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 0.0, -1.0, 0.0, -1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
        double node2mu[27] = {-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};

        // Construct shape functions as products of 1D Lagrange polynomials
        for (int i = 0; i < 27; i++)
        {
            N[i] = getLagrange1DPoly(2, node2xi[i], 0, tcoord[0]) *
                   getLagrange1DPoly(2, node2eta[i], 0, tcoord[1]) *
                   getLagrange1DPoly(2, node2mu[i], 0, tcoord[2]);
        }

        break;
    }
    default:
        return -1;
    }
    return 1;
}

// Matrix of derivatives with respect to physical coordinates of basis functions for hexahedrons
int getHexaBeTLF(const int np, const double tcoord[3],
                 const double *x, const double *y, const double *z, double &Jdet,
                 double **Be)
{
    int i, j, errorcode;
    double *DxiN = new double[np];
    double *DetaN = new double[np];
    double *DzetaN = new double[np];
    double *DxN = new double[np];
    double *DyN = new double[np];
    double *DzN = new double[np];
    double J[3][3];
    double IJ[3][3];

    // Derivatives of the basis functions with respect to natural coordinates
    errorcode = getHexaShapeFunGradNat(np, tcoord, DxiN, DetaN, DzetaN);

    // Jacobian
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            J[i][j] = 0.0;
        }
    for (i = 0; i < np; i++)
    {
        J[0][0] += DxiN[i] * x[i];
        J[0][1] += DetaN[i] * x[i];
        J[0][2] += DzetaN[i] * x[i];
        J[1][0] += DxiN[i] * y[i];
        J[1][1] += DetaN[i] * y[i];
        J[1][2] += DzetaN[i] * y[i];
        J[2][0] += DxiN[i] * z[i];
        J[2][1] += DetaN[i] * z[i];
        J[2][2] += DzetaN[i] * z[i];
    }
    if (!getInverse33(J, Jdet, IJ))
        return -4;
    if (Jdet < 0)
        return -3;

    // Derivatives of basis functions (DxN - derivative w.r.t. x, DyN - derivatives w.r.t. y, DzN - derivatives w.r.t. z)
    for (i = 0; i < np; i++)
    {
        DxN[i] = IJ[0][0] * DxiN[i] + IJ[1][0] * DetaN[i] + IJ[2][0] * DzetaN[i];
        DyN[i] = IJ[0][1] * DxiN[i] + IJ[1][1] * DetaN[i] + IJ[2][1] * DzetaN[i];
        DzN[i] = IJ[0][2] * DxiN[i] + IJ[1][2] * DetaN[i] + IJ[2][2] * DzetaN[i];
    }

    // Matrix of derivatives of basis functions
    for (i = 0; i < np; i++)
    {
        Be[0][3 * i + 0] = DxN[i];
        Be[0][3 * i + 1] = 0.0;
        Be[0][3 * i + 2] = 0.0;

        Be[1][3 * i + 0] = 0.0;
        Be[1][3 * i + 1] = DxN[i];
        Be[1][3 * i + 2] = 0.0;

        Be[2][3 * i + 0] = 0.0;
        Be[2][3 * i + 1] = 0.0;
        Be[2][3 * i + 2] = DxN[i];

        Be[3][3 * i + 0] = DyN[i];
        Be[3][3 * i + 1] = 0.0;
        Be[3][3 * i + 2] = 0.0;

        Be[4][3 * i + 0] = 0.0;
        Be[4][3 * i + 1] = DyN[i];
        Be[4][3 * i + 2] = 0.0;

        Be[5][3 * i + 0] = 0.0;
        Be[5][3 * i + 1] = 0.0;
        Be[5][3 * i + 2] = DyN[i];

        Be[6][3 * i + 0] = DzN[i];
        Be[6][3 * i + 1] = 0.0;
        Be[6][3 * i + 2] = 0.0;

        Be[7][3 * i + 0] = 0.0;
        Be[7][3 * i + 1] = DzN[i];
        Be[7][3 * i + 2] = 0.0;

        Be[8][3 * i + 0] = 0.0;
        Be[8][3 * i + 1] = 0.0;
        Be[8][3 * i + 2] = DzN[i];
    }

    // Clear memory
    delete[] DxiN;
    delete[] DetaN;
    delete[] DzetaN;
    delete[] DxN;
    delete[] DyN;
    delete[] DzN;

    return errorcode;
}

// Matrix of derivatives with respect to physical coordinates of basis functions for hexahedrons
int getHexaBeTLE(const int np, const double tcoord[4], const double *x,
                 const double *y, const double *z, const double *uel, double &Jdet, double **Be,
                 double **Bf, double F[9])
{
    int i, j, errorcode;
    double *DxiN = new double[np];
    double *DetaN = new double[np];
    double *DzetaN = new double[np];
    double *DxN = new double[np];
    double *DyN = new double[np];
    double *DzN = new double[np];
    double J[3][3];
    double IJ[3][3];

    // Derivatives of the basis functions with respect to natural coordinates
    errorcode = getHexaShapeFunGradNat(np, tcoord, DxiN, DetaN, DzetaN);

    // Jacobian
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            J[i][j] = 0.0;
        }
    }
    for (i = 0; i < np; i++)
    {
        J[0][0] += DxiN[i] * x[i];
        J[0][1] += DetaN[i] * x[i];
        J[0][2] += DzetaN[i] * x[i];
        J[1][0] += DxiN[i] * y[i];
        J[1][1] += DetaN[i] * y[i];
        J[1][2] += DzetaN[i] * y[i];
        J[2][0] += DxiN[i] * z[i];
        J[2][1] += DetaN[i] * z[i];
        J[2][2] += DzetaN[i] * z[i];
    }
    if (!getInverse33(J, Jdet, IJ))
        return -4;
    if (Jdet < 0)
        return -3;

    // Derivatives of basis functions (DxN - derivative w.r.t. x, DyN - derivatives w.r.t. y, DzN - derivatives w.r.t. z)
    for (i = 0; i < np; i++)
    {
        DxN[i] = IJ[0][0] * DxiN[i] + IJ[1][0] * DetaN[i] + IJ[2][0] * DzetaN[i];
        DyN[i] = IJ[0][1] * DxiN[i] + IJ[1][1] * DetaN[i] + IJ[2][1] * DzetaN[i];
        DzN[i] = IJ[0][2] * DxiN[i] + IJ[1][2] * DetaN[i] + IJ[2][2] * DzetaN[i];
    }

    // Bf matrix
    for (i = 0; i < np; i++)
    {
        Bf[0][3 * i + 0] = DxN[i];
        Bf[0][3 * i + 1] = 0.0;
        Bf[0][3 * i + 2] = 0.0;

        Bf[1][3 * i + 0] = 0.0;
        Bf[1][3 * i + 1] = DxN[i];
        Bf[1][3 * i + 2] = 0.0;

        Bf[2][3 * i + 0] = 0.0;
        Bf[2][3 * i + 1] = 0.0;
        Bf[2][3 * i + 2] = DxN[i];

        Bf[3][3 * i + 0] = DyN[i];
        Bf[3][3 * i + 1] = 0.0;
        Bf[3][3 * i + 2] = 0.0;

        Bf[4][3 * i + 0] = 0.0;
        Bf[4][3 * i + 1] = DyN[i];
        Bf[4][3 * i + 2] = 0.0;

        Bf[5][3 * i + 0] = 0.0;
        Bf[5][3 * i + 1] = 0.0;
        Bf[5][3 * i + 2] = DyN[i];

        Bf[6][3 * i + 0] = DzN[i];
        Bf[6][3 * i + 1] = 0.0;
        Bf[6][3 * i + 2] = 0.0;

        Bf[7][3 * i + 0] = 0.0;
        Bf[7][3 * i + 1] = DzN[i];
        Bf[7][3 * i + 2] = 0.0;

        Bf[8][3 * i + 0] = 0.0;
        Bf[8][3 * i + 1] = 0.0;
        Bf[8][3 * i + 2] = DzN[i];
    }

    // Get F
    getF3d(Bf, uel, np, F);

    // Be matrix
    for (i = 0; i < np; i++)
    {
        Be[0][3 * i + 0] = F[0] * DxN[i];
        Be[0][3 * i + 1] = F[1] * DxN[i];
        Be[0][3 * i + 2] = F[2] * DxN[i];

        Be[1][3 * i + 0] = F[3] * DyN[i];
        Be[1][3 * i + 1] = F[4] * DyN[i];
        Be[1][3 * i + 2] = F[5] * DyN[i];

        Be[2][3 * i + 0] = F[6] * DzN[i];
        Be[2][3 * i + 1] = F[7] * DzN[i];
        Be[2][3 * i + 2] = F[8] * DzN[i];

        Be[3][3 * i + 0] = F[0] * DyN[i] + F[3] * DxN[i];
        Be[3][3 * i + 1] = F[1] * DyN[i] + F[4] * DxN[i];
        Be[3][3 * i + 2] = F[2] * DyN[i] + F[5] * DxN[i];

        Be[4][3 * i + 0] = F[3] * DzN[i] + F[6] * DyN[i];
        Be[4][3 * i + 1] = F[4] * DzN[i] + F[7] * DyN[i];
        Be[4][3 * i + 2] = F[5] * DzN[i] + F[8] * DyN[i];

        Be[5][3 * i + 0] = F[6] * DxN[i] + F[0] * DzN[i];
        Be[5][3 * i + 1] = F[7] * DxN[i] + F[1] * DzN[i];
        Be[5][3 * i + 2] = F[8] * DxN[i] + F[2] * DzN[i];
    }

    // Clear memory
    delete[] DxiN;
    delete[] DetaN;
    delete[] DzetaN;
    delete[] DxN;
    delete[] DyN;
    delete[] DzN;

    return errorcode;
}

// Convert coordinates from natural to physical for quadrangles
int ConvertHexaNat2Phys(const int np, const double tcoord[4],
                        const double *x, const double *y, const double *z, double &X,
                        double &Y, double &Z)
{
    int errorcode;
    double *N = new double[np];

    // Get basis functions evaluations
    errorcode = getHexaShapeFun(np, tcoord, N);

    // Physical coordinates
    X = 0.0;
    Y = 0.0;
    Z = 0.0;
    for (int i = 0; i < np; i++)
    {
        X += N[i] * x[i];
        Y += N[i] * y[i];
        Z += N[i] * z[i];
    }

    // Clear memory
    delete[] N;

    return errorcode;
}

// Convert coordinates from physical to natural for hexahedrons
int ConvertHexaPhys2Nat(const int np, const double X, const double Y,
                        const double Z, const double *x, const double *y, const double *z,
                        const double TOL_g, double tcoord[4])
{

    // Initialize variables
    int i, j, errorcode;
    double f[3], J[3][3], IJ[3][3], dtcoord[3], detJ;
    double *N = new double[np];
    double *DxiN = new double[np];
    double *DetaN = new double[np];
    double *DzetaN = new double[np];

    // Get an initial guess (zero natural coordinates)
    for (i = 0; i < 4; i++)
        tcoord[i] = 0.0;

    // Get shape function evaluations at current tcoord
    errorcode = getHexaShapeFun(np, tcoord, N);

    // Update the residual vector
    for (i = 0; i < 3; i++)
        f[i] = 0.0; // initialize f
    for (i = 0; i < np; i++)
    { // compute f
        f[0] += x[i] * N[i];
        f[1] += y[i] * N[i];
        f[2] += z[i] * N[i];
    }
    f[0] -= X;
    f[1] -= Y;
    f[2] -= Z;

    // Update error
    double eps = 0.0;
    for (i = 0; i < 3; i++)
        eps += fabs(f[i]);

    // Newton loop
    int Niter = 0;
    int Maxiter = 100;
    while (eps > TOL_g && Niter < Maxiter)
    {
        Niter++;

        // Get derivatives of basis functions with respect to natural coordinates evaluated at current tcoord
        errorcode = getHexaShapeFunGradNat(np, tcoord, DxiN, DetaN, DzetaN);

        // Jacobian of the nonlinear system of equations f = 0
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                J[i][j] = 0.0;
        for (i = 0; i < np; i++)
        {
            J[0][0] += x[i] * DxiN[i];
            J[0][1] += x[i] * DetaN[i];
            J[0][2] += x[i] * DzetaN[i];

            J[1][0] += y[i] * DxiN[i];
            J[1][1] += y[i] * DetaN[i];
            J[1][2] += y[i] * DzetaN[i];

            J[2][0] += z[i] * DxiN[i];
            J[2][1] += z[i] * DetaN[i];
            J[2][2] += z[i] * DzetaN[i];
        }

        // Inverse Jacobian
        if (!getInverse33(J, detJ, IJ))
            break;

        // Solve the system tcoord = tcoord - J\f
        for (i = 0; i < 3; i++)
        {
            dtcoord[i] = 0.0;
            for (j = 0; j < 3; j++)
                dtcoord[i] -= IJ[i][j] * f[j];
        }

        // Update tcoord
        for (i = 0; i < 3; i++)
            tcoord[i] += dtcoord[i];

        // Get shape function evaluations at current tcoord
        errorcode = getHexaShapeFun(np, tcoord, N);

        // Update the residual vector
        for (i = 0; i < 3; i++)
            f[i] = 0.0; // initialize f
        for (i = 0; i < np; i++)
        { // compute f
            f[0] += x[i] * N[i];
            f[1] += y[i] * N[i];
            f[2] += z[i] * N[i];
        }
        f[0] -= X;
        f[1] -= Y;
        f[2] -= Z;

        // Update error
        eps = 0.0;
        for (i = 0; i < 3; i++)
            eps += fabs(f[i]);
    }

    // Print warning message if maximum number of iterations was exceeded
    if (Niter >= Maxiter)
    {
        // printf("ConvertTriagCoordinates: %i Nwtn. iter. exceeded\n", Maxiter);
        for (i = 0; i < 4; i++)
            tcoord[i] = NAN; // return NANs
        errorcode = -5;
    }
    else
    { // check if given point is inside element or not
        if ((fabs(tcoord[0]) < 1.0 + TOL_g) && (fabs(tcoord[1]) < 1.0 + TOL_g) &&
            (fabs(tcoord[2]) < 1.0 + TOL_g))
        {
            // If it is inside, do nothing
        }
        else
        {
            for (i = 0; i < 4; i++)
                tcoord[i] = NAN; // return NANs
        }
    }

    // Clear memory
    delete[] N;
    delete[] DxiN;
    delete[] DetaN;
    delete[] DzetaN;

    return errorcode;
}

// Returns gradient of shape functions for quadrangles with respect to natural coordinates
int getHexaShapeFunGradNat(const int np, const double tcoord[3],
                           double *DxiN, double *DetaN, double *DmuN)
{

    // Choose element type
    switch (np)
    {
    case 8: // eight node hexahedron
    {
        // Derivatives of the basis functions (DxiN - derivatives w.r.t. xi, DetaN - derivatives w.r.t eta)
        DxiN[0] = -1.0 / 8.0 * (1 - tcoord[1]) * (1 - tcoord[2]);
        DxiN[1] = +1.0 / 8.0 * (1 - tcoord[1]) * (1 - tcoord[2]);
        DxiN[2] = +1.0 / 8.0 * (1 + tcoord[1]) * (1 - tcoord[2]);
        DxiN[3] = -1.0 / 8.0 * (1 + tcoord[1]) * (1 - tcoord[2]);
        DxiN[4] = -1.0 / 8.0 * (1 - tcoord[1]) * (1 + tcoord[2]);
        DxiN[5] = +1.0 / 8.0 * (1 - tcoord[1]) * (1 + tcoord[2]);
        DxiN[6] = +1.0 / 8.0 * (1 + tcoord[1]) * (1 + tcoord[2]);
        DxiN[7] = -1.0 / 8.0 * (1 + tcoord[1]) * (1 + tcoord[2]);

        DetaN[0] = -1.0 / 8.0 * (1 - tcoord[0]) * (1 - tcoord[2]);
        DetaN[1] = -1.0 / 8.0 * (1 + tcoord[0]) * (1 - tcoord[2]);
        DetaN[2] = +1.0 / 8.0 * (1 + tcoord[0]) * (1 - tcoord[2]);
        DetaN[3] = +1.0 / 8.0 * (1 - tcoord[0]) * (1 - tcoord[2]);
        DetaN[4] = -1.0 / 8.0 * (1 - tcoord[0]) * (1 + tcoord[2]);
        DetaN[5] = -1.0 / 8.0 * (1 + tcoord[0]) * (1 + tcoord[2]);
        DetaN[6] = +1.0 / 8.0 * (1 + tcoord[0]) * (1 + tcoord[2]);
        DetaN[7] = +1.0 / 8.0 * (1 - tcoord[0]) * (1 + tcoord[2]);

        DmuN[0] = -1.0 / 8.0 * (1 - tcoord[0]) * (1 - tcoord[1]);
        DmuN[1] = -1.0 / 8.0 * (1 + tcoord[0]) * (1 - tcoord[1]);
        DmuN[2] = -1.0 / 8.0 * (1 + tcoord[0]) * (1 + tcoord[1]);
        DmuN[3] = -1.0 / 8.0 * (1 - tcoord[0]) * (1 + tcoord[1]);
        DmuN[4] = +1.0 / 8.0 * (1 - tcoord[0]) * (1 - tcoord[1]);
        DmuN[5] = +1.0 / 8.0 * (1 + tcoord[0]) * (1 - tcoord[1]);
        DmuN[6] = +1.0 / 8.0 * (1 + tcoord[0]) * (1 + tcoord[1]);
        DmuN[7] = +1.0 / 8.0 * (1 - tcoord[0]) * (1 + tcoord[1]);

        break;
    }

    case 20: // twenty node hexahedron
    {
        // Derivatives of the basis functions (DxiN - derivatives w.r.t. xi, DetaN - derivatives w.r.t eta)
        DxiN[0] = 1.0 / 8.0 * (-1.0 + tcoord[1]) * (1.0 + 2.0 * tcoord[0] + tcoord[1] + tcoord[2]) * (-1.0 + tcoord[2]);
        DxiN[1] = -1.0 / 8.0 * (-1.0 + tcoord[1]) * (1.0 - 2.0 * tcoord[0] + tcoord[1] + tcoord[2]) * (-1.0 + tcoord[2]);
        DxiN[2] = -1.0 / 8.0 * (1.0 + tcoord[1]) * (-1.0 + 2.0 * tcoord[0] + tcoord[1] - tcoord[2]) * (-1.0 + tcoord[2]);
        DxiN[3] = 1.0 / 8.0 * (1.0 + tcoord[1]) * (-1.0 - 2.0 * tcoord[0] + tcoord[1] - tcoord[2]) * (-1.0 + tcoord[2]);
        DxiN[4] = -1.0 / 8.0 * (-1.0 + tcoord[1]) * (1.0 + 2.0 * tcoord[0] + tcoord[1] - tcoord[2]) * (1.0 + tcoord[2]);
        DxiN[5] = 1.0 / 8.0 * (-1.0 + tcoord[1]) * (1.0 - 2.0 * tcoord[0] + tcoord[1] - tcoord[2]) * (1.0 + tcoord[2]);
        DxiN[6] = 1.0 / 8.0 * (1.0 + tcoord[1]) * (-1.0 + 2.0 * tcoord[0] + tcoord[1] + tcoord[2]) * (1.0 + tcoord[2]);
        DxiN[7] = -1.0 / 8.0 * (1.0 + tcoord[1]) * (-1.0 - 2.0 * tcoord[0] + tcoord[1] + tcoord[2]) * (1.0 + tcoord[2]);
        DxiN[8] = -1.0 / 2.0 * tcoord[0] * (-1.0 + tcoord[1]) * (-1.0 + tcoord[2]);
        DxiN[9] = -1.0 / 4.0 * (-1.0 + tcoord[1] * tcoord[1]) * (-1.0 + tcoord[2]);
        DxiN[10] = -1.0 / 4.0 * (-1.0 + tcoord[1]) * (-1.0 + tcoord[2] * tcoord[2]);
        DxiN[11] = 1.0 / 4.0 * (-1.0 + tcoord[1] * tcoord[1]) * (-1.0 + tcoord[2]);
        DxiN[12] = 1.0 / 4.0 * (-1.0 + tcoord[1]) * (-1.0 + tcoord[2] * tcoord[2]);
        DxiN[13] = 1.0 / 2.0 * tcoord[0] * (1.0 + tcoord[1]) * (-1.0 + tcoord[2]);
        DxiN[14] = -1.0 / 4.0 * (1.0 + tcoord[1]) * (-1.0 + tcoord[2] * tcoord[2]);
        DxiN[15] = 1.0 / 4.0 * (1.0 + tcoord[1]) * (-1.0 + tcoord[2] * tcoord[2]);
        DxiN[16] = 1.0 / 2.0 * tcoord[0] * (-1.0 + tcoord[1]) * (1.0 + tcoord[2]);
        DxiN[17] = 1.0 / 4.0 * (-1.0 + tcoord[1] * tcoord[1]) * (1.0 + tcoord[2]);
        DxiN[18] = -1.0 / 4.0 * (-1.0 + tcoord[1] * tcoord[1]) * (1.0 + tcoord[2]);
        DxiN[19] = -1.0 / 2.0 * tcoord[0] * (1.0 + tcoord[1]) * (1.0 + tcoord[2]);

        DetaN[0] = 1.0 / 8.0 * (-1.0 + tcoord[0]) * (1.0 + tcoord[0] + 2.0 * tcoord[1] + tcoord[2]) * (-1.0 + tcoord[2]);
        DetaN[1] = 1.0 / 8.0 * (1.0 + tcoord[0]) * (-1.0 + tcoord[0] - 2.0 * tcoord[1] - tcoord[2]) * (-1.0 + tcoord[2]);
        DetaN[2] = -1.0 / 8.0 * (1.0 + tcoord[0]) * (-1.0 + tcoord[0] + 2.0 * tcoord[1] - tcoord[2]) * (-1.0 + tcoord[2]);
        DetaN[3] = -1.0 / 8.0 * (-1.0 + tcoord[0]) * (1.0 + tcoord[0] - 2.0 * tcoord[1] + tcoord[2]) * (-1.0 + tcoord[2]);
        DetaN[4] = -1.0 / 8.0 * (-1.0 + tcoord[0]) * (1.0 + tcoord[0] + 2.0 * tcoord[1] - tcoord[2]) * (1.0 + tcoord[2]);
        DetaN[5] = -1.0 / 8.0 * (1.0 + tcoord[0]) * (-1.0 + tcoord[0] - 2.0 * tcoord[1] + tcoord[2]) * (1.0 + tcoord[2]);
        DetaN[6] = 1.0 / 8.0 * (1.0 + tcoord[0]) * (-1.0 + tcoord[0] + 2.0 * tcoord[1] + tcoord[2]) * (1.0 + tcoord[2]);
        DetaN[7] = 1.0 / 8.0 * (-1.0 + tcoord[0]) * (1.0 + tcoord[0] - 2.0 * tcoord[1] - tcoord[2]) * (1.0 + tcoord[2]);
        DetaN[8] = -1.0 / 4.0 * (-1.0 + tcoord[0] * tcoord[0]) * (-1.0 + tcoord[2]);
        DetaN[9] = -1.0 / 2.0 * (-1.0 + tcoord[0]) * tcoord[1] * (-1.0 + tcoord[2]);
        DetaN[10] = -1.0 / 4.0 * (-1.0 + tcoord[0]) * (-1.0 + tcoord[2] * tcoord[2]);
        DetaN[11] = 1.0 / 2.0 * (1.0 + tcoord[0]) * tcoord[1] * (-1.0 + tcoord[2]);
        DetaN[12] = 1.0 / 4.0 * (1.0 + tcoord[0]) * (-1.0 + tcoord[2] * tcoord[2]);
        DetaN[13] = 1.0 / 4.0 * (-1.0 + tcoord[0] * tcoord[0]) * (-1.0 + tcoord[2]);
        DetaN[14] = -1.0 / 4.0 * (1.0 + tcoord[0]) * (-1.0 + tcoord[2] * tcoord[2]);
        DetaN[15] = 1.0 / 4.0 * (-1.0 + tcoord[0]) * (-1.0 + tcoord[2] * tcoord[2]);
        DetaN[16] = 1.0 / 4.0 * (-1.0 + tcoord[0] * tcoord[0]) * (1.0 + tcoord[2]);
        DetaN[17] = 1.0 / 2.0 * (-1.0 + tcoord[0]) * tcoord[1] * (1.0 + tcoord[2]);
        DetaN[18] = -1.0 / 2.0 * (1.0 + tcoord[0]) * tcoord[1] * (1.0 + tcoord[2]);
        DetaN[19] = -1.0 / 4.0 * (-1.0 + tcoord[0] * tcoord[0]) * (1.0 + tcoord[2]);

        DmuN[0] = 1.0 / 8.0 * (-1.0 + tcoord[0]) * (-1.0 + tcoord[1]) * (1.0 + tcoord[0] + tcoord[1] + 2.0 * tcoord[2]);
        DmuN[1] = 1.0 / 8.0 * (1.0 + tcoord[0]) * (-1.0 + tcoord[1]) * (-1.0 + tcoord[0] - tcoord[1] - 2.0 * tcoord[2]);
        DmuN[2] = -1.0 / 8.0 * (1.0 + tcoord[0]) * (1.0 + tcoord[1]) * (-1.0 + tcoord[0] + tcoord[1] - 2.0 * tcoord[2]);
        DmuN[3] = -1.0 / 8.0 * (-1.0 + tcoord[0]) * (1.0 + tcoord[1]) * (1.0 + tcoord[0] - tcoord[1] + 2.0 * tcoord[2]);
        DmuN[4] = -1.0 / 8.0 * (-1.0 + tcoord[0]) * (-1.0 + tcoord[1]) * (1.0 + tcoord[0] + tcoord[1] - 2.0 * tcoord[2]);
        DmuN[5] = -1.0 / 8.0 * (1.0 + tcoord[0]) * (-1.0 + tcoord[1]) * (-1.0 + tcoord[0] - tcoord[1] + 2.0 * tcoord[2]);
        DmuN[6] = 1.0 / 8.0 * (1.0 + tcoord[0]) * (1.0 + tcoord[1]) * (-1.0 + tcoord[0] + tcoord[1] + 2.0 * tcoord[2]);
        DmuN[7] = 1.0 / 8.0 * (-1.0 + tcoord[0]) * (1.0 + tcoord[1]) * (1.0 + tcoord[0] - tcoord[1] - 2.0 * tcoord[2]);
        DmuN[8] = -1.0 / 4.0 * (-1.0 + tcoord[0] * tcoord[0]) * (-1.0 + tcoord[1]);
        DmuN[9] = -1.0 / 4.0 * (-1.0 + tcoord[0]) * (-1.0 + tcoord[1] * tcoord[1]);
        DmuN[10] = -1.0 / 2.0 * (-1.0 + tcoord[0]) * (-1.0 + tcoord[1]) * tcoord[2];
        DmuN[11] = 1.0 / 4.0 * (1.0 + tcoord[0]) * (-1.0 + tcoord[1] * tcoord[1]);
        DmuN[12] = 1.0 / 2.0 * (1.0 + tcoord[0]) * (-1.0 + tcoord[1]) * tcoord[2];
        DmuN[13] = 1.0 / 4.0 * (-1.0 + tcoord[0] * tcoord[0]) * (1.0 + tcoord[1]);
        DmuN[14] = -1.0 / 2.0 * (1.0 + tcoord[0]) * (1.0 + tcoord[1]) * tcoord[2];
        DmuN[15] = 1.0 / 2.0 * (-1.0 + tcoord[0]) * (1.0 + tcoord[1]) * tcoord[2];
        DmuN[16] = 1.0 / 4.0 * (-1.0 + tcoord[0] * tcoord[0]) * (-1.0 + tcoord[1]);
        DmuN[17] = 1.0 / 4.0 * (-1.0 + tcoord[0]) * (-1.0 + tcoord[1] * tcoord[1]);
        DmuN[18] = -1.0 / 4.0 * (1.0 + tcoord[0]) * (-1.0 + tcoord[1] * tcoord[1]);
        DmuN[19] = -1.0 / 4.0 * (-1.0 + tcoord[0] * tcoord[0]) * (1.0 + tcoord[1]);

        break;
    }

    case 27: // twenty-seven node hexahedron
    {
        // Derivatives of the basis functions (DxiN - derivatives w.r.t. xi, DetaN - derivatives w.r.t eta)
        // GMSH: mapping from node numbering to xi-, eta-, and mu-coordinates
        double node2xi[27] = {-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 0.0, -1.0, -1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0};
        double node2eta[27] = {-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 0.0, -1.0, 0.0, -1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
        double node2mu[27] = {-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};

        // Construct derivatives as products of 1D Lagrange polynomials
        for (int i = 0; i < 27; i++)
        {
            DxiN[i] = getLagrange1DPoly(2, node2xi[i], 1, tcoord[0]) *
                      getLagrange1DPoly(2, node2eta[i], 0, tcoord[1]) *
                      getLagrange1DPoly(2, node2mu[i], 0, tcoord[2]);
            DetaN[i] = getLagrange1DPoly(2, node2xi[i], 0, tcoord[0]) *
                       getLagrange1DPoly(2, node2eta[i], 1, tcoord[1]) *
                       getLagrange1DPoly(2, node2mu[i], 0, tcoord[2]);
            DmuN[i] = getLagrange1DPoly(2, node2xi[i], 0, tcoord[0]) *
                      getLagrange1DPoly(2, node2eta[i], 0, tcoord[1]) *
                      getLagrange1DPoly(2, node2mu[i], 1, tcoord[2]);
        }

        break;
    }

    default:
        return -2;
    }
    return 1;
}

/******************** 2d Constitutive laws formulated in F  ********************/

// Get derivatives of invariants with respect to F
int getInvariantsDerivativesF2d(const double F[4],
                                const double IF[4], const double J,                   // inputs
                                double &I1b, double &I2b, double &I132, double &logJ, // outputs
                                double &J13, double &J23, double DI1b[4], double DI2b[4],
                                double DDI1b[4][4], double DDI2b[4][4])
{

    // Allocate indices
    int i, j, k, l;

    // The left Cauchy-Green tensor B = F*F'
    double B[4];
    B[0] = F[0] * F[0] + F[2] * F[2];
    B[1] = F[1] * F[0] + F[3] * F[2];
    B[2] = F[0] * F[1] + F[2] * F[3];
    B[3] = F[1] * F[1] + F[3] * F[3];

    // Get B^2
    double B2[4];
    B2[0] = B[0] * B[0] + B[2] * B[1];
    B2[1] = B[1] * B[0] + B[3] * B[1];
    B2[2] = B[0] * B[2] + B[2] * B[3];
    B2[3] = B[1] * B[2] + B[3] * B[3];

    // The first invariant of B
    double I1 = B[0] + B[3] + 1.0;

    // The second invariant of B
    double I2 = 1.0 / 2.0 * (pow(I1, 2.0) - (B2[0] + B2[3] + 1.0));

    // The righ Cauchy-Green tensor C = F'*F
    double C[4];
    C[0] = F[0] * F[0] + F[1] * F[1];
    C[1] = F[2] * F[0] + F[3] * F[1];
    C[2] = F[0] * F[2] + F[1] * F[3];
    C[3] = F[2] * F[2] + F[3] * F[3];

    // FC = F*C
    double FC[4];
    FC[0] = F[0] * C[0] + F[2] * C[1];
    FC[1] = F[1] * C[0] + F[3] * C[1];
    FC[2] = F[0] * C[2] + F[2] * C[3];
    FC[3] = F[1] * C[2] + F[3] * C[3];

    // Normalized invariants and constants
    J13 = pow(J, 1.0 / 3.0);
    J23 = pow(J, 2.0 / 3.0);
    double J43 = pow(J, 4.0 / 3.0);
    logJ = log(J);
    I1b = I1 / J23;
    I2b = I2 / J43;
    I132 = pow(I1b - 3.0, 2.0);

    // Identity second-order tensor
    double Id2[4];
    Id2[0] = 1.0;
    Id2[1] = 0.0;
    Id2[2] = 0.0;
    Id2[3] = 1.0;

    // Get the first order derivatives of I1/J23 and I2/J43 w.r.t. F
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 2; j++)
        {
            DI1b[i + 2 * j] = 2.0 / J23 * F[i + 2 * j] - 2.0 / 3.0 * I1b * IF[j + 2 * i];
            DI2b[i + 2 * j] = 2.0 / J23 * I1b * F[i + 2 * j] - 4.0 / 3.0 * I2b * IF[j + 2 * i] - 2.0 / J43 * FC[i + 2 * j];
        }
    }

    // Get the second order derivatives of I1/J23 and I2/J43
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 2; j++)
        {
            for (k = 0; k < 2; k++)
            {
                for (l = 0; l < 2; l++)
                {
                    DDI1b[i + 2 * j][k + 2 * l] = 2.0 / J23 * Id2[i + 2 * k] * Id2[j + 2 * l] +
                                                  2.0 / 3.0 * I1b * IF[l + 2 * i] * IF[j + 2 * k] -
                                                  2.0 / 3.0 * DI1b[k + 2 * l] * IF[j + 2 * i] -
                                                  4.0 / (3.0 * J23) * F[i + 2 * j] * IF[l + 2 * k];
                    DDI2b[i + 2 * j][k + 2 * l] = 2.0 / J23 * I1b * Id2[i + 2 * k] * Id2[j + 2 * l] +
                                                  2.0 / J23 * DI1b[k + 2 * l] * F[i + 2 * j] +
                                                  4.0 / 3.0 * I2b * IF[l + 2 * i] * IF[j + 2 * k] +
                                                  8.0 / (3.0 * J43) * IF[l + 2 * k] * FC[i + 2 * j] -
                                                  4.0 / (3.0 * J23) * I1b * IF[l + 2 * k] * F[i + 2 * j] -
                                                  4.0 / 3.0 * DI2b[k + 2 * l] * IF[j + 2 * i] -
                                                  2.0 / J43 * (Id2[i + 2 * k] * C[l + 2 * j] + F[i + 2 * l] * F[k + 2 * j] + B[i + 2 * k] * Id2[j + 2 * l]);
                }
            }
        }
    }

    return 1;
}

// OOFEM (compressible Neo-hookean model)
int material_oofemF2d(const double matconst[8], const double F[4],
                      double &Wd, double P[4], double D[4][4])
{

    // Get material constants
    int errorcode;
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // Inverse of the deformation gradient and determinant of the deformation gradient
    double J;
    double IF[4];
    if (!getInverse22(F, J, IF))
        return -4;

    // Get derivatives of invariants with respect to F
    int i, j, k, l;
    double J13, J23, I1b, I2b, I132, logJ;
    double DI1b[4], DI2b[4], DDI1b[4][4], DDI2b[4][4];
    errorcode = getInvariantsDerivativesF2d(F, IF, J,                                                  // inputs
                                            I1b, I2b, I132, logJ, J13, J23, DI1b, DI2b, DDI1b, DDI2b); // outputs

    // Energy density W(F) = m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+1/2*kappa(ln(J))^2
    Wd = m1 * (I1b - 3.0) + m2 * (I2b - 3.0) + 1.0 / 2.0 * kappa * logJ * logJ;

    // The first Piola Kirchhoff stress tensor P
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            P[i + 2 * j] = m1 * DI1b[i + 2 * j] + m2 * DI2b[i + 2 * j] + kappa * logJ * IF[j + 2 * i];

    // Material stiffness D
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            for (k = 0; k < 2; k++)
                for (l = 0; l < 2; l++)
                    D[i + 2 * j][k + 2 * l] = m1 * DDI1b[i + 2 * j][k + 2 * l] +
                                              m2 * DDI2b[i + 2 * j][k + 2 * l] +
                                              kappa * (IF[j + 2 * i] * IF[l + 2 * k] -
                                                       logJ * IF[j + 2 * k] * IF[l + 2 * i]);

    return errorcode;
}

// Bertoldi, Boyce
int material_bbF2d(const double matconst[8], const double F[4],
                   double &Wd, double P[4], double D[4][4])
{

    // Get material constants
    int i, j, k, l;
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // Identity second-order tensor
    double Id2[4];
    Id2[0] = 1.0;
    Id2[1] = 0.0;
    Id2[2] = 0.0;
    Id2[3] = 1.0;

    // Inverse of the deformation gradient and determinant of the deformation gradient
    double J;
    double IF[4];
    if (!getInverse22(F, J, IF))
        return -4;

    // The left Cauchy-Green tensor B = F*F'
    double B[4];
    B[0] = F[0] * F[0] + F[2] * F[2];
    B[1] = F[1] * F[0] + F[3] * F[2];
    B[2] = F[0] * F[1] + F[2] * F[3];
    B[3] = F[1] * F[1] + F[3] * F[3];

    // The first invariant of B
    double I1 = B[0] + B[3] + 1.0;

    // Energy density W(F) = m1(I1-3)+m2(I1-3)^2-2m1*ln(J)+1/2*kappa(J-1)^2
    Wd = m1 * (I1 - 3.0) + m2 * pow(I1 - 3.0, 2.0) - 2.0 * m1 * log(J) + kappa / 2.0 * pow(J - 1.0, 2.0);

    // The first Piola Kirchhoff stress tensor P
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            P[i + 2 * j] = 2.0 * m1 * F[i + 2 * j] + 4.0 * m2 * (I1 - 3.0) * F[i + 2 * j] - 2.0 * m1 * IF[j + 2 * i] +
                           kappa * (J - 1.0) * J * IF[j + 2 * i];

    // Material stiffness D
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            for (k = 0; k < 2; k++)
                for (l = 0; l < 2; l++)
                    D[i + 2 * j][k + 2 * l] = 2.0 * m1 * Id2[i + 2 * k] * Id2[j + 2 * l] +
                                              8.0 * m2 * F[i + 2 * j] * F[k + 2 * l] +
                                              4.0 * m2 * (I1 - 3) * Id2[i + 2 * k] * Id2[j + 2 * l] +
                                              2.0 * m1 * IF[l + 2 * i] * IF[j + 2 * k] +
                                              kappa * J * J * IF[j + 2 * i] * IF[l + 2 * k] +
                                              kappa * (J - 1.0) * J * IF[j + 2 * i] * IF[l + 2 * k] -
                                              kappa * (J - 1.0) * J * IF[l + 2 * i] * IF[j + 2 * k];

    return 1;
}

// Jamus, Green, Simpson
int material_jgsF2d(const double matconst[8], const double F[4],
                    double &Wd, double P[4], double D[4][4])
{

    // Get material constants
    int errorcode;
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double m3 = matconst[3];    // material parameter
    double m4 = matconst[4];    // material parameter
    double m5 = matconst[5];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // Inverse of the deformation gradient and determinant of the deformation gradient
    double J;
    double IF[4];
    if (!getInverse22(F, J, IF))
        return -4;

    // Get derivatives of invariants with respect to F
    int i, j, k, l;
    double J13, J23, I1b, I2b, I132, logJ;
    double DI1b[4], DI2b[4], DDI1b[4][4], DDI2b[4][4];
    errorcode = getInvariantsDerivativesF2d(F, IF, J,                                                  // inputs
                                            I1b, I2b, I132, logJ, J13, J23, DI1b, DI2b, DDI1b, DDI2b); // outputs

    // Energy density W(F) = m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I1/J^{2/3}-3)^3+9/2*kappa(J^{1/3}-1)^2
    Wd = m1 * (I1b - 3.0) + m2 * (I2b - 3.0) + m3 * (I1b - 3.0) * (I2b - 3.0) + m4 * I132 + m5 * pow(I1b - 3.0, 3.0) + 9.0 / 2.0 * kappa * pow(J13 - 1.0, 2.0);

    // The first Piola Kirchhoff stress tensor P
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            P[i + 2 * j] = m1 * DI1b[i + 2 * j] +
                           m2 * DI2b[i + 2 * j] +
                           m3 * ((I2b - 3.0) * DI1b[i + 2 * j] +
                                 (I1b - 3.0) * DI2b[i + 2 * j]) +
                           2.0 * m4 * (I1b - 3.0) * DI1b[i + 2 * j] +
                           3.0 * m5 * I132 * DI1b[i + 2 * j] +
                           3.0 * kappa * J13 * (J13 - 1.0) * IF[j + 2 * i];

    // Material stiffness D
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            for (k = 0; k < 2; k++)
                for (l = 0; l < 2; l++)
                    D[i + 2 * j][k + 2 * l] = m1 * DDI1b[i + 2 * j][k + 2 * l] +
                                              m2 * DDI2b[i + 2 * j][k + 2 * l] +
                                              m3 * ((I2b - 3.0) * DDI1b[i + 2 * j][k + 2 * l] +
                                                    DI1b[i + 2 * j] * DI2b[k + 2 * l] +
                                                    DI1b[k + 2 * l] * DI2b[i + 2 * j] +
                                                    (I1b - 3.0) * DDI2b[i + 2 * j][k + 2 * l]) +
                                              2.0 * m4 * (DI1b[k + 2 * l] * DI1b[i + 2 * j] + (I1b - 3.0) * DDI1b[i + 2 * j][k + 2 * l]) +
                                              3.0 * m5 * (2.0 * (I1b - 3.0) * DI1b[k + 2 * l] * DI1b[i + 2 * j] + I132 * DDI1b[i + 2 * j][k + 2 * l]) +
                                              kappa * (J13 * (J13 - 1.0) * IF[j + 2 * i] * IF[l + 2 * k] +
                                                       J23 * IF[j + 2 * i] * IF[l + 2 * k] -
                                                       3.0 * J13 * (J13 - 1.0) * IF[l + 2 * i] * IF[j + 2 * k]);

    return errorcode;
}

// five-term Mooney-Rivlin
int material_5mrF2d(const double matconst[8], const double F[4],
                    double &Wd, double P[4], double D[4][4])
{

    // Get material constants
    int errorcode;
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double m3 = matconst[3];    // material parameter
    double m4 = matconst[4];    // material parameter
    double m5 = matconst[5];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // Inverse of the deformation gradient and determinant of the deformation gradient
    double J;
    double IF[4];
    if (!getInverse22(F, J, IF))
        return -4;

    // Get derivatives of invariants with respect to F
    int i, j, k, l;
    double J13, J23, I1b, I2b, I132, logJ;
    double DI1b[4], DI2b[4], DDI1b[4][4], DDI2b[4][4];
    errorcode = getInvariantsDerivativesF2d(F, IF, J,                                                  // inputs
                                            I1b, I2b, I132, logJ, J13, J23, DI1b, DI2b, DDI1b, DDI2b); // outputs

    // Energy density W(F) = m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I2/J^{2/3}-3)^2+kappa*(J-1)^2
    Wd = m1 * (I1b - 3.0) + m2 * (I2b - 3.0) + m3 * (I1b - 3.0) * (I2b - 3.0) + m4 * pow(I1b - 3.0, 2.0) + m5 * pow(I2b - 3.0, 2.0) + kappa * pow(J - 1.0, 2.0);

    // The first Piola Kirchhoff stress tensor P
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            P[i + 2 * j] = m1 * DI1b[i + 2 * j] +
                           m2 * DI2b[i + 2 * j] +
                           m3 * ((I2b - 3.0) * DI1b[i + 2 * j] +
                                 (I1b - 3.0) * DI2b[i + 2 * j]) +
                           2.0 * m4 * (I1b - 3.0) * DI1b[i + 2 * j] +
                           2.0 * m5 * (I2b - 3.0) * DI2b[i + 2 * j] +
                           2.0 * kappa * J * (J - 1.0) * IF[j + 2 * i];

    // Material stiffness D
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            for (k = 0; k < 2; k++)
                for (l = 0; l < 2; l++)
                    D[i + 2 * j][k + 2 * l] = m1 * DDI1b[i + 2 * j][k + 2 * l] +
                                              m2 * DDI2b[i + 2 * j][k + 2 * l] +
                                              m3 * ((I2b - 3.0) * DDI1b[i + 2 * j][k + 2 * l] +
                                                    DI1b[i + 2 * j] * DI2b[k + 2 * l] +
                                                    DI1b[k + 2 * l] * DI2b[i + 2 * j] +
                                                    (I1b - 3.0) * DDI2b[i + 2 * j][k + 2 * l]) +
                                              2.0 * m4 * (DI1b[k + 2 * l] * DI1b[i + 2 * j] + (I1b - 3.0) * DDI1b[i + 2 * j][k + 2 * l]) +
                                              2.0 * m5 * (DI2b[k + 2 * l] * DI2b[i + 2 * j] + (I2b - 3.0) * DDI2b[i + 2 * j][k + 2 * l]) +
                                              2.0 * kappa * (J * J * IF[j + 2 * i] * IF[l + 2 * k] + J * (J - 1.0) * IF[j + 2 * i] * IF[l + 2 * k] - J * (J - 1.0) * IF[l + 2 * i] * IF[j + 2 * k]);

    return errorcode;
}

// Linear elastic
int material_leF2d(const double matconst[8], const double F[4],
                   double &Wd, double P[4], double D[4][4])
{

    // Get material constants
    int i, j, k, l, m, n;
    double mu = matconst[1];     // Lame's parameter (shear modulus G)
    double lambda = matconst[6]; // Lame's parameter (K-2/3*G)

    // The Green-Lagrange strain tensor E = 0.5*(F'*F-I)
    double C[3];
    double E[4];
    getC2d(F, C);
    E[0] = 0.5 * (C[0] - 1.0);
    E[1] = 0.5 * C[2];
    E[2] = 0.5 * C[2];
    E[3] = 0.5 * (C[1] - 1.0);

    // Identity second-order tensor
    double Id2[4];
    Id2[0] = 1.0;
    Id2[1] = 0.0;
    Id2[2] = 0.0;
    Id2[3] = 1.0;

    // Material stiffness C
    double CD[4][4];
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            for (k = 0; k < 2; k++)
                for (l = 0; l < 2; l++)
                    CD[i + 2 * j][k + 2 * l] = lambda * Id2[i + 2 * j] * Id2[k + 2 * l] +
                                               mu * (Id2[i + 2 * k] * Id2[j + 2 * l] + Id2[i + 2 * l] * Id2[j + 2 * k]);

    // The second P-K stress
    double S[4];
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
        {
            S[i + 2 * j] = 0.0;
            for (k = 0; k < 2; k++)
                for (l = 0; l < 2; l++)
                    S[i + 2 * j] += CD[i + 2 * j][k + 2 * l] * E[k + 2 * l];
        }

    // Energy density W(F) = 0.5*(0.5*(C-I)*(m1*)*0.5*(C-I)) (linear elastic material)
    Wd = 0.0;
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            Wd += 0.5 * S[i + 2 * j] * E[i + 2 * j];

    // The first Piola Kirchhoff stress tensor P
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
        {
            P[i + 2 * j] = 0.0;
            for (k = 0; k < 2; k++)
                P[i + 2 * j] += F[i + 2 * k] * S[k + 2 * j];
        }

    // Material stiffness D
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            for (k = 0; k < 2; k++)
                for (l = 0; l < 2; l++)
                {
                    D[i + 2 * j][k + 2 * l] = Id2[i + 2 * k] * S[j + 2 * l];
                    for (m = 0; m < 2; m++)
                        for (n = 0; n < 2; n++)
                            D[i + 2 * j][k + 2 * l] += F[i + 2 * m] * F[k + 2 * n] * CD[m + 2 * j][n + 2 * l];
                }

    return 1;
}

/******************** 2d Constitutive laws formulated in E  ********************/

// Get derivatives of invariants with respect to C
int getInvariantsDerivativesC2d(const double C[3], double &I1, double &I2, double &J,
                                double &J2, double dI1dC[3], double dI2dC[3],
                                double dJdC[3], double d2I2dCC[3][3],
                                double d2JdCC[3][3])
{

    // Get invariants
    int i, j, errorcode;
    I1 = C[0] + C[1] + 1.0;
    J2 = C[0] * C[1] - C[2] * C[2];
    J = sqrt(J2);
    I2 = 0.5 * (I1 * I1 - (C[0] * C[0] + C[1] * C[1] + 2.0 * C[2] * C[2] + 1.0));

    // Get inverse of C
    double IC[3];
    if (!getInverse22sym(C, J2, IC)) // det(C) = J2, where det(F) = J
        return -4;

    // Get first derivatives
    getId2d(dI1dC);
    for (i = 0; i < 3; i++)
    {
        dI2dC[i] = I1 * dI1dC[i] - C[i];
        dJdC[i] = 0.5 * J * IC[i];
    }

    // Get second derivatives
    double Id4[3][3];
    errorcode = getSymmetricProduct2d(Id4, dI1dC, dI1dC);
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            d2I2dCC[i][j] = dI1dC[i] * dI1dC[j] - Id4[i][j];

    double IC4[3][3];
    errorcode = getSymmetricProduct2d(IC4, IC, IC);
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            d2JdCC[i][j] = J * (IC[i] * IC[j] - 2.0 * IC4[i][j]);

    return errorcode;
}

// OOFEM (compressible Neo-hookean model)
int material_oofemE2d(const double matconst[8], const double C[3], // inputs
                      double &w, double S[3], double D[3][3])
{

    // Get material constants
    int i, j, errorcode;
    double I1, I2, J, J2, dI1dC[3], dI2dC[3], dJdC[3], d2I2dCC[3][3], d2JdCC[3][3];
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // The invariants of C and their derivatives
    errorcode = getInvariantsDerivativesC2d(C, I1, I2, J, J2, dI1dC, dI2dC, dJdC, d2I2dCC, d2JdCC);

    // Normalized invariants and constants
    double J13 = 1 / pow(J, 1.0 / 3.0);
    double J23 = J13 * J13;
    double J43 = J23 * J23;
    double J53 = J43 * J13;
    double J73 = J53 * J23;
    double J83 = J73 * J13;
    double J103 = J83 * J23;
    double logJ = log(J);
    double In1 = I1 * J23;
    double In2 = I2 * J43;

    // Modified invariants of C: In1 = I1/J^{2/3}, In2 = I2/J^{4/3}
    // Energy density W(F) = m1(In1-3)+m2(In2-3)+1/2*kappa(ln(J))^2
    w = m1 * (In1 - 3.0) +
        m2 * (In2 - 3.0) +
        1.0 / 2.0 * kappa * logJ * logJ;

    // The second Piola-Kirchhoff stress tensor S
    double dwdI1 = m1 * J23;
    double dwdI2 = m2 * J43;
    double dwdJ = -2.0 / 3.0 * m1 * I1 * J53 -
                  4.0 / 3.0 * m2 * I2 * J73 +
                  kappa * logJ / J;
    for (i = 0; i < 3; i++)
        S[i] = 2.0 * (dwdI1 * dI1dC[i] + dwdI2 * dI2dC[i] + dwdJ * dJdC[i]);

    // Material stiffness D
    double d2wdJJ = 10.0 / 9.0 * m1 * I1 * J83 +
                    28.0 / 9.0 * m2 * I2 * J103 +
                    kappa * (1.0 - logJ) / J2;
    double d2wdI1J = -2.0 / 3.0 * m1 * J53;
    double d2wdI2J = -4.0 / 3.0 * m2 * J73;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            D[i][j] = 4.0 * (d2wdJJ * dJdC[i] * dJdC[j] +
                             d2wdI1J * (dI1dC[i] * dJdC[j] + dJdC[i] * dI1dC[j]) +
                             d2wdI2J * (dI2dC[i] * dJdC[j] + dJdC[i] * dI2dC[j])) +
                      4.0 * dwdI2 * d2I2dCC[i][j] +
                      dwdJ * d2JdCC[i][j];

    return errorcode;
}

// Bertoldi, Boyce
int material_bbE2d(const double matconst[8], const double C[3],
                   double &w, double S[3], double D[3][3])
{
    // Get material constants
    int i, j, errorcode;
    double I1, I2, J, J2, dI1dC[3], dI2dC[3], dJdC[3], d2I2dCC[3][3], d2J3dCC[3][3];
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // The invariants of C and their derivatives
    errorcode = getInvariantsDerivativesC2d(C, I1, I2, J, J2, dI1dC, dI2dC, dJdC, d2I2dCC, d2J3dCC);

    // Energy density W(F) = m1(I1-3)+m2(I1-3)^2-2m1*ln(J)+1/2*kappa(J-1)^2
    w = m1 * (I1 - 3.0) +
        m2 * pow(I1 - 3.0, 2.0) -
        2.0 * m1 * log(J) +
        kappa / 2.0 * pow(J - 1.0, 2.0);

    // The second Piola-Kirchhoff stress tensor S
    double dwdI1 = m1 + 2.0 * m2 * (I1 - 3.0);
    double dwdJ = -2.0 * m1 / J + kappa * (J - 1.0);
    for (i = 0; i < 3; i++)
        S[i] = 2.0 * (dwdI1 * dI1dC[i] + dwdJ * dJdC[i]);

    // Material stiffness D
    double d2wdI1I1 = 2.0 * m2;
    double d2wdJJ = 2.0 * m1 / J2 + kappa;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            D[i][j] = 4.0 * (d2wdI1I1 * dI1dC[i] * dI1dC[j] +
                             d2wdJJ * dJdC[i] * dJdC[j]) +
                      dwdJ * d2J3dCC[i][j];

    return errorcode;
}

// Jamus, Green, Simpson
int material_jgsE2d(const double matconst[8], const double C[3],
                    double &w, double S[3], double D[3][3])
{

    // Get material constants
    int i, j, errorcode;
    double I1, I2, J, J2, dI1dC[3], dI2dC[3], dJdC[3], d2I2dCC[3][3], d2JdCC[3][3];
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double m3 = matconst[3];    // material parameter
    double m4 = matconst[4];    // material parameter
    double m5 = matconst[5];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // The invariants of C and their derivatives
    errorcode = getInvariantsDerivativesC2d(C, I1, I2, J, J2, dI1dC, dI2dC, dJdC, d2I2dCC, d2JdCC);

    // Normalized invariants and constants
    double J13 = 1 / pow(J, 1.0 / 3.0);
    double J23 = J13 * J13;
    double J43 = J23 * J23;
    double J53 = J43 * J13;
    double J63 = J53 * J13;
    double J73 = J63 * J13;
    double J83 = J73 * J13;
    double J93 = J83 * J13;
    double J103 = J93 * J13;
    double J123 = J103 * J23;
    double In1 = I1 * J23;
    double In2 = I2 * J43;

    // Modified invariants of C: In1 = I1/J^{2/3}, In2 = I2/J^{4/3}
    // Energy density W(F) = m1(In1-3)+m2(In2-3)+m3(In1-3)*(In2-3)+m4(In1-3)^2+m5(In1-3)^3+9/2*kappa(J^{1/3}-1)^2
    w = m1 * (In1 - 3.0) + m2 * (In2 - 3.0) +
        m3 * (In1 - 3.0) * (In2 - 3.0) +
        m4 * pow(In1 - 3.0, 2.0) +
        m5 * pow(In1 - 3.0, 3.0) +
        9.0 / 2.0 * kappa * pow(1 / J13 - 1.0, 2.0);

    // The second Piola-Kirchhoff stress tensor S
    double dwdI1 = m1 * J23 +
                   m3 * (In2 - 3.0) * J23 +
                   2.0 * m4 * (In1 - 3.0) * J23 +
                   3.0 * m5 * pow(In1 - 3.0, 2.0) * J23;
    double dwdI2 = m2 * J43 +
                   m3 * (In1 - 3.0) * J43;
    double dwdJ = -2.0 / 3.0 * m1 * I1 * J53 -
                  4.0 / 3.0 * m2 * I2 * J73 -
                  2.0 / 3.0 * m3 * (I1 * (In2 - 3.0) * J53 + 2.0 * (In1 - 3.0) * I2 * J73) -
                  4.0 / 3.0 * m4 * (In1 - 3.0) * I1 * J53 -
                  2.0 * m5 * pow(In1 - 3.0, 2.0) * I1 * J53 +
                  3.0 * kappa * (1 / J13 - 1.0) * J23;
    for (i = 0; i < 3; i++)
        S[i] = 2.0 * (dwdI1 * dI1dC[i] + dwdI2 * dI2dC[i] + dwdJ * dJdC[i]);

    // Material stiffness D
    double d2wdI1I1 = 2.0 * m4 * J43 +
                      6.0 * m5 * (In1 - 3.0) * J43;
    double d2wdJJ = 10.0 / 9.0 * m1 * I1 * J83 +
                    28.0 / 9.0 * m2 * I2 * J103 +
                    2.0 / 9.0 * m3 * (4.0 * I1 * I2 * J123 + 5.0 * (In2 - 3.0) * I1 * J83 + 4.0 * I1 * I2 * J123 + 14.0 * (In1 - 3.0) * I2 * J103) +
                    8.0 / 9.0 * m4 * I1 * I1 * J103 +
                    20.0 / 9.0 * m4 * (In1 - 3.0) * I1 * J83 +
                    8.0 / 3.0 * m5 * (In1 - 3.0) * I1 * I1 * J103 +
                    10.0 / 3.0 * m5 * pow(In1 - 3.0, 2.0) * I1 * J83 +
                    kappa * (1.0 * J43 - 2.0 * (1 / J13 - 1.0) * J53);
    double d2wdI1I2 = m3 * J63;
    double d2wdI1J = -2.0 / 3.0 * m1 * J53 -
                     m3 * 2.0 / 3.0 * (2.0 * I2 * J93 + (In2 - 3.0) * J53) -
                     4.0 / 3.0 * m4 * (I1 * J73 + (In1 - 3.0) * J53) -
                     2.0 * m5 * (2.0 * (In1 - 3.0) * I1 * J73 + pow(In1 - 3.0, 2.0) * J53);
    double d2wdI2J = -4.0 / 3.0 * m2 * J73 -
                     2.0 / 3.0 * m3 * (I1 * J93 + 2.0 * (In1 - 3.0) * J73);
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            D[i][j] = 4.0 * (d2wdI1I1 * dI1dC[i] * dI1dC[j] +
                             d2wdJJ * dJdC[i] * dJdC[j] +
                             d2wdI1I2 * (dI1dC[i] * dI2dC[j] + dI2dC[i] * dI1dC[j]) +
                             d2wdI1J * (dI1dC[i] * dJdC[j] + dJdC[i] * dI1dC[j]) +
                             d2wdI2J * (dI2dC[i] * dJdC[j] + dJdC[i] * dI2dC[j])) +
                      4.0 * dwdI2 * d2I2dCC[i][j] +
                      dwdJ * d2JdCC[i][j];

    return errorcode;
}

// five-term Mooney-Rivlin
int material_5mrE2d(const double matconst[8], const double C[3],
                    double &w, double S[3], double D[3][3])
{

    // Get material constants
    int i, j, errorcode;
    double I1, I2, J, J2, dI1dC[3], dI2dC[3], dJdC[3], d2I2dCC[3][3], d2JdCC[3][3];
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double m3 = matconst[3];    // material parameter
    double m4 = matconst[4];    // material parameter
    double m5 = matconst[5];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // The invariants of C and their derivatives
    errorcode = getInvariantsDerivativesC2d(C, I1, I2, J, J2, dI1dC, dI2dC, dJdC, d2I2dCC, d2JdCC);

    // Normalized invariants and constants
    double J13 = 1 / pow(J, 1.0 / 3.0);
    double J23 = J13 * J13;
    double J43 = J23 * J23;
    double J53 = J43 * J13;
    double J63 = J53 * J13;
    double J73 = J63 * J13;
    double J83 = J73 * J13;
    double J93 = J83 * J13;
    double J103 = J93 * J13;
    double J113 = J103 * J13;
    double J123 = J103 * J23;
    double J143 = J73 * J73;
    double In1 = I1 * J23;
    double In2 = I2 * J43;

    // Modified invariants of C: In1 = I1/J^{2/3}, In2 = I2/J^{4/3}
    // Energy density W(F) = m1(In1-3)+m2(In2-3)+m3(In1-3)*(In2-3)+m4(In1-3)^2+m5(In2-3)^2+kappa*(J-1)^2
    w = m1 * (In1 - 3.0) +
        m2 * (In2 - 3.0) +
        m3 * (In1 - 3.0) * (In2 - 3.0) +
        m4 * pow(In1 - 3.0, 2.0) +
        m5 * pow(In2 - 3.0, 2.0) +
        kappa * pow(J - 1.0, 2.0);

    // The second Piola-Kirchhoff stress tensor S
    double dwdI1 = m1 * J23 +
                   m3 * (In2 - 3.0) * J23 +
                   2.0 * m4 * (In1 - 3.0) * J23;
    double dwdI2 = m2 * J43 +
                   m3 * (In1 - 3.0) * J43 +
                   2 * m5 * (In2 - 3.0) * J43;
    double dwdJ = -2.0 / 3.0 * m1 * I1 * J53 -
                  4.0 / 3.0 * m2 * I2 * J73 -
                  2.0 / 3.0 * m3 * (I1 * (In2 - 3.0) * J53 + 2.0 * (In1 - 3.0) * I2 * J73) -
                  4.0 / 3.0 * m4 * (In1 - 3.0) * I1 * J53 -
                  8.0 / 3.0 * m5 * (In2 - 3.0) * I2 * J73 +
                  2.0 * kappa * (J - 1.0);
    for (i = 0; i < 3; i++)
        S[i] = 2.0 * (dwdI1 * dI1dC[i] + dwdI2 * dI2dC[i] + dwdJ * dJdC[i]);

    // Material stiffness D
    double d2wdI1I1 = 2.0 * m4 * J43;
    double d2wdI2I2 = 2.0 * m5 * J83;
    double d2wdJJ = 10.0 / 9.0 * m1 * I1 * J83 +
                    28.0 / 9.0 * m2 * I2 * J103 +
                    2.0 / 9.0 * m3 * (4.0 * I1 * I2 * J123 + 5.0 * (In2 - 3.0) * I1 * J83 + 4.0 * I1 * I2 * J123 + 14.0 * (In1 - 3.0) * I2 * J103) +
                    8.0 / 9.0 * m4 * I1 * I1 * J103 +
                    20.0 / 9.0 * m4 * (In1 - 3.0) * I1 * J83 +
                    32.0 / 9.0 * m5 * I2 * I2 * J143 +
                    56.0 / 9.0 * m5 * (In2 - 3.0) * I2 * J103 +
                    2.0 * kappa;
    double d2wdI1I2 = m3 * J63;
    double d2wdI1J = -2.0 / 3.0 * m1 * J53 -
                     m3 * 2.0 / 3.0 * (2.0 * I2 * J93 + (In2 - 3.0) * J53) -
                     4.0 / 3.0 * m4 * (I1 * J73 + (In1 - 3.0) * J53);
    double d2wdI2J = -4.0 / 3.0 * m2 * J73 -
                     2.0 / 3.0 * m3 * (I1 * J93 + 2.0 * (In1 - 3.0) * J73) -
                     8.0 / 3.0 * m5 * (I2 * J113 + (In2 - 3.0) * J73);
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            D[i][j] = 4.0 * (d2wdI1I1 * dI1dC[i] * dI1dC[j] +
                             d2wdI2I2 * dI2dC[i] * dI2dC[j] +
                             d2wdJJ * dJdC[i] * dJdC[j] +
                             d2wdI1I2 * (dI1dC[i] * dI2dC[j] + dI2dC[i] * dI1dC[j]) +
                             d2wdI1J * (dI1dC[i] * dJdC[j] + dJdC[i] * dI1dC[j]) +
                             d2wdI2J * (dI2dC[i] * dJdC[j] + dJdC[i] * dI2dC[j])) +
                      4.0 * dwdI2 * d2I2dCC[i][j] +
                      dwdJ * d2JdCC[i][j];

    return errorcode;
}

// Linear elastic
int material_leE2d(const double matconst[8], const double C[3],
                   double &w, double S[3], double D[3][3])
{

    // Get material constants
    int i, j;
    double mu = matconst[1];     // Lame's parameter (shear modulus G)
    double lambda = matconst[6]; // Lame's parameter (K-2/3*G)

    // Get stiffness matrix
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            D[i][j] = 0.0;
    double nu = lambda / (2.0 * (lambda + mu));
    double Et = mu * (3.0 * lambda + 2.0 * mu) * (1.0 - nu) / ((lambda + mu) * (1 - 2.0 * nu) * (1 + nu));
    D[0][0] = Et;
    D[0][1] = Et * nu / (1.0 - nu);
    D[1][0] = Et * nu / (1.0 - nu);
    D[1][1] = Et;
    D[2][2] = mu;

    // Get the second P-K stress
    double E[3];
    E[0] = 0.5 * (C[0] - 1.0);
    E[1] = 0.5 * (C[1] - 1.0);
    E[2] = C[2];
    for (i = 0; i < 3; i++)
    {
        S[i] = 0.0;
        for (j = 0; j < 3; j++)
            S[i] += D[i][j] * E[j];
    }

    // Get energy
    w = 0.0;
    for (i = 0; i < 3; i++)
        w += 0.5 * S[i] * E[i];

    return 1;
}

// Ogden
int material_ogdenE2d(const double matconst[8], const double C[3], double &w,
                      double S[3], double D[3][3])
{

    // Get material constants
    int i, j, k, l, errorcode;
    double Stretch[2], J, J2;
    double c1 = matconst[1];    // material parameter
    double m1 = matconst[2];    // material parameter
    double c2 = matconst[3];    // material parameter
    double m2 = matconst[4];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // Get spectral decomposition of C
    Eigen::Matrix2d Cd;
    Cd << C[0], C[2],
        C[2], C[1];
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(Cd);
    Eigen::Matrix2d Veig = eigensolver.eigenvectors();
    Eigen::Vector2d Deig = eigensolver.eigenvalues();
    J2 = Cd.determinant(); // det(C) = J2, where det(F) = J
    J = sqrt(J2);
    Stretch[0] = sqrt(Deig[0]);
    Stretch[1] = sqrt(Deig[1]);

    // Energy density W(F) = c1/m1^2*(lambda1^m1+lambda2^m1+lambda3^m1-3-m1*log(J))+c2/m2^2*(lambda1^m2+lambda2^m2+lambda3^m2-3-m2*log(J))+kappa/2*(J-1)^2 (Ogden)
    double logJ = log(J);
    w = c1 / (m1 * m1) * (pow(Stretch[0], m1) + pow(Stretch[1], m1) + 1.0 - 3.0 - m1 * logJ) +
        c2 / (m2 * m2) * (pow(Stretch[0], m2) + pow(Stretch[1], m2) + 1.0 - 3.0 - m2 * logJ) +
        kappa / 2.0 * (J - 1.0) * (J - 1.0);

    // Compute derivatives of the energy density with respect to stretches and constants (treating J = lambda1*lambda2*1.0)
    double dwdL[2];
    dwdL[0] = c1 / m1 * pow(Stretch[0], m1 - 1.0) + c2 / m2 * pow(Stretch[0], m2 - 1.0) - (c1 / m1 + c2 / m2 - J * kappa * (J - 1.0)) / J * Stretch[1];
    dwdL[1] = c1 / m1 * pow(Stretch[1], m1 - 1.0) + c2 / m2 * pow(Stretch[1], m2 - 1.0) - (c1 / m1 + c2 / m2 - J * kappa * (J - 1.0)) / J * Stretch[0];

    double d2wd2L[2][2];
    d2wd2L[0][0] = c1 / m1 * (m1 - 1.0) * pow(Stretch[0], m1 - 2.0) + c2 / m2 * (m2 - 1.0) * pow(Stretch[0], m2 - 2.0) + (c1 / m1 + c2 / m2 + J2 * kappa) / J2 * Deig[1];
    d2wd2L[1][1] = c1 / m1 * (m1 - 1.0) * pow(Stretch[1], m1 - 2.0) + c2 / m2 * (m2 - 1.0) * pow(Stretch[1], m2 - 2.0) + (c1 / m1 + c2 / m2 + J2 * kappa) / J2 * Deig[0];
    d2wd2L[0][1] = kappa * (2.0 * J - 1.0);
    d2wd2L[1][0] = d2wd2L[0][1];

    // The second Piola-Kirchhoff stress tensor S
    double tS[2][2];
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            tS[i][j] = dwdL[0] / Stretch[0] * Veig(i, 0) * Veig(j, 0) +
                       dwdL[1] / Stretch[1] * Veig(i, 1) * Veig(j, 1);

    // Map tS on Voight notation
    int idmap[2][3] = {{0, 1, 0}, {0, 1, 1}}; // map from tensor to Voight notation
    for (i = 0; i < 3; i++)
        S[i] = tS[idmap[0][i]][idmap[1][i]];

    // Material stiffness D
    double tD[2][2][2][2];
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            for (k = 0; k < 2; k++)
                for (l = 0; l < 2; l++)
                    tD[i][j][k][l] = 1.0 / Stretch[0] * (-1.0 / Deig[0] * dwdL[0] + 1.0 / Stretch[0] * d2wd2L[0][0]) * Veig(i, 0) * Veig(j, 0) * Veig(k, 0) * Veig(l, 0) +
                                     1.0 / Stretch[0] * (1.0 / Stretch[1] * d2wd2L[0][1]) * Veig(i, 0) * Veig(j, 0) * Veig(k, 1) * Veig(l, 1) +
                                     1.0 / Stretch[1] * (1.0 / Stretch[0] * d2wd2L[1][0]) * Veig(i, 1) * Veig(j, 1) * Veig(k, 0) * Veig(l, 0) +
                                     1.0 / Stretch[1] * (-1.0 / Deig[1] * dwdL[1] + 1.0 / Stretch[1] * d2wd2L[1][1]) * Veig(i, 1) * Veig(j, 1) * Veig(k, 1) * Veig(l, 1);
    if (fabs(Stretch[1] - Stretch[0]) > EPS) // two distinct eigenvalues
    {
        for (i = 0; i < 2; i++)
            for (j = 0; j < 2; j++)
                for (k = 0; k < 2; k++)
                    for (l = 0; l < 2; l++)
                        tD[i][j][k][l] += (dwdL[1] / Stretch[1] - dwdL[0] / Stretch[0]) / (Deig[1] - Deig[0]) * (Veig(i, 0) * Veig(j, 1) * Veig(k, 0) * Veig(l, 1) + Veig(i, 0) * Veig(j, 1) * Veig(k, 1) * Veig(l, 0)) +
                                          (dwdL[0] / Stretch[0] - dwdL[1] / Stretch[1]) / (Deig[0] - Deig[1]) * (Veig(i, 1) * Veig(j, 0) * Veig(k, 1) * Veig(l, 0) + Veig(i, 1) * Veig(j, 0) * Veig(k, 0) * Veig(l, 1));
    }
    else // multiple eigenvalues
    {
        for (i = 0; i < 2; i++)
            for (j = 0; j < 2; j++)
                for (k = 0; k < 2; k++)
                    for (l = 0; l < 2; l++)
                        tD[i][j][k][l] += 0.5 / Stretch[1] * ((-1.0 / Deig[1] * dwdL[1] + 1.0 / Stretch[1] * d2wd2L[1][1]) - (1.0 / Stretch[0] * d2wd2L[0][1])) * (Veig(i, 0) * Veig(j, 1) * Veig(k, 0) * Veig(l, 1) + Veig(i, 0) * Veig(j, 1) * Veig(k, 1) * Veig(l, 0)) +
                                          0.5 / Stretch[0] * ((-1.0 / Deig[0] * dwdL[0] + 1.0 / Stretch[0] * d2wd2L[0][0]) - (1.0 / Stretch[1] * d2wd2L[1][0])) * (Veig(i, 1) * Veig(j, 0) * Veig(k, 1) * Veig(l, 0) + Veig(i, 1) * Veig(j, 0) * Veig(k, 0) * Veig(l, 1));
    }

    // Map tD on Voight notation
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            D[i][j] = tD[idmap[0][i]][idmap[1][i]][idmap[0][j]][idmap[1][j]];

    errorcode = 1;
    return errorcode;
}

// Ogden nematic
int material_ogdennematicE2d(const double matconst[8], const double C[3], double &w,
                             double S[3], double D[3][3])
{

    // Get material constants
    int i, j, k, l, errorcode;
    double Stretch[2], J, J2;
    double c = matconst[1]; // material parameter
    double a = matconst[2]; // material parameter
    double a13 = pow(a, 1.0 / 3.0);
    double kappa = matconst[6]; // bulk modulus

    // Get spectral decomposition of C
    Eigen::Matrix2d Cd;
    Cd << C[0], C[2],
        C[2], C[1];
    // By default sorted in increasing order
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(Cd);
    Eigen::Matrix2d Veig = eigensolver.eigenvectors();
    Eigen::Vector2d Deig = eigensolver.eigenvalues();
    if (Deig[0] > Deig[1])
        errorcode = -1;
    J2 = Cd.determinant(); // det(C) = J2, where det(F) = J
    J = sqrt(J2);
    double J13 = pow(J, 1.0 / 3.0);
    Stretch[0] = sqrt(Deig[0]);
    double Stretcha13 = pow(Stretch[0], 1.0 / 3.0);
    double Stretcha23 = Stretcha13 * Stretcha13;
    double Stretcha43 = Stretcha23 * Stretcha23;
    double Stretcha53 = Stretcha43 * Stretcha13;
    double Stretcha83 = Stretcha43 * Stretcha43;
    Stretch[1] = sqrt(Deig[1]);
    double Stretchb13 = pow(Stretch[1], 1.0 / 3.0);
    double Stretchb23 = Stretchb13 * Stretchb13;
    double Stretchb43 = Stretchb23 * Stretchb23;
    double Stretchb53 = Stretchb43 * Stretchb13;
    double Stretchb83 = Stretchb43 * Stretchb43;

    // Energy density W(F) = m1/2*m2^(1/3)*(tlambda1^2+tlambda2^2+tlambda3^2/m2-3*m2^(-1/3))+kappa/2*(J-1)^2 (Nematic materials Martin)
    w = c / 2.0 * a13 * (pow(Stretch[0] / J13, 2.0) + pow(Stretch[1] / J13, 2.0) / a + pow(1.0 / J13, 2.0) - 3.0 / a13) + kappa / 2.0 * (J - 1.0) * (J - 1.0);

    // Compute derivatives of the energy density with respect to stretches and constants (treating J = lambda1*lambda2*1.0)
    double dwdL[2];
    dwdL[0] = c * a13 * (2.0 / 3.0 * Stretcha13 / Stretchb23 - 1.0 / (3.0 * a) * Stretchb43 / Stretcha53 - 1.0 / (3.0 * Stretcha53 * Stretchb23)) + kappa * (J - 1.0) * Stretch[1];
    dwdL[1] = c * a13 * (-1.0 / 3.0 * Stretcha43 / Stretchb53 + 2.0 / (3.0 * a) * Stretchb13 / Stretcha23 - 1.0 / (3.0 * Stretcha23 * Stretchb53)) + kappa * (J - 1.0) * Stretch[0];

    double d2wd2L[2][2];
    d2wd2L[0][0] = c * a13 * (2.0 / 9.0 * 1.0 / pow(Stretch[0] * Stretch[1], 2.0 / 3.0) + 5.0 / (9.0 * a) * Stretchb43 / Stretcha83 + 5.0 / (9.0 * Stretcha83 * Stretchb23)) + kappa * Deig[1];
    d2wd2L[1][1] = c * a13 * (5.0 / (9.0 * a) * Stretcha43 / Stretchb83 + 2.0 / (9.0 * a) * 1.0 / pow(Stretch[0] * Stretch[1], 2.0 / 3.0) + 5.0 / (9.0 * Stretchb83 * Stretcha23)) + kappa * Deig[0];
    d2wd2L[0][1] = c * a13 * (-4.0 / 9.0 * Stretcha13 / Stretchb53 - 4.0 / (9.0 * a) * Stretchb13 / Stretcha53 + 2.0 / (9.0 * pow(Stretch[0] * Stretch[1], 5.0 / 3.0))) + kappa * (2.0 * J - 1.0);
    d2wd2L[1][0] = d2wd2L[0][1];

    // The second Piola-Kirchhoff stress tensor S
    double tS[2][2];
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            tS[i][j] = dwdL[0] / Stretch[0] * Veig(i, 0) * Veig(j, 0) +
                       dwdL[1] / Stretch[1] * Veig(i, 1) * Veig(j, 1);

    // Map tS on Voight notation
    int idmap[2][3] = {{0, 1, 0}, {0, 1, 1}}; // map from tensor to Voight notation
    for (i = 0; i < 3; i++)
        S[i] = tS[idmap[0][i]][idmap[1][i]];

    // Material stiffness D
    double tD[2][2][2][2];
    for (i = 0; i < 2; i++)
        for (j = 0; j < 2; j++)
            for (k = 0; k < 2; k++)
                for (l = 0; l < 2; l++)
                    tD[i][j][k][l] = 1.0 / Stretch[0] * (-1.0 / Deig[0] * dwdL[0] + 1.0 / Stretch[0] * d2wd2L[0][0]) * Veig(i, 0) * Veig(j, 0) * Veig(k, 0) * Veig(l, 0) +
                                     1.0 / Stretch[0] * (1.0 / Stretch[1] * d2wd2L[0][1]) * Veig(i, 0) * Veig(j, 0) * Veig(k, 1) * Veig(l, 1) +
                                     1.0 / Stretch[1] * (1.0 / Stretch[0] * d2wd2L[1][0]) * Veig(i, 1) * Veig(j, 1) * Veig(k, 0) * Veig(l, 0) +
                                     1.0 / Stretch[1] * (-1.0 / Deig[1] * dwdL[1] + 1.0 / Stretch[1] * d2wd2L[1][1]) * Veig(i, 1) * Veig(j, 1) * Veig(k, 1) * Veig(l, 1);
    if (fabs(Stretch[1] - Stretch[0]) > EPS) // two distinct eigenvalues
    {
        for (i = 0; i < 2; i++)
            for (j = 0; j < 2; j++)
                for (k = 0; k < 2; k++)
                    for (l = 0; l < 2; l++)
                        tD[i][j][k][l] += (dwdL[1] / Stretch[1] - dwdL[0] / Stretch[0]) / (Deig[1] - Deig[0]) * (Veig(i, 0) * Veig(j, 1) * Veig(k, 0) * Veig(l, 1) + Veig(i, 0) * Veig(j, 1) * Veig(k, 1) * Veig(l, 0)) +
                                          (dwdL[0] / Stretch[0] - dwdL[1] / Stretch[1]) / (Deig[0] - Deig[1]) * (Veig(i, 1) * Veig(j, 0) * Veig(k, 1) * Veig(l, 0) + Veig(i, 1) * Veig(j, 0) * Veig(k, 0) * Veig(l, 1));
    }
    else // multiple eigenvalues
    {
        for (i = 0; i < 2; i++)
            for (j = 0; j < 2; j++)
                for (k = 0; k < 2; k++)
                    for (l = 0; l < 2; l++)
                        tD[i][j][k][l] += 0.5 / Stretch[1] * ((-1.0 / Deig[1] * dwdL[1] + 1.0 / Stretch[1] * d2wd2L[1][1]) - (1.0 / Stretch[0] * d2wd2L[0][1])) * (Veig(i, 0) * Veig(j, 1) * Veig(k, 0) * Veig(l, 1) + Veig(i, 0) * Veig(j, 1) * Veig(k, 1) * Veig(l, 0)) +
                                          0.5 / Stretch[0] * ((-1.0 / Deig[0] * dwdL[0] + 1.0 / Stretch[0] * d2wd2L[0][0]) - (1.0 / Stretch[1] * d2wd2L[1][0])) * (Veig(i, 1) * Veig(j, 0) * Veig(k, 1) * Veig(l, 0) + Veig(i, 1) * Veig(j, 0) * Veig(k, 0) * Veig(l, 1));
    }

    // Map tD on Voight notation
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            D[i][j] = tD[idmap[0][i]][idmap[1][i]][idmap[0][j]][idmap[1][j]];

    return errorcode;
}

/******************** 3d Constitutive laws formulated in F  ********************/

// Get derivatives of invariants with respect to F
int getInvariantsDerivativesF3d(const double F[9],
                                const double IF[9], const double J,                   // inputs
                                double &I1b, double &I2b, double &I132, double &logJ, // outputs
                                double &J13, double &J23, double DI1b[9], double DI2b[9],
                                double DDI1b[9][9], double DDI2b[9][9])
{

    // Allocate indices
    int i, j, k, l;

    // The left Cauchy-Green tensor B = F*F'
    double B[9];
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            B[i + 3 * j] = 0.0;
            for (k = 0; k < 3; k++)
                B[i + 3 * j] += F[i + 3 * k] * F[j + 3 * k];
        }

    // Get B^2
    double B2[9];
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            B2[i + 3 * j] = 0.0;
            for (k = 0; k < 3; k++)
                B2[i + 3 * j] += B[i + 3 * k] * B[k + 3 * j];
        }

    // The first invariant of B
    double I1 = B[0] + B[4] + B[8];

    // The second invariant of B
    double I2 = 1.0 / 2.0 * (I1 * I1 - (B2[0] + B2[4] + B2[8]));

    // The righ Cauchy-Green tensor C = F'*F
    double C[9];
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            C[i + 3 * j] = 0.0;
            for (k = 0; k < 3; k++)
                C[i + 3 * j] += F[k + 3 * i] * F[k + 3 * j];
        }

    // FC = F*C
    double FC[9];
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            FC[i + 3 * j] = 0;
            for (k = 0; k < 3; k++)
                FC[i + 3 * j] += F[i + 3 * k] * C[k + 3 * j];
        }

    // Normalized invariants and constants
    J13 = pow(J, 1.0 / 3.0);
    J23 = pow(J, 2.0 / 3.0);
    double J43 = pow(J, 4.0 / 3.0);
    logJ = log(J);
    I1b = I1 / J23;
    I2b = I2 / J43;
    I132 = pow(I1b - 3.0, 2.0);

    // Identity second-order tensor
    double Id3[9];
    for (i = 0; i < 9; i++)
        Id3[i] = 0.0;
    Id3[0] = 1.0;
    Id3[4] = 1.0;
    Id3[8] = 1.0;

    // Get the first order derivatives of I1/J23 and I2/J43
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            DI1b[i + 3 * j] = 2.0 / J23 * F[i + 3 * j] - 2.0 / 3.0 * I1b * IF[j + 3 * i];
            DI2b[i + 3 * j] = 2.0 / J23 * I1b * F[i + 3 * j] - 4.0 / 3.0 * I2b * IF[j + 3 * i] - 2.0 / J43 * FC[i + 3 * j];
        }

    // Get the second order derivatives of I1/J23 and I2/J43
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            for (k = 0; k < 3; k++)
                for (l = 0; l < 3; l++)
                {
                    DDI1b[i + 3 * j][k + 3 * l] = 2.0 / J23 * Id3[i + 3 * k] * Id3[j + 3 * l] +
                                                  2.0 / 3.0 * I1b * IF[l + 3 * i] * IF[j + 3 * k] -
                                                  2.0 / 3.0 * DI1b[k + 3 * l] * IF[j + 3 * i] -
                                                  4.0 / (3.0 * J23) * F[i + 3 * j] * IF[l + 3 * k];
                    DDI2b[i + 3 * j][k + 3 * l] = 2.0 / J23 * I1b * Id3[i + 3 * k] * Id3[j + 3 * l] +
                                                  2.0 / J23 * DI1b[k + 3 * l] * F[i + 3 * j] +
                                                  4.0 / 3.0 * I2b * IF[l + 3 * i] * IF[j + 3 * k] +
                                                  8.0 / (3.0 * J43) * IF[l + 3 * k] * FC[i + 3 * j] -
                                                  4.0 / (3.0 * J23) * I1b * IF[l + 3 * k] * F[i + 3 * j] -
                                                  4.0 / 3.0 * DI2b[k + 3 * l] * IF[j + 3 * i] -
                                                  2.0 / J43 * (Id3[i + 3 * k] * C[l + 3 * j] + F[i + 3 * l] * F[k + 3 * j] + B[i + 3 * k] * Id3[j + 3 * l]);
                }

    return 1;
}

// OOFEM (compressible Neo-hookean model)
int material_oofemF3d(const double matconst[8], const double F[9],
                      double &Wd, double P[9], double D[9][9])
{

    // Get material constants
    int errorcode;
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // Inverse of the deformation gradient and determinant of the deformation gradient
    double J;
    double IF[9];
    if (!getInverse33(F, J, IF))
        return -4;

    // Get derivatives of invariants with respect to F
    int i, j, k, l;
    double J13, J23, I1b, I2b, I132, logJ;
    double DI1b[9], DI2b[9], DDI1b[9][9], DDI2b[9][9];
    errorcode = getInvariantsDerivativesF3d(F, IF, J,                                                  // inputs
                                            I1b, I2b, I132, logJ, J13, J23, DI1b, DI2b, DDI1b, DDI2b); // outputs

    // Energy density W(F) = m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+1/2*kappa(ln(J))^2
    Wd = m1 * (I1b - 3.0) + m2 * (I2b - 3.0) + 1.0 / 2.0 * kappa * logJ * logJ;

    // The first Piola Kirchhoff stress tensor P
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            P[i + 3 * j] = m1 * DI1b[i + 3 * j] + m2 * DI2b[i + 3 * j] + kappa * logJ * IF[j + 3 * i];

    // Material stiffness D
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            for (k = 0; k < 3; k++)
                for (l = 0; l < 3; l++)
                    D[i + 3 * j][k + 3 * l] = m1 * DDI1b[i + 3 * j][k + 3 * l] +
                                              m2 * DDI2b[i + 3 * j][k + 3 * l] +
                                              kappa * (IF[j + 3 * i] * IF[l + 3 * k] -
                                                       logJ * IF[j + 3 * k] * IF[l + 3 * i]);

    return errorcode;
}

// Bertoldi, Boyce
int material_bbF3d(const double matconst[8], const double F[9],
                   double &Wd, double P[9], double D[9][9])
{

    // Get material constants
    int i, j, k, l;
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // Identity second-order tensor
    double Id3[9];
    for (i = 0; i < 9; i++)
        Id3[i] = 0.0;
    Id3[0] = 1.0;
    Id3[4] = 1.0;
    Id3[8] = 1.0;

    // Inverse of the deformation gradient and determinant of the deformation gradient
    double J;
    double IF[9];
    if (!getInverse33(F, J, IF))
        return -4;

    // The left Cauchy-Green tensor B = F*F'
    double B[9];
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            B[i + 3 * j] = 0.0;
            for (k = 0; k < 3; k++)
                B[i + 3 * j] += F[i + 3 * k] * F[j + 3 * k];
        }

    // Get B^2
    double B2[9];
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            B2[i + 3 * j] = 0.0;
            for (k = 0; k < 3; k++)
                B2[i + 3 * j] += B[i + 3 * k] * B[k + 3 * j];
        }

    // The first invariant of B
    double I1 = B[0] + B[4] + B[8];

    // Energy density W(F) = m1(I1-3)+m2(I1-3)^2-2m1*ln(J)+1/2*kappa(J-1)^2
    Wd = m1 * (I1 - 3.0) + m2 * pow(I1 - 3.0, 2.0) - 2.0 * m1 * log(J) + kappa / 2.0 * pow(J - 1.0, 2.0);

    // The first Piola Kirchhoff stress tensor P
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            P[i + 3 * j] = 2.0 * m1 * F[i + 3 * j] + 4.0 * m2 * (I1 - 3.0) * F[i + 3 * j] - 2.0 * m1 * IF[j + 3 * i] +
                           kappa * (J - 1.0) * J * IF[j + 3 * i];

    // Material stiffness D
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            for (k = 0; k < 3; k++)
                for (l = 0; l < 3; l++)
                    D[i + 3 * j][k + 3 * l] = 2.0 * m1 * Id3[i + 3 * k] * Id3[j + 3 * l] +
                                              8.0 * m2 * F[i + 3 * j] * F[k + 3 * l] +
                                              4.0 * m2 * (I1 - 3) * Id3[i + 3 * k] * Id3[j + 3 * l] +
                                              2.0 * m1 * IF[l + 3 * i] * IF[j + 3 * k] +
                                              kappa * J * J * IF[j + 3 * i] * IF[l + 3 * k] +
                                              kappa * (J - 1.0) * J * IF[j + 3 * i] * IF[l + 3 * k] -
                                              kappa * (J - 1.0) * J * IF[l + 3 * i] * IF[j + 3 * k];

    return 1;
}

// Jamus, Green, Simpson
int material_jgsF3d(const double matconst[8], const double F[9],
                    double &Wd, double P[9], double D[9][9])
{

    // Get material constants
    int errorcode;
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double m3 = matconst[3];    // material parameter
    double m4 = matconst[4];    // material parameter
    double m5 = matconst[5];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // Inverse of the deformation gradient and determinant of the deformation gradient
    double J;
    double IF[9];
    if (!getInverse33(F, J, IF))
        return -4;

    // Get derivatives of invariants with respect to F
    int i, j, k, l;
    double J13, J23, I1b, I2b, I132, logJ;
    double DI1b[9], DI2b[9], DDI1b[9][9], DDI2b[9][9];
    errorcode = getInvariantsDerivativesF3d(F, IF, J,                                                  // inputs
                                            I1b, I2b, I132, logJ, J13, J23, DI1b, DI2b, DDI1b, DDI2b); // outputs

    // Energy density W(F) = m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I1/J^{2/3}-3)^3+9/2*kappa(J^{1/3}-1)^2
    Wd = m1 * (I1b - 3.0) + m2 * (I2b - 3.0) + m3 * (I1b - 3.0) * (I2b - 3.0) + m4 * I132 + m5 * pow(I1b - 3.0, 3.0) + 9.0 / 2.0 * kappa * pow(J13 - 1.0, 2.0);

    // The first Piola Kirchhoff stress tensor P
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            P[i + 3 * j] = m1 * DI1b[i + 3 * j] +
                           m2 * DI2b[i + 3 * j] +
                           m3 * ((I2b - 3.0) * DI1b[i + 3 * j] +
                                 (I1b - 3.0) * DI2b[i + 3 * j]) +
                           2.0 * m4 * (I1b - 3.0) * DI1b[i + 3 * j] +
                           3.0 * m5 * I132 * DI1b[i + 3 * j] +
                           3.0 * kappa * J13 * (J13 - 1.0) * IF[j + 3 * i];

    // Material stiffness D
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            for (k = 0; k < 3; k++)
                for (l = 0; l < 3; l++)
                    D[i + 3 * j][k + 3 * l] = m1 * DDI1b[i + 3 * j][k + 3 * l] +
                                              m2 * DDI2b[i + 3 * j][k + 3 * l] +
                                              m3 * ((I2b - 3.0) * DDI1b[i + 3 * j][k + 3 * l] +
                                                    DI1b[i + 3 * j] * DI2b[k + 3 * l] +
                                                    DI1b[k + 3 * l] * DI2b[i + 3 * j] +
                                                    (I1b - 3.0) * DDI2b[i + 3 * j][k + 3 * l]) +
                                              2.0 * m4 * (DI1b[k + 3 * l] * DI1b[i + 3 * j] + (I1b - 3.0) * DDI1b[i + 3 * j][k + 3 * l]) +
                                              3.0 * m5 * (2.0 * (I1b - 3.0) * DI1b[k + 3 * l] * DI1b[i + 3 * j] + I132 * DDI1b[i + 3 * j][k + 3 * l]) +
                                              kappa * (J13 * (J13 - 1.0) * IF[j + 3 * i] * IF[l + 3 * k] +
                                                       J23 * IF[j + 3 * i] * IF[l + 3 * k] -
                                                       3.0 * J13 * (J13 - 1.0) * IF[l + 3 * i] * IF[j + 3 * k]);

    return errorcode;
}

// five-term Mooney-Rivlin
int material_5mrF3d(const double matconst[8], const double F[9],
                    double &Wd, double P[9], double D[9][9])
{

    // Get material constants
    int errorcode;
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double m3 = matconst[3];    // material parameter
    double m4 = matconst[4];    // material parameter
    double m5 = matconst[5];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // Inverse of the deformation gradient and determinant of the deformation gradient
    double J;
    double IF[9];
    if (!getInverse33(F, J, IF))
        return -4;

    // Get derivatives of invariants with respect to F
    int i, j, k, l;
    double J13, J23, I1b, I2b, I132, logJ;
    double DI1b[9], DI2b[9], DDI1b[9][9], DDI2b[9][9];
    errorcode = getInvariantsDerivativesF3d(F, IF, J,                                                  // inputs
                                            I1b, I2b, I132, logJ, J13, J23, DI1b, DI2b, DDI1b, DDI2b); // outputs

    // Energy density W(F) = m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I2/J^{2/3}-3)^2+kappa*(J-1)^2
    Wd = m1 * (I1b - 3.0) + m2 * (I2b - 3.0) + m3 * (I1b - 3.0) * (I2b - 3.0) + m4 * pow(I1b - 3.0, 2.0) + m5 * pow(I2b - 3.0, 2.0) + kappa * pow(J - 1.0, 2.0);

    // The first Piola Kirchhoff stress tensor P
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            P[i + 3 * j] = m1 * DI1b[i + 3 * j] +
                           m2 * DI2b[i + 3 * j] +
                           m3 * ((I2b - 3.0) * DI1b[i + 3 * j] +
                                 (I1b - 3.0) * DI2b[i + 3 * j]) +
                           2.0 * m4 * (I1b - 3.0) * DI1b[i + 3 * j] +
                           2.0 * m5 * (I2b - 3.0) * DI2b[i + 3 * j] +
                           2.0 * kappa * J * (J - 1.0) * IF[j + 3 * i];

    // Material stiffness D
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            for (k = 0; k < 3; k++)
                for (l = 0; l < 3; l++)
                    D[i + 3 * j][k + 3 * l] = m1 * DDI1b[i + 3 * j][k + 3 * l] +
                                              m2 * DDI2b[i + 3 * j][k + 3 * l] +
                                              m3 * ((I2b - 3.0) * DDI1b[i + 3 * j][k + 3 * l] +
                                                    DI1b[i + 3 * j] * DI2b[k + 3 * l] +
                                                    DI1b[k + 3 * l] * DI2b[i + 3 * j] +
                                                    (I1b - 3.0) * DDI2b[i + 3 * j][k + 3 * l]) +
                                              2.0 * m4 * (DI1b[k + 3 * l] * DI1b[i + 3 * j] + (I1b - 3.0) * DDI1b[i + 3 * j][k + 3 * l]) +
                                              2.0 * m5 * (DI2b[k + 3 * l] * DI2b[i + 3 * j] + (I2b - 3.0) * DDI2b[i + 3 * j][k + 3 * l]) +
                                              2.0 * kappa * (J * J * IF[j + 3 * i] * IF[l + 3 * k] + J * (J - 1.0) * IF[j + 3 * i] * IF[l + 3 * k] - J * (J - 1.0) * IF[l + 3 * i] * IF[j + 3 * k]);

    return errorcode;
}

// Linear elastic
int material_leF3d(const double matconst[8], const double F[9],
                   double &Wd, double P[9], double D[9][9])
{

    // Get material constants
    int i, j, k, l, m, n;
    double mu = matconst[1];     // Lame's parameter (shear modulus G)
    double lambda = matconst[6]; // Lame's parameter (K-2/3*G)

    // The Green-Lagrange strain tensor E = 0.5*(F'*F-I)
    double C[6], E[9];
    getC3d(F, C);
    E[0] = 0.5 * (C[0] - 1.0);
    E[1] = 0.5 * C[3];
    E[2] = 0.5 * C[5];
    E[3] = 0.5 * C[3];
    E[4] = 0.5 * (C[1] - 1.0);
    E[5] = 0.5 * C[4];
    E[6] = 0.5 * C[5];
    E[7] = 0.5 * C[4];
    E[8] = 0.5 * (C[2] - 1.0);

    // Identity second-order tensor
    double Id3[9];
    for (i = 0; i < 9; i++)
        Id3[i] = 0.0;
    Id3[0] = 1.0;
    Id3[4] = 1.0;
    Id3[8] = 1.0;

    // Material stiffness C
    double CD[9][9];
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            for (k = 0; k < 3; k++)
                for (l = 0; l < 3; l++)
                    CD[i + 3 * j][k + 3 * l] = lambda * Id3[i + 3 * j] * Id3[k + 3 * l] +
                                               mu * (Id3[i + 3 * k] * Id3[j + 3 * l] + Id3[i + 3 * l] * Id3[j + 3 * k]);

    // The second P-K stress
    double S[9];
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            S[i + 3 * j] = 0.0;
            for (k = 0; k < 3; k++)
                for (l = 0; l < 3; l++)
                    S[i + 3 * j] += CD[i + 3 * j][k + 3 * l] * E[k + 3 * l];
        }

    // Energy density W(F) = 0.5*(0.5*(C-I)*(m1*)*0.5*(C-I)) (linear elastic material)
    Wd = 0.0;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            Wd += 0.5 * S[i + 3 * j] * E[i + 3 * j];

    // The first Piola Kirchhoff stress tensor P
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            P[i + 3 * j] = 0.0;
            for (k = 0; k < 3; k++)
                P[i + 3 * j] += F[i + 3 * k] * S[k + 3 * j];
        }

    // Material stiffness D
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            for (k = 0; k < 3; k++)
                for (l = 0; l < 3; l++)
                {
                    D[i + 3 * j][k + 3 * l] = Id3[i + 3 * k] * S[j + 3 * l];
                    for (m = 0; m < 3; m++)
                        for (n = 0; n < 3; n++)
                            D[i + 3 * j][k + 3 * l] += F[i + 3 * m] * F[k + 3 * n] * CD[m + 3 * j][n + 3 * l];
                }

    return 1;
}

/******************** 3d Constitutive laws formulated in E  ********************/

// Get derivatives of invariants with respect to C
int getInvariantsDerivativesC3d(const double C[6], double &I1, double &I2, double &J,
                                double &J2, double dI1dC[6], double dI2dC[6],
                                double dJdC[6], double d2I2dCC[6][6],
                                double d2JdCC[6][6])
{

    // Fix some variables
    int i, j, errorcode;

    // Get invariants
    I1 = C[0] + C[1] + C[2];
    double IC[6];
    if (!getInverse33sym(C, J2, IC)) // det(C) = J2, where det(F) = J
        return -4;
    J = sqrt(J2);
    I2 = 0.5 * (I1 * I1 - (C[0] * C[0] + C[1] * C[1] + C[2] * C[2] + 2.0 * C[3] * C[3] + 2.0 * C[4] * C[4] + 2.0 * C[5] * C[5]));

    // Get first derivatives
    getId3d(dI1dC);
    for (i = 0; i < 6; i++)
    {
        dI2dC[i] = I1 * dI1dC[i] - C[i];
        dJdC[i] = 0.5 * J * IC[i];
    }

    // Get second derivatives
    double Id4[6][6];
    errorcode = getSymmetricProduct3d(Id4, dI1dC, dI1dC);
    for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
            d2I2dCC[i][j] = dI1dC[i] * dI1dC[j] - Id4[i][j];

    double IC4[6][6];
    errorcode = getSymmetricProduct3d(IC4, IC, IC);
    for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
            d2JdCC[i][j] = J * (IC[i] * IC[j] - 2.0 * IC4[i][j]);

    return errorcode;
}

// OOFEM (compressible Neo-hookean model)
int material_oofemE3d(const double matconst[8], const double C[6],
                      double &w, double S[6], double D[6][6])
{

    // Get material constants
    int i, j, errorcode;
    double I1, I2, J, J2, dI1dC[6], dI2dC[6], dJdC[6], d2I2dCC[6][6], d2JdCC[6][6];
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // The invariants of C and their derivatives
    errorcode = getInvariantsDerivativesC3d(C, I1, I2, J, J2, dI1dC, dI2dC, dJdC, d2I2dCC, d2JdCC);

    // Normalized invariants and constants
    double J13 = 1 / pow(J, 1.0 / 3.0);
    double J23 = J13 * J13;
    double J43 = J23 * J23;
    double J53 = J43 * J13;
    double J73 = J53 * J23;
    double J83 = J73 * J13;
    double J103 = J83 * J23;
    double logJ = log(J);
    double In1 = I1 * J23;
    double In2 = I2 * J43;

    // Modified invariants of C: In1 = I1/J^{2/3}, In2 = I2/J^{4/3}
    // Energy density W(F) = m1(In1-3)+m2(In2-3)+1/2*kappa(ln(J))^2
    w = m1 * (In1 - 3.0) +
        m2 * (In2 - 3.0) +
        1.0 / 2.0 * kappa * logJ * logJ;

    // The second Piola-Kirchhoff stress tensor S
    double dwdI1 = m1 * J23;
    double dwdI2 = m2 * J43;
    double dwdJ = -2.0 / 3.0 * m1 * I1 * J53 -
                  4.0 / 3.0 * m2 * I2 * J73 +
                  kappa * logJ / J;
    for (i = 0; i < 6; i++)
        S[i] = 2.0 * (dwdI1 * dI1dC[i] + dwdI2 * dI2dC[i] + dwdJ * dJdC[i]);

    // Material stiffness D
    double d2wdJJ = 10.0 / 9.0 * m1 * I1 * J83 +
                    28.0 / 9.0 * m2 * I2 * J103 +
                    kappa * (1.0 - logJ) / J2;
    double d2wdI1J = -2.0 / 3.0 * m1 * J53;
    double d2wdI2J = -4.0 / 3.0 * m2 * J73;
    for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
            D[i][j] = 4.0 * (d2wdJJ * dJdC[i] * dJdC[j] +
                             d2wdI1J * (dI1dC[i] * dJdC[j] + dJdC[i] * dI1dC[j]) +
                             d2wdI2J * (dI2dC[i] * dJdC[j] + dJdC[i] * dI2dC[j])) +
                      4.0 * dwdI2 * d2I2dCC[i][j] +
                      dwdJ * d2JdCC[i][j];

    return errorcode;
}

// Bertoldi, Boyce
int material_bbE3d(const double matconst[8], const double C[6],
                   double &w, double S[6], double D[6][6])
{
    // Get material constants
    int i, j, errorcode;
    double I1, I2, J, J2, dI1dC[6], dI2dC[6], dJdC[6], d2I2dCC[6][6], d2J3dCC[6][6];
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // The invariants of C and their derivatives
    errorcode = getInvariantsDerivativesC3d(C, I1, I2, J, J2, dI1dC, dI2dC, dJdC, d2I2dCC, d2J3dCC);

    // Energy density W(F) = m1(I1-3)+m2(I1-3)^2-2m1*ln(J)+1/2*kappa(J-1)^2
    w = m1 * (I1 - 3.0) +
        m2 * pow(I1 - 3.0, 2.0) -
        2.0 * m1 * log(J) +
        kappa / 2.0 * pow(J - 1.0, 2.0);

    // The second Piola-Kirchhoff stress tensor S
    double dwdI1 = m1 + 2.0 * m2 * (I1 - 3.0);
    double dwdJ = -2.0 * m1 / J + kappa * (J - 1.0);
    for (i = 0; i < 6; i++)
        S[i] = 2.0 * (dwdI1 * dI1dC[i] + dwdJ * dJdC[i]);

    // Material stiffness D
    double d2wdI1I1 = 2.0 * m2;
    double d2wdJJ = 2.0 * m1 / J2 + kappa;
    for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
            D[i][j] = 4.0 * (d2wdI1I1 * dI1dC[i] * dI1dC[j] +
                             d2wdJJ * dJdC[i] * dJdC[j]) +
                      dwdJ * d2J3dCC[i][j];

    return errorcode;
}

// Jamus, Green, Simpson
int material_jgsE3d(const double matconst[8], const double C[6],
                    double &w, double S[6], double D[6][6])
{

    // Get material constants
    int i, j, errorcode;
    double I1, I2, J, J2, dI1dC[6], dI2dC[6], dJdC[6], d2I2dCC[6][6], d2JdCC[6][6];
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double m3 = matconst[3];    // material parameter
    double m4 = matconst[4];    // material parameter
    double m5 = matconst[5];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // The invariants of C and their derivatives
    errorcode = getInvariantsDerivativesC3d(C, I1, I2, J, J2, dI1dC, dI2dC, dJdC, d2I2dCC, d2JdCC);

    // Normalized invariants and constants
    double J13 = 1 / pow(J, 1.0 / 3.0);
    double J23 = J13 * J13;
    double J43 = J23 * J23;
    double J53 = J43 * J13;
    double J63 = J53 * J13;
    double J73 = J63 * J13;
    double J83 = J73 * J13;
    double J93 = J83 * J13;
    double J103 = J93 * J13;
    double J123 = J103 * J23;
    double In1 = I1 * J23;
    double In2 = I2 * J43;

    // Modified invariants of C: In1 = I1/J^{2/3}, In2 = I2/J^{4/3}
    // Energy density W(F) = m1(In1-3)+m2(In2-3)+m3(In1-3)*(In2-3)+m4(In1-3)^2+m5(In1-3)^3+9/2*kappa(J^{1/3}-1)^2
    w = m1 * (In1 - 3.0) + m2 * (In2 - 3.0) +
        m3 * (In1 - 3.0) * (In2 - 3.0) +
        m4 * pow(In1 - 3.0, 2.0) +
        m5 * pow(In1 - 3.0, 3.0) +
        9.0 / 2.0 * kappa * pow(1 / J13 - 1.0, 2.0);

    // The second Piola-Kirchhoff stress tensor S
    double dwdI1 = m1 * J23 +
                   m3 * (In2 - 3.0) * J23 +
                   2.0 * m4 * (In1 - 3.0) * J23 +
                   3.0 * m5 * pow(In1 - 3.0, 2.0) * J23;
    double dwdI2 = m2 * J43 +
                   m3 * (In1 - 3.0) * J43;
    double dwdJ = -2.0 / 3.0 * m1 * I1 * J53 -
                  4.0 / 3.0 * m2 * I2 * J73 -
                  2.0 / 3.0 * m3 * (I1 * (In2 - 3.0) * J53 + 2.0 * (In1 - 3.0) * I2 * J73) -
                  4.0 / 3.0 * m4 * (In1 - 3.0) * I1 * J53 -
                  2.0 * m5 * pow(In1 - 3.0, 2.0) * I1 * J53 +
                  3.0 * kappa * (1 / J13 - 1.0) * J23;
    for (i = 0; i < 6; i++)
        S[i] = 2.0 * (dwdI1 * dI1dC[i] + dwdI2 * dI2dC[i] + dwdJ * dJdC[i]);

    // Material stiffness D
    double d2wdI1I1 = 2.0 * m4 * J43 +
                      6.0 * m5 * (In1 - 3.0) * J43;
    double d2wdJJ = 10.0 / 9.0 * m1 * I1 * J83 +
                    28.0 / 9.0 * m2 * I2 * J103 +
                    2.0 / 9.0 * m3 * (4.0 * I1 * I2 * J123 + 5.0 * (In2 - 3.0) * I1 * J83 + 4.0 * I1 * I2 * J123 + 14.0 * (In1 - 3.0) * I2 * J103) +
                    8.0 / 9.0 * m4 * I1 * I1 * J103 +
                    20.0 / 9.0 * m4 * (In1 - 3.0) * I1 * J83 +
                    8.0 / 3.0 * m5 * (In1 - 3.0) * I1 * I1 * J103 +
                    10.0 / 3.0 * m5 * pow(In1 - 3.0, 2.0) * I1 * J83 +
                    kappa * (1.0 * J43 - 2.0 * (1 / J13 - 1.0) * J53);
    double d2wdI1I2 = m3 * J63;
    double d2wdI1J = -2.0 / 3.0 * m1 * J53 -
                     m3 * 2.0 / 3.0 * (2.0 * I2 * J93 + (In2 - 3.0) * J53) -
                     4.0 / 3.0 * m4 * (I1 * J73 + (In1 - 3.0) * J53) -
                     2.0 * m5 * (2.0 * (In1 - 3.0) * I1 * J73 + pow(In1 - 3.0, 2.0) * J53);
    double d2wdI2J = -4.0 / 3.0 * m2 * J73 -
                     2.0 / 3.0 * m3 * (I1 * J93 + 2.0 * (In1 - 3.0) * J73);
    for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
            D[i][j] = 4.0 * (d2wdI1I1 * dI1dC[i] * dI1dC[j] +
                             d2wdJJ * dJdC[i] * dJdC[j] +
                             d2wdI1I2 * (dI1dC[i] * dI2dC[j] + dI2dC[i] * dI1dC[j]) +
                             d2wdI1J * (dI1dC[i] * dJdC[j] + dJdC[i] * dI1dC[j]) +
                             d2wdI2J * (dI2dC[i] * dJdC[j] + dJdC[i] * dI2dC[j])) +
                      4.0 * dwdI2 * d2I2dCC[i][j] +
                      dwdJ * d2JdCC[i][j];

    return errorcode;
}

// five-term Mooney-Rivlin
int material_5mrE3d(const double matconst[8], const double C[6],
                    double &w, double S[6], double D[6][6])
{

    // Get material constants
    int i, j, errorcode;
    double I1, I2, J, J2, dI1dC[6], dI2dC[6], dJdC[6], d2I2dCC[6][6], d2JdCC[6][6];
    double m1 = matconst[1];    // material parameter
    double m2 = matconst[2];    // material parameter
    double m3 = matconst[3];    // material parameter
    double m4 = matconst[4];    // material parameter
    double m5 = matconst[5];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // The invariants of C and their derivatives
    errorcode = getInvariantsDerivativesC3d(C, I1, I2, J, J2, dI1dC, dI2dC, dJdC, d2I2dCC, d2JdCC);

    // Normalized invariants and constants
    double J13 = 1 / pow(J, 1.0 / 3.0);
    double J23 = J13 * J13;
    double J43 = J23 * J23;
    double J53 = J43 * J13;
    double J63 = J53 * J13;
    double J73 = J63 * J13;
    double J83 = J73 * J13;
    double J93 = J83 * J13;
    double J103 = J93 * J13;
    double J113 = J103 * J13;
    double J123 = J103 * J23;
    double J143 = J73 * J73;
    double In1 = I1 * J23;
    double In2 = I2 * J43;

    // Modified invariants of C: In1 = I1/J^{2/3}, In2 = I2/J^{4/3}
    // Energy density W(F) = m1(In1-3)+m2(In2-3)+m3(In1-3)*(In2-3)+m4(In1-3)^2+m5(In2-3)^2+kappa*(J-1)^2
    w = m1 * (In1 - 3.0) +
        m2 * (In2 - 3.0) +
        m3 * (In1 - 3.0) * (In2 - 3.0) +
        m4 * pow(In1 - 3.0, 2.0) +
        m5 * pow(In2 - 3.0, 2.0) +
        kappa * pow(J - 1.0, 2.0);

    // The second Piola-Kirchhoff stress tensor S
    double dwdI1 = m1 * J23 +
                   m3 * (In2 - 3.0) * J23 +
                   2.0 * m4 * (In1 - 3.0) * J23;
    double dwdI2 = m2 * J43 +
                   m3 * (In1 - 3.0) * J43 +
                   2 * m5 * (In2 - 3.0) * J43;
    double dwdJ = -2.0 / 3.0 * m1 * I1 * J53 -
                  4.0 / 3.0 * m2 * I2 * J73 -
                  2.0 / 3.0 * m3 * (I1 * (In2 - 3.0) * J53 + 2.0 * (In1 - 3.0) * I2 * J73) -
                  4.0 / 3.0 * m4 * (In1 - 3.0) * I1 * J53 -
                  8.0 / 3.0 * m5 * (In2 - 3.0) * I2 * J73 +
                  2.0 * kappa * (J - 1.0);
    for (i = 0; i < 6; i++)
        S[i] = 2.0 * (dwdI1 * dI1dC[i] + dwdI2 * dI2dC[i] + dwdJ * dJdC[i]);

    // Material stiffness D
    double d2wdI1I1 = 2.0 * m4 * J43;
    double d2wdI2I2 = 2.0 * m5 * J83;
    double d2wdJJ = 10.0 / 9.0 * m1 * I1 * J83 +
                    28.0 / 9.0 * m2 * I2 * J103 +
                    2.0 / 9.0 * m3 * (4.0 * I1 * I2 * J123 + 5.0 * (In2 - 3.0) * I1 * J83 + 4.0 * I1 * I2 * J123 + 14.0 * (In1 - 3.0) * I2 * J103) +
                    8.0 / 9.0 * m4 * I1 * I1 * J103 +
                    20.0 / 9.0 * m4 * (In1 - 3.0) * I1 * J83 +
                    32.0 / 9.0 * m5 * I2 * I2 * J143 +
                    56.0 / 9.0 * m5 * (In2 - 3.0) * I2 * J103 +
                    2.0 * kappa;
    double d2wdI1I2 = m3 * J63;
    double d2wdI1J = -2.0 / 3.0 * m1 * J53 -
                     m3 * 2.0 / 3.0 * (2.0 * I2 * J93 + (In2 - 3.0) * J53) -
                     4.0 / 3.0 * m4 * (I1 * J73 + (In1 - 3.0) * J53);
    double d2wdI2J = -4.0 / 3.0 * m2 * J73 -
                     2.0 / 3.0 * m3 * (I1 * J93 + 2.0 * (In1 - 3.0) * J73) -
                     8.0 / 3.0 * m5 * (I2 * J113 + (In2 - 3.0) * J73);
    for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
            D[i][j] = 4.0 * (d2wdI1I1 * dI1dC[i] * dI1dC[j] +
                             d2wdI2I2 * dI2dC[i] * dI2dC[j] +
                             d2wdJJ * dJdC[i] * dJdC[j] +
                             d2wdI1I2 * (dI1dC[i] * dI2dC[j] + dI2dC[i] * dI1dC[j]) +
                             d2wdI1J * (dI1dC[i] * dJdC[j] + dJdC[i] * dI1dC[j]) +
                             d2wdI2J * (dI2dC[i] * dJdC[j] + dJdC[i] * dI2dC[j])) +
                      4.0 * dwdI2 * d2I2dCC[i][j] +
                      dwdJ * d2JdCC[i][j];

    return errorcode;
}

// Linear elastic
int material_leE3d(const double matconst[8], const double C[6],
                   double &w, double S[6], double D[6][6])
{

    // Get material constants
    int i, j;
    double mu = matconst[1];     // Lame's parameter (shear modulus G)
    double lambda = matconst[6]; // Lame's parameter (K-2/3*G)

    // Get stiffness matrix
    for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
            D[i][j] = 0.0;
    D[0][0] = lambda + 2.0 * mu;
    D[1][1] = lambda + 2.0 * mu;
    D[2][2] = lambda + 2.0 * mu;
    D[3][3] = mu;
    D[4][4] = mu;
    D[5][5] = mu;
    D[0][1] = lambda;
    D[0][2] = lambda;
    D[1][0] = lambda;
    D[1][2] = lambda;
    D[2][0] = lambda;
    D[2][1] = lambda;

    // Get the second P-K stress
    double E[6];
    E[0] = 0.5 * (C[0] - 1.0);
    E[1] = 0.5 * (C[1] - 1.0);
    E[2] = 0.5 * (C[2] - 1.0);
    E[3] = C[3];
    E[4] = C[4];
    E[5] = C[5];

    for (i = 0; i < 6; i++)
    {
        S[i] = 0.0;
        for (j = 0; j < 6; j++)
            S[i] += D[i][j] * E[j];
    }

    // Get energy
    w = 0.0;
    for (i = 0; i < 6; i++)
        w += 0.5 * S[i] * E[i];

    return 1;
}

// Ogden
int material_ogdenE3d(const double matconst[8], const double C[6], double &w,
                      double S[6], double D[6][6])
{

    // Get material constants
    int i, j, k, l, errorcode;
    double Stretch[3], J, J2;
    double c1 = matconst[1];    // material parameter
    double m1 = matconst[2];    // material parameter
    double c2 = matconst[3];    // material parameter
    double m2 = matconst[4];    // material parameter
    double kappa = matconst[6]; // bulk modulus

    // Get spectral decomposition of C
    Eigen::Matrix3d Cd;
    Cd << C[0], C[3], C[5],
        C[3], C[1], C[4],
        C[5], C[4], C[2];
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(Cd);
    Eigen::Matrix3d Veig = eigensolver.eigenvectors();
    Eigen::Vector3d Deig = eigensolver.eigenvalues();
    J2 = Cd.determinant(); // det(C) = J2, where det(F) = J
    J = sqrt(J2);
    Stretch[0] = sqrt(Deig[0]);
    Stretch[1] = sqrt(Deig[1]);
    Stretch[2] = sqrt(Deig[2]);

    // Energy density W(F) = c1/m1^2*(lambda1^m1+lambda2^m1+lambda3^m1-3-m1*log(J))+c2/m2^2*(lambda1^m2+lambda2^m2+lambda3^m2-3-m2*log(J))+kappa/2*(J-1)^2 (Ogden)
    double logJ = log(J);
    w = c1 / (m1 * m1) * (pow(Stretch[0], m1) + pow(Stretch[1], m1) + pow(Stretch[2], m1) - 3.0 - m1 * logJ) +
        c2 / (m2 * m2) * (pow(Stretch[0], m2) + pow(Stretch[1], m2) + pow(Stretch[2], m2) - 3.0 - m2 * logJ) +
        kappa / 2.0 * (J - 1.0) * (J - 1.0);

    // Compute derivatives of the energy density with respect to stretches and constants (treating J = lambda1*lambda2*1.0)
    double dwdL[3];
    dwdL[0] = c1 / m1 * pow(Stretch[0], m1 - 1.0) + c2 / m2 * pow(Stretch[0], m2 - 1.0) - (c1 / m1 + c2 / m2 - J * kappa * (J - 1.0)) / J * Stretch[1] * Stretch[2];
    dwdL[1] = c1 / m1 * pow(Stretch[1], m1 - 1.0) + c2 / m2 * pow(Stretch[1], m2 - 1.0) - (c1 / m1 + c2 / m2 - J * kappa * (J - 1.0)) / J * Stretch[0] * Stretch[2];
    dwdL[2] = c1 / m1 * pow(Stretch[2], m1 - 1.0) + c2 / m2 * pow(Stretch[2], m2 - 1.0) - (c1 / m1 + c2 / m2 - J * kappa * (J - 1.0)) / J * Stretch[0] * Stretch[1];

    double d2wd2L[3][3];
    d2wd2L[0][0] = c1 / m1 * (m1 - 1.0) * pow(Stretch[0], m1 - 2.0) + c2 / m2 * (m2 - 1.0) * pow(Stretch[0], m2 - 2.0) + (c1 / m1 + c2 / m2 + J2 * kappa) / J2 * Deig[1] * Deig[2];
    d2wd2L[1][1] = c1 / m1 * (m1 - 1.0) * pow(Stretch[1], m1 - 2.0) + c2 / m2 * (m2 - 1.0) * pow(Stretch[1], m2 - 2.0) + (c1 / m1 + c2 / m2 + J2 * kappa) / J2 * Deig[0] * Deig[2];
    d2wd2L[2][2] = c1 / m1 * (m1 - 1.0) * pow(Stretch[2], m1 - 2.0) + c2 / m2 * (m2 - 1.0) * pow(Stretch[2], m2 - 2.0) + (c1 / m1 + c2 / m2 + J2 * kappa) / J2 * Deig[0] * Deig[1];
    d2wd2L[0][1] = kappa * Stretch[2] * (2.0 * J - 1.0);
    d2wd2L[1][0] = d2wd2L[0][1];
    d2wd2L[0][2] = kappa * Stretch[1] * (2.0 * J - 1.0);
    d2wd2L[2][0] = d2wd2L[0][2];
    d2wd2L[1][2] = kappa * Stretch[0] * (2.0 * J - 1.0);
    d2wd2L[2][1] = d2wd2L[1][2];

    // The second Piola-Kirchhoff stress tensor S
    double tS[3][3];
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            tS[i][j] = dwdL[0] / Stretch[0] * Veig(i, 0) * Veig(j, 0) +
                       dwdL[1] / Stretch[1] * Veig(i, 1) * Veig(j, 1) +
                       dwdL[2] / Stretch[2] * Veig(i, 2) * Veig(j, 2);

    // Map tS on Voight notation
    int idmap[2][6] = {{0, 1, 2, 0, 1, 2}, {0, 1, 2, 1, 2, 0}}; // map from tensor to Voight notation
    for (i = 0; i < 6; i++)
        S[i] = tS[idmap[0][i]][idmap[1][i]];

    // Material stiffness D
    double tD[3][3][3][3];
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            for (k = 0; k < 3; k++)
                for (l = 0; l < 3; l++)
                    tD[i][j][k][l] = 1.0 / Stretch[0] * (-1.0 / Deig[0] * dwdL[0] + 1.0 / Stretch[0] * d2wd2L[0][0]) * Veig(i, 0) * Veig(j, 0) * Veig(k, 0) * Veig(l, 0) +
                                     1.0 / Stretch[0] * (1.0 / Stretch[1] * d2wd2L[0][1]) * Veig(i, 0) * Veig(j, 0) * Veig(k, 1) * Veig(l, 1) +
                                     1.0 / Stretch[0] * (1.0 / Stretch[2] * d2wd2L[0][2]) * Veig(i, 0) * Veig(j, 0) * Veig(k, 2) * Veig(l, 2) +
                                     1.0 / Stretch[1] * (1.0 / Stretch[0] * d2wd2L[1][0]) * Veig(i, 1) * Veig(j, 1) * Veig(k, 0) * Veig(l, 0) +
                                     1.0 / Stretch[1] * (-1.0 / Deig[1] * dwdL[1] + 1.0 / Stretch[1] * d2wd2L[1][1]) * Veig(i, 1) * Veig(j, 1) * Veig(k, 1) * Veig(l, 1) +
                                     1.0 / Stretch[1] * (1.0 / Stretch[2] * d2wd2L[1][2]) * Veig(i, 1) * Veig(j, 1) * Veig(k, 2) * Veig(l, 2) +
                                     1.0 / Stretch[2] * (1.0 / Stretch[0] * d2wd2L[2][0]) * Veig(i, 2) * Veig(j, 2) * Veig(k, 0) * Veig(l, 0) +
                                     1.0 / Stretch[2] * (1.0 / Stretch[1] * d2wd2L[2][1]) * Veig(i, 2) * Veig(j, 2) * Veig(k, 1) * Veig(l, 1) +
                                     1.0 / Stretch[2] * (-1.0 / Deig[2] * dwdL[2] + 1.0 / Stretch[2] * d2wd2L[2][2]) * Veig(i, 2) * Veig(j, 2) * Veig(k, 2) * Veig(l, 2);

    // Stretch[0] \neq Stretch[1]
    if (fabs(Stretch[1] - Stretch[0]) > EPS)
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                        tD[i][j][k][l] += (dwdL[1] / Stretch[1] - dwdL[0] / Stretch[0]) / (Deig[1] - Deig[0]) * (Veig(i, 0) * Veig(j, 1) * Veig(k, 0) * Veig(l, 1) + Veig(i, 0) * Veig(j, 1) * Veig(k, 1) * Veig(l, 0)) +
                                          (dwdL[0] / Stretch[0] - dwdL[1] / Stretch[1]) / (Deig[0] - Deig[1]) * (Veig(i, 1) * Veig(j, 0) * Veig(k, 1) * Veig(l, 0) + Veig(i, 1) * Veig(j, 0) * Veig(k, 0) * Veig(l, 1));
    }
    // Stretch[0] = Stretch[1]
    else
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                        tD[i][j][k][l] += 0.5 / Stretch[1] * ((-1.0 / Deig[1] * dwdL[1] + 1.0 / Stretch[1] * d2wd2L[1][1]) - (1.0 / Stretch[0] * d2wd2L[0][1])) * (Veig(i, 0) * Veig(j, 1) * Veig(k, 0) * Veig(l, 1) + Veig(i, 0) * Veig(j, 1) * Veig(k, 1) * Veig(l, 0)) +
                                          0.5 / Stretch[0] * ((-1.0 / Deig[0] * dwdL[0] + 1.0 / Stretch[0] * d2wd2L[0][0]) - (1.0 / Stretch[1] * d2wd2L[1][0])) * (Veig(i, 1) * Veig(j, 0) * Veig(k, 1) * Veig(l, 0) + Veig(i, 1) * Veig(j, 0) * Veig(k, 0) * Veig(l, 1));
    }

    // Stretch[0] \neq Stretch[2]
    if (fabs(Stretch[2] - Stretch[0]) > EPS)
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                        tD[i][j][k][l] += (dwdL[2] / Stretch[2] - dwdL[0] / Stretch[0]) / (Deig[2] - Deig[0]) * (Veig(i, 0) * Veig(j, 2) * Veig(k, 0) * Veig(l, 2) + Veig(i, 0) * Veig(j, 2) * Veig(k, 2) * Veig(l, 0)) +
                                          (dwdL[0] / Stretch[0] - dwdL[2] / Stretch[2]) / (Deig[0] - Deig[2]) * (Veig(i, 2) * Veig(j, 0) * Veig(k, 2) * Veig(l, 0) + Veig(i, 2) * Veig(j, 0) * Veig(k, 0) * Veig(l, 2));
    }
    // Stretch[0] = Stretch[2]
    else
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                        tD[i][j][k][l] += 0.5 / Stretch[2] * ((-1.0 / Deig[2] * dwdL[2] + 1.0 / Stretch[2] * d2wd2L[2][2]) - (1.0 / Stretch[0] * d2wd2L[0][2])) * (Veig(i, 0) * Veig(j, 2) * Veig(k, 0) * Veig(l, 2) + Veig(i, 0) * Veig(j, 2) * Veig(k, 2) * Veig(l, 0)) +
                                          0.5 / Stretch[0] * ((-1.0 / Deig[0] * dwdL[0] + 1.0 / Stretch[0] * d2wd2L[0][0]) - (1.0 / Stretch[2] * d2wd2L[2][0])) * (Veig(i, 2) * Veig(j, 0) * Veig(k, 2) * Veig(l, 0) + Veig(i, 2) * Veig(j, 0) * Veig(k, 0) * Veig(l, 2));
    }

    // Stretch[1] \neq Stretch[2]
    if (fabs(Stretch[2] - Stretch[1]) > EPS)
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                        tD[i][j][k][l] += (dwdL[2] / Stretch[2] - dwdL[1] / Stretch[1]) / (Deig[2] - Deig[1]) * (Veig(i, 1) * Veig(j, 2) * Veig(k, 1) * Veig(l, 2) + Veig(i, 1) * Veig(j, 2) * Veig(k, 2) * Veig(l, 1)) +
                                          (dwdL[1] / Stretch[1] - dwdL[2] / Stretch[2]) / (Deig[1] - Deig[2]) * (Veig(i, 2) * Veig(j, 1) * Veig(k, 2) * Veig(l, 1) + Veig(i, 2) * Veig(j, 1) * Veig(k, 1) * Veig(l, 2));
    }
    // Stretch[1] = Stretch[2]
    else
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                        tD[i][j][k][l] += 0.5 / Stretch[2] * ((-1.0 / Deig[2] * dwdL[2] + 1.0 / Stretch[2] * d2wd2L[2][2]) - (1.0 / Stretch[1] * d2wd2L[1][2])) * (Veig(i, 1) * Veig(j, 2) * Veig(k, 1) * Veig(l, 2) + Veig(i, 1) * Veig(j, 2) * Veig(k, 2) * Veig(l, 1)) +
                                          0.5 / Stretch[1] * ((-1.0 / Deig[1] * dwdL[1] + 1.0 / Stretch[1] * d2wd2L[1][1]) - (1.0 / Stretch[2] * d2wd2L[2][1])) * (Veig(i, 2) * Veig(j, 1) * Veig(k, 2) * Veig(l, 1) + Veig(i, 2) * Veig(j, 1) * Veig(k, 1) * Veig(l, 2));
    }

    // Map tD on Voight notation
    for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
            D[i][j] = tD[idmap[0][i]][idmap[1][i]][idmap[0][j]][idmap[1][j]];

    errorcode = 1;
    return errorcode;
}

// Ogden
int material_ogdennematicE3d(const double matconst[8], const double C[6], double &w,
                             double S[6], double D[6][6])
{

    // Get material constants
    int i, j, k, l, errorcode;
    double Stretch[3], J, J2;
    double c = matconst[1]; // material parameter
    double a = matconst[2]; // material parameter
    double a13 = pow(a, 1.0 / 3.0);
    double kappa = matconst[6]; // bulk modulus

    // Get spectral decomposition of C
    Eigen::Matrix3d Cd;
    Cd << C[0], C[3], C[5],
        C[3], C[1], C[4],
        C[5], C[4], C[2];
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(Cd);
    Eigen::Matrix3d Veig = eigensolver.eigenvectors();
    Eigen::Vector3d Deig = eigensolver.eigenvalues();
    if ((Deig[0] > Deig[2]) || (Deig[1] > Deig[2]))
        errorcode = -1;
    J2 = Cd.determinant(); // det(C) = J2, where det(F) = J
    J = sqrt(J2);
    double J13 = pow(J, 1.0 / 3.0);
    double J23 = J13 * J13;
    Stretch[0] = sqrt(Deig[0]);
    double Stretcha13 = pow(Stretch[0], 1.0 / 3.0);
    double Stretcha23 = Stretcha13 * Stretcha13;
    double Stretcha43 = Stretcha23 * Stretcha23;
    double Stretcha53 = Stretcha43 * Stretcha13;
    double Stretcha83 = Stretcha43 * Stretcha43;
    Stretch[1] = sqrt(Deig[1]);
    double Stretchb13 = pow(Stretch[1], 1.0 / 3.0);
    double Stretchb23 = Stretchb13 * Stretchb13;
    double Stretchb43 = Stretchb23 * Stretchb23;
    double Stretchb53 = Stretchb43 * Stretchb13;
    double Stretchb83 = Stretchb43 * Stretchb43;
    Stretch[2] = sqrt(Deig[2]);
    double Stretchc13 = pow(Stretch[2], 1.0 / 3.0);
    double Stretchc23 = Stretchc13 * Stretchc13;
    double Stretchc43 = Stretchc23 * Stretchc23;
    double Stretchc53 = Stretchc43 * Stretchc13;
    double Stretchc83 = Stretchc43 * Stretchc43;

    // Energy density W(F) = m1/2*m2^(1/3)*(tlambda1^2+tlambda2^2+tlambda3^2/m2-3*m2^(-1/3))+kappa/2*(J-1)^2 (Nematic materials Martin)
    w = c / 2.0 * a13 * (pow(Stretch[0] / J13, 2.0) + pow(Stretch[1] / J13, 2.0) + pow(Stretch[2] / J13, 2.0) / a - 3.0 / a13) + kappa / 2.0 * (J - 1.0) * (J - 1.0);

    // Compute derivatives of the energy density with respect to stretches and constants (treating J = lambda1*lambda2*1.0)
    double dwdL[3];
    dwdL[0] = c * a13 * (2.0 * Stretcha13 / (3.0 * Stretchb23 * Stretchc23) - Stretchb43 / (3.0 * Stretcha53 * Stretchc23) - Stretchc43 / (3.0 * a * Stretcha53 * Stretchb23)) + kappa * (J - 1.0) * Stretch[1] * Stretch[2];
    dwdL[1] = c * a13 * (-Stretcha43 / (3.0 * Stretchb53 * Stretchc23) + 2.0 * Stretchb13 / (3.0 * Stretcha23 * Stretchc23) - Stretchc43 / (3.0 * a * Stretcha23 * Stretchb53)) + kappa * (J - 1.0) * Stretch[0] * Stretch[2];
    dwdL[2] = c * a13 * (-Stretcha43 / (3.0 * Stretchb23 * Stretchc53) - Stretchb43 / (3.0 * Stretcha23 * Stretchc53) + 2.0 * Stretchc13 / (3.0 * a * Stretcha23 * Stretchb23)) + kappa * (J - 1.0) * Stretch[0] * Stretch[1];

    // Compute derivatives of the energy density with respect to stretches and constants (treating J = lambda1*lambda2*1.0)
    double d2wd2L[3][3];
    d2wd2L[0][0] = c * a13 * (2.0 / (9.0 * J23) + 5.0 * Stretchb43 / (9.0 * Stretcha83 * Stretchc23) + 5.0 * Stretchc43 / (9.0 * a * Stretcha83 * Stretchb23)) + kappa * Deig[1] * Deig[2];
    d2wd2L[1][1] = c * a13 * (5.0 * Stretcha43 / (9.0 * Stretchb83 * Stretchc23) + 2.0 / (9.0 * J23) + 5.0 * Stretchc43 / (9.0 * a * Stretcha23 * Stretchb83)) + kappa * Deig[0] * Deig[2];
    d2wd2L[2][2] = c * a13 * (5.0 * Stretcha43 / (9.0 * Stretchb23 * Stretchc83) + 5.0 * Stretchb43 / (9.0 * Stretcha23 * Stretchc83) + 2.0 / (9.0 * a * J23)) + kappa * Deig[0] * Deig[1];
    d2wd2L[0][1] = c * a13 * (-4.0 * Stretcha13 / (9.0 * Stretchb53 * Stretchc23) - 4.0 * Stretchb13 / (9.0 * Stretcha53 * Stretchc23) + 2.0 * Stretchc43 / (9.0 * a * Stretcha53 * Stretchb53)) + kappa * Stretch[2] * (2.0 * J - 1.0);
    d2wd2L[1][0] = d2wd2L[0][1];
    d2wd2L[0][2] = c * a13 * (-4.0 * Stretcha13 / (9.0 * Stretchb23 * Stretchc53) + 4.0 * Stretchb43 / (9.0 * Stretcha53 * Stretchc53) - 4.0 * Stretchc13 / (9.0 * a * Stretcha53 * Stretchb23)) + kappa * Stretch[1] * (2.0 * J - 1.0);
    d2wd2L[2][0] = d2wd2L[0][2];
    d2wd2L[1][2] = c * a13 * (2.0 * Stretcha43 / (9.0 * Stretchb53 * Stretchc53) - 4.0 * Stretchb13 / (9.0 * Stretcha23 * Stretchc53) - 4.0 * Stretchc13 / (9.0 * a * Stretcha23 * Stretchb53)) + kappa * Stretch[0] * (2.0 * J - 1.0);
    d2wd2L[2][1] = d2wd2L[1][2];

    // The second Piola-Kirchhoff stress tensor S
    double tS[3][3];
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            tS[i][j] = dwdL[0] / Stretch[0] * Veig(i, 0) * Veig(j, 0) +
                       dwdL[1] / Stretch[1] * Veig(i, 1) * Veig(j, 1) +
                       dwdL[2] / Stretch[2] * Veig(i, 2) * Veig(j, 2);

    // Map tS on Voight notation
    int idmap[2][6] = {{0, 1, 2, 0, 1, 2}, {0, 1, 2, 1, 2, 0}}; // map from tensor to Voight notation
    for (i = 0; i < 6; i++)
        S[i] = tS[idmap[0][i]][idmap[1][i]];

    // Material stiffness D
    double tD[3][3][3][3];
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            for (k = 0; k < 3; k++)
                for (l = 0; l < 3; l++)
                    tD[i][j][k][l] = 1.0 / Stretch[0] * (-1.0 / Deig[0] * dwdL[0] + 1.0 / Stretch[0] * d2wd2L[0][0]) * Veig(i, 0) * Veig(j, 0) * Veig(k, 0) * Veig(l, 0) +
                                     1.0 / Stretch[0] * (1.0 / Stretch[1] * d2wd2L[0][1]) * Veig(i, 0) * Veig(j, 0) * Veig(k, 1) * Veig(l, 1) +
                                     1.0 / Stretch[0] * (1.0 / Stretch[2] * d2wd2L[0][2]) * Veig(i, 0) * Veig(j, 0) * Veig(k, 2) * Veig(l, 2) +
                                     1.0 / Stretch[1] * (1.0 / Stretch[0] * d2wd2L[1][0]) * Veig(i, 1) * Veig(j, 1) * Veig(k, 0) * Veig(l, 0) +
                                     1.0 / Stretch[1] * (-1.0 / Deig[1] * dwdL[1] + 1.0 / Stretch[1] * d2wd2L[1][1]) * Veig(i, 1) * Veig(j, 1) * Veig(k, 1) * Veig(l, 1) +
                                     1.0 / Stretch[1] * (1.0 / Stretch[2] * d2wd2L[1][2]) * Veig(i, 1) * Veig(j, 1) * Veig(k, 2) * Veig(l, 2) +
                                     1.0 / Stretch[2] * (1.0 / Stretch[0] * d2wd2L[2][0]) * Veig(i, 2) * Veig(j, 2) * Veig(k, 0) * Veig(l, 0) +
                                     1.0 / Stretch[2] * (1.0 / Stretch[1] * d2wd2L[2][1]) * Veig(i, 2) * Veig(j, 2) * Veig(k, 1) * Veig(l, 1) +
                                     1.0 / Stretch[2] * (-1.0 / Deig[2] * dwdL[2] + 1.0 / Stretch[2] * d2wd2L[2][2]) * Veig(i, 2) * Veig(j, 2) * Veig(k, 2) * Veig(l, 2);

    // Stretch[0] \neq Stretch[1]
    if (fabs(Stretch[1] - Stretch[0]) > EPS)
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                        tD[i][j][k][l] += (dwdL[1] / Stretch[1] - dwdL[0] / Stretch[0]) / (Deig[1] - Deig[0]) * (Veig(i, 0) * Veig(j, 1) * Veig(k, 0) * Veig(l, 1) + Veig(i, 0) * Veig(j, 1) * Veig(k, 1) * Veig(l, 0)) +
                                          (dwdL[0] / Stretch[0] - dwdL[1] / Stretch[1]) / (Deig[0] - Deig[1]) * (Veig(i, 1) * Veig(j, 0) * Veig(k, 1) * Veig(l, 0) + Veig(i, 1) * Veig(j, 0) * Veig(k, 0) * Veig(l, 1));
    }
    // Stretch[0] = Stretch[1]
    else
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                        tD[i][j][k][l] += 0.5 / Stretch[1] * ((-1.0 / Deig[1] * dwdL[1] + 1.0 / Stretch[1] * d2wd2L[1][1]) - (1.0 / Stretch[0] * d2wd2L[0][1])) * (Veig(i, 0) * Veig(j, 1) * Veig(k, 0) * Veig(l, 1) + Veig(i, 0) * Veig(j, 1) * Veig(k, 1) * Veig(l, 0)) +
                                          0.5 / Stretch[0] * ((-1.0 / Deig[0] * dwdL[0] + 1.0 / Stretch[0] * d2wd2L[0][0]) - (1.0 / Stretch[1] * d2wd2L[1][0])) * (Veig(i, 1) * Veig(j, 0) * Veig(k, 1) * Veig(l, 0) + Veig(i, 1) * Veig(j, 0) * Veig(k, 0) * Veig(l, 1));
    }

    // Stretch[0] \neq Stretch[2]
    if (fabs(Stretch[2] - Stretch[0]) > EPS)
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                        tD[i][j][k][l] += (dwdL[2] / Stretch[2] - dwdL[0] / Stretch[0]) / (Deig[2] - Deig[0]) * (Veig(i, 0) * Veig(j, 2) * Veig(k, 0) * Veig(l, 2) + Veig(i, 0) * Veig(j, 2) * Veig(k, 2) * Veig(l, 0)) +
                                          (dwdL[0] / Stretch[0] - dwdL[2] / Stretch[2]) / (Deig[0] - Deig[2]) * (Veig(i, 2) * Veig(j, 0) * Veig(k, 2) * Veig(l, 0) + Veig(i, 2) * Veig(j, 0) * Veig(k, 0) * Veig(l, 2));
    }
    // Stretch[0] = Stretch[2]
    else
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                        tD[i][j][k][l] += 0.5 / Stretch[2] * ((-1.0 / Deig[2] * dwdL[2] + 1.0 / Stretch[2] * d2wd2L[2][2]) - (1.0 / Stretch[0] * d2wd2L[0][2])) * (Veig(i, 0) * Veig(j, 2) * Veig(k, 0) * Veig(l, 2) + Veig(i, 0) * Veig(j, 2) * Veig(k, 2) * Veig(l, 0)) +
                                          0.5 / Stretch[0] * ((-1.0 / Deig[0] * dwdL[0] + 1.0 / Stretch[0] * d2wd2L[0][0]) - (1.0 / Stretch[2] * d2wd2L[2][0])) * (Veig(i, 2) * Veig(j, 0) * Veig(k, 2) * Veig(l, 0) + Veig(i, 2) * Veig(j, 0) * Veig(k, 0) * Veig(l, 2));
    }

    // Stretch[1] \neq Stretch[2]
    if (fabs(Stretch[2] - Stretch[1]) > EPS)
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                        tD[i][j][k][l] += (dwdL[2] / Stretch[2] - dwdL[1] / Stretch[1]) / (Deig[2] - Deig[1]) * (Veig(i, 1) * Veig(j, 2) * Veig(k, 1) * Veig(l, 2) + Veig(i, 1) * Veig(j, 2) * Veig(k, 2) * Veig(l, 1)) +
                                          (dwdL[1] / Stretch[1] - dwdL[2] / Stretch[2]) / (Deig[1] - Deig[2]) * (Veig(i, 2) * Veig(j, 1) * Veig(k, 2) * Veig(l, 1) + Veig(i, 2) * Veig(j, 1) * Veig(k, 1) * Veig(l, 2));
    }
    // Stretch[1] = Stretch[2]
    else
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                        tD[i][j][k][l] += 0.5 / Stretch[2] * ((-1.0 / Deig[2] * dwdL[2] + 1.0 / Stretch[2] * d2wd2L[2][2]) - (1.0 / Stretch[1] * d2wd2L[1][2])) * (Veig(i, 1) * Veig(j, 2) * Veig(k, 1) * Veig(l, 2) + Veig(i, 1) * Veig(j, 2) * Veig(k, 2) * Veig(l, 1)) +
                                          0.5 / Stretch[1] * ((-1.0 / Deig[1] * dwdL[1] + 1.0 / Stretch[1] * d2wd2L[1][1]) - (1.0 / Stretch[2] * d2wd2L[2][1])) * (Veig(i, 2) * Veig(j, 1) * Veig(k, 2) * Veig(l, 1) + Veig(i, 2) * Veig(j, 1) * Veig(k, 1) * Veig(l, 2));
    }

    // Map tD on Voight notation
    for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
            D[i][j] = tD[idmap[0][i]][idmap[1][i]][idmap[0][j]][idmap[1][j]];

    errorcode = 1;
    return errorcode;
}

/******************** Basic linear algebra ********************/

// Computes inverse of 2x2 matrix with determinant
bool getInverse22(const double m[2][2], double &detm, double minv[2][2])
{

    // Get determinant of m and its inverse
    detm = (m[0][0] * m[1][1] - m[0][1] * m[1][0]);

    if (fabs(detm) < EPS)
        return false;

    // Inverse of matrix m
    minv[0][0] = m[1][1] / detm;
    minv[1][0] = -m[1][0] / detm;
    minv[0][1] = -m[0][1] / detm;
    minv[1][1] = m[0][0] / detm;

    return true;
}

bool getInverse22(const double m[4], double &detm, double minv[4])
{

    // Get determinant of m and its inverse
    detm = (m[0] * m[3] - m[2] * m[1]);

    if (fabs(detm) < EPS)
        return false;

    // Inverse of matrix m
    minv[0] = m[3] / detm;
    minv[1] = -m[1] / detm;
    minv[2] = -m[2] / detm;
    minv[3] = m[0] / detm;

    return true;
}

bool getInverse22sym(const double m[3], double &detm, double minv[3])
{

    // Get determinant of m and its inverse
    detm = (m[0] * m[1] - m[2] * m[2]);

    if (fabs(detm) < EPS)
        return false;

    // Inverse of matrix m
    minv[0] = m[1] / detm;
    minv[1] = m[0] / detm;
    minv[2] = -m[2] / detm;

    return true;
}

// Computes eigenvalues and eigenvectors of a 2x2 symmetric matrix
bool getEigendecomposition22(const double m[3], double D[2], double V[2][2])
{
    double a = 1.0;
    double b = -(m[0] + m[1]);            // minus trace
    double c = m[0] * m[1] - m[2] * m[2]; // det

    // Get eigenvalues
    double d = b * b - 4 * a * c;
    if (d >= 0) // if sqrt(d) is real; complex would violate physics
    {
        double sqrtd = sqrt(d);
        D[0] = (-b - sqrtd) / (2 * a);
        D[1] = (-b + sqrtd) / (2 * a);

        // Get eigenvectors
        if (D[0] == D[1]) // isotropic tensor, b == 0
        {
            V[0][0] = 1.0;
            V[1][0] = 0.0;

            V[0][1] = 0.0;
            V[1][1] = 1.0;

            return true;
        }
        else // preferred direction exists, b !=0
        {
            V[0][0] = m[2];
            V[1][0] = D[0] - m[0];

            V[0][1] = D[1] - m[1];
            V[1][1] = m[2];
        }

        // Nnormalize eigenvectors to 1 in L2 norm
        double normv1 = sqrt(V[0][0] * V[0][0] + V[1][0] * V[1][0]);
        V[0][0] /= normv1;
        V[1][0] /= normv1;

        double normv2 = sqrt(V[0][1] * V[0][1] + V[1][1] * V[1][1]);
        V[0][1] /= normv2;
        V[1][1] /= normv2;
        return true;
    }
    else
        return false;
}

// Computes inverse of 3x3 matrix
bool getInverse33(const double m[3][3], double &detm, double minv[3][3])
{

    // Get determinant of m
    detm = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if (fabs(detm) < EPS)
        return false;

    // Inverse of matrix m
    minv[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) / detm;
    minv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / detm;
    minv[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) / detm;
    minv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / detm;
    minv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / detm;
    minv[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) / detm;
    minv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / detm;
    minv[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) / detm;
    minv[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) / detm;

    return true;
}

bool getInverse33(const double m[9], double &detm, double minv[9])
{

    // Get determinant of m
    detm = m[0] * (m[4] * m[8] - m[5] * m[7]) -
           m[3] * (m[1] * m[8] - m[7] * m[2]) +
           m[6] * (m[1] * m[5] - m[4] * m[2]);

    if (fabs(detm) < EPS)
        return false;

    // Inverse of matrix m
    minv[0] = (m[4] * m[8] - m[5] * m[7]) / detm;
    minv[1] = (m[7] * m[2] - m[1] * m[8]) / detm;
    minv[2] = (m[1] * m[5] - m[2] * m[4]) / detm;
    minv[3] = (m[6] * m[5] - m[3] * m[8]) / detm;
    minv[4] = (m[0] * m[8] - m[6] * m[2]) / detm;
    minv[5] = (m[2] * m[3] - m[0] * m[5]) / detm;
    minv[6] = (m[3] * m[7] - m[6] * m[4]) / detm;
    minv[7] = (m[1] * m[6] - m[0] * m[7]) / detm;
    minv[8] = (m[0] * m[4] - m[1] * m[3]) / detm;

    return true;
}

bool getInverse33sym(const double m[6], double &detm, double minv[6])
{

    // Get determinant of m
    detm = m[0] * (m[1] * m[2] - m[4] * m[4]) -
           m[3] * (m[3] * m[2] - m[4] * m[5]) +
           m[5] * (m[3] * m[4] - m[1] * m[5]);

    if (fabs(detm) < EPS)
        return false;

    // Inverse of matrix m
    minv[0] = (m[1] * m[2] - m[4] * m[4]) / detm;
    minv[1] = (m[0] * m[2] - m[5] * m[5]) / detm;
    minv[2] = (m[0] * m[1] - m[3] * m[3]) / detm;
    minv[3] = (m[5] * m[4] - m[3] * m[2]) / detm;
    minv[4] = (m[3] * m[5] - m[0] * m[4]) / detm;
    minv[5] = (m[3] * m[4] - m[5] * m[1]) / detm;

    return true;
}

// Computes inverse of 4x4 matrix
bool getInverse44(const double mIn[4][4], double &detm, double minv[4][4])
{

    double m[16], inv[16];
    int i, j;

    // Reshape mIn
    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            m[i + 4 * j] = mIn[i][j];
        }
    }

    inv[0] = m[5] * m[10] * m[15] -
             m[5] * m[11] * m[14] -
             m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] +
             m[13] * m[6] * m[11] -
             m[13] * m[7] * m[10];

    inv[4] = -m[4] * m[10] * m[15] +
             m[4] * m[11] * m[14] +
             m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] -
             m[12] * m[6] * m[11] +
             m[12] * m[7] * m[10];

    inv[8] = m[4] * m[9] * m[15] -
             m[4] * m[11] * m[13] -
             m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4] * m[9] * m[14] +
              m[4] * m[10] * m[13] +
              m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] -
              m[12] * m[5] * m[10] +
              m[12] * m[6] * m[9];

    inv[1] = -m[1] * m[10] * m[15] +
             m[1] * m[11] * m[14] +
             m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] -
             m[13] * m[2] * m[11] +
             m[13] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] -
             m[0] * m[11] * m[14] -
             m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0] * m[9] * m[15] +
             m[0] * m[11] * m[13] +
             m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] -
             m[12] * m[1] * m[11] +
             m[12] * m[3] * m[9];

    inv[13] = m[0] * m[9] * m[14] -
              m[0] * m[10] * m[13] -
              m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1] * m[6] * m[15] -
             m[1] * m[7] * m[14] -
             m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] +
             m[0] * m[7] * m[14] +
             m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] -
             m[12] * m[2] * m[7] +
             m[12] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] -
              m[0] * m[7] * m[13] -
              m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0] * m[5] * m[14] +
              m[0] * m[6] * m[13] +
              m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] -
              m[12] * m[1] * m[6] +
              m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
             m[1] * m[7] * m[10] +
             m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] +
             m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
              m[0] * m[7] * m[9] +
              m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] +
              m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    detm = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (fabs(detm) < EPS)
        return false;

    for (i = 0; i < 16; i++)
        inv[i] = inv[i] / detm;

    // Reshape minv
    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            minv[i][j] = inv[i + 4 * j];
        }
    }

    return true;
}

/******************** Interpolation ********************/

// Return evaluation of Lagrange polynomial of given order at xi
double getLagrange1DPoly(const int order, const double xinode, const int derivative,
                         const double xi)
{
    double eval = 0.0;

    // So far only quadratic and cubic polynomials implemented
    switch (order)
    {
    case 2:
    {
        if (derivative == 0)
        {                                  // get function evaluation
            if (fabs(-1.0 - xinode) < EPS) // N1
                eval = 1.0 / 2.0 * xi * (xi - 1.0);
            else if (fabs(0.0 - xinode) < EPS) // N2
                eval = (1.0 + xi) * (1.0 - xi);
            else if (fabs(1.0 - xinode) < EPS) // N3
                eval = 1.0 / 2.0 * xi * (1.0 + xi);
        }
        else if (derivative == 1)
        {                                  // get the first derivative evaluation
            if (fabs(-1.0 - xinode) < EPS) // dN1
                eval = xi - 1.0 / 2.0;
            else if (fabs(0.0 - xinode) < EPS) // dN2
                eval = -2.0 * xi;
            else if (fabs(1.0 - xinode) < EPS) // dN3
                eval = 1.0 / 2.0 + xi;
        }
        break;
    }

    case 3:
    {
        if (derivative == 0)
        {                                  // get function evaluation
            if (fabs(-1.0 - xinode) < EPS) // N1
                eval = 1.0 / 16.0 * (xi - 1.0) * (1.0 - 9.0 * xi * xi);
            else if (fabs(-1.0 / 3.0 - xinode) < EPS) // N2
                eval = 9.0 / 16.0 * (1.0 - xi * xi) * (1.0 - 3.0 * xi);
            else if (fabs(1.0 / 3.0 - xinode) < EPS) // N3
                eval = 9.0 / 16.0 * (1.0 - xi * xi) * (1.0 + 3.0 * xi);
            else if (fabs(1.0 - xinode) < EPS) // N4
                eval = 1.0 / 16.0 * (xi + 1.0) * (9.0 * xi * xi - 1.0);
        }
        else if (derivative == 1)
        {                                  // get first derivative evaluation
            if (fabs(-1.0 - xinode) < EPS) // dN1
                eval = 1.0 / 16.0 * (1.0 + 9.0 * xi * (2.0 - 3.0 * xi));
            else if (fabs(-1.0 / 3.0 - xinode) < EPS) // dN2
                eval = 9.0 / 16.0 * (-3.0 + xi * (-2.0 + 9.0 * xi));
            else if (fabs(1.0 / 3.0 - xinode) < EPS) // dN3
                eval = -9.0 / 16.0 * (-3.0 + xi * (2.0 + 9.0 * xi));
            else if (fabs(1.0 - xinode) < EPS) // dN4
                eval = 1.0 / 16.0 * (-1.0 + 9.0 * xi * (2.0 + 3.0 * xi));
        }
        break;
    }

    default:
        mexErrMsgTxt("Wrong Lagrange polynomial order.");
    }
    return eval;
}

/******************** Geometric features ********************/

// Check if two bounding boxes specified through xa (left x), wa (width x), ya (left y), ha (height y), and
// xb, wb, yb, hb intersect
bool intersect2d(const double xa, const double wa, const double ya,
                 const double ha, const double xb, const double wb,
                 const double yb, const double hb)
{
    return xa + wa > xb &&
           xa < xb + wb &&
           ya + ha > yb &&
           ya < yb + hb;
}

// Check if two bounding boxes specified through xa (min x), wa (width x), ya (min y), ha (height y), za (min z), da (depth z), and
// xb, wb, yb, hb, zb, db intersect
bool intersect3d(const double xa, const double wa, const double ya, const double ha, const double za, const double da,
                 const double xb, const double wb, const double yb, const double hb, const double zb, const double db)
{
    return xa + wa > xb &&
           xa < xb + wb &&
           ya + ha > yb &&
           ya < yb + hb &&
           za + da > zb &&
           za < zb + db;
}
