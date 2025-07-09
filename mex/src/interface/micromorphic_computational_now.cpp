// micromorphic_stress_computational: compute homogenized quantities for the
// micromorphic formulation corresponding to an ar arbitrary amount of modes
// nmode and FE2 formulation

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

  // Test for eight input arguments
  if (nrhs != 8)
    mexErrMsgTxt("Eight input arguments required.");

  // Get the input data
  double *prp = mxGetPr(prhs[0]);
  double *prt = mxGetPr(prhs[1]);
  double *prmat = mxGetPr(prhs[2]);
  int ngauss = (int)mxGetScalar(prhs[3]);
  double *pru = mxGetPr(prhs[4]);
  int nmodes = (int)mxGetScalar(prhs[5]);
  double *prphi = mxGetPr(prhs[6]);
  int maxNumThreads = (int)mxGetScalar(prhs[7]);
  int nnodes = (int)mxGetN(prhs[0]);
  int nelem = (int)mxGetN(prhs[1]);
  int mt = (int)mxGetM(prhs[1]);
  int np = mt - 1; // number of points in an element
  int mmat = (int)mxGetM(prhs[2]);
  int ndof = 2 * nnodes;

  // Test consistency of nmodes
  if (nmodes != (int)mxGetN(prhs[6]))
    mexErrMsgTxt("nmodes is incosistent. Check inputs for phi.");

  // Test the number of dofs
  if (ndof != (int)mxGetM(prhs[4]) || ndof != (int)mxGetM(prhs[6]))
    mexErrMsgTxt("Wrong number of dofs.");

  // Allocate outputs
  // Energy
  nlhs = 10;
  // Homoenized stresses
  plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(4, 1, mxREAL);
  plhs[2] = mxCreateDoubleMatrix(1, nmodes, mxREAL);
  plhs[3] = mxCreateDoubleMatrix(2, nmodes, mxREAL);
  // Homogenized stiffnessess
  plhs[4] = mxCreateDoubleMatrix(4, 4, mxREAL);
  plhs[5] = mxCreateDoubleMatrix(4, 2 * nmodes, mxREAL);
  plhs[6] = mxCreateDoubleMatrix(4, nmodes, mxREAL);
  plhs[7] = mxCreateDoubleMatrix(2 * nmodes, 2 * nmodes, mxREAL);
  plhs[8] = mxCreateDoubleMatrix(2 * nmodes, nmodes, mxREAL);
  plhs[9] = mxCreateDoubleMatrix(nmodes, nmodes, mxREAL);
  // Get pointers
  double *prW = mxGetPr(plhs[0]);
  double *prP1 = mxGetPr(plhs[1]);
  double *prQi = mxGetPr(plhs[2]);
  double *prRi = mxGetPr(plhs[3]);
  double *prC00 = mxGetPr(plhs[4]);
  double *prC0iBB = mxGetPr(plhs[5]);
  double *prC0iBN = mxGetPr(plhs[6]);
  double *prCiiBB = mxGetPr(plhs[7]);
  double *prCiiBN = mxGetPr(plhs[8]);
  double *prCiiNN = mxGetPr(plhs[9]);

  // Compute the gradient and Hessian, populate prG, prI, prJ, prS
  int tr; // triangle ID
  int errCountGauss =
      0;               // integration rule: error control inside parallel region
  int errCountMat = 0; // constitutive law: error control inside parallel region
  int errCountEl = 0;  // constitutive law: error control inside parallel region
#pragma omp parallel num_threads(maxNumThreads)
  {
    int _i, _j, _k, _l, _ig, _h, _g;
    double _Wd, _W, _Jdet, _thickness;
    double _F[4], _P[4], _D[4][4], _tcoord[3], _matconst[8]; // use total Lagrangian formulation
    double _P1[4], _X[2], _C00[4][4];
    double *_Qi = new double[nmodes];
    double *_tQi = new double[nmodes];
    double **_Ri = new double *[2];
    double **_tRi = new double *[2];
    for (_i = 0; _i < 2; _i++)
    {
      _Ri[_i] = new double[nmodes];
      _tRi[_i] = new double[nmodes];
    }
    double **_C0iBB = new double *[4];
    double **_C0iBN = new double *[4];
    double **_Dp = new double *[4];
    double **_Dgp = new double *[4];
    for (_i = 0; _i < 4; _i++)
    {
      _C0iBB[_i] = new double[2 * nmodes];
      _C0iBN[_i] = new double[nmodes];
      _Dp[_i] = new double[2 * nmodes];
      _Dgp[_i] = new double[nmodes];
    }
    double **_CiiBB = new double *[2 * nmodes];
    double **_CiiBN = new double *[2 * nmodes];
    double **_pDp = new double *[2 * nmodes];
    double **_pDgp = new double *[2 * nmodes];
    for (_i = 0; _i < 2 * nmodes; _i++)
    {
      _CiiBB[_i] = new double[2 * nmodes];
      _CiiBN[_i] = new double[nmodes];
      _pDp[_i] = new double[2 * nmodes];
      _pDgp[_i] = new double[nmodes];
    }
    double **_CiiNN = new double *[nmodes];
    double **_gpDp = new double *[nmodes];
    double **_gpDgp = new double *[nmodes];
    for (_i = 0; _i < nmodes; _i++)
    {
      _CiiNN[_i] = new double[nmodes];
      _gpDp[_i] = new double[2 * nmodes];
      _gpDgp[_i] = new double[nmodes];
    }
    double *_phi = new double[2 * nmodes];
    double *_gradphi = new double[4 * nmodes];
    int *_idel = new int[np];    // ids of nodes of given element
    int *_cod = new int[2 * np]; // code numbers of given element
    double *_uel =
        new double[2 * np]; // displacements associated with given element
    double *_phiel =
        new double[nmodes * 2 *
                   np];          // displacements associated with given element
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
    int (*_getWPD)(const double matconst[8], const double F[4], double &Wd,
                   double P[4], double D[4][4]) =
        NULL; // pointer to constitutive law functions
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
        _uel[_i] = pru[_cod[_i] - 1];

      // Get phi_i
      for (_i = 0; _i < nmodes; _i++)
        for (_j = 0; _j < 2 * np; _j++)
          _phiel[2 * np * _i + _j] = prphi[ndof * _i + _cod[_j] - 1];

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
      case 1: // W(F) = m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+1/2*kappa(ln(J))^2
              // (OOFEM)
        _getWPD = &material_oofemF2d;
        break;
      case 2: // W(F) = m1(I1-3)+m2(I1-3)^2-2m1*ln(J)+1/2*kappa(J-1)^2
              // (Bertoldi, Boyce)
        _getWPD = &material_bbF2d;
        break;
      case 3: // W(F) =
              // m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I1/J^{2/3}-3)^3+9/2*kappa(J^{1/3}-1)^2
              // (Jamus, Green, Simpson)
        _getWPD = &material_jgsF2d;
        break;
      case 4: // W(F) =
              // m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I2/J^{2/3}-3)^2+kappa*(J-1)^2
              // (five-term Mooney-Rivlin)
        _getWPD = &material_5mrF2d;
        break;
      case 5: // W(F) = 0.5*(0.5*(C-I)*(m1*IxI+2*kappa*I)*0.5*(C-I)) (linear
              // elastic material)
        _getWPD = &material_leF2d;
        break;
      default: // if unrecognized material used, use OOFEM in order not to
               // crash, but throw an error
        _getWPD = &material_oofemF2d;
#pragma omp atomic
        errCountMat++;
        break;
      }

      // Gauss integration rule
      // Initialize energy, stresses, and stiffnesses
      _W = 0.0;
      for (_i = 0; _i < 4; _i++)
      {
        _P1[_i] = 0.0;
        for (_k = 0; _k < 4; _k++)
          _C00[_i][_k] = 0.0;
        for (_l = 0; _l < nmodes; _l++)
        {
          _C0iBN[_i][_l] = 0.0;
          for (_k = 0; _k < 2; _k++)
            _C0iBB[_i][_k + 2 * _l] = 0.0;
        }
      }
      for (_l = 0; _l < nmodes; _l++)
      {
        _Qi[_l] = 0.0;
        for (_j = 0; _j < 2; _j++)
          _Ri[_j][_l] = 0.0;
        for (_j = 0; _j < nmodes; _j++)
        {
          _CiiNN[_j][_l] = 0.0;
          for (_i = 0; _i < 2; _i++)
          {
            _CiiBN[2 * _j + _i][_l] = 0.0;
            for (_k = 0; _k < 2; _k++)
              _CiiBB[2 * _j + _i][_k + 2 * _l] = 0.0;
          }
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
        _getBe(np, _tcoord, _x, _y, _Jdet, _Be);

        // Get deformation gradient
        getF2d(_Be, _uel, np, _F);

        // Get energy density, stress, and material stiffness evaluated at
        // current Gauss point
        _getWPD(_matconst, _F, // inputs
                _Wd, _P, _D);  // outputs

        // Get shape function evaluations
        _getShapeFun(np, _tcoord, _N);

        // Get phi
        for (_i = 0; _i < 2 * nmodes; _i++)
          _phi[_i] = 0.0; // initialize phi
        for (_i = 0; _i < nmodes; _i++)
          for (_j = 0; _j < np; _j++)
          {
            _phi[2 * _i + 0] += _phiel[2 * np * _i + 2 * _j + 0] * _N[_j];
            _phi[2 * _i + 1] += _phiel[2 * np * _i + 2 * _j + 1] * _N[_j];
          }

        // Get grad(phi)
        for (_i = 0; _i < 4 * nmodes; _i++)
          _gradphi[_i] = 0.0; // initialize phi
        for (_i = 0; _i < nmodes; _i++)
          for (_j = 0; _j < 2; _j++)
            for (_k = 0; _k < 2; _k++)
              for (_l = 0; _l < 2 * np; _l++)
                _gradphi[4 * _i + _j + 2 * _k] +=
                    _Be[_j + 2 * _k][_l] * _phiel[2 * np * _i + _l];

        // Get X_m
        _X[0] = 0.0; // initialize X_m
        _X[1] = 0.0;
        for (_i = 0; _i < np; _i++)
        {
          _X[0] += _x[_i] * _N[_i];
          _X[1] += _y[_i] * _N[_i];
        }

        // Integrate element's energy W = W + alpha(ig)*Wd*_Jdet*h
        _W += _alpha[_ig] * _thickness * _Jdet * _Wd;

        // Integrate homogenized stress P1 = P1 + alpha(ig)*P(:)*Jdet*h
        for (_i = 0; _i < 2; _i++)
          for (_j = 0; _j < 2; _j++)
            _P1[_i + 2 * _j] +=
                _alpha[_ig] * _thickness * _Jdet * _P[_i + 2 * _j];

        // Integrate homogenized stress Qi = Qi +
        // alpha(ig)*P:grad(phi_i)*Jdet*h, tQi = P:grad(phi_i)
        for (_i = 0; _i < nmodes; _i++)
        {
          _tQi[_i] = 0.0;
          for (_j = 0; _j < 2; _j++)
            for (_k = 0; _k < 2; _k++)
              _tQi[_i] += _P[_j + 2 * _k] * _gradphi[4 * _i + _j + 2 * _k];
        }
        for (_i = 0; _i < nmodes; _i++)
          _Qi[_i] += _alpha[_ig] * _thickness * _Jdet * _tQi[_i];

        // Integrate homogenized stress R1 = R1 + alpha(ig)*P*phi1*Jdet*h, tRi = P'.phi_i
        for (_i = 0; _i < nmodes; _i++)
          for (_j = 0; _j < 2; _j++)
          {
            _tRi[_j][_i] = 0.0;
            for (_k = 0; _k < 2; _k++)
              _tRi[_j][_i] += _P[_k + 2 * _j] * _phi[2 * _i + _k];
          }
        for (_i = 0; _i < nmodes; _i++)
          for (_j = 0; _j < 2; _j++)
            _Ri[_j][_i] += _alpha[_ig] * _thickness * _Jdet * (_tRi[_j][_i] + _X[_j] * _tQi[_i]);

        // Integrate homogenized stiffness C00 = C00 + alpha(ig)*D*Jdet*h
        for (_i = 0; _i < 4; _i++)
          for (_j = 0; _j < 4; _j++)
            _C00[_i][_j] += _alpha[_ig] * _thickness * _Jdet * _D[_i][_j];

        // Integrate terms that occur multiple times
        for (_i = 0; _i < 4; _i++)
        {
          for (_j = 0; _j < 2 * nmodes; _j++)
            _Dp[_i][_j] = 0.0;
          for (_j = 0; _j < nmodes; _j++)
            _Dgp[_i][_j] = 0.0;
        }
        for (_i = 0; _i < 2 * nmodes; _i++)
        {
          for (_j = 0; _j < 2 * nmodes; _j++)
            _pDp[_i][_j] = 0.0;
          for (_j = 0; _j < nmodes; _j++)
          {
            _pDgp[_i][_j] = 0.0;
            _gpDp[_j][_i] = 0.0;
          }
        }
        for (_i = 0; _i < nmodes; _i++)
          for (_j = 0; _j < nmodes; _j++)
            _gpDgp[_i][_j] = 0.0;
        for (_h = 0; _h < nmodes; _h++)
          for (_i = 0; _i < 2; _i++)
            for (_j = 0; _j < 2; _j++)
              for (_k = 0; _k < 2; _k++)
                for (_l = 0; _l < 2; _l++)
                {
                  _Dp[_i + 2 * _j][_l + 2 * _h] +=
                      _D[_i + 2 * _j][_k + 2 * _l] * _phi[2 * _h + _k];
                  _Dgp[_i + 2 * _j][_h] +=
                      _D[_i + 2 * _j][_k + 2 * _l] * _gradphi[_k + 2 * _l + 4 * _h];
                  for (_g = 0; _g < nmodes; _g++)
                  {
                    _pDp[_j + 2 * _h][_l + 2 * _g] +=
                        _phi[_i + 2 * _h] * _D[_i + 2 * _j][_k + 2 * _l] * _phi[_k + 2 * _g];
                    _pDgp[_j + 2 * _h][_g] +=
                        _phi[_i + 2 * _h] * _D[_i + 2 * _j][_k + 2 * _l] * _gradphi[_k + 2 * _l + 4 * _g];
                    _gpDp[_h][_l + 2 * _g] += _gradphi[_i + 2 * _j + 4 * _h] * _D[_i + 2 * _j][_k + 2 * _l] * _phi[_k + 2 * _g];
                    _gpDgp[_h][_g] +=
                        _gradphi[_i + 2 * _j + 4 * _h] * _D[_i + 2 * _j][_k + 2 * _l] * _gradphi[_k + 2 * _l + 4 * _g];
                  }
                }

        // Integrate homogenized stiffness C0iBB = C0iBB + alpha(ig)*(D.phi + D:gradphi*X)*Jdet*h
        for (_h = 0; _h < nmodes; _h++)
          for (_i = 0; _i < 2; _i++)
            for (_j = 0; _j < 2; _j++)
              for (_l = 0; _l < 2; _l++)
                _C0iBB[_i + 2 * _j][_l + 2 * _h] += _alpha[_ig] * _thickness * _Jdet * (_Dp[_i + 2 * _j][_l + 2 * _h] + _Dgp[_i + 2 * _j][_h] * _X[_l]);

        // Integrate homogenized stiffness C0iBN = C0iBN + alpha(ig)*(D:gradphi)*Jdet*h
        for (_h = 0; _h < nmodes; _h++)
          for (_i = 0; _i < 2; _i++)
            for (_j = 0; _j < 2; _j++)
              _C0iBN[_i + 2 * _j][_h] += _alpha[_ig] * _thickness * _Jdet * _Dgp[_i + 2 * _j][_h];

        // Integrate homogenized stiffness CiiBB = CiiBB + alpha(ig)*Jdet*h*(phi.D.phi +
        // phi.D:gradphi*X + X*gradphi:D.phi + X*gradphi:D:gradphi*X)
        for (_g = 0; _g < nmodes; _g++)
          for (_h = 0; _h < nmodes; _h++)
            for (_j = 0; _j < 2; _j++)
              for (_l = 0; _l < 2; _l++)
                _CiiBB[_j + 2 * _h][_l + 2 * _g] += _alpha[_ig] * _thickness * _Jdet *
                                                    (_pDp[_j + 2 * _h][_l + 2 * _g] + _pDgp[_j + 2 * _h][_g] * _X[_l] +
                                                     _X[_j] * _gpDp[_h][_l + 2 * _g] + _X[_j] * _gpDgp[_h][_g] * _X[_l]);

        // Integrate homogenized stiffness CiiBN = CiiBN + alpha(ig)*Jdet*h*(phi.D:gradphi + X*gradphi:D:gradphi)
        for (_h = 0; _h < nmodes; _h++)
          for (_g = 0; _g < nmodes; _g++)
            for (_j = 0; _j < 2; _j++)
              _CiiBN[_j + 2 * _h][_g] += _alpha[_ig] * _thickness * _Jdet * (_pDgp[_j + 2 * _h][_g] + _X[_j] * _gpDgp[_h][_g]);

        // Integrate homogenized stiffness CiiNN = CiiNN + alpha(ig)*Jdet*h*(gradphi:D:gradphi)
        for (_h = 0; _h < nmodes; _h++)
          for (_g = 0; _g < nmodes; _g++)
            _CiiNN[_h][_g] += _alpha[_ig] * _thickness * _Jdet * _gpDgp[_h][_g];
      }

      // Allocate energy
#pragma omp atomic
      prW[0] += _W;

      // Allocate homogenized stress P1
      for (_i = 0; _i < 4; _i++)
      {
#pragma omp atomic
        prP1[_i] += _P1[_i];
      }

      // Allocate homogenized stress Qi
      for (_i = 0; _i < nmodes; _i++)
      {
#pragma omp atomic
        prQi[_i] += _Qi[_i];
      }

      // Allocate homogenized stress Ri
      for (_i = 0; _i < nmodes; _i++)
      {
        for (_j = 0; _j < 2; _j++)
        {
#pragma omp atomic
          prRi[2 * _i + _j] += _Ri[_j][_i];
        }
      }

      // Allocate homogenized stiffness C00
      for (_i = 0; _i < 4; _i++)
      {
        for (_j = 0; _j < 4; _j++)
        {
#pragma omp atomic
          prC00[_i + 4 * _j] += _C00[_i][_j];
        }
      }

      // Allocate homogenized stiffness C0iBB
      for (_i = 0; _i < 4; _i++)
      {
        for (_j = 0; _j < 2 * nmodes; _j++)
        {
#pragma omp atomic
          prC0iBB[_i + 4 * _j] += _C0iBB[_i][_j];
        }
      }

      // Allocate homogenized stiffness C0iBN
      for (_i = 0; _i < 4; _i++)
      {
        for (_j = 0; _j < nmodes; _j++)
        {
#pragma omp atomic
          prC0iBN[_i + 4 * _j] += _C0iBN[_i][_j];
        }
      }
      // Allocate homogenized stiffness CiiBB
      for (_i = 0; _i < 2 * nmodes; _i++)
      {
        for (_j = 0; _j < 2 * nmodes; _j++)
        {
#pragma omp atomic
          prCiiBB[_i + (2 * nmodes) * _j] += _CiiBB[_i][_j];
        }
      }

      // Allocate homogenized stiffness CiiBN
      for (_i = 0; _i < 2 * nmodes; _i++)
      {
        for (_j = 0; _j < nmodes; _j++)
        {
#pragma omp atomic
          prCiiBN[_i + (2 * nmodes) * _j] += _CiiBN[_i][_j];
        }
      }

      // Allocate homogenized stiffness CiiNN
      for (_i = 0; _i < nmodes; _i++)
      {
        for (_j = 0; _j < nmodes; _j++)
        {
#pragma omp atomic
          prCiiNN[_i + nmodes * _j] += _CiiNN[_i][_j];
        }
      }
    }

    // Delete dynamically allocated memory
    delete[] _Qi;
    delete[] _tQi;
    for (_i = 0; _i < 2; _i++)
    {
      delete[] _Ri[_i];
      delete[] _tRi[_i];
    }
    delete[] _Ri;
    delete[] _tRi;
    for (_i = 0; _i < 4; _i++)
    {
      delete[] _C0iBB[_i];
      delete[] _C0iBN[_i];
      delete[] _Dp[_i];
      delete[] _Dgp[_i];
    }
    delete[] _C0iBB;
    delete[] _C0iBN;
    delete[] _Dp;
    delete[] _Dgp;
    for (_i = 0; _i < 2 * nmodes; _i++)
    {
      delete[] _CiiBB[_i];
      delete[] _CiiBN[_i];
      delete[] _pDp[_i];
      delete[] _pDgp[_i];
    }
    delete[] _CiiBB;
    delete[] _CiiBN;
    delete[] _pDp;
    delete[] _pDgp;
    for (_i = 0; _i < nmodes; _i++)
    {
      delete[] _CiiNN[_i];
      delete[] _gpDp[_i];
      delete[] _gpDgp[_i];
    }
    delete[] _gpDgp;
    delete[] _CiiNN;
    delete[] _gpDp;
    delete[] _phi;
    delete[] _gradphi;
    delete[] _idel;
    delete[] _cod;
    delete[] _uel;
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

  // Catch errors
  if (errCountMat != 0)
    mexErrMsgTxt("Unrecognized material chosen.");
  if (errCountGauss != 0)
    mexErrMsgTxt("Unrecognized Gauss integration rule chosen.");
  if (errCountEl != 0)
    mexErrMsgTxt("Unrecognized element type chosen.");
}
