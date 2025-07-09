// sample_UFP: interpolates U, F and P at query points,
// midpoints of elements, or at all Gauss integration points

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

  // Test for ten input arguments
  if (nrhs != 11)
    mexErrMsgTxt("Eleven input arguments required.");

  // Get the input data
  double *prp = mxGetPr(prhs[0]);
  double *prt = mxGetPr(prhs[1]);
  double *prmat = mxGetPr(prhs[2]);
  int ngauss = (int)mxGetScalar(prhs[3]);
  double *pru = mxGetPr(prhs[4]);
  double *prx = mxGetPr(prhs[5]);
  double *pry = mxGetPr(prhs[6]);
  double *prz = mxGetPr(prhs[7]);
  double TOL_g = mxGetScalar(prhs[8]);
  int SWpoints = (int)mxGetScalar(prhs[9]);
  int maxNumThreads = (int)mxGetScalar(prhs[10]);
  int nnodes = (int)mxGetN(prhs[0]);
  int nelem = (int)mxGetN(prhs[1]);
  int mt = (int)mxGetM(prhs[1]);
  int np = mt - 1; // number of points in an element
  int mmat = (int)mxGetM(prhs[2]);
  int ndof = 3 * nnodes;
  int npoints = std::max((int)mxGetM(prhs[5]), (int)mxGetN(prhs[5]));

  // Test the number of dofs
  if (ndof != (int)mxGetM(prhs[4]))
    mexErrMsgTxt("Wrong number of dofs.");

  // Test the number of points
  nlhs = 7;
  if ((npoints != std::max((int)mxGetM(prhs[6]), (int)mxGetN(prhs[6]))) ||
      (npoints != std::max((int)mxGetM(prhs[7]), (int)mxGetN(prhs[7]))))
    mexErrMsgTxt("Inconsistent X and Y coordinates of query points.");
  if ((0 == SWpoints) &&
      (0 == std::min((int)mxGetM(prhs[6]), (int)mxGetN(prhs[6]))))
  {
    plhs[0] = mxCreateDoubleMatrix(1, 0, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(3, 0, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(9, 0, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(9, 0, mxREAL);
    plhs[4] = mxCreateDoubleMatrix(1, 0, mxREAL);
    plhs[5] = mxCreateDoubleMatrix(1, 0, mxREAL);
    plhs[6] = mxCreateDoubleMatrix(1, 0, mxREAL);
    return;
  }

  // Change the number of Gauss points when data at centres are required
  if (SWpoints == 1)
    ngauss = 1;

  // Allocate outputs
  int i, maxPoints = 0;
  double minxq = 0.0, maxxq = 0.0, minyq = 0.0, maxyq = 0.0, minzq = 0.0, maxzq = 0.0;
  if (SWpoints == 0)
  { // interpolate data at query points prx, pry, prz
    maxPoints = npoints;

    // Get bounding box for query points
    minxq = prx[0];
    maxxq = prx[0];
    minyq = pry[0];
    maxyq = pry[0];
    minzq = prz[0];
    maxzq = prz[0];
    for (i = 1; i < npoints; i++)
    {
      minxq = (minxq < prx[i]) ? minxq : prx[i];
      maxxq = (maxxq > prx[i]) ? maxxq : prx[i];
      minyq = (minyq < pry[i]) ? minyq : pry[i];
      maxyq = (maxyq > pry[i]) ? maxyq : pry[i];
      minzq = (minzq < prz[i]) ? minzq : prz[i];
      maxzq = (maxzq > prz[i]) ? maxzq : prz[i];
    }
    minxq -= TOL_g;
    maxxq += TOL_g;
    minyq -= TOL_g;
    maxyq += TOL_g;
    minzq -= TOL_g;
    maxzq += TOL_g;
  }
  else if (SWpoints == 1 ||
           SWpoints == 2) // interpolate data at zero natural coordinates
                          // (midpoints of elements) or at all Gauss points
    maxPoints = ngauss * nelem;
  else
    mexErrMsgTxt("Wrong type of query points specified.");
  plhs[0] = mxCreateDoubleMatrix(1, maxPoints, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(3 * maxPoints, 1, mxREAL);
  plhs[2] = mxCreateDoubleMatrix(9, maxPoints, mxREAL);
  plhs[3] = mxCreateDoubleMatrix(9, maxPoints, mxREAL);
  plhs[4] = mxCreateDoubleMatrix(1, maxPoints, mxREAL);
  plhs[5] = mxCreateDoubleMatrix(1, maxPoints, mxREAL);
  plhs[6] = mxCreateDoubleMatrix(1, maxPoints, mxREAL);
  double *prW = mxGetPr(plhs[0]);
  double *prU = mxGetPr(plhs[1]);
  double *prF = mxGetPr(plhs[2]);
  double *prP = mxGetPr(plhs[3]);
  double *prX = mxGetPr(plhs[4]);
  double *prY = mxGetPr(plhs[5]);
  double *prZ = mxGetPr(plhs[6]);

  // Sample U, F, and P at selected points
  int tr; // triangle ID
  int *multiplicity = new int[maxPoints];
  for (i = 0; i < maxPoints; i++)
    multiplicity[i] = 0;
  int errCountGauss =
      0;               // integration rule: error control inside parallel region
  int errCountMat = 0; // constitutive law: error control inside parallel region
  int errCountEl = 0;  // constitutive law: error control inside parallel region
#pragma omp parallel num_threads(maxNumThreads)
  {
    int _i, _ig, _point;
    double _Wd, _Jdet, _X, _Y, _Z, _radius2, _tradius2;
    double _U[3], _F[9], _P[9], _D[9][9], _tcoord[4], _Ce[3], _matconst[8]; // use total Lagrangian formulation
    double _minxe, _maxxe, _minye, _maxye, _minze, _maxze;
    int *_idel = new int[np];    // ids of nodes of given element
    int *_cod = new int[3 * np]; // code numbers of given element
    double *_uel =
        new double[3 * np];      // displacements associated with given element
    double *_x = new double[np]; // x-coordinates of all nodes of given element
    double *_y = new double[np]; // y-coordinates of all nodes of given element
    double *_z = new double[np]; // z-coordinates of all nodes of given element
    double **_Be =
        new double *[9]; // element's matrix of derivatives of basis functions
    for (_i = 0; _i < 9; _i++)
      _Be[_i] = new double[3 * np];
    double **_IP = new double *[abs(ngauss)];
    for (_i = 0; _i < abs(ngauss); _i++)
      _IP[_i] = new double[4];
    double *_alpha = new double[abs(ngauss)];
    int (*_getBe)(const int np, const double tcoord[4], const double *x,
                  const double *y, const double *z, double &Jdet,
                  double **Be) = NULL; // pointer to Be functions
    int (*_getWPD)(
        const double matconst[8], const double F[9], double &Wd, double P[9],
        double D[9][9]) = NULL; // pointer to constitutive law functions
    int (*_getShapeFun)(const int np, const double tcoord[4],
                        double *N) =
        NULL; // pointer to shape functions function
    int (*_natural2physical)(
        const int np, const double tcoord[4], const double *x, const double *y,
        const double *z, double &X, double &Y,
        double &Z) = NULL; // pointer to functions converting from natural to
                           // physical coordinates
    int (*_physical2natural)(
        const int np, const double X, const double Y, const double Z,
        const double *x, const double *y, const double *z, const double TOL_g,
        double tcoord[4]) = NULL; // pointer to functions converting from
                                  // physical to natural coordinates
    if (np == 4 || np == 10)
    {

      // Choose Gauss integration rule
      if (ngauss == 1 || ngauss == 4 || ngauss == 5 || ngauss == 11 ||
          ngauss == 15)
        getTetraGaussInfo(ngauss, _alpha,
                          _IP); // tetrahendra: ngauss = 1, 4, 5, 11, 15
      else
      {
#pragma omp atomic
        errCountGauss++; // throw error if unrecognized integration rule used
      }

      // Choose functions
      _getBe = &getTetraBeTLF;
      _getShapeFun = &getTetraShapeFun;
      _natural2physical = &ConvertTetraNat2Phys;
      _physical2natural = &ConvertTetraPhys2Nat;
    }
    else if (np == 8 || np == 20 || np == 27)
    {
      // Choose Gauss integration rule
      if (ngauss == 1 || ngauss == 8 || ngauss == 27 || ngauss == 64)
        getHexaGaussInfo(ngauss, _alpha,
                         _IP); // hexahedra: ngauss = 1, 8, 27, 64
      else
      {
#pragma omp atomic
        errCountGauss++; // throw error if unrecognized integration rule used
      }

      // Choose functions
      _getBe = &getHexaBeTLF;
      _getShapeFun = &getHexaShapeFun;
      _natural2physical = &ConvertHexaNat2Phys;
      _physical2natural = &ConvertHexaPhys2Nat;
    }
    else
    {
#pragma omp atomic
      errCountEl++; // throw error if unrecognized element type used
    }
    double *_N = new double[np]; // shape function evaluations

    // SAMPLE U, F, P AT QUERY POINTS
    if (SWpoints == 0)
    { // interpolate data at query points prx, pry
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

        // Get element's bounding box
        _minxe = _x[0];
        _maxxe = _x[0];
        _minye = _y[0];
        _maxye = _y[0];
        _minze = _z[0];
        _maxze = _z[0];
        for (_i = 1; _i < np; _i++)
        {
          _minxe = (_minxe < _x[_i]) ? _minxe : _x[_i];
          _maxxe = (_maxxe > _x[_i]) ? _maxxe : _x[_i];
          _minye = (_minye < _y[_i]) ? _minye : _y[_i];
          _maxye = (_maxye > _y[_i]) ? _maxye : _y[_i];
          _minze = (_minze < _z[_i]) ? _minze : _z[_i];
          _maxze = (_maxze > _z[_i]) ? _maxze : _z[_i];
        }
        _minxe -= TOL_g;
        _maxxe += TOL_g;
        _minye -= TOL_g;
        _maxye += TOL_g;
        _minze -= TOL_g;
        _maxze += TOL_g;

        // If element is within the query points' bounding box
        if (intersect3d(_minxe, _maxxe - _minxe, _minye, _maxye - _minye, _minze,
                        _maxze - _minze, minxq, maxxq - minxq, minyq,
                        maxyq - minyq, minzq, maxzq - minzq))
        {

          // Get material constants
          for (_i = 0; _i < 8; _i++)
            _matconst[_i] = prmat[int(prt[np + mt * tr] - 1) + _i * mmat];

          // Choose constitutive law
          switch ((int)_matconst[0])
          {
          case 1: // W(F) = m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+1/2*kappa(ln(J))^2
            // (OOFEM)
            _getWPD = &material_oofemF3d;
            break;
          case 2: // W(F) = m1(I1-3)+m2(I1-3)^2-2m1*ln(J)+1/2*kappa(J-1)^2
            // (Bertoldi, Boyce)
            _getWPD = &material_bbF3d;
            break;
          case 3: // W(F) =
            // m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I1/J^{2/3}-3)^3+9/2*kappa(J^{1/3}-1)^2
            // (Jamus, Green, Simpson)
            _getWPD = &material_jgsF3d;
            break;
          case 4: // W(F) =
            // m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I2/J^{2/3}-3)^2+kappa*(J-1)^2
            // (five-term Mooney-Rivlin)
            _getWPD = &material_5mrF3d;
            break;
          case 5: // W(F) = 0.5*(0.5*(C-I)*(m1*IxI+2*kappa*I)*0.5*(C-I)) (linear
            // elastic material)
            _getWPD = &material_leF3d;
            break;
          default: // if unrecognized material used, use OOFEM in order not to crash, but throw an error
            _getWPD = &material_oofemF3d;
#pragma omp atomic
            errCountMat++;
            break;
          }

          // Estimate element's center and radius
          _Ce[0] = 0;
          _Ce[1] = 0;
          _Ce[2] = 0;
          for (_i = 0; _i < np; _i++)
          {
            _Ce[0] += _x[_i] / np;
            _Ce[1] += _y[_i] / np;
            _Ce[2] += _z[_i] / np;
          }
          _radius2 = 0;
          for (_i = 0; _i < np; _i++)
          {
            _tradius2 = pow(_Ce[0] - _x[_i], 2) + pow(_Ce[1] - _y[_i], 2) +
                        pow(_Ce[2] - _z[_i], 2);
            if (_tradius2 > _radius2)
              _radius2 = _tradius2;
          }

          // Loop over all points
          for (_point = 0; _point < npoints; _point++)
          {

            // Get coordinates of query points
            _X = prx[_point];
            _Y = pry[_point];
            _Z = prz[_point];

            // If pixel is inside bounding circle
            if (pow(_Ce[0] - _X, 2) + pow(_Ce[1] - _Y, 2) + pow(_Ce[2] - _Z, 2) <
                1.25 * _radius2 + TOL_g)
            {

              // Get shape function information evaluated at given query point
              _physical2natural(np, _X, _Y, _Z, _x, _y, _z, TOL_g, // inputs
                                _tcoord);                          // outputs

              // Check if query point is inside given triangle
              if (_tcoord[0] == _tcoord[0])
              { // otherwise it is NAN

                // Get shape functions
                _getShapeFun(np, _tcoord, _N);

                // Get Be matrix
                _getBe(np, _tcoord, _x, _y, _z, _Jdet, _Be);

                // Get deformation gradient, its determinant
                getF3d(_Be, _uel, np, _F);

                // Get energy density, stress, and material stiffness evaluated at
                // current Gauss point
                _getWPD(_matconst, _F, // inputs
                        _Wd, _P, _D);  // outputs

                // Take physical coordinates associated with tcoord, just in case
                _natural2physical(np, _tcoord, _x, _y, _z, // inputs
                                  _X, _Y, _Z);             // outputs

                // Get displacements
                _U[0] = 0.0;
                _U[1] = 0.0;
                _U[2] = 0.0;
                for (_i = 0; _i < np; _i++)
                {
                  _U[0] += _N[_i] * _uel[3 * _i + 0];
                  _U[1] += _N[_i] * _uel[3 * _i + 1];
                  _U[2] += _N[_i] * _uel[3 * _i + 2];
                }

                // Allocate U, F, P, X, Y directly to outputs to guarantee proper
                // ordering of all fields
#pragma omp atomic
                prW[_point] += _Wd;
#pragma omp atomic
                prX[_point] += _X;
#pragma omp atomic
                prY[_point] += _Y;
#pragma omp atomic
                prZ[_point] += _Z;
#pragma omp atomic
                prU[_point * 3 + 0] += _U[0];
#pragma omp atomic
                prU[_point * 3 + 1] += _U[1];
#pragma omp atomic
                prU[_point * 3 + 2] += _U[2];
                for (_i = 0; _i < 9; _i++)
                {
#pragma omp atomic
                  prF[_point * 9 + _i] += _F[_i];
#pragma omp atomic
                  prP[_point * 9 + _i] += _P[_i];
                }

                // Increase multiplicity
#pragma omp atomic
                multiplicity[_point]++;
              }
            }
          }
        }
      }

// Divide by multiplicity
#pragma omp barrier
#pragma omp single
      {
        for (_point = 0; _point < npoints; _point++)
        {
          if (multiplicity[_point] ==
              0)
          { // assign NaN because given point is not located within any
            // of the elements
            prX[_point] += NAN;
            prY[_point] += NAN;
            prZ[_point] += NAN;
            prU[3 * _point] += NAN;
            prU[3 * _point + 1] += NAN;
            prU[3 * _point + 2] += NAN;
            for (_i = 0; _i < 9; _i++)
            {
              prF[9 * _point + _i] += NAN;
              prP[9 * _point + _i] += NAN;
            }
          }
          else if (multiplicity[_point] >
                   1)
          { // divide by multiplicity if point is located within
            // more than one element
            prX[_point] /= double(multiplicity[_point]);
            prY[_point] /= double(multiplicity[_point]);
            prZ[_point] /= double(multiplicity[_point]);
            prU[3 * _point] /= double(multiplicity[_point]);
            prU[3 * _point + 1] /= double(multiplicity[_point]);
            prU[3 * _point + 2] /= double(multiplicity[_point]);
            for (_i = 0; _i < 3; _i++)
            {
              prF[9 * _point + _i] /= double(multiplicity[_point]);
              prP[9 * _point + _i] /= double(multiplicity[_point]);
            }
          }
        }
      }
    }

    // SAMPLE U, F, P AT GAUSS INTEGRATION POINTS
    else if (SWpoints == 1 || SWpoints == 2)
    {
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
        for (_i = 0; _i < 7; _i++)
          _matconst[_i] = prmat[int(prt[np + mt * tr] - 1) + _i * mmat];

        // Choose constitutive law
        switch ((int)_matconst[0])
        {
        case 1: // W(F) = m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+1/2*kappa(ln(J))^2
          // (OOFEM)
          _getWPD = &material_oofemF3d;
          break;
        case 2: // W(F) = m1(I1-3)+m2(I1-3)^2-2m1*ln(J)+1/2*kappa(J-1)^2
          // (Bertoldi, Boyce)
          _getWPD = &material_bbF3d;
          break;
        case 3: // W(F) =
          // m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I1/J^{2/3}-3)^3+9/2*kappa(J^{1/3}-1)^2
          // (Jamus, Green, Simpson)
          _getWPD = &material_jgsF3d;
          break;
        case 4: // W(F) =
          // m1(I1/J^{2/3}-3)+m2(I2/J^{4/3}-3)+m3(I1/J^{2/3}-3)*(I2/J^{4/3}-3)+m4(I1/J^{2/3}-3)^2+m5(I2/J^{2/3}-3)^2+kappa*(J-1)^2
          // (five-term Mooney-Rivlin)
          _getWPD = &material_5mrF3d;
          break;
        case 5: // W(F) = 0.5*(0.5*(C-I)*(m1*IxI+2*kappa*I)*0.5*(C-I)) (linear
          // elastic material)
          _getWPD = &material_leF3d;
          break;
        default: // throw error if unrecognized material used
#pragma omp atomic
          errCountMat++;
          break;
        }

        // Loop over all Gauss integratio npoints
        for (_ig = 0; _ig < abs(ngauss); _ig++)
        {

          // Get natural coordinates of Gauss integration rule
          for (_i = 0; _i < 4; _i++)
            _tcoord[_i] = _IP[_ig][_i];

          // Get shape functions
          _getShapeFun(np, _tcoord, _N);

          // Get Be matrix
          _getBe(np, _tcoord, _x, _y, _z, _Jdet, _Be);

          // Get deformation gradient, its determinant
          getF3d(_Be, _uel, np, _F);

          // Get energy density, stress, and material stiffness evaluated at
          // current Gauss point
          _getWPD(_matconst, _F, // inputs
                  _Wd, _P, _D);  // outputs

          // Take physical coordinates associated with tcoord, just in case
          _natural2physical(np, _tcoord, _x, _y, _z, // inputs
                            _X, _Y, _Z);             // outputs

          // Get displacements
          _U[0] = 0.0;
          _U[1] = 0.0;
          _U[2] = 0.0;
          for (_i = 0; _i < np; _i++)
          {
            _U[0] += _N[_i] * _uel[3 * _i + 0];
            _U[1] += _N[_i] * _uel[3 * _i + 1];
            _U[2] += _N[_i] * _uel[3 * _i + 2];
          }

          // Allocate U, F, P, X, Y directly to outputs to guarantee proper
          // ordering of all fields
#pragma omp atomic
          prW[tr * abs(ngauss) + _ig] += _Wd;
#pragma omp atomic
          prX[tr * abs(ngauss) + _ig] += _X;
#pragma omp atomic
          prY[tr * abs(ngauss) + _ig] += _Y;
#pragma omp atomic
          prZ[tr * abs(ngauss) + _ig] += _Z;
#pragma omp atomic
          prU[tr * abs(ngauss) * 3 + 3 * _ig + 0] += _U[0];
#pragma omp atomic
          prU[tr * abs(ngauss) * 3 + 3 * _ig + 1] += _U[1];
#pragma omp atomic
          prU[tr * abs(ngauss) * 3 + 3 * _ig + 2] += _U[2];
          for (_i = 0; _i < 9; _i++)
          {
#pragma omp atomic
            prF[tr * abs(ngauss) * 9 + 9 * _ig + _i] += _F[_i];
#pragma omp atomic
            prP[tr * abs(ngauss) * 9 + 9 * _ig + _i] += _P[_i];
          }
        }
      }
    }

    // Delete dynamically allocated memory
    delete[] _idel;
    delete[] _cod;
    delete[] _uel;
    delete[] _x;
    delete[] _y;
    for (_i = 0; _i < 9; _i++)
      delete[] _Be[_i];
    delete[] _Be;
    for (_i = 0; _i < abs(ngauss); _i++)
      delete[] _IP[_i];
    delete[] _IP;
    delete[] _alpha;
    delete[] _N;
  }

  // Catch errors
  if (errCountMat != 0)
    mexErrMsgTxt("Unrecognized material chosen.");
  if (errCountGauss != 0)
    mexErrMsgTxt("Unrecognized Gauss integration rule chosen.");
  if (errCountEl != 0)
    mexErrMsgTxt("Unrecognized element type chosen.");

  // Delete dynamically allocated memory
  delete[] multiplicity;
}
