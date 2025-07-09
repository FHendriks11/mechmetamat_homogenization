// build_phi_higher: builds I, J, and S arrays for the COO
// representation of the Phi matrix for higher-order triangular elements.

#include <algorithm>
#include <math.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "matrix.h"
#include "mex.h"
#include "myfem.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#define EIGEN_DONT_PARALLELIZE

typedef Eigen::Triplet<double> T;

/************************ Main program ************************/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

  // Test for six input arguments
  if (nrhs != 7)
    mexErrMsgTxt("Seven input arguments required.");

  // Get the input data
  double *prp = mxGetPr(prhs[0]);
  double *prt = mxGetPr(prhs[1]);
  double *prx = mxGetPr(prhs[2]);
  double *pry = mxGetPr(prhs[3]);
  double *prz = mxGetPr(prhs[4]);
  double TOL_g = mxGetScalar(prhs[5]);
  int maxNumThreads = mxGetScalar(prhs[6]);
  int nelem = (int)mxGetN(prhs[1]);
  int mt = (int)mxGetM(prhs[1]);
  int np = mt - 1; // number of points in an element
  int npoints = std::max((int)mxGetM(prhs[2]), (int)mxGetN(prhs[2]));
  int ndim = (int)mxGetM(prhs[0]);
  int nnode = (int)mxGetN(prhs[0]);

  // Test dimensions
  if (ndim != 3)
    mexErrMsgTxt("p matrix is of wrong dimensionality (data for a 3d mesh expected).");

  // Test the number of points
  if ((npoints != std::max((int)mxGetM(prhs[3]), (int)mxGetN(prhs[3]))) ||
      (npoints != std::max((int)mxGetM(prhs[4]), (int)mxGetN(prhs[4]))))
    mexErrMsgTxt("Inconsistent X, Y, and Z coordinates of query points.");

  // Get bounding box for query points
  double minxq = prx[0];
  double maxxq = prx[0];
  double minyq = pry[0];
  double maxyq = pry[0];
  double minzq = prz[0];
  double maxzq = prz[0];
  for (int i = 1; i < npoints; i++)
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

  // Sample U, F, and P at selected points
  Eigen::SparseMatrix<double> Phi(npoints, nnode);
  int tr;             // triangle ID
  int errCountEl = 0; // constitutive law: error control inside parallel region
#pragma omp parallel num_threads(maxNumThreads)
  {
    int _i, _point;
    double _X, _Y, _Z, _radius2, _tradius2, _tcoord[4], _Ce[3];
    double _minxe, _maxxe, _minye, _maxye, _minze, _maxze;
    Eigen::SparseMatrix<double> _Phi(npoints, nnode);
    auto _tripletList = std::vector<T>{};
    _tripletList.reserve(30 * npoints);
    int *_idel = new int[np];    // ids of nodes of given element
    double *_x = new double[np]; // x-coordinates of all nodes of given element
    double *_y = new double[np]; // y-coordinates of all nodes of given element
    double *_z = new double[np]; // z-coordinates of all nodes of given element
    double *_N = new double[np]; // shape function evaluations
    int _counter = 0;
    // Choose functions
    int (*_physical2natural)(
        const int np, const double X, const double Y, const double Z,
        const double *x, const double *y, const double *z, const double TOL_g,
        double tcoord[4]) = NULL; // pointer to functions converting from
                                  // physical to natural coordinates
    int (*_getShapeFun)(const int np, const double tcoord[4], double *N) =
        NULL; // pointer to shape functions function
    if (np == 4 || np == 10)
    {
      _physical2natural = &ConvertTetraPhys2Nat;
      _getShapeFun = &getTetraShapeFun;
    }
    else if (np == 8 || np == 20 || np == 27)
    {
      _physical2natural = &ConvertHexaPhys2Nat;
      _getShapeFun = &getHexaShapeFun;
    }
    else
    {
#pragma omp atomic
      errCountEl++; // throw error if unrecognized element type used
    }

#pragma omp for
    for (tr = 0; tr < nelem; tr++)
    { // loop over all elements

      // Get code numbers of i-th element
      for (_i = 0; _i < np; _i++)
        _idel[_i] = (int)prt[_i + mt * tr];

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

          // If pixel is inside bounding sphere
          if (pow(_Ce[0] - _X, 2) + pow(_Ce[1] - _Y, 2) + pow(_Ce[2] - _Z, 2) <
              1.25 * _radius2 + TOL_g)
          {

            // Get shape function information evaluated at given query point
            _physical2natural(np, _X, _Y, _Z, _x, _y, _z, TOL_g, // inputs
                              _tcoord);                          // outputs

            // Check if query point is inside given triangle
            if (_tcoord[0] == _tcoord[0])
            { // otherwise it is NAN

              // Get shape function evaluations
              _getShapeFun(np, _tcoord, _N);

              // Allocate to Phi matrix
              for (_i = 0; _i < np; _i++)
                _tripletList.push_back(T(_point, _idel[_i] - 1, _N[_i]));
            }
          }
        }
      }
    }

    // Assemble _K from triplets
    _Phi.setFromTriplets(_tripletList.begin(), _tripletList.end(), [](const double &, const double &b) { return b; });

    // Collect _Phi from all threads
#pragma omp critical(ALLOCATE)
    {
      Phi += _Phi;
    }

    // Delete dynamically allocated memory
    delete[] _idel;
    delete[] _x;
    delete[] _y;
    delete[] _z;
    delete[] _N;
  }

  // Catch errors
  if (errCountEl != 0)
    mexErrMsgTxt("Unrecognized element type chosen.");

  // Send out data back to matlab
  nlhs = 1;
  plhs[0] = mxCreateSparse(Phi.rows(), Phi.cols(), Phi.nonZeros(), mxREAL);
  int *ir = Phi.innerIndexPtr();
  int *jc = Phi.outerIndexPtr();
  double *pr = Phi.valuePtr();
  mwIndex *ir2 = mxGetIr(plhs[0]);
  mwIndex *jc2 = mxGetJc(plhs[0]);
  double *pr2 = mxGetPr(plhs[0]);
  for (auto i = 0; i < Phi.nonZeros(); ++i)
  {
    pr2[i] = pr[i];
    ir2[i] = ir[i];
  }
  for (auto i = 0; i < Phi.cols() + 1; ++i)
  {
    jc2[i] = jc[i];
  }
}
