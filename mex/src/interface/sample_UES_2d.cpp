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
	if (nrhs != 10)
		mexErrMsgTxt("Ten input arguments required.");

	// Get the input data
	double *prp = mxGetPr(prhs[0]);
	double *prt = mxGetPr(prhs[1]);
	double *prmat = mxGetPr(prhs[2]);
	int ngauss = (int)mxGetScalar(prhs[3]);
	double *pru = mxGetPr(prhs[4]);
	double *prx = mxGetPr(prhs[5]);
	double *pry = mxGetPr(prhs[6]);
	double TOL_g = mxGetScalar(prhs[7]);
	int SWpoints = (int)mxGetScalar(prhs[8]);
	int maxNumThreads = (int)mxGetScalar(prhs[9]);
	int nnodes = (int)mxGetN(prhs[0]);
	int nelem = (int)mxGetN(prhs[1]);
	int mt = (int)mxGetM(prhs[1]);
	int np = mt - 1; // number of points in an element
	int mmat = (int)mxGetM(prhs[2]);
	int ndof = 2 * nnodes;
	int npoints = std::max((int)mxGetM(prhs[5]), (int)mxGetN(prhs[5]));

	// Test the number of dofs
	if (ndof != (int)mxGetM(prhs[4]))
        //find some way to print ndof and mxGetM(prhs[4]))
        //function to_string is not declared in current scope
		mexErrMsgTxt("Wrong number of dofs.");

	// Test the number of points
	nlhs = 6;
	if (npoints != std::max((int)mxGetM(prhs[6]), (int)mxGetN(prhs[6])))
		mexErrMsgTxt("Inconsistent X and Y coordinates of query points.");
	if ((0 == SWpoints) &&
		(0 == std::min((int)mxGetM(prhs[6]), (int)mxGetN(prhs[6]))))
	{
		plhs[0] = mxCreateDoubleMatrix(1, 0, mxREAL);
		plhs[1] = mxCreateDoubleMatrix(2, 0, mxREAL);
		plhs[2] = mxCreateDoubleMatrix(3, 0, mxREAL);
		plhs[3] = mxCreateDoubleMatrix(3, 0, mxREAL);
		plhs[4] = mxCreateDoubleMatrix(1, 0, mxREAL);
		plhs[5] = mxCreateDoubleMatrix(1, 0, mxREAL);
		return;
	}

	// Change the number of Gauss points when data at centres are required
	if (SWpoints == 1)
		ngauss = 1;

	// Allocate outputs
	int i, maxPoints = 0;
	double minxq = 0.0, maxxq = 0.0, minyq = 0.0, maxyq = 0.0;
	if (SWpoints == 0)
	{ // interpolate data at query points prx, pry
		maxPoints = npoints;

		// Get bounding box for query points
		minxq = prx[0];
		maxxq = prx[0];
		minyq = pry[0];
		maxyq = pry[0];
		for (i = 1; i < npoints; i++)
		{
			minxq = (minxq < prx[i]) ? minxq : prx[i];
			maxxq = (maxxq > prx[i]) ? maxxq : prx[i];
			minyq = (minyq < pry[i]) ? minyq : pry[i];
			maxyq = (maxyq > pry[i]) ? maxyq : pry[i];
		}
		minxq -= TOL_g;
		maxxq += TOL_g;
		minyq -= TOL_g;
		maxyq += TOL_g;
	}
	else if (SWpoints == 1 ||
			 SWpoints == 2) // interpolate data at zero natural coordinates
							// (midpoints of elements) or at all Gauss points
		maxPoints = ngauss * nelem;
	else
		mexErrMsgTxt("Wrong type of query points specified.");
	plhs[0] = mxCreateDoubleMatrix(maxPoints, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(2 * maxPoints, 1, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(3, maxPoints, mxREAL);
	plhs[3] = mxCreateDoubleMatrix(3, maxPoints, mxREAL);
	plhs[4] = mxCreateDoubleMatrix(1, maxPoints, mxREAL);
	plhs[5] = mxCreateDoubleMatrix(1, maxPoints, mxREAL);
	double *prW = mxGetPr(plhs[0]);
	double *prU = mxGetPr(plhs[1]);
	double *prE = mxGetPr(plhs[2]);
	double *prS = mxGetPr(plhs[3]);
	double *prX = mxGetPr(plhs[4]);
	double *prY = mxGetPr(plhs[5]);

	// Sample U, F, and P at selected points
	int tr; // triangle ID
	int *multiplicity = new int[maxPoints];
	for (i = 0; i < maxPoints; i++)
		multiplicity[i] = 0;
	int errCountGauss =
		0;				 // integration rule: error control inside parallel region
	int errCountMat = 0; // constitutive law: error control inside parallel region
	int errCountEl = 0;  // constitutive law: error control inside parallel region
#pragma omp parallel num_threads(maxNumThreads)
	{
		int _i, _ig, _point;
		double _w, _Jdet, _X, _Y, _radius2, _tradius2, _matconst[8];
		double _U[2], _F[4], _C[3], _E[3], _S[3], _CD[3][3], _tcoord[3], _Ce[2]; // use total Lagrangian formulation
		double _minxe, _maxxe, _minye, _maxye;
		int *_idel = new int[np];	// ids of nodes of given element
		int *_cod = new int[2 * np]; // code numbers of given element
		double *_uel =
			new double[2 * np];			// displacements associated with given element
		double *_x = new double[np];	// x-coordinates of all nodes of given element
		double *_y = new double[np];	// y-coordinates of all nodes of given element
		double **_Be = new double *[3]; // element's matrix of derivatives of basis
										// functions (Green-Lagrange strain)
		for (_i = 0; _i < 3; _i++)
			_Be[_i] = new double[2 * np];
		double **_Bf = new double *[4]; // element's matrix of derivatives of basis
										// functions (deformation gradient)
		for (_i = 0; _i < 4; _i++)
			_Bf[_i] = new double[2 * np];
		double **_IP = new double *[abs(ngauss)];
		for (_i = 0; _i < abs(ngauss); _i++)
			_IP[_i] = new double[3];
		double *_alpha = new double[abs(ngauss)];
		int (*_getBe)(const int np, const double tcoord[3], const double *x,
					  const double *y, const double *uel, double &Jdet, double **Be,
					  double **Bf, double F[4]) = NULL; // pointer to Be functions
		int (*_getWSC)(
			const double matconst[8], const double C[3], double &w, double S[3],
			double D[3][3]) = NULL; // pointer to constitutive law functions
		int (*_getShapeFun)(const int np, const double tcoord[3], double *N) =
			NULL; // pointer to shape functions function
		int (*_natural2physical)(
			const int np, const double tcoord[3], const double *x, const double *y,
			double &X, double &Y) = NULL; // pointer to functions converting from
										  // natural to physical coordinates
		int (*_physical2natural)(const int np, const double X, const double Y,
								 const double *x, const double *y,
								 const double TOL_g, double tcoord[3]) =
			NULL; // pointer to functions converting from physical to natural
			// coordinates
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
			_getBe = &getTriagBeTLE;
			_getShapeFun = &getTriagShapeFun;
			_natural2physical = &ConvertTriagNat2Phys;
			_physical2natural = &ConvertTriagPhys2Nat;
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
			_getBe = &getQuadBeTLE;
			_getShapeFun = &getQuadShapeFun;
			_natural2physical = &ConvertQuadNat2Phys;
			_physical2natural = &ConvertQuadPhys2Nat;
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
					_cod[2 * _i] = 2 * _idel[_i] - 1;
					_cod[2 * _i + 1] = 2 * _idel[_i];
				}

				// Get element's displacements
				for (_i = 0; _i < 2 * np; _i++)
					_uel[_i] = pru[_cod[_i] - 1];

				// Get element's coordinates
				for (_i = 0; _i < np; _i++)
				{
					_x[_i] = prp[0 + (_idel[_i] - 1) * 2];
					_y[_i] = prp[1 + (_idel[_i] - 1) * 2];
				}

				// Get element's bounding box
				_minxe = _x[0];
				_maxxe = _x[0];
				_minye = _y[0];
				_maxye = _y[0];
				for (_i = 1; _i < np; _i++)
				{
					_minxe = (_minxe < _x[_i]) ? _minxe : _x[_i];
					_maxxe = (_maxxe > _x[_i]) ? _maxxe : _x[_i];
					_minye = (_minye < _y[_i]) ? _minye : _y[_i];
					_maxye = (_maxye > _y[_i]) ? _maxye : _y[_i];
				}
				_minxe -= TOL_g;
				_maxxe += TOL_g;
				_minye -= TOL_g;
				_maxye += TOL_g;

				// If element is within the query points' bounding box
				if (intersect2d(_minxe, _maxxe - _minxe, _minye, _maxye - _minye,
								minxq, maxxq - minxq, minyq, maxyq - minyq))
				{

					// Get material constants
					for (_i = 0; _i < 8; _i++)
						_matconst[_i] = prmat[int(prt[np + mt * tr] - 1) + _i * mmat];

					// Choose constitutive law
					switch ((int)_matconst[0])
					{
					case 1: // OOFEM
						_getWSC = &material_oofemE2d;
						break;
					case 2: // Bertoldi, Boyce
						_getWSC = &material_bbE2d;
						break;
					case 3: // Jamus, Green, Simpson
						_getWSC = &material_jgsE2d;
						break;
					case 4: // five-term Mooney-Rivlin
						_getWSC = &material_5mrE2d;
						break;
					case 5: // elastic material
						_getWSC = &material_leE2d;
						break;
					case 6: // Ogden material
						_getWSC = &material_ogdenE2d;
						break;
					case 7: // Ogden nematic material
						_getWSC = &material_ogdennematicE2d;
						break;
					default: // if unrecognized material used, use OOFEM in order not to
							 // crash, but throw an error
						_getWSC = &material_oofemE2d;
#pragma omp atomic
						errCountMat++;
						break;
					}

					// Estimate element's center and radius
					_Ce[0] = 0;
					_Ce[1] = 0;
					for (_i = 0; _i < np; _i++)
					{
						_Ce[0] += _x[_i] / np;
						_Ce[1] += _y[_i] / np;
					}
					_radius2 = 0;
					for (_i = 0; _i < np; _i++)
					{
						_tradius2 = pow(_Ce[0] - _x[_i], 2) + pow(_Ce[1] - _y[_i], 2);
						if (_tradius2 > _radius2)
							_radius2 = _tradius2;
					}

					// Loop over all points
					for (_point = 0; _point < npoints; _point++)
					{

						// Get coordinates of query points
						_X = prx[_point];
						_Y = pry[_point];

						// If pixel is inside bounding circle
						if (pow(_Ce[0] - _X, 2) + pow(_Ce[1] - _Y, 2) <
							1.25 * _radius2 + TOL_g)
						{

							// Get shape function information evaluated at given query point
							_physical2natural(np, _X, _Y, _x, _y, TOL_g, _tcoord);

							if (_tcoord[0] == _tcoord[0])
							{ // otherwise it is NAN

								_getShapeFun(np, _tcoord, _N);
								_getBe(np, _tcoord, _x, _y, _uel, _Jdet, _Be, _Bf, _F);
								getC2d(_F, _C);
								getE2d(_F, _E);
								_getWSC(_matconst, _C, _w, _S, _CD);
								_natural2physical(np, _tcoord, _x, _y, _X, _Y);

								// Get displacements
								_U[0] = 0.0;
								_U[1] = 0.0;
								for (_i = 0; _i < np; _i++)
								{
									_U[0] += _N[_i] * _uel[2 * _i];
									_U[1] += _N[_i] * _uel[2 * _i + 1];
								}

								// Allocate U, F, P, X, Y directly to outputs to guarantee proper
								// ordering of all fields
#pragma omp atomic
								prX[_point] += _X;
#pragma omp atomic
								prY[_point] += _Y;
#pragma omp atomic
								prW[_point] += _w;
#pragma omp atomic
								prU[2 * _point] += _U[0];
#pragma omp atomic
								prU[2 * _point + 1] += _U[1];
								for (_i = 0; _i < 3; _i++)
								{
#pragma omp atomic
									prE[3 * _point + _i] += _E[_i];
#pragma omp atomic
									prS[3 * _point + _i] += _S[_i];
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
						prW[_point] += NAN;
						prU[2 * _point] += NAN;
						prU[2 * _point + 1] += NAN;
						for (_i = 0; _i < 3; _i++)
						{
							prE[3 * _point + _i] += NAN;
							prS[3 * _point + _i] += NAN;
						}
					}
					else if (multiplicity[_point] >
							 1)
					{ // divide by multiplicity if point is located within
						// more than one element
						prX[_point] /= double(multiplicity[_point]);
						prY[_point] /= double(multiplicity[_point]);
						prW[_point] /= double(multiplicity[_point]);
						prU[2 * _point] /= double(multiplicity[_point]);
						prU[2 * _point + 1] /= double(multiplicity[_point]);
						for (_i = 0; _i < 3; _i++)
						{
							prE[3 * _point + _i] /= double(multiplicity[_point]);
							prS[3 * _point + _i] /= double(multiplicity[_point]);
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
					_cod[2 * _i] = 2 * _idel[_i] - 1;
					_cod[2 * _i + 1] = 2 * _idel[_i];
				}

				// Get element's displacements
				for (_i = 0; _i < 2 * np; _i++)
					_uel[_i] = pru[_cod[_i] - 1];

				// Get element's coordinates
				for (_i = 0; _i < np; _i++)
				{
					_x[_i] = prp[0 + (_idel[_i] - 1) * 2];
					_y[_i] = prp[1 + (_idel[_i] - 1) * 2];
				}

				// Get material constants
				for (_i = 0; _i < 8; _i++)
					_matconst[_i] = prmat[int(prt[np + mt * tr] - 1) + _i * mmat];

				// Choose constitutive law
				switch ((int)_matconst[0])
				{
				case 1: // OOFEM
					_getWSC = &material_oofemE2d;
					break;
				case 2: // Bertoldi, Boyce
					_getWSC = &material_bbE2d;
					break;
				case 3: // Jamus, Green, Simpson
					_getWSC = &material_jgsE2d;
					break;
				case 4: // five-term Mooney-Rivlin
					_getWSC = &material_5mrE2d;
					break;
				case 5: // elastic material
					_getWSC = &material_leE2d;
					break;
				case 6: // Ogden material
					_getWSC = &material_ogdenE2d;
					break;
				case 7: // Ogden material
					_getWSC = &material_ogdennematicE2d;
					break;
				default: // if unrecognized material used, use OOFEM in order not to
						 // crash, but throw an error
					_getWSC = &material_oofemE2d;
#pragma omp atomic
					errCountMat++;
					break;
				}

				// Loop over all Gauss integratio npoints
				for (_ig = 0; _ig < abs(ngauss); _ig++)
				{

					// Get shape function information evaluated at given Gauss point:
					// jacobian _Jdet, matrix of basis functions' derivatives _Be
					for (_i = 0; _i < 3; _i++)
						_tcoord[_i] =
							_IP[_ig]
							   [_i]; // get natural coordinates of Gauss integration rule

					_getShapeFun(np, _tcoord, _N);
					_getBe(np, _tcoord, _x, _y, _uel, _Jdet, _Be, _Bf, _F);
					getC2d(_F, _C);
					getE2d(_F, _E);
					_getWSC(_matconst, _C, _w, _S, _CD);
					_natural2physical(np, _tcoord, _x, _y, _X, _Y);

					// Get displacements
					_U[0] = 0.0;
					_U[1] = 0.0;
					for (_i = 0; _i < np; _i++)
					{
						_U[0] += _N[_i] * _uel[2 * _i];
						_U[1] += _N[_i] * _uel[2 * _i + 1];
					}

					// Allocate U, F, P, X, Y directly to outputs to guarantee proper
					// ordering of all fields
#pragma omp atomic
					prX[tr * abs(ngauss) + _ig] += _X;
#pragma omp atomic
					prY[tr * abs(ngauss) + _ig] += _Y;
#pragma omp atomic
					prW[tr * abs(ngauss) + _ig] += _w;
#pragma omp atomic
					prU[tr * abs(ngauss) * 2 + 2 * _ig] += _U[0];
#pragma omp atomic
					prU[tr * abs(ngauss) * 2 + 2 * _ig + 1] += _U[1];
					for (_i = 0; _i < 3; _i++)
					{
#pragma omp atomic
						prE[tr * abs(ngauss) * 3 + 3 * _ig + _i] += _E[_i];
#pragma omp atomic
						prS[tr * abs(ngauss) * 3 + 3 * _ig + _i] += _S[_i];
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
		for (_i = 0; _i < 3; _i++)
			delete[] _Be[_i];
		delete[] _Be;
		for (_i = 0; _i < 4; _i++)
			delete[] _Bf[_i];
		delete[] _Bf;
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