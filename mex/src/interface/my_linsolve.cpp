/* newmark_nonproportional_mex: integrate system of ODEs using Newmark's method */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <type_traits>

#if (defined _OPENMP) && (!defined EIGEN_DONT_PARALLELIZE)
#define EIGEN_HAS_OPENMP
#endif

#ifdef EIGEN_HAS_OPENMP
#include <omp.h>
#endif

#include "mex.h"
#include "matrix.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace Eigen;
typedef SparseMatrix<double, ColMajor, std::make_signed<mwIndex>::type> MatlabSparse;

Map<MatlabSparse> matlab2eigen(const mxArray *A)
{
	mxAssert(mxGetClassID(A) == mxDOUBLE_CLASS, "Type of the input matrix needs to be double.");
	MatlabSparse::StorageIndex *ir = reinterpret_cast<MatlabSparse::StorageIndex *>(mxGetIr(A));
	MatlabSparse::StorageIndex *jc = reinterpret_cast<MatlabSparse::StorageIndex *>(mxGetJc(A));
	Map<MatlabSparse> eigA(mxGetM(A), mxGetN(A), mxGetNzmax(A), jc, ir, mxGetPr(A));
	return eigA;
}

/************************ Main program ************************/
// [dx,relres,iter,resvec] = linsolve(H,-G,TOL_r,maxiter,SW);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	// Test for seven input arguments
	if (nrhs != 7)
	{
		mexErrMsgTxt("Seven input arguments required.");
	}

	// Init OpenMP in Eigen
	int maxNumThreads = (int)mxGetScalar(prhs[6]);
	initParallel();
	setNbThreads(maxNumThreads);

	// Get input data
	int ndof = (int)mxGetM(prhs[0]);
	// 	Map<MatlabSparse> A = matlab2eigen(prhs[0]);
	SparseMatrix<double> A = SparseMatrix<double>(matlab2eigen(prhs[0]));
	Map<VectorXd> b(mxGetPr(prhs[1]), ndof);
	double TOL_r = mxGetScalar(prhs[2]);
	int FillFactor = (int)mxGetScalar(prhs[3]);
	int maxiter = (int)mxGetScalar(prhs[4]);
	int SW = (int)mxGetScalar(prhs[5]);

	// Allocate outputs
	nlhs = 4;
	plhs[0] = mxCreateDoubleMatrix(ndof, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
	plhs[3] = mxCreateDoubleMatrix(1, 1, mxREAL);
	Map<VectorXd> dx(mxGetPr(plhs[0]), ndof);
	double *prflag = mxGetPr(plhs[1]);
	double *prrelres = mxGetPr(plhs[2]);
	double *priter = mxGetPr(plhs[3]);

	// Solve the system
	switch (SW)
	{
	case 0: // perform Cholesky decomposition LLT
	{
		SimplicialLLT<SparseMatrix<double>> llt(A);
		dx = llt.solve(b);
		prflag[0] = 0;
		if (llt.info() != Success)
			prflag[0] = -1;
		break;
	}
	case 1: // perform Cholesky decomposition LDLT
	{
		SimplicialLDLT<SparseMatrix<double>> ldlt(A);
		dx = ldlt.solve(b);
		prflag[0] = 0;
		if (ldlt.info() != Success)
			prflag[0] = -1;
		break;
	}
	case 2: // perform sparse LU decomposition
	{
		SparseLU<SparseMatrix<double>> lu(A);
		dx = lu.solve(b);
		prflag[0] = 0;
		if (lu.info() != Success)
			prflag[0] = -1;
		break;
	}
	case 3: // perform sparse QR decomposition
	{
		SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> qr(A);
		dx = qr.solve(b);
		prflag[0] = 0;
		if (qr.info() != Success)
			prflag[0] = -1;
		break;
	}
	case 4: // conjugate gradients with identity preconditioner
	{
		ConjugateGradient<SparseMatrix<double>, Lower | Upper, IdentityPreconditioner> cg;
		cg.setTolerance(TOL_r);
		cg.setMaxIterations(maxiter);
		cg.compute(A);
		dx = cg.solve(b);
		prflag[0] = 0;
		if (cg.info() != Success)
			prflag[0] = -1;
		priter[0] = (double)cg.iterations();
		prrelres[0] = cg.error();
		break;
	}
	case 5: // conjugate gradients with diagonal preconditioner
	{
		ConjugateGradient<SparseMatrix<double>, Lower | Upper, DiagonalPreconditioner<double>> cg;
		cg.setTolerance(TOL_r);
		cg.setMaxIterations(maxiter);
		cg.compute(A);
		dx = cg.solve(b);
		prflag[0] = 0;
		if (cg.info() != Success)
			prflag[0] = -1;
		priter[0] = (double)cg.iterations();
		prrelres[0] = cg.error();
		break;
	}
	case 6: // conjugate gradients with ICHOL preconditioner
	{
		ConjugateGradient<SparseMatrix<double>, Lower | Upper, IncompleteCholesky<double>> cg;
		cg.setTolerance(TOL_r);
		cg.setMaxIterations(maxiter);
		cg.compute(A);
		dx = cg.solve(b);
		prflag[0] = 0;
		if (cg.info() != Success)
			prflag[0] = -1;
		priter[0] = (double)cg.iterations();
		prrelres[0] = cg.error();
		break;
	}
	case 7: // stabilized bi-conjugate gradient with incomplete LU preconditioner
	{
		BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double>> bicg;
		bicg.preconditioner().setFillfactor(FillFactor);
		bicg.setTolerance(TOL_r);
		bicg.setMaxIterations(maxiter);
		bicg.compute(A);
		dx = bicg.solve(b);
		prflag[0] = 0;
		if (bicg.info() != Success)
			prflag[0] = -1;
		priter[0] = (double)bicg.iterations();
		prrelres[0] = bicg.error();
		break;
	}
	default:
		mexErrMsgTxt("Wrong solver type chosen.");
		break;
	}
}
