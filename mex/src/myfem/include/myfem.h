#pragma once

/******************** General error codes returned as integers ********************/
//  1 everything proceeds smoothly
// -1 wrong Gauss integration rule
// -2 wrong basis function evaluation
// -3 negative element jacobian
// -4 zero determinant during matrix inversion
// -5 maximum number of iterations reached

/******************** Deformation ********************/

// Get deformation gradient (F = [F11,F21,F12,F22], F = [F11 F21 F31 F12 F22 F32
// F13 F23 F33])
int getF2d(double **Be, const double *uel, const int np, double F[4]);
int getF3d(double **Be, const double *uel, const int np, double F[9]);

// Get the right Cauchy-Green strain tensor C = F'*F, Voigt notation (i.e C =
// [C11, C22, C12], C = [C11, C22, C33, C12, C23, C31])
int getC2d(const double F[4], double C[3]);
int getC3d(const double F[9], double C[6]);

// Get the Green-Lagrange strain tensor E = 0.5*(F'*F-I), Voigt notation (i.e E
// = [E11, E22, 2*E12], E = [E11, E22, E33, 2*E12, 2*E23, 2*E31])
int getE2d(const double F[4], double E[3]);
int getE3d(const double F[9], double E[6]);

/******************** Tensor operations ********************/

// Get second-order identity tensor tensor in Voigt notation
int getId2d(double Id[3]);
int getId3d(double Id[6]);

// Take symmetric producdt of two second-order tensors in Voigt notation ( Cijkl
// = 0.5*(Aik*Bjl+Ail*Bjk) )
int getSymmetricProduct2d(double C[3][3], const double A[3], const double B[3]);
int getSymmetricProduct3d(double C[6][6], const double A[6], const double B[6]);

/******************** Triangles ********************/

// Weights and natural coordinates for integration rule corresponding to ngauss
// Gauss points for triangles
int getTriagGaussInfo(const int ngauss, double *alpha, double **IP);

// Returns shape functions evaluated at tcoord for triangular elements
int getTriagShapeFun(const int np, const double tcoord[3], double *N);

// Matrix of derivatives with respect to physical coordinates of basis functions
// for TL formulation using F
int getTriagBeTLF(const int np, const double tcoord[3], const double *x,
                  const double *y, double &Jdet, double **Be);

// Matrix of derivatives with respect to physical coordinates of basis functions
// for TL formulation using E
int getTriagBeTLE(const int np, const double tcoord[3], const double *x,
                  const double *y, const double *uel, double &Jdet, double **Be,
                  double **Bf, double F[4]);

// Convert coordinates from natural to physical for triangles
int ConvertTriagNat2Phys(const int np, const double tcoord[3], const double *x,
                         const double *y, double &X, double &Y);

// Convert coordinates from physical to natural for triangles
int ConvertTriagPhys2Nat(const int np, const double X, const double Y,
                         const double *x, const double *y, const double TOL_g,
                         double tcoord[3]);

// Returns gradient of shape functions for triangles with respect to natural
// coordinates
int getTriagShapeFunGradNat(const int np, const double tcoord[3],
                            double *DalphaN, double *DbetaN, double *DgammaN);

// Returns gradient of shape functions for triangles with respect to physical
// coordinates
int getTriagShapeFunGradPhys(const int np, const double tcoord[3],
                             const double *x, const double *y, double *DxN,
                             double *DyN, double &Jdet);

/******************** Quadrangles ********************/

// Weights and natural coordinates for integration rule corresponding to ngauss
// Gauss points for quadrangles
int getQuadGaussInfo(const int ngauss, double *alpha, double **IP);

// Returns shape functions evaluated at tcoord for quadrangles
int getQuadShapeFun(const int np, const double tcoord[3], double *N);

// Matrix of derivatives with respect to physical coordinates of basis functions
// for TL formulation using F
int getQuadBeTLF(const int np, const double tcoord[3], const double *x,
                 const double *y, double &Jdet, double **Be);

// Matrix of derivatives with respect to physical coordinates of basis functions
// for TL formulation using E
int getQuadBeTLE(const int np, const double tcoord[3], const double *x,
                 const double *y, const double *uel, double &Jdet, double **Be,
                 double **Bf, double F[4]);

// Convert coordinates from natural to physical for quadrangles
int ConvertQuadNat2Phys(const int np, const double tcoord[3], const double *x,
                        const double *y, double &X, double &Y);

// Convert coordinates from physical to natural for quadrangles
int ConvertQuadPhys2Nat(const int np, const double X, const double Y,
                        const double *x, const double *y, const double TOL_g,
                        double tcoord[3]);

// Returns gradient of shape functions for quadrangles with respect to natural
// coordinates
int getQuadShapeFunGradNat(const int np, const double tcoord[3], double *DNxi,
                           double *DNeta);

/******************** Tetrahedrons ********************/

// Weights and natural coordinates for integration rule corresponding to ngauss
// Gauss points for tetrahedrons
int getTetraGaussInfo(const int ngauss, double *alpha, double **IP);

// Returns shape functions evaluated at tcoord for tetrahedrons
int getTetraShapeFun(const int np, const double tcoord[4], double *N);

// Matrix of derivatives with respect to physical coordinates of basis functions
// for tetrahedrons
int getTetraBeTLF(const int np, const double tcoord[4], const double *x,
                  const double *y, const double *z, double &Jdet, double **Be);

// Matrix of derivatives with respect to physical coordinates of basis functions
// for TL formulation using E
int getTetraBeTLE(const int np, const double tcoord[4], const double *x,
                  const double *y, const double *z, const double *uel,
                  double &Jdet, double **Be, double **Bf, double F[9]);

// Convert coordinates from natural to physical for tetrahedrons
int ConvertTetraNat2Phys(const int np, const double tcoord[4], const double *x,
                         const double *y, const double *z, double &X, double &Y,
                         double &Z);

// Convert coordinates from physical to natural for tetrahedrons
int ConvertTetraPhys2Nat(const int np, const double X, const double Y,
                         const double Z, const double *x, const double *y,
                         const double *z, const double TOL_g, double tcoord[4]);

// Returns gradient of shape functions for triangles with respect to natural
// coordinates
int getTetraShapeFunGradNat(const int np, const double tcoord[4],
                            double *DalphaN, double *DbetaN, double *DgammaN,
                            double *DdeltaN);

/******************** Hexahedrons ********************/

// Weights and natural coordinates for integration rule corresponding to ngauss
// Gauss points for hexahedrons
int getHexaGaussInfo(const int ngauss, double *alpha, double **IP);

// Returns shape functions evaluated at tcoord for hexahedrons
int getHexaShapeFun(const int np, const double tcoord[4], double *N);

// Matrix of derivatives with respect to physical coordinates of basis functions
// for hexahedrons
int getHexaBeTLF(const int np, const double tcoord[4], const double *x,
                 const double *y, const double *z, double &Jdet, double **Be);

// Matrix of derivatives with respect to physical coordinates of basis functions
// for TL formulation using E
int getHexaBeTLE(const int np, const double tcoord[4], const double *x,
                 const double *y, const double *z, const double *uel,
                 double &Jdet, double **Be, double **Bf, double F[9]);

// Convert coordinates from natural to physical for hexahedrons
int ConvertHexaNat2Phys(const int np, const double tcoord[4], const double *x,
                        const double *y, const double *z, double &X, double &Y,
                        double &Z);

// Convert coordinates from physical to natural for hexahedrons
int ConvertHexaPhys2Nat(const int np, const double X, const double Y,
                        const double Z, const double *x, const double *y,
                        const double *z, const double TOL_g, double tcoord[4]);

// Returns gradient of shape functions for hexahedra with respect to natural
// coordinates
int getHexaShapeFunGradNat(const int np, const double tcoord[3], double *DNxi,
                           double *DNeta, double *DNzeta);

/******************** Constitutive laws ********************/

// Get derivatives of invariants with respect to F
int getInvariantsDerivativesF2d(const double F[4], const double IF[4],
                                const double J, // inputs
                                double &I1b, double &I2b, double &I132,
                                double &logJ, // outputs
                                double &J13, double DI1b[4], double DI2b[4],
                                double DDI1b[4][4], double DDI2b[4][4]);

int getInvariantsDerivativesF3d(const double F[9], const double IF[9],
                                const double J, // inputs
                                double &I1b, double &I2b, double &I132,
                                double &logJ, // outputs
                                double &J13, double DI1b[9], double DI2b[9],
                                double DDI1b[9][9], double DDI2b[2][2][2][2]);

// Get derivatives of invariants with respect to C
int getInvariantsDerivativesC2d(const double C[3], double &I1, double &I2,
                                double &J, double &J2, double dI1dC[3],
                                double dI2dC[3], double dJdC[3],
                                double d2I2dCC[3][3], double d2JdCC[3][3]);

int getInvariantsDerivativesC3d(const double C[6], double &I1, double &I2,
                                double &J, double &J2, double dI1dC[6],
                                double dI2dC[6], double dJdC[6],
                                double d2I2dCC[6][6], double d2JdCC[6][6]);

// OOFEM (compressible Neo-hookean model)
int material_oofemF2d(const double matconst[8],
                      const double F[4],                       // inputs
                      double &w, double P[4], double D[4][4]); // outputs

int material_oofemE2d(const double matconst[8], const double C[3], // inputs
                      double &w, double S[3], double D[3][3]);     // outputs

int material_oofemF3d(const double matconst[8], const double F[9], // inputs
                      double &w, double P[9], double D[9][9]);     // outputs

int material_oofemE3d(const double matconst[8], const double C[6], // inputs
                      double &w, double S[6], double D[6][6]);     // outputs

// Bertoldi, Boyce
int material_bbF2d(const double matconst[8], const double F[4], double &w,
                   double P[4], double D[4][4]);

int material_bbE2d(const double matconst[8], const double C[3], double &w,
                   double S[3], double D[3][3]);

int material_bbF3d(const double matconst[8], const double F[9], double &w,
                   double P[9], double D[9][9]);

int material_bbE3d(const double matconst[8], const double C[6], double &w,
                   double S[6], double D[6][6]);

// Jamus, Green, Simpson
int material_jgsF2d(const double matconst[8], const double F[4], double &w,
                    double P[4], double D[4][4]);

int material_jgsE2d(const double matconst[8], const double C[3], double &w,
                    double S[3], double D[3][3]);

int material_jgsF3d(const double matconst[8], const double F[9], double &w,
                    double P[9], double D[9][9]);

int material_jgsE3d(const double matconst[8], const double C[6], double &w,
                    double S[6], double D[6][6]);

// five-term Mooney-Rivlin
int material_5mrF2d(const double matconst[8], const double F[4], double &w,
                    double P[4], double D[4][4]);

int material_5mrE2d(const double matconst[8], const double C[3], double &w,
                    double S[3], double D[3][3]);

int material_5mrF3d(const double matconst[8], const double F[9], double &w,
                    double P[9], double D[9][9]);

int material_5mrE3d(const double matconst[8], const double C[6], double &w,
                    double S[6], double D[6][6]);

// Linear elastic
int material_leF2d(const double matconst[8], const double F[4], double &w,
                   double P[4], double D[4][4]);

int material_leE2d(const double matconst[8], const double C[3], double &w,
                   double S[3], double D[3][3]);

int material_leF3d(const double matconst[8], const double F[9], double &w,
                   double P[9], double D[9][9]);

int material_leE3d(const double matconst[8], const double C[6], double &w,
                   double S[6], double D[6][6]);

// Ogden
int material_ogdenE2d(const double matconst[8], const double C[3], double &w,
                      double S[3], double D[3][3]);

int material_ogdenE3d(const double matconst[8], const double C[6], double &w,
                      double S[6], double D[6][6]);

// Ogden nematic
int material_ogdennematicE2d(const double matconst[8], const double C[3], double &w,
                      double S[3], double D[3][3]);

int material_ogdennematicE3d(const double matconst[8], const double C[6], double &w,
                      double S[6], double D[6][6]);

/******************** Basic linear algebra ********************/

// Computes inverse of 2x2 matrix with determinant (true for success, false for
// detm = 0)
bool getInverse22(const double m[2][2], double &detm, double minv[2][2]);
bool getInverse22(const double m[4], double &detm, double minv[4]);
bool getInverse22sym(const double m[3], double &detm, double minv[3]);

// Computes eigenvalues and eigenvectors of a 2x2 symmetric matrix
bool getEigendecomposition22(const double m[3], double D[2], double V[2][2]);

// Computes inverse of 3x3 matrix with determinant (true for success, false for
// detm = 0)
bool getInverse33(const double m[3][3], double &detm, double minv[3][3]);
bool getInverse33(const double m[9], double &detm, double minv[9]);
bool getInverse33sym(const double m[6], double &detm, double minv[6]);

// Computes inverse of 4x4 matrix with determinant (true for success, false for
// detm = 0)
bool getInverse44(const double m[4][4], double &detm, double minv[4][4]);

/******************** Interpolation ********************/

// Evaluation of the Lagrangian polynomial for derivative = 0 (or its derivative
// for derivative = 1) of given order evaluated at xi
double getLagrange1DPoly(const int order, const double xinode,
                         const int derivative, const double xi);

/******************** Geometric features ********************/

// Check if two bounding boxes specified through xa (min x), wa (width x), ya (min y), ha (height y), and
// xb, wb, yb, hb intersect
bool intersect2d(const double xa, const double wa, const double ya, const double ha,
                 const double xb, const double wb, const double yb, const double hb);

// Check if two bounding boxes specified through xa (min x), wa (width x), ya (min y), ha (height y), za (min z), da (depth z), and
// xb, wb, yb, hb, zb, db intersect
bool intersect3d(const double xa, const double wa, const double ya, const double ha, const double za, const double da,
                 const double xb, const double wb, const double yb, const double hb, const double zb, const double db);