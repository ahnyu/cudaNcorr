#ifndef _DEVICE_HPP_
#define _DEVICE_HPP_

#include <cmath>
#include <vector_types.h>

extern __constant__ int3 d_shifts[27];
extern __constant__ double3 d_box;
extern __constant__ double3 d_ori;
extern __constant__ double d_sqrRMax;
extern __constant__ double d_sqrRMin;
extern __constant__ double d_sqrRpMax;
extern __constant__ double d_sqrRpMin;
extern __constant__ double d_sqrPiMax;
extern __constant__ int3 d_numCells;
extern __constant__ double3 d_cellSize;


__device__ double dotProduct_d3(double3 a, double3 b); 
__device__ double dotProduct_d4(double4 a, double4 b); 
__device__ double dotProduct_d3d4(double3 a, double4 b); 

__device__ double sqrSep(double4 p1, double4 p2);

__device__ double sqrRpBox(double4 p1, double4 p2);
__device__ double sqrPiBox(double4 p1, double4 p2);

__device__ double3 getLOS(double4 p1, double4 p2, double4 p3, double &sqrL);
__device__ double getSqrSdotL(double4 p1, double4 p2, double3 los);
__device__ void getSqrRpSqrPi(double sqrS, double sqrL, double sqrSDotL, double &sqrRp, double &sqrPi);

__device__ int4 initCellIndexBox(double4 particle);
__device__ int4 initCellIndexSurvey(double4 particle);
__device__ int4 shiftCellIndexBox(int4 p_cell, int i, double3 &rShift);
__device__ int4 shiftCellIndexSurvey(int4 p_cell, int i);

__device__ int getTriangleIndexThreeD(
    double sqrR1,
    double sqrR2,
    double sqrR3,
    const double *sqrBins,
    int numBins
);

__device__ int getTriangleIndexRppi(
    double sqrRp1,
    double sqrRp2,
    double sqrRp3,
    double sqrPi1,
    double sqrPi2,
    const double *sqrBins,
    int numBins,
    int numPiBins
);

__global__ void countTrianglesThreeDBox(
    double4 *d_p1,
    int Np1,
    double4 **d_p2Cell,
    double4 **d_p3Cell,
    int *d_p2CellSize,
    int *d_p3CellSize,
    double *d_sqrBins,
    int numBins,
    double *d_triangle_counts
);

__global__ void countTrianglesRppiBox(
    double4 *d_p1,
    int Np1,
    double4 **d_p2Cell,
    double4 **d_p3Cell,
    int *d_p2CellSize,
    int *d_p3CellSize,
    double *d_sqrBins,
    int numBins,
    int numPiBins,
    double *d_triangle_counts
);

__global__ void countTrianglesThreeDSurvey(
    double4 *d_p1,
    int Np1,
    double4 **d_p2Cell,
    double4 **d_p3Cell,
    int *d_p2CellSize,
    int *d_p3CellSize,
    double *d_sqrBins,
    int numBins,
    double *d_triangle_counts
);

__global__ void countTrianglesRppiSurvey(
    double4 *d_p1,
    int Np1,
    double4 **d_p2Cell,
    double4 **d_p3Cell,
    int *d_p2CellSize,
    int *d_p3CellSize,
    double *d_sqrBins,
    int numBins,
    int numPiBins,
    double *d_triangle_counts
);


#endif
