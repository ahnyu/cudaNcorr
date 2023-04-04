#include <vector_types.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdio>
#include "../include/device.hpp"
#include "../include/gpuerrchk.h"

//----device functions----//
__constant__ int3 d_shifts[27];
__constant__ double3 d_box;
__constant__ double3 d_ori;
__constant__ double d_sqrRMax;
__constant__ double d_sqrRMin;
__constant__ double d_sqrRpMax;
__constant__ double d_sqrRpMin;
__constant__ double d_sqrPiMax;
__constant__ int3 d_numCells;
__constant__ double3 d_cellSize;

__device__ double dotProduct_d4(double4 a, double4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ double dotProduct_d3(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ double dotProduct_d3d4(double3 a, double4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ double sqrSep(double4 p1, double4 p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;

    return dx * dx + dy * dy + dz * dz;
}

__device__ double sqrRpBox(double4 p1, double4 p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;

    return dx * dx + dy * dy;
}

__device__ double sqrPiBox(double4 p1, double4 p2) {
    double dz = p1.z - p2.z;

    return dz * dz;
}

__device__ double3 getLOS(double4 p1, double4 p2, double4 p3, double &sqrL) {
    double3 pos1, pos2, pos3;
    pos1.x = p1.x-d_ori.x; pos1.y = p1.y-d_ori.y; pos1.z = p1.z-d_ori.z;
    pos2.x = p2.x-d_ori.x; pos2.y = p2.y-d_ori.y; pos2.z = p2.z-d_ori.z;
    pos3.x = p3.x-d_ori.x; pos3.y = p3.y-d_ori.y; pos3.z = p3.z-d_ori.z;
    
    double3 los=make_double3(pos1.x+pos2.x+pos3.x,pos1.y+pos2.y+pos3.y,pos1.z+pos2.z+pos3.z);
    sqrL=dotProduct_d3(los,los);
    return los;
}

__device__ double getSqrSdotL(double4 p1, double4 p2, double3 los) {
    double3 s = make_double3(p1.x-p2.x, p1.y-p2.y, p1.z-p2.z);
    double sDotL=dotProduct_d3(los,s);
    double sqrSDotL=sDotL*sDotL;
    return sqrSDotL;
}

__device__ void getSqrRpSqrPi(double sqrS, double sqrL, double sqrSDotL, double &sqrRp, double &sqrPi) {
    sqrPi=sqrSDotL/sqrL;
    sqrRp=sqrS-sqrPi;
}
    

__device__ int4 initCellIndexBox(double4 particle) {
    int4 cellIndex={int(particle.x/d_cellSize.x), int(particle.y/d_cellSize.y), int(particle.z/d_cellSize.z), 0};

    if(cellIndex.x == d_numCells.x) cellIndex.x--;
    if(cellIndex.y == d_numCells.y) cellIndex.y--;
    if(cellIndex.z == d_numCells.z) cellIndex.z--;
    return cellIndex;
}

__device__ int4 initCellIndexSurvey(double4 particle) {
    int4 cellIndex={int(particle.x/d_cellSize.x), int(particle.y/d_cellSize.y), int(particle.z/d_cellSize.z), 0};
    return cellIndex;
}

__device__ int4 shiftCellIndexBox(int4 p_cell, int i, double3 &rShift) {
    p_cell.x += d_shifts[i].x;
    p_cell.y += d_shifts[i].y;
    p_cell.z += d_shifts[i].z;
    rShift.x = 0.0;
    rShift.y = 0.0;
    rShift.z = 0.0;
    if (p_cell.x == d_numCells.x) {
        p_cell.x = 0;
        rShift.x = d_box.x;
    }
    if (p_cell.y == d_numCells.y) {
        p_cell.y = 0;
        rShift.y = d_box.y;
    }
    if (p_cell.z == d_numCells.z) {
        p_cell.z = 0;
        rShift.z = d_box.z;
    }
    if (p_cell.x == -1) {
        p_cell.x = d_numCells.x - 1;
        rShift.x = -d_box.x;
    }
    if (p_cell.y == -1) {
        p_cell.y = d_numCells.y - 1;
        rShift.y = -d_box.y;
    }
    if (p_cell.z == -1) {
        p_cell.z = d_numCells.z - 1;
        rShift.z = -d_box.z;
    }
    p_cell.w = p_cell.z + d_numCells.z*(p_cell.y + d_numCells.y*p_cell.x);
    return p_cell;
}
__device__ int4 shiftCellIndexSurvey(int4 p_cell, int i) {
    p_cell.x += d_shifts[i].x;
    p_cell.y += d_shifts[i].y;
    p_cell.z += d_shifts[i].z;
    if (p_cell.x == d_numCells.x) {
        p_cell.w=-1;
        return p_cell;
    }
    if (p_cell.y == d_numCells.y) {
        p_cell.w=-1;
        return p_cell;
    }
    if (p_cell.z == d_numCells.z) {
        p_cell.w=-1;
        return p_cell;
    }
    if (p_cell.x == -1) {
        p_cell.w=-1;
        return p_cell;
    }
    if (p_cell.y == -1) {
        p_cell.w=-1;
        return p_cell;
    }
    if (p_cell.z == -1) {
        p_cell.w=-1;
        return p_cell;
    }
    p_cell.w = p_cell.z + d_numCells.z*(p_cell.y + d_numCells.y*p_cell.x);
    return p_cell;
}
__device__ int getTriangleIndexThreeD(
    double sqrR1,
    double sqrR2,
    double sqrR3,
    const double *sqrBins,
    int numBins
) {
    // Sort the lengths in ascending order
    double temp;
    if (sqrR1 > sqrR2) {
        temp = sqrR1; sqrR1 = sqrR2; sqrR2 = temp;
    }
    if (sqrR1 > sqrR3) {
        temp = sqrR1; sqrR1 = sqrR3; sqrR3 = temp;
    }
    if (sqrR2 > sqrR3) {
        temp = sqrR2; sqrR2 = sqrR3; sqrR3 = temp;
    }
    // Check if the given lengths can form a triangle
    if (sqrtf(sqrR1) + sqrtf(sqrR2) <= sqrtf(sqrR3)) {
        return -1; // The lengths cannot form a triangle
    }



    // Find the intervals for each length
    int idx1, idx2, idx3;
    for (int i = 0; i < numBins; ++i) {
        if (sqrR1 >= sqrBins[i] && sqrR1 < sqrBins[i + 1]) {
            idx1 = i;
        }
        if (sqrR2 >= sqrBins[i] && sqrR2 < sqrBins[i + 1]) {
            idx2 = i;
        }
        if (sqrR3 >= sqrBins[i] && sqrR3 < sqrBins[i + 1]) {
            idx3 = i;
        }
    }

    if (idx1 == -1 || idx2 == -1 || idx3 == -1) {
        return -1; // At least one of the lengths is not in the bin range
    }

    // Flatten the 3D index to 1D
    int index = idx3 + idx2 * numBins + idx1 * numBins * numBins;

    return index;
}

__device__ int getTriangleIndexRppi(
    double sqrRp1,
    double sqrRp2,
    double sqrRp3,
    double sqrPi1,
    double sqrPi2,
    double sqrPi3,
    const double *sqrBins,
    int numBins,
    int numPiBins
) {

    // Sort the lengths in ascending order
    double temp;
    if (sqrRp1 > sqrRp2) {
        temp = sqrRp1; sqrRp1 = sqrRp2; sqrRp2 = temp;
        temp = sqrPi1; sqrPi1 = sqrPi2; sqrPi2 = temp;
    }
    if (sqrRp1 > sqrRp3) {
        temp = sqrRp1; sqrRp1 = sqrRp3; sqrRp3 = temp;
        temp = sqrPi1; sqrPi1 = sqrPi3; sqrPi3 = temp;
    }
    if (sqrRp2 > sqrRp3) {
        temp = sqrRp2; sqrRp2 = sqrRp3; sqrRp3 = temp;
        temp = sqrPi2; sqrPi2 = sqrPi3; sqrPi3 = temp;
    }
    if (sqrtf(sqrRp1) + sqrtf(sqrRp2) <= sqrtf(sqrRp3)) {
        return -1; // The lengths cannot form a triangle
    }


    // Find the intervals for each length
    int idx1, idx2, idx3, idxPi1, idxPi2;
    for (int i = 0; i < numBins; ++i) {
        if (sqrRp1 >= sqrBins[i] && sqrRp1 < sqrBins[i + 1]) {
            idx1 = i;
        }
        if (sqrRp2 >= sqrBins[i] && sqrRp2 < sqrBins[i + 1]) {
            idx2 = i;
        }
        if (sqrRp3 >= sqrBins[i] && sqrRp3 < sqrBins[i + 1]) {
            idx3 = i;
        }
    }

    idxPi1 = floor(sqrtf(sqrPi1));
    idxPi2 = floor(sqrtf(sqrPi2));

    // Flatten the 3D index to 1D
    int index = idxPi2 + idxPi1*numPiBins + idx3*numPiBins*numPiBins + idx2*numBins*numPiBins*numPiBins + idx1*numBins*numBins*numPiBins*numPiBins;

    return index;
}

//----end device functions----//
//----global cuda kernels----//
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
) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx1 < Np1) {
        double4 particle1=d_p1[idx1];
        int4 p1_cell=initCellIndexBox(particle1);
        for (int nci1 = 0; nci1 < 27; ++nci1) {
            double3 rShift2;
            int4 p2_cell = shiftCellIndexBox(p1_cell,nci1,rShift2);
            int cell_counts2 = d_p2CellSize[p2_cell.w];

            for (int idx2 = 0; idx2 < cell_counts2; ++idx2) {
                double4 particle2 = d_p2Cell[p2_cell.w][idx2];
                particle2.x += rShift2.x;
                particle2.y += rShift2.y;
                particle2.z += rShift2.z;
                double sqrR1 = sqrSep(particle1, particle2);
            
                if(sqrR1 < d_sqrRMax && sqrR1 > d_sqrRMin) {
                    for (int nci2 = 0; nci2 < 27; ++nci2) {
                        double3 rShift3;
                        int4 p3_cell = shiftCellIndexBox(p1_cell,nci2,rShift3);
                        int cell_counts3 = d_p3CellSize[p3_cell.w];

                        for (int idx3 = 0; idx3 < cell_counts3; ++idx3) {
                            double4 particle3 = d_p3Cell[p3_cell.w][idx3];
                            particle3.x += rShift3.x;
                            particle3.y += rShift3.y;
                            particle3.z += rShift3.z;

                            double sqrR2 = sqrSep(particle1, particle3);
                            double sqrR3 = sqrSep(particle2, particle3);
                            if(sqrR2 < d_sqrRMax && sqrR3 < d_sqrRMax && sqrR2 > d_sqrRMin && sqrR3 > d_sqrRMin) {
                                int triangle_index = getTriangleIndexThreeD(sqrR1, sqrR2, sqrR3, d_sqrBins, numBins);

                                if (triangle_index != -1) {
                                    double weight_product = particle1.w * particle2.w * particle3.w;
                                    atomicAdd(&d_triangle_counts[triangle_index], weight_product);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

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
) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx1 < Np1) {
        double4 particle1=d_p1[idx1];
        int4 p1_cell=initCellIndexBox(particle1);
        for (int nci1 = 0; nci1 < 27; ++nci1) {
            double3 rShift2;
            int4 p2_cell = shiftCellIndexBox(p1_cell,nci1,rShift2);
            int cell_counts2 = d_p2CellSize[p2_cell.w];

            for (int idx2 = 0; idx2 < cell_counts2; ++idx2) {
                double4 particle2 = d_p2Cell[p2_cell.w][idx2];
                particle2.x += rShift2.x;
                particle2.y += rShift2.y;
                particle2.z += rShift2.z;
                double sqrPi1 = sqrPiBox(particle1, particle2);
                if(sqrPi1<d_sqrPiMax) {

                    double sqrRp1 = sqrRpBox(particle1, particle2);
                    if(sqrRp1 < d_sqrRpMax && sqrRp1 > d_sqrRpMin) {
                        for (int nci2 = 0; nci2 < 27; ++nci2) {
                            double3 rShift3;
                            int4 p3_cell = shiftCellIndexBox(p1_cell,nci2,rShift3);
                            int cell_counts3 = d_p3CellSize[p3_cell.w];
    
                            for (int idx3 = 0; idx3 < cell_counts3; ++idx3) {
                                double4 particle3 = d_p3Cell[p3_cell.w][idx3];
                                particle3.x += rShift3.x;
                                particle3.y += rShift3.y;
                                particle3.z += rShift3.z;
                                double sqrPi2 = sqrPiBox(particle1, particle3);
                                double sqrPi3 = sqrPiBox(particle2, particle3);
                                if(sqrPi2<d_sqrPiMax && sqrPi3<d_sqrPiMax) {
                                    double sqrRp2 = sqrRpBox(particle1, particle3);
                                    double sqrRp3 = sqrRpBox(particle2, particle3);
                                    if(sqrRp2 < d_sqrRpMax && sqrRp3 < d_sqrRpMax && sqrRp2 > d_sqrRpMin && sqrRp3 > d_sqrRpMin) {
                                        int triangle_index = getTriangleIndexRppi(sqrRp1, sqrRp2, sqrRp3, sqrPi1, sqrPi2, sqrPi3, d_sqrBins, numBins, numPiBins);
                                        if (triangle_index != -1) {
                                            double weight_product = particle1.w * particle2.w * particle3.w;
                                            atomicAdd(&d_triangle_counts[triangle_index], weight_product);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

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
) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx1 < Np1) {
        double4 particle1=d_p1[idx1];
        int4 p1_cell=initCellIndexSurvey(particle1);
        for (int nci1 = 0; nci1 < 27; ++nci1) {
            int4 p2_cell = shiftCellIndexSurvey(p1_cell,nci1);
            if (p2_cell.w==-1) continue;
            int cell_counts2 = d_p2CellSize[p2_cell.w];

            for (int idx2 = 0; idx2 < cell_counts2; ++idx2) {
                double4 particle2 = d_p2Cell[p2_cell.w][idx2];
                double sqrR1 = sqrSep(particle1, particle2);

                if(sqrR1 < d_sqrRMax && sqrR1 > d_sqrRMin) {
                    for (int nci2 = 0; nci2 < 27; ++nci2) {
                        int4 p3_cell = shiftCellIndexSurvey(p1_cell,nci2);
                        if (p3_cell.w==-1) continue;
                        int cell_counts3 = d_p3CellSize[p3_cell.w];

                        for (int idx3 = 0; idx3 < cell_counts3; ++idx3) {
                            double4 particle3 = d_p3Cell[p3_cell.w][idx3];

                            double sqrR2 = sqrSep(particle1, particle3);
                            double sqrR3 = sqrSep(particle2, particle3);
                            if(sqrR2 < d_sqrRMax && sqrR3 < d_sqrRMax && sqrR2 > d_sqrRMin && sqrR3 > d_sqrRMin) {
                                int triangle_index = getTriangleIndexThreeD(sqrR1, sqrR2, sqrR3, d_sqrBins, numBins);

                                if (triangle_index != -1) {
                                    double weight_product = particle1.w * particle2.w * particle3.w;
                                    atomicAdd(&d_triangle_counts[triangle_index], weight_product);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


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
) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx1 < Np1) {
        double4 particle1=d_p1[idx1];
        int4 p1_cell=initCellIndexSurvey(particle1);
        for (int nci1 = 0; nci1 < 27; ++nci1) {
            int4 p2_cell = shiftCellIndexSurvey(p1_cell,nci1);
            if (p2_cell.w==-1) continue;
            int cell_counts2 = d_p2CellSize[p2_cell.w];

            for (int idx2 = 0; idx2 < cell_counts2; ++idx2) {
                double4 particle2 = d_p2Cell[p2_cell.w][idx2];
                double sqrS1 = sqrSep(particle1, particle2);
                if(sqrS1<d_sqrRMax && sqrS1>d_sqrRMin) {
                    for (int nci2 = 0; nci2 < 27; ++nci2) {
                        int4 p3_cell = shiftCellIndexSurvey(p1_cell,nci2);
                        if (p3_cell.w==-1) continue;
                        int cell_counts3 = d_p3CellSize[p3_cell.w];
    
                        for (int idx3 = 0; idx3 < cell_counts3; ++idx3) {
                            double4 particle3 = d_p3Cell[p3_cell.w][idx3];
                            double sqrS2 = sqrSep(particle1, particle3); 
                            double sqrS3 = sqrSep(particle2, particle3);
                            if(sqrS2<d_sqrRMax && sqrS3<d_sqrRMax && sqrS2>d_sqrRMin &&sqrS3>d_sqrRMin) {
                                double sqrL;
                                double3 los=getLOS(particle1, particle2, particle3, sqrL);
                                double sqrPiMaxTimesSqrL=d_sqrPiMax*sqrL;
                                double sqrS1dotL=getSqrSdotL(particle1, particle2, los);
                                if(sqrPiMaxTimesSqrL<sqrS1dotL) continue;
                                double sqrS2dotL=getSqrSdotL(particle1, particle3, los);
                                if(sqrPiMaxTimesSqrL<sqrS2dotL) continue;
                                double sqrS3dotL=getSqrSdotL(particle2, particle3, los);
                                if(sqrPiMaxTimesSqrL<sqrS3dotL) continue;
                                double sqrRp1, sqrRp2, sqrRp3, sqrPi1, sqrPi2, sqrPi3;
                                getSqrRpSqrPi(sqrS1, sqrL, sqrS1dotL,sqrRp1, sqrPi1);
                                getSqrRpSqrPi(sqrS2, sqrL, sqrS2dotL,sqrRp2, sqrPi2);
                                getSqrRpSqrPi(sqrS3, sqrL, sqrS3dotL,sqrRp3, sqrPi3);
                                if(sqrRp1 < d_sqrRpMax && 
                                   sqrRp2 < d_sqrRpMax && 
                                   sqrRp3 < d_sqrRpMax && 
                                   sqrRp1 > d_sqrRpMin && 
                                   sqrRp2 > d_sqrRpMin && 
                                   sqrRp3 > d_sqrRpMin &&
                                   sqrPi1 < d_sqrPiMax &&
                                   sqrPi2 < d_sqrPiMax) {
                                    int triangle_index = getTriangleIndexRppi(sqrRp1, sqrRp2, sqrRp3, sqrPi1, sqrPi2, sqrPi3, d_sqrBins, numBins, numPiBins);
                                    if (triangle_index != -1) {
                                        double weight_product = particle1.w * particle2.w * particle3.w;
                                        atomicAdd(&d_triangle_counts[triangle_index], weight_product);
                                    
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//----end global cuda kernels----//

