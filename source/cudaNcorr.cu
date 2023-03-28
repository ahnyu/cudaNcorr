#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <vector_types.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdexcept>
#include <sstream>
#include <string>
#include "../include/cudaNcorr.hpp"

//----device functions----//
__constant__ int3 d_shifts[27];
__constant__ double3 d_box;
__constant__ double d_rMax;
__constant__ double d_rMin;
__constant__ int3 d_numCells;
__constant__ double3 d_cellSize;

__device__ double distance(double4 p1, double4 p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;

    return sqrtf(dx * dx + dy * dy + dz * dz);
}

__device__ int4 initCellIndex(double4 particle) {
    int4 cellIndex={int(particle.x/d_cellSize.x), int(particle.y/d_cellSize.y), int(particle.z/d_cellSize.z), 0};
    if(cellIndex.x == d_numCells.x) cellIndex.x--;
    if(cellIndex.y == d_numCells.y) cellIndex.y--;
    if(cellIndex.z == d_numCells.z) cellIndex.z--;
    return cellIndex;
}

__device__ int4 shiftCellIndex(int4 p_cell, int i, double3 &rShift) {
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

__device__ int getTriangleIndex(
    double r1,
    double r2,
    double r3,
    const double *bins,
    int numBins
) {
    // Check if the given lengths can form a triangle
    if (r1 + r2 <= r3 || r1 + r3 <= r2 || r2 + r3 <= r1) {
        return -1; // The lengths cannot form a triangle
    }

    // Sort the lengths in ascending order
    double temp;
    if (r1 > r2) {
        temp = r1; r1 = r2; r2 = temp;
    }
    if (r1 > r3) {
        temp = r1; r1 = r3; r3 = temp;
    }
    if (r2 > r3) {
        temp = r2; r2 = r3; r3 = temp;
    }

    // Find the intervals for each length
    int idx1 = -1, idx2 = -1, idx3 = -1;
    for (int i = 0; i < numBins; ++i) {
        if (r1 >= bins[i] && r1 < bins[i + 1]) {
            idx1 = i;
        }
        if (r2 >= bins[i] && r2 < bins[i + 1]) {
            idx2 = i;
        }
        if (r3 >= bins[i] && r3 < bins[i + 1]) {
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

//----end device functions----//

//----global cuda kernels----//
__global__ void countTriangles(
    double4 *d_p1,
    int Np1,
    double4 **d_p2Cell,
    double4 **d_p3Cell,
    int *d_p2CellSize,
    int *d_p3CellSize,
    double *d_bins,
    int numBins,
    double *d_triangle_counts
) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx1 < Np1) {
        double4 particle1=d_p1[idx1];
        int4 p1_cell=initCellIndex(particle1);
        for (int nci1 = 0; nci1 < 27; ++nci1) {
            double3 rShift2;
            int4 p2_cell = shiftCellIndex(p1_cell,nci1,rShift2);
            int cell_counts2 = d_p2CellSize[p2_cell.w];

            for (int idx2 = 0; idx2 < cell_counts2; ++idx2) {
                double4 particle2 = d_p2Cell[p2_cell.w][idx2];
                particle2.x += rShift2.x;
                particle2.y += rShift2.y;
                particle2.z += rShift2.z;
                double r1 = distance(particle1, particle2);

                if(r1 < d_rMax && r1 > d_rMin) {
                    for (int nci2 = 0; nci2 < 27; ++nci2) {
                        double3 rShift3;
                        int4 p3_cell = shiftCellIndex(p1_cell,nci2,rShift3);
                        int cell_counts3 = d_p3CellSize[p3_cell.w];

                        for (int idx3 = 0; idx3 < cell_counts3; ++idx3) {
                            double4 particle3 = d_p3Cell[p3_cell.w][idx3];
                            particle3.x += rShift3.x;
                            particle3.y += rShift3.y;
                            particle3.z += rShift3.z;

                            double r2 = distance(particle1, particle3);
                            double r3 = distance(particle2, particle3);
                            if(r2 < d_rMax && r3 < d_rMax && r2 > d_rMin && r3 > d_rMin) {
                                int triangle_index = getTriangleIndex(r1, r2, r3, d_bins, numBins);

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

//----end global cuda kernels----//

//----host functions----//

void ncorr::rezeroVectors() {;
#pragma omp parallel for
    for (int i = 0; i < triCountsAll_.size(); ++i) {
        triCountsAll_[i] = 0;
    }
}

std::vector<double> ncorr::getBinsMid(const std::vector<double>& bins) {
    std::vector<double> binsMid;

    for (size_t i = 1; i < bins.size(); ++i) {
        double mid = (bins[i] + bins[i - 1]) / 2.0;
        binsMid.push_back(mid);
    }

    return binsMid;
}

std::vector<int3> ncorr::getShifts() {
    std::vector<int3> shifts;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            for (int k = -1; k <= 1; ++k) {
                int3 temp = {i, j, k};
                shifts.push_back(temp);
            }
        }
    }
    return shifts;
}

int ncorr::getCellIndex(const double4& p) {
    int x_cell = static_cast<int>(p.x / cellSize_.x);
    if(x_cell == numCells_.x) x_cell--;
    int y_cell = static_cast<int>(p.y / cellSize_.y);
    if(y_cell == numCells_.y) y_cell--;
    int z_cell = static_cast<int>(p.z / cellSize_.z);
    if(z_cell == numCells_.z) z_cell--;
    return z_cell + numCells_.z*(y_cell + numCells_.y*x_cell);
}

void ncorr::gridParticles(int Np, std::vector<double4> &particles, std::vector<std::vector<double4>> &particle_cells) {
    #pragma omp parallel
    {   
        #pragma omp for
        for (int i = 0; i < Np; i++) {
            int cell_index = ncorr::getCellIndex(particles[i]);
            #pragma omp critical
            particle_cells[cell_index].push_back(particles[i]);
        }   
    }   
}

std::vector<double3> ncorr::getTriBins() const {
    return triBins_;
}

std::vector<double> ncorr::getTriCountsOut() const {
    return triCountsOut_;
}

ncorr::ncorr(std::vector<double> r_bins, double3 box_size){
    bins_=r_bins;
    binsMid_=getBinsMid(bins_);
    rMax_=r_bins.back();
    rMin_=r_bins[0];
    box_=box_size;
    numBins_=bins_.size()-1;
    numBinsCounts_=numBins_*numBins_*numBins_;
    cellSize_={bins_.back(),bins_.back(),bins_.back()};
    numCells_.x=static_cast<int>(std::ceil(box_.x / cellSize_.x));
    numCells_.y=static_cast<int>(std::ceil(box_.y / cellSize_.y));
    numCells_.z=static_cast<int>(std::ceil(box_.z / cellSize_.z));
    shifts_=getShifts();
    for (int i=0; i<numBins_; ++i) {
        for (int j=i; j<numBins_; ++j) {
            for (int k=j; k<numBins_; ++k) {
                if(binsMid_[k]<=binsMid_[i]+binsMid_[j]) {
                    triCountsOut_.push_back(0);
                    triBins_.push_back({binsMid_[i],binsMid_[j],binsMid_[k]});
                }
            }
        }
    }
    triCountsAll_.resize(numBinsCounts_);
}

void ncorr::calculateD1D2D3(std::vector<double4> &p1, std::vector<double4> &p2, std::vector<double4> &p3) {

    rezeroVectors();

    std::cout <<"cellsizex = "<< cellSize_.x <<"cellsizey "<< cellSize_.y <<"cellsizez = "<<cellSize_.z<< std::endl;
    std::cout <<"numCellsx = "<< numCells_.x <<"numCellsy "<< numCells_.y <<"numCellsz = "<<numCells_.z<< std::endl;

    cudaMemcpyToSymbol(d_rMax, &rMax_, sizeof(double));
    cudaMemcpyToSymbol(d_rMin, &rMin_, sizeof(double));
    cudaMemcpyToSymbol(d_cellSize, &cellSize_, sizeof(double3));
    cudaMemcpyToSymbol(d_box, &box_, sizeof(double3));
    cudaMemcpyToSymbol(d_numCells, &numCells_, sizeof(int3));
    cudaMemcpyToSymbol(d_shifts, shifts_.data(), shifts_.size()*sizeof(int3));

    int Np1=p1.size();
    int Np2=p2.size();
    int Np3=p3.size();

    std::cout <<"Np1 = "<< Np1 <<"Np2 = "<< Np2 <<"Np3 = "<<Np3<< std::endl;

    std::vector<std::vector<double4>> p1Cell(numCells_.x * numCells_.y * numCells_.z);
    std::vector<std::vector<double4>> p2Cell(numCells_.x * numCells_.y * numCells_.z);
    std::vector<std::vector<double4>> p3Cell(numCells_.x * numCells_.y * numCells_.z);

    double4 **h_p1Cell = (double4 **)malloc(p1Cell.size()*sizeof(double4 *));
    double4 **h_p2Cell = (double4 **)malloc(p2Cell.size()*sizeof(double4 *));
    double4 **h_p3Cell = (double4 **)malloc(p3Cell.size()*sizeof(double4 *));

    std::vector<int> p1CellSize;
    std::vector<int> p2CellSize;
    std::vector<int> p3CellSize;

    gridParticles(Np1,p1,p1Cell);
    gridParticles(Np2,p2,p2Cell);
    gridParticles(Np3,p3,p3Cell);

    for (int i=0; i < p1Cell.size(); ++i) {
        p1CellSize.push_back(p1Cell[i].size());
        cudaMalloc((void **)&h_p1Cell[i], p1Cell[i].size()*sizeof(double4));
        cudaMemcpy(h_p1Cell[i], p1Cell[i].data(), p1Cell[i].size()*sizeof(double4), cudaMemcpyHostToDevice);
    }

    for (int i=0; i < p2Cell.size(); ++i) {
        p2CellSize.push_back(p2Cell[i].size());
        cudaMalloc((void **)&h_p2Cell[i], p2Cell[i].size()*sizeof(double4));
        cudaMemcpy(h_p2Cell[i], p2Cell[i].data(), p2Cell[i].size()*sizeof(double4), cudaMemcpyHostToDevice);
    }

    for (int i=0; i < p3Cell.size(); ++i) {
        p3CellSize.push_back(p3Cell[i].size());
        cudaMalloc((void **)&h_p3Cell[i], p3Cell[i].size()*sizeof(double4));
        cudaMemcpy(h_p3Cell[i], p3Cell[i].data(), p3Cell[i].size()*sizeof(double4), cudaMemcpyHostToDevice);
    }

    std::cout <<"Np1 = "<< Np1 <<"Np2 = "<< Np2 <<"Np3 = "<<Np3<< std::endl;

    double4 *d_p1;
    double4 **d_p2Cell;
    double4 **d_p3Cell;
    int *d_p2CellSize;
    int *d_p3CellSize;
    cudaMalloc((void **)&d_p1, Np1 * sizeof(double4));
    cudaMemcpy(d_p1, p1.data(), Np1 * sizeof(double4), cudaMemcpyHostToDevice);
    cudaMalloc(&d_p2Cell,p2Cell.size()*sizeof(double4 *));
    cudaMemcpy(d_p2Cell,h_p2Cell,p2Cell.size()*sizeof(double4 *), cudaMemcpyHostToDevice);
    cudaMalloc(&d_p3Cell,p3Cell.size()*sizeof(double4 *));
    cudaMemcpy(d_p3Cell,h_p3Cell,p3Cell.size()*sizeof(double4 *), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_p2CellSize, p2CellSize.size()*sizeof(int));
    cudaMemcpy(d_p2CellSize,p2CellSize.data(),p2CellSize.size()*sizeof(int),cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_p3CellSize, p3CellSize.size()*sizeof(int));
    cudaMemcpy(d_p3CellSize,p3CellSize.data(),p3CellSize.size()*sizeof(int),cudaMemcpyHostToDevice);

    double *d_bins;
    double *d_triCountsAll;

    cudaMalloc((void **)&d_bins, bins_.size()*sizeof(double));
    cudaMemcpy(d_bins, bins_.data(), bins_.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_triCountsAll, numBinsCounts_ * sizeof(double));
    cudaMemcpy(d_triCountsAll, triCountsAll_.data(), triCountsAll_.size()*sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 64;
    int gridSize = (Np1 + blockSize - 1) / blockSize;

    countTriangles<<<gridSize, blockSize>>>(
            d_p1,
            Np1,
            d_p2Cell,
            d_p3Cell,
            d_p2CellSize,
            d_p3CellSize,
            d_bins,
            numBins_,
            d_triCountsAll);

    cudaDeviceSynchronize();

    cudaMemcpy(triCountsAll_.data(), d_triCountsAll, numBinsCounts_ * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_p1);
    cudaFree(d_bins);
    cudaFree(d_triCountsAll);
    for (int i=0; i<p1Cell.size(); ++i) {
        cudaFree(h_p1Cell);
    }
    for (int i=0; i<p2Cell.size(); ++i) {
        cudaFree(h_p2Cell);
    }
    for (int i=0; i<p3Cell.size(); ++i) {
        cudaFree(h_p3Cell);
    }
    cudaFree(d_p2Cell);
    cudaFree(d_p3Cell);
    delete[] h_p1Cell;
    delete[] h_p2Cell;
    delete[] h_p3Cell;
    cudaFree(d_p2CellSize);
    cudaFree(d_p3CellSize);

    int iout = 0;
    for (int ibin1 = 0; ibin1 < numBins_; ++ibin1) {
        for (int ibin2 = ibin1; ibin2 < numBins_; ++ibin2) {
            for (int ibin3 = ibin2; ibin3 < numBins_; ++ibin3) {
                if (binsMid_[ibin3]<=binsMid_[ibin1]+binsMid_[ibin2]) {
                    int triidx=ibin3 + ibin2*numBins_ + ibin1*numBins_*numBins_;
                    std::cout << "Triangle count for bin " << binsMid_[ibin1] << ", "<< binsMid_[ibin2] << ", " << binsMid_[ibin3] << " = " << triCountsAll_[triidx] << std::endl;
                    triCountsOut_[iout]=triCountsAll_[triidx];
                    iout++;
                }
            }
        }
    }
}

//----end host functions----//
