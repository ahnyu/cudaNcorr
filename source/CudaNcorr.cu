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
#include "../include/CudaNcorr.hpp"

//----device functions----//
__constant__ int3 d_shifts[27];
__constant__ float3 d_box;
__constant__ float d_rMax;
__constant__ float d_rMin;
__constant__ int3 d_numCells;
__constant__ float3 d_cellSize;

__device__ float distance(float3 p1, float3 p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;

    return sqrtf(dx * dx + dy * dy + dz * dz);
}

__device__ int4 initCellIndex(float3 particle) {
    int4 cellIndex={int(particle.x/d_cellSize.x), int(particle.y/d_cellSize.y), int(particle.z/d_cellSize.z), 0};
    if(cellIndex.x == d_numCells.x) cellIndex.x--;
    if(cellIndex.y == d_numCells.y) cellIndex.y--;
    if(cellIndex.z == d_numCells.z) cellIndex.z--;
    return cellIndex;
}

__device__ int4 shiftCellIndex(int4 p_cell, int i, float3 &rShift) {
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
    float r1,
    float r2,
    float r3,
    const float *bins,
    int numBins
) {
    // Check if the given lengths can form a triangle
    if (r1 + r2 <= r3 || r1 + r3 <= r2 || r2 + r3 <= r1) {
        return -1; // The lengths cannot form a triangle
    }

    // Sort the lengths in ascending order
    float temp;
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
    float3 *d_p1,
    int Np1,
    float3 **d_p2Cell,
    float3 **d_p3Cell,
    int *d_p2CellSize,
    int *d_p3CellSize,
    float *d_bins,
    int numBins,
    int *d_triangle_counts
) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx1 < Np1) {
        float3 particle1=d_p1[idx1];
        int4 p1_cell=initCellIndex(particle1);
        for (int nci1 = 0; nci1 < 27; ++nci1) {
            float3 rShift2;
            int4 p2_cell = shiftCellIndex(p1_cell,nci1,rShift2);
            int cell_counts2 = d_p2CellSize[p2_cell.w];

            for (int idx2 = 0; idx2 < cell_counts2; ++idx2) {
                float3 particle2 = d_p2Cell[p2_cell.w][idx2];
                particle2.x += rShift2.x;
                particle2.y += rShift2.y;
                particle2.z += rShift2.z;
                float r1 = distance(particle1, particle2);

                if(r1 < d_rMax && r1 > d_rMin) {
                    for (int nci2 = 0; nci2 < 27; ++nci2) {
                        float3 rShift3;
                        int4 p3_cell = shiftCellIndex(p1_cell,nci2,rShift3);
                        int cell_counts3 = d_p3CellSize[p3_cell.w];

                        for (int idx3 = 0; idx3 < cell_counts3; ++idx3) {
                            float3 particle3 = d_p3Cell[p3_cell.w][idx3];
                            particle3.x += rShift3.x;
                            particle3.y += rShift3.y;
                            particle3.z += rShift3.z;

                            float r2 = distance(particle1, particle3);
                            float r3 = distance(particle2, particle3);
                            if(r2 < d_rMax && r3 < d_rMax && r2 > d_rMin && r3 > d_rMin) {
                                int triangle_index = getTriangleIndex(r1, r2, r3, d_bins, numBins);

                                if (triangle_index != -1) {
                                    atomicAdd(&d_triangle_counts[triangle_index], 1);
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
    for (int i = 0; i < ncorr::triCountsAll.size(); ++i) {
        ncorr::triCountsAll[i] = 0;
    }
}

std::vector<float> ncorr::getBinsMid(const std::vector<float>& bins) {
    std::vector<float> binsMid;

    for (size_t i = 1; i < bins.size(); ++i) {
        float mid = (bins[i] + bins[i - 1]) / 2.0;
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

int ncorr::getCellIndex(const float3& pos) {
    int x_cell = static_cast<int>(pos.x / ncorr::cellSize.x);
    if(x_cell == numCells.x) x_cell--;
    int y_cell = static_cast<int>(pos.y / ncorr::cellSize.y);
    if(y_cell == numCells.y) y_cell--;
    int z_cell = static_cast<int>(pos.z / ncorr::cellSize.z);
    if(z_cell == numCells.z) z_cell--;
    return z_cell + ncorr::numCells.z*(y_cell + ncorr::numCells.y*x_cell);
}

void ncorr::gridParticles(int Np, std::vector<float3> &particles, std::vector<std::vector<float3>> &particle_cells) {
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

float3 ncorr::generateRandomPosition(std::mt19937& gen) {
    std::uniform_real_distribution<float> distx(0, ncorr::box.x);
    std::uniform_real_distribution<float> disty(0, ncorr::box.y);
    std::uniform_real_distribution<float> distz(0, ncorr::box.z);
    float3 position;
    position.x = distx(gen);
    position.y = disty(gen);
    position.z = distz(gen);
    return position;
}

std::vector<float3> ncorr::generateRandomCatalog(int Np) {
    std::vector<float3> p(Np);
    #pragma omp parallel
    {
        std::mt19937 gen(omp_get_thread_num() + static_cast<unsigned>(time(0)));
        #pragma omp for
        for (int i = 0; i < Np; i++) {
            p[i] = generateRandomPosition(gen);
        }
    }
    return p;
}


std::vector<float3> ncorr::readCatalogFromFile(const std::string& filename) {
    std::vector<float3> p;
    std::ifstream input_file(filename);

    if (!input_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::string line;
    size_t line_number = 0;

    while (std::getline(input_file, line)) {
        ++line_number;
        std::istringstream line_stream(line);
        float3 temp;
        line_stream >> temp.x >> temp.y >> temp.z;

        if (temp.x < 0 || temp.y < 0 || temp.z < 0 || temp.x > ncorr::box.x || temp.y > ncorr::box.y || temp.z > ncorr::box.z) {
            input_file.close();
            throw std::runtime_error("Error at line " + std::to_string(line_number) + ": float3 value is out of bounds.");
        }
        p.push_back(temp);
    }

    input_file.close();
    return p;
}

void ncorr::writeToFile(const std::string& filename, const std::vector<float3>& float3_vec, const std::vector<int>& int_vec) {
    if (float3_vec.size() != int_vec.size()) {
        throw std::runtime_error("Vectors have different sizes.");
    }

    std::ofstream output_file(filename);
    
    if (!output_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    for (size_t i = 0; i < float3_vec.size(); ++i) {
        output_file << float3_vec[i].x << " " << float3_vec[i].y << " " << float3_vec[i].z << " " << int_vec[i] << std::endl;
    }

    output_file.close();
}


ncorr::ncorr(std::vector<float> r_bins, float3 box_size){
    ncorr::bins=r_bins;
    ncorr::binsMid=getBinsMid(ncorr::bins);
    ncorr::rMax=r_bins.back();
    ncorr::rMin=r_bins[0];
    ncorr::box=box_size;
    ncorr::numBins=bins.size()-1;
    ncorr::numBinsCounts=numBins*numBins*numBins;
    ncorr::cellSize={bins.back(),bins.back(),bins.back()};
    ncorr::numCells.x=static_cast<int>(std::ceil(box.x / cellSize.x));
    ncorr::numCells.y=static_cast<int>(std::ceil(box.y / cellSize.y));
    ncorr::numCells.z=static_cast<int>(std::ceil(box.z / cellSize.z));
    ncorr::shifts=getShifts();
    for (int i=0; i<ncorr::numBins; ++i) {
        for (int j=i; j<ncorr::numBins; ++j) {
            for (int k=j; k<ncorr::numBins; ++k) {
                if(ncorr::binsMid[k]<=ncorr::binsMid[i]+ncorr::binsMid[j]) {
                    ncorr::triCountsOut.push_back(0);
                    ncorr::triBins.push_back({ncorr::binsMid[i],ncorr::binsMid[j],ncorr::binsMid[k]});
                }
            }
        }
    }
    ncorr::triCountsAll.resize(ncorr::numBinsCounts);
}

void ncorr::calculateD1D2D3(std::vector<float3> &p1, std::vector<float3> &p2, std::vector<float3> &p3) {

    rezeroVectors();

    float h_rMax = ncorr::rMax;
    float h_rMin = ncorr::rMin;
    float3 h_cellSize = ncorr::cellSize;
    float3 h_box = ncorr::box;
    int3 h_numCells = ncorr::numCells;

    std::cout <<"cellsizex = "<< ncorr::cellSize.x <<"cellsizey "<< ncorr::cellSize.y <<"cellsizez = "<<ncorr::cellSize.z<< std::endl;
    std::cout <<"numCellsx = "<< ncorr::numCells.x <<"numCellsy "<< ncorr::numCells.y <<"numCellsz = "<<ncorr::numCells.z<< std::endl;

    cudaMemcpyToSymbol(d_rMax, &h_rMax, sizeof(float));
    cudaMemcpyToSymbol(d_rMin, &h_rMin, sizeof(float));
    cudaMemcpyToSymbol(d_cellSize, &h_cellSize, sizeof(float3));
    cudaMemcpyToSymbol(d_box, &h_box, sizeof(float3));
    cudaMemcpyToSymbol(d_numCells, &h_numCells, sizeof(int3));
    cudaMemcpyToSymbol(d_shifts, ncorr::shifts.data(), ncorr::shifts.size()*sizeof(int3));


    int Np1=p1.size();
    int Np2=p2.size();
    int Np3=p3.size();

    std::cout <<"Np1 = "<< Np1 <<"Np2 = "<< Np2 <<"Np3 = "<<Np3<< std::endl;

    std::vector<std::vector<float3>> p1Cell(ncorr::numCells.x * ncorr::numCells.y * ncorr::numCells.z);
    std::vector<std::vector<float3>> p2Cell(ncorr::numCells.x * ncorr::numCells.y * ncorr::numCells.z);
    std::vector<std::vector<float3>> p3Cell(ncorr::numCells.x * ncorr::numCells.y * ncorr::numCells.z);

    float3 **h_p1Cell = (float3 **)malloc(p1Cell.size()*sizeof(float3 *));
    float3 **h_p2Cell = (float3 **)malloc(p2Cell.size()*sizeof(float3 *));
    float3 **h_p3Cell = (float3 **)malloc(p3Cell.size()*sizeof(float3 *));

    std::vector<int> p1CellSize;
    std::vector<int> p2CellSize;
    std::vector<int> p3CellSize;

    gridParticles(Np1,p1,p1Cell);
    gridParticles(Np2,p2,p2Cell);
    gridParticles(Np3,p3,p3Cell);

    for (int i=0; i < p1Cell.size(); ++i) {
        p1CellSize.push_back(p1Cell[i].size());
        cudaMalloc((void **)&h_p1Cell[i], p1Cell[i].size()*sizeof(float3));
        cudaMemcpy(h_p1Cell[i], p1Cell[i].data(), p1Cell[i].size()*sizeof(float3), cudaMemcpyHostToDevice);
    }

    for (int i=0; i < p2Cell.size(); ++i) {
        p2CellSize.push_back(p2Cell[i].size());
        cudaMalloc((void **)&h_p2Cell[i], p2Cell[i].size()*sizeof(float3));
        cudaMemcpy(h_p2Cell[i], p2Cell[i].data(), p2Cell[i].size()*sizeof(float3), cudaMemcpyHostToDevice);
    }

    for (int i=0; i < p3Cell.size(); ++i) {
        p3CellSize.push_back(p3Cell[i].size());
        cudaMalloc((void **)&h_p3Cell[i], p3Cell[i].size()*sizeof(float3));
        cudaMemcpy(h_p3Cell[i], p3Cell[i].data(), p3Cell[i].size()*sizeof(float3), cudaMemcpyHostToDevice);
    }

    std::cout <<"Np1 = "<< Np1 <<"Np2 = "<< Np2 <<"Np3 = "<<Np3<< std::endl;

    float3 *d_p1;
    float3 **d_p2Cell;
    float3 **d_p3Cell;
    int *d_p2CellSize;
    int *d_p3CellSize;
    cudaMalloc((void **)&d_p1, Np1 * sizeof(float3));
    cudaMemcpy(d_p1, p1.data(), Np1 * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&d_p2Cell,p2Cell.size()*sizeof(float3 *));
    cudaMemcpy(d_p2Cell,h_p2Cell,p2Cell.size()*sizeof(float3 *), cudaMemcpyHostToDevice);
    cudaMalloc(&d_p3Cell,p3Cell.size()*sizeof(float3 *));
    cudaMemcpy(d_p3Cell,h_p3Cell,p3Cell.size()*sizeof(float3 *), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_p2CellSize, p2CellSize.size()*sizeof(int));
    cudaMemcpy(d_p2CellSize,p2CellSize.data(),p2CellSize.size()*sizeof(int),cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_p3CellSize, p3CellSize.size()*sizeof(int));
    cudaMemcpy(d_p3CellSize,p3CellSize.data(),p3CellSize.size()*sizeof(int),cudaMemcpyHostToDevice);

    float *d_bins;
    int *d_triCountsAll;

    cudaMalloc((void **)&d_bins, ncorr::bins.size()*sizeof(float));
    cudaMemcpy(d_bins, ncorr::bins.data(), ncorr::bins.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_triCountsAll, ncorr::numBinsCounts * sizeof(int));
    cudaMemcpy(d_triCountsAll, ncorr::triCountsAll.data(), ncorr::triCountsAll.size()*sizeof(int), cudaMemcpyHostToDevice);

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
            ncorr::numBins,
            d_triCountsAll);

    cudaDeviceSynchronize();

    cudaMemcpy(triCountsAll.data(), d_triCountsAll, ncorr::numBinsCounts * sizeof(int), cudaMemcpyDeviceToHost);

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
    for (int ibin1 = 0; ibin1 < ncorr::numBins; ++ibin1) {
        for (int ibin2 = ibin1; ibin2 < ncorr::numBins; ++ibin2) {
            for (int ibin3 = ibin2; ibin3 < ncorr::numBins; ++ibin3) {
                if (ncorr::binsMid[ibin3]<=ncorr::binsMid[ibin1]+ncorr::binsMid[ibin2]) {
                    int triidx=ibin3 + ibin2*ncorr::numBins + ibin1*ncorr::numBins*ncorr::numBins;
                    std::cout << "Triangle count for bin " << ncorr::binsMid[ibin1] << ", "<< ncorr::binsMid[ibin2] << ", " << ncorr::binsMid[ibin3] << " = " << triCountsAll[triidx] << std::endl;
                    ncorr::triCountsOut[iout]=ncorr::triCountsAll[triidx];
                    iout++;
                }
            }
        }
    }
}

//----end host functions----//
