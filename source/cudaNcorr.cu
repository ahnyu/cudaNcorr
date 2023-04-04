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
#include "../include/device.hpp"
#include "../include/cudaNcorr.hpp"
#include "../include/gpuerrchk.h"

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

void ncorr::prepTriCountsOutThreeD() {
    int iout = 0;

    for (int ibin1 = 0; ibin1 < numBins_; ++ibin1) {
        for (int ibin2 = ibin1; ibin2 < numBins_; ++ibin2) {
            for (int ibin3 = ibin2; ibin3 < numBins_; ++ibin3) {
                if (binsMid_[ibin3]<=binsMid_[ibin1]+binsMid_[ibin2]) {
                    int triidx=ibin3 + ibin2*numBins_ + ibin1*numBins_*numBins_;
                    triCountsOut_[iout]=triCountsAll_[triidx];
                    iout++;
                }
            }
        }
    }
}

void ncorr::prepTriCountsOutRppi() {
    int iout = 0;
    for (int i=0; i<numBins_; ++i) {
        for (int j=i; j<numBins_; ++j) {
            for (int k=j; k<numBins_; ++k) {
                if(binsMid_[k]<=binsMid_[i]+binsMid_[j]) {
                    for(int l=0; l<numPiBins_; ++l) {
                        for(int m=0; m<numPiBins_; ++m) {
                            int triidx=m+l*numPiBins_+k*numPiBins_*numPiBins_+j*numPiBins_*numPiBins_*numBins_+i*numPiBins_*numPiBins_*numBins_*numBins_;
                            triCountsOut_[iout]=triCountsAll_[triidx];
                            iout++;
                        }
                    }
                }
            }
        }
    }
}
std::vector<double3> ncorr::getTriBinsOut() const {
    return triBinsOut_;
}

std::vector<double2> ncorr::getPiBinsOut() const {
    return piBinsOut_;
}

std::vector<double> ncorr::getTriCountsOut() const {
    return triCountsOut_;
}

ncorr::ncorr(double3 box, int bin_type, std::vector<double> &bins, bool isSurvey, double3 ori={0.0,0.0,0.0}): box_(box), binType_(static_cast<binType>(bin_type)), bins_(bins), isSurvey_(isSurvey), ori_(ori){
    std::cout << "Debug message at line " << __LINE__ << std::endl;
    if (binType_ != threeD) {
        throw std::runtime_error("Invalid constructor call for bin_type.");
    }
    std::cout << "Debug message at line " << __LINE__ << std::endl;
    for (int i=0; i<bins_.size(); ++i) {
        sqrBins_.push_back(bins_[i]*bins_[i]);
        std::cout << "sqrBins "<<i<<"="<<sqrBins_[i]<< std::endl;
    }
    std::cout << "Debug message at line " << __LINE__ << std::endl;

    binsMid_=getBinsMid(bins_);
    rMax_=bins_.back();
    rMin_=bins_[0];
    sqrRMax_=rMax_*rMax_;
    sqrRMin_=rMin_*rMin_;
    numBins_=bins_.size()-1;
    numBinsCounts_=numBins_*numBins_*numBins_;
    cellSize_={bins_.back(),bins_.back(),bins_.back()};
    numCells_.x=static_cast<int>(std::ceil(box_.x / cellSize_.x));
    numCells_.y=static_cast<int>(std::ceil(box_.y / cellSize_.y));
    numCells_.z=static_cast<int>(std::ceil(box_.z / cellSize_.z));
    std::cout << "cellSize = "<<cellSize_.x<<","<<cellSize_.y<<","<<cellSize_.z<<","<<std::endl;
    std::cout << "numCells = "<<numCells_.x<<","<<numCells_.y<<","<<numCells_.z<<","<<std::endl;
    shifts_=getShifts();
    std::cout << "Debug message at line " << __LINE__ << std::endl;
    for (int i=0; i<numBins_; ++i) {
        for (int j=i; j<numBins_; ++j) {
            for (int k=j; k<numBins_; ++k) {
                if(binsMid_[k]<=binsMid_[i]+binsMid_[j]) {
                    triCountsOut_.push_back(0);
                    triBinsOut_.push_back({binsMid_[i],binsMid_[j],binsMid_[k]});
                }
            }
        }
    }
    std::cout << "Debug message at line " << __LINE__ << std::endl;
    triCountsAll_.resize(numBinsCounts_);
}

ncorr::ncorr(double3 box, int bin_type, std::vector<double> &bins, int numMinorBins, bool isSurvey, double3 ori={0.0,0.0,0.0}): box_(box), binType_(static_cast<binType>(bin_type)), bins_(bins), numMinorBins_(numMinorBins), isSurvey_(isSurvey), ori_(ori){
    if (binType_ !=rppi) {
        throw std::runtime_error("Invalid constructor call for bin_type.");
    }
    for (int i=0; i<bins_.size(); ++i) {
        sqrBins_.push_back(bins_[i]*bins_[i]);
    }
    numBins_=bins_.size()-1;
    binsMid_=getBinsMid(bins_);
    rpMax_=bins_.back();
    rpMin_=bins_[0];
    sqrRpMax_=rpMax_*rpMax_;
    sqrRpMin_=rpMin_*rpMin_;

    for (double i=0; i<=static_cast<double>(numMinorBins); i=i+1.0){
        piBins_.push_back(i);
    }
    numPiBins_=piBins_.size()-1;
    if (numPiBins_ != numMinorBins_) {
        throw std::runtime_error("number of piBins do not match input, could be a bug");
    }
    piMax_=piBins_.back();
    sqrPiMax_=piMax_*piMax_;
//    sqrPiMin_=piMin_*piMin_;

    sqrRMax_=rpMax_*rpMax_ + piMax_*piMax_;
    rMax_=sqrtf(sqrRMax_);
//    sqrRMin_=rpMin_*rpMin_ + piMin_*piMin_;

    numBinsCounts_=numBins_*numBins_*numBins_*numPiBins_*numPiBins_;
    if (isSurvey_) {
        cellSize_={rMax_, rMax_, rMax_};
    } else {
        cellSize_={rpMax_, rpMax_, piMax_};
    }
    numCells_.x=static_cast<int>(std::ceil(box_.x / cellSize_.x));
    numCells_.y=static_cast<int>(std::ceil(box_.y / cellSize_.y));
    numCells_.z=static_cast<int>(std::ceil(box_.z / cellSize_.z));

    shifts_=getShifts();

    for (int i=0; i<numBins_; ++i) {
        for (int j=i; j<numBins_; ++j) {
            for (int k=j; k<numBins_; ++k) {
                if(binsMid_[k]<=binsMid_[i]+binsMid_[j]) {
                    for(int l=0; l<numPiBins_; ++l) {
                        for(int m=0; m<numPiBins_; ++m) {
                            triCountsOut_.push_back(0);
                            triBinsOut_.push_back({binsMid_[i],binsMid_[j],binsMid_[k]});
                            piBinsOut_.push_back({piBins_[l],piBins_[m]});
                        }
                    }
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
    printf("sqrRMax = %f\n", sqrRMax_);

    gpuErrchk(cudaMemcpyToSymbol(d_sqrRMax, &sqrRMax_, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(d_sqrRMin, &sqrRMin_, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(d_cellSize, &cellSize_, sizeof(double3)));
    gpuErrchk(cudaMemcpyToSymbol(d_box, &box_, sizeof(double3)));
    gpuErrchk(cudaMemcpyToSymbol(d_numCells, &numCells_, sizeof(int3)));
    gpuErrchk(cudaMemcpyToSymbol(d_shifts, shifts_.data(), shifts_.size()*sizeof(int3)));

    double *devPtr;
    cudaError_t err = cudaGetSymbolAddress((void **)&devPtr, d_sqrRMax);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetSymbolAddress error: %s\n", cudaGetErrorString(err));
    }
    double value;
    err = cudaMemcpy(&value, devPtr, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy error: %s\n", cudaGetErrorString(err));
    }
    printf("The value of d_sqrRMax on the device is %f\n", value);

    if (binType_==rppi) {
        cudaMemcpyToSymbol(d_sqrRpMax, &sqrRpMax_, sizeof(double));
        cudaMemcpyToSymbol(d_sqrRpMin, &sqrRpMin_, sizeof(double));
        cudaMemcpyToSymbol(d_sqrPiMax, &sqrPiMax_, sizeof(double));
    };
    
    if (isSurvey_) {
        cudaMemcpyToSymbol(d_ori, &ori_, sizeof(double));
    }

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

//    double *d_bins;
    double *d_sqrBins;
    double *d_triCountsAll;

//    cudaMalloc((void **)&d_bins, bins_.size()*sizeof(double));
//    cudaMemcpy(d_bins, bins_.data(), bins_.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_sqrBins, sqrBins_.size()*sizeof(double));
    cudaMemcpy(d_sqrBins, sqrBins_.data(), sqrBins_.size()*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_triCountsAll, numBinsCounts_ * sizeof(double));
    cudaMemcpy(d_triCountsAll, triCountsAll_.data(), triCountsAll_.size()*sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 64;
    int gridSize = (Np1 + blockSize - 1) / blockSize;

    if(binType_==threeD) {
        if (isSurvey_) {
            std::cout << "Debug message at line " << __LINE__ << std::endl;
            countTrianglesThreeDSurvey<<<gridSize, blockSize>>>(
                    d_p1,
                    Np1,
                    d_p2Cell,
                    d_p3Cell,
                    d_p2CellSize,
                    d_p3CellSize,
                    d_sqrBins,
                    numBins_,
                    d_triCountsAll);
        } else {
            std::cout << "Debug message at line " << __LINE__ << std::endl;
            countTrianglesThreeDBox<<<gridSize, blockSize>>>(
                    d_p1,
                    Np1,
                    d_p2Cell,
                    d_p3Cell,
                    d_p2CellSize,
                    d_p3CellSize,
                    d_sqrBins,
                    numBins_,
                    d_triCountsAll);
        }
    } 
    if(binType_==rppi) {
        if (isSurvey_) {
            std::cout << "Debug message at line " << __LINE__ << std::endl;
            countTrianglesRppiSurvey<<<gridSize, blockSize>>>(
                    d_p1,
                    Np1,
                    d_p2Cell,
                    d_p3Cell,
                    d_p2CellSize,
                    d_p3CellSize,
                    d_sqrBins,
                    numBins_,
                    numPiBins_,
                    d_triCountsAll);

        } else {
            std::cout << "Debug message at line " << __LINE__ << std::endl;
            countTrianglesRppiBox<<<gridSize, blockSize>>>(
                    d_p1,
                    Np1,
                    d_p2Cell,
                    d_p3Cell,
                    d_p2CellSize,
                    d_p3CellSize,
                    d_sqrBins,
                    numBins_,
                    numPiBins_,
                    d_triCountsAll);
        }
    }
        

    cudaDeviceSynchronize();

    cudaMemcpy(triCountsAll_.data(), d_triCountsAll, numBinsCounts_ * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_p1);
//    cudaFree(d_bins);
    cudaFree(d_sqrBins);
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

    if(binType_==threeD) prepTriCountsOutThreeD();
    if(binType_==rppi) prepTriCountsOutRppi();
//    int iout = 0;
//    
//    for (int ibin1 = 0; ibin1 < numBins_; ++ibin1) {
//        for (int ibin2 = ibin1; ibin2 < numBins_; ++ibin2) {
//            for (int ibin3 = ibin2; ibin3 < numBins_; ++ibin3) {
//                if (binsMid_[ibin3]<=binsMid_[ibin1]+binsMid_[ibin2]) {
//                    int triidx=ibin3 + ibin2*numBins_ + ibin1*numBins_*numBins_;
////                    std::cout << "Triangle count for bin " << binsMid_[ibin1] << ", "<< binsMid_[ibin2] << ", " << binsMid_[ibin3] << " = " << triCountsAll_[triidx] << std::endl;
//                    triCountsOut_[iout]=triCountsAll_[triidx];
//                    iout++;
//                }
//            }
//        }
//    }
}

//----end host functions----//
