#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <omp.h>
#include <random>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <vector_types.h>
#include "../include/CudaNcorr.hpp"


int main() {
    std::vector<float> r_bins;
    for (float i=0.0f; i<=32.0f; i+=1.0f) {
        r_bins.push_back(i);
    }
    float3 box_size={1024.f,1024.f,1024.f};

    ncorr corr(r_bins,box_size);

    std::string inpath = "/global/homes/h/hanyuz/cudaNcorr/LNKNLogsVelFortran_01.dat";
    std::vector<float3> p1 = corr.readCatalogFromFile(inpath);
    std::vector<float3> p2 = corr.readCatalogFromFile(inpath);
    std::vector<float3> p3 = corr.readCatalogFromFile(inpath);

    corr.calculateD1D2D3(p1, p2, p3);

    std::string outpath = "/global/homes/h/hanyuz/cudaNcorr/DDDtest.dat";
    corr.writeToFile(outpath,corr.triBins,corr.triCountsOut);


    return 0;
}
