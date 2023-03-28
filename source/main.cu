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
#include "../include/cudaNcorr.hpp"
#include "../include/utils.hpp"
#include "../include/survey.hpp"


int main() {
    std::vector<double> bins;
    for (double i=0.0f; i<=32.0f; i+=1.0f) {
        bins.push_back(i);
    }
    double3 box={1024.f,1024.f,1024.f};


    std::string inpath = "/global/homes/h/hanyuz/cudaNcorr/data/testData.txt";
    std::vector<double4> p1 = readCatalogTxt(inpath,box,{0,1,2,3});
    std::vector<double4> p2 = readCatalogTxt(inpath,box,{0,1,2,3});
    std::vector<double4> p3 = readCatalogTxt(inpath,box,{0,1,2,3});

    ncorr corr(bins,box);
    corr.calculateD1D2D3(p1, p2, p3);

    std::string outpath = "/global/homes/h/hanyuz/cudaNcorr/DDDtest.dat";
    std::vector<double3> triBins = corr.getTriBins();
    std::vector<double> triCountsOut = corr.getTriCountsOut();
    writeToFile(outpath,triBins,triCountsOut);


    return 0;
}
