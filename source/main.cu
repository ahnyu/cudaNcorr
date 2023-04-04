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
#include "../include/device.hpp"
#include "../include/cudaNcorr.hpp"
#include "../include/utils.hpp"
#include "../include/survey.hpp"


int main() {
    std::vector<double> bins = logspace(0.1, 20.0, 13);
//    for (double i=0.0f; i<=32.0f; i+=1.0f) {
//        bins.push_back(i);
//    }
//----example code for mock catalog in a box----//
    double3 box={2000.0,2000.0,2000.0};
//
    std::string datPath = "/global/homes/h/hanyuz/cudaNcorr/data/c000_mock.dat";
    std::vector<double4> p1 = readCatalogTxt(datPath,box,{0,1,2,3});
    std::vector<double4> r1 = generateRandomCatalog(20000000,box);

    ncorr corr(box,0,bins,false,{0.0,0.0,0.0});
    std::vector<double3> triBinsOut = corr.getTriBinsOut();
//    std::vector<double2> piBins = corr.getPiBinsOut();
    std::vector<double> triCountsOut;

    corr.calculateD1D2D3(p1, p1, p1);
    std::string DDDPath = "/global/homes/h/hanyuz/cudaNcorr/DDDMock3D.dat";
    triCountsOut = corr.getTriCountsOut();
    writeToFileThreeD(DDDPath,triBinsOut,triCountsOut);
//    writeToFileRppi(DDDPath,triBinsOut,piBins,triCountsOut);
    corr.calculateD1D2D3(r1, r1, r1);
    std::string RRRPath = "/global/homes/h/hanyuz/cudaNcorr/RRRMock3D.dat";
    triCountsOut = corr.getTriCountsOut();
    writeToFileThreeD(RRRPath,triBinsOut,triCountsOut);
//    writeToFileRppi(RRRPath,triBinsOut,piBins,triCountsOut);

//----end example code for mock catalog in a box----//
    
//----example code for survey catalog----//
    
//    std::string lrgDatN = "/global/cfs/cdirs/desi/survey/catalogs/edav1/sv3/LSScats/clustering/LRG_main_N_clustering.dat.fits";
//    std::string lrgRanN = "/global/cfs/cdirs/desi/survey/catalogs/edav1/sv3/LSScats/clustering/LRG_main_N_0_clustering.ran.fits";
//    std::vector<double4> lrgDatN_radecz = readCatalogFits(lrgDatN,{"RA","DEC","Z","WEIGHT"},2, 0.4, 0.6);
//    std::vector<double4> lrgRanN_radecz = readCatalogFits(lrgRanN,{"RA","DEC","Z","WEIGHT"},2, 0.4, 0.6);
//
//    cosmology fiducial{67.36, 0.313772,0.686228};
//
//    survey fid(fiducial);
//
//    std::vector<double4> lrgDatN_cartesian = fid.convert_coordinates(lrgDatN_radecz);
//    std::vector<double4> lrgRanN_cartesian = fid.convert_coordinates(lrgRanN_radecz);
//
//    double3 box,ori;
//
//    fid.prep_cat_survey(lrgDatN_cartesian, lrgRanN_cartesian, box, ori);
//
//    std::string dddout = "/global/homes/h/hanyuz/cudaNcorr/data/DDD3D_lrg_z0.4_0.6.dat";
//    std::string ddrout = "/global/homes/h/hanyuz/cudaNcorr/data/DDR3D_lrg_z0.4_0.6.dat";
//    std::string drrout = "/global/homes/h/hanyuz/cudaNcorr/data/DRR3D_lrg_z0.4_0.6.dat";
//    std::string rrrout = "/global/homes/h/hanyuz/cudaNcorr/data/RRR3D_lrg_z0.4_0.6.dat";
//    ncorr corr(box,0,bins,true,ori);
//    std::vector<double3> triBins = corr.getTriBinsOut();
////    std::vector<double2> piBins = corr.getPiBinsOut();
//
//    std::vector<double> triCountsOut;
//
//    corr.calculateD1D2D3(lrgDatN_cartesian,lrgDatN_cartesian,lrgDatN_cartesian);
//    triCountsOut = corr.getTriCountsOut();
//    writeToFileThreeD(dddout,triBins,triCountsOut);
////    writeToFileRppi(dddout,triBins,piBins,triCountsOut);
//
//    corr.calculateD1D2D3(lrgDatN_cartesian,lrgDatN_cartesian,lrgRanN_cartesian);
//    triCountsOut = corr.getTriCountsOut();
//    writeToFileThreeD(ddrout,triBins,triCountsOut);
////    writeToFileRppi(ddrout,triBins,piBins,triCountsOut);
//
//    corr.calculateD1D2D3(lrgDatN_cartesian,lrgRanN_cartesian,lrgRanN_cartesian);
//    triCountsOut = corr.getTriCountsOut();
//    writeToFileThreeD(drrout,triBins,triCountsOut);
////    writeToFileRppi(drrout,triBins,piBins,triCountsOut);
//
//    corr.calculateD1D2D3(lrgRanN_cartesian,lrgRanN_cartesian,lrgRanN_cartesian);
//    triCountsOut = corr.getTriCountsOut();
//    writeToFileThreeD(rrrout,triBins,triCountsOut);
//    writeToFileRppi(rrrout,triBins,piBins,triCountsOut);
//----end example code for survey catalog----//


    return 0;
}
