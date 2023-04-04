#ifndef _CUDANCORR_HPP_
#define _CUDANCORR_HPP_

#include <vector>
#include <vector_types.h>
#include <random>
#include <string>
#include "../include/device.hpp"

class ncorr{
    public:
        enum binType {threeD, rppi, smu};
        //Constructor
        ncorr(double3 box, int bin_type, std::vector<double> &bins, bool isSurvey, double3 ori);
        ncorr(double3 box, int bin_type, std::vector<double> &bins, int numMinorBins, bool isSurvey, double3 ori);

        //public class interface

        void calculateD1D2D3(std::vector<double4> &p1, std::vector<double4> &p2, std::vector<double4> &p3);

        std::vector<double3> getTriBinsOut() const;

        std::vector<double2> getPiBinsOut() const;

        std::vector<double> getTriCountsOut() const;

    private:
        //private data members
        binType binType_;
        bool isSurvey_;
        std::vector<double> bins_,binsMid_;
        std::vector<double> sqrBins_;
        std::vector<double> piBins_, muBins_;
        double rMin_,rMax_;
        double rpMin_,rpMax_;
        double piMax_;
//        double piMin_;
        double sqrRMin_, sqrRMax_;
        double sqrRpMin_, sqrRpMax_;
        double sqrPiMax_;
        int numBins_, numBinsCounts_;
        int numMinorBins_;
        int numPiBins_, numMuBins_;
        double3 box_, cellSize_, ori_;
        int3 numCells_;
        std::vector<int3> shifts_;
        std::vector<double3> triBinsOut_;
        std::vector<double2> piBinsOut_;
        std::vector<double> triCountsAll_;
        std::vector<double> triCountsOut_;

        //private member functions
        void setBins(std::vector<double> &bin);

        void rezeroVectors();

        std::vector<double> getBinsMid(const std::vector<double> &bins);

        std::vector<int3> getShifts();

        int getCellIndex(const double4 &p);

        void gridParticles(int Np, std::vector<double4> &particles, std::vector<std::vector<double4>> &particle_cells);

        void prepTriCountsOutThreeD();

        void prepTriCountsOutRppi();
};


#endif
