#ifndef _CUDANCORR_HPP_
#define _CUDANCORR_HPP_

#include <vector>
#include <vector_types.h>
#include <random>
#include <string>

class ncorr{
    public:
        //Constructor
        ncorr(std::vector<double> r_bins, double3 box_size);

        //public class interface

        void calculateD1D2D3(std::vector<double4> &p1, std::vector<double4> &p2, std::vector<double4> &p3);

        std::vector<double3> getTriBins() const;

        std::vector<double> getTriCountsOut() const;

    private:
        //private data members
        std::vector<double> bins_,binsMid_;
        double rMin_,rMax_;
        int numBins_, numBinsCounts_;
        double3 box_, cellSize_;
        int3 numCells_;
        std::vector<int3> shifts_;
        std::vector<double3> triBins_;
        std::vector<double> triCountsAll_;
        std::vector<double> triCountsOut_;

        //private member functions
        void rezeroVectors();

        std::vector<double> getBinsMid(const std::vector<double> &bins);

        std::vector<int3> getShifts();

        int getCellIndex(const double4 &p);

        void gridParticles(int Np, std::vector<double4> &particles, std::vector<std::vector<double4>> &particle_cells);
};


#endif
