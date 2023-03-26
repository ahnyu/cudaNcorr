#ifndef _CUDANCORR_HPP_
#define _CUDANCORR_HPP_

#include <vector>
#include <vector_types.h>
#include <random>
#include <string>

class ncorr{
    public:
        //Constructor
        ncorr(std::vector<float> r_bins, float3 box_size);

        //public class interface
        std::vector<float3> triBins;
        std::vector<int> triCountsOut;

        std::vector<float3> generateRandomCatalog(int Np);

        std::vector<float3> readCatalogFromFile(const std::string& filename);

        void calculateD1D2D3(std::vector<float3> &p1, std::vector<float3> &p2, std::vector<float3> &p3);

        void writeToFile(const std::string& filename, const std::vector<float3>& float3_vec, const std::vector<int>& int_vec);

    private:
        //private data members
        std::vector<float> bins,binsMid;
        float rMin,rMax;
        int numBins, numBinsCounts;
        float3 box, cellSize;
        int3 numCells;
        std::vector<int3> shifts;
//        std::vector<float3> triBins;
        std::vector<int> triCountsAll;
//        std::vector<int> triCountsOut;

        //private member functions
//        void initVectors();

        void rezeroVectors();

        std::vector<float> getBinsMid(const std::vector<float> &bins);

        std::vector<int3> getShifts();

        float3 generateRandomPosition(std::mt19937 &gen);

        int getCellIndex(const float3 &pos);

        void gridParticles(int Np, std::vector<float3> &particles, std::vector<std::vector<float3>> &particle_cells);
};


#endif
