#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <vector>
#include <vector_types.h>
#include <string>
#include <random>

std::vector<double> logspace(double start, double end, int Nedge);

std::vector<double4> generateRandomCatalog(int Np, const double3 &box);

double4 generateRandomPosition(std::mt19937 &gen, const double3 &box);

void writeToFileThreeD(
    const std::string& filename, 
    const std::vector<double3>& double3_vec, 
    const std::vector<double>& double_vec
);

void writeToFileRppi(
    const std::string& filename, 
    const std::vector<double3>& double3_vec, 
    const std::vector<double2>& double2_vec, 
    const std::vector<double>& double_vec
);

std::vector<double4> readCatalogFits(
    const std::string& filename,
    const std::vector<std::string>& column_names,
    int hdu_index,
    double zmin,
    double zmax
);

std::vector<double4> readCatalogTxt(
    const std::string& filename,
    const double3& box,
    const std::vector<int>& pos_columns
);

#endif
