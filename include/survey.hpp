// Survey.hpp

#ifndef SURVEY_HPP
#define SURVEY_HPP

#include <vector>
#include <vector_types.h>

// Define the cosmological parameters
struct Cosmology {
    double H0;
    double OmegaM;
    double OmegaLambda;
};

class Survey {
public:
    Survey(const Cosmology &cosmo);

    std::vector<double4> convert_coordinates(const std::vector<double4> &catalog);

private:
    static double integrand(double a, void *params);

    double comoving_distance(double z);

    Cosmology cosmo_;
};

#endif // SURVEY_HPP

