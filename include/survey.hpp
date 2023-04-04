// survey.hpp

#ifndef SURVEY_HPP
#define SURVEY_HPP

#include <vector>
#include <vector_types.h>

// Define the cosmological parameters
struct cosmology {
    double H0;
    double OmegaM;
    double OmegaLambda;
};

class survey {
public:
    survey(const cosmology &cosmo);

    std::vector<double4> convert_coordinates(const std::vector<double4> &catalog);

    void prep_cat_survey(std::vector<double4> &catalog, double3 &box, double3 &ori);
    void prep_cat_survey(std::vector<double4> &catalog1, std::vector<double4> &catalog2, double3 &box, double3 &ori);

private:
    static double integrand(double a, void *params);

    double comoving_distance(double z);

    void find_min_max_elements(const std::vector<double4>& data, double3 &catMin, double3 &catMax);

    void shift_catalog(std::vector<double4> &catalog, double3 catMin);

    cosmology cosmo_;
};

#endif // SURVEY_HPP

