// survey.cpp

#include "../include/survey.hpp"
#include <vector>
#include <vector_types.h>
#include <cmath>
#include <omp.h>
#include <limits>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

constexpr double SPEED_OF_LIGHT = 299792.458; // Speed of light in km/s
constexpr double DEG_TO_RAD = M_PI / 180.0; // degr

survey::survey(const cosmology &cosmo) : cosmo_(cosmo) {};

double survey::integrand(double z, void *params) {
    cosmology *cosmo = static_cast<cosmology *>(params);
//    double H0 = cosmo->H0;
    double OmegaM = cosmo->OmegaM;
    double OmegaLambda = cosmo->OmegaLambda;

    return 1 / sqrt(OmegaM * pow((1+z), 3) + OmegaLambda);
}

double survey::comoving_distance(double z) {
    gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(1000);
    gsl_function F;
    F.function = &survey::integrand;
    F.params = &cosmo_;

    double result, error;
    gsl_integration_qags(&F, 0, z, 0, 1e-7, 1000, workspace, &result, &error);
    gsl_integration_workspace_free(workspace);

    return SPEED_OF_LIGHT * result / 100.0; // Divide by 100 to get the output in Mpc/h
}

std::vector<double4> survey::convert_coordinates(const std::vector<double4> &catalog) {
    std::vector<double4> cartesian_coords(catalog.size());
    #pragma omp parallel for
    for (size_t i = 0; i < catalog.size(); ++i) {
        double ra = catalog[i].x * DEG_TO_RAD; // Convert RA from degrees to radians
        double dec = catalog[i].y * DEG_TO_RAD; // Convert Dec from degrees to radians
        double z = catalog[i].z;

        double d = comoving_distance(z);
        double x = d * cos(dec) * cos(ra);
        double y = d * cos(dec) * sin(ra);
        double z_cart = d * sin(dec);
        double weight = catalog[i].w;

        cartesian_coords[i] = make_double4(x, y, z_cart,weight);
    }

    return cartesian_coords;
}


void survey::find_min_max_elements(const std::vector<double4>& data, double3 &catMin, double3 &catMax) {
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();

    #pragma omp parallel for reduction(min : min_x, min_y, min_z) reduction(max : max_x, max_y, max_z)
    for (size_t i = 0; i < data.size(); ++i) {
        const double4& element = data[i];
        if (element.x < min_x) min_x = element.x;
        if (element.y < min_y) min_y = element.y;
        if (element.z < min_z) min_z = element.z;

        if (element.x > max_x) max_x = element.x;
        if (element.y > max_y) max_y = element.y;
        if (element.z > max_z) max_z = element.z;
    }

    catMin = make_double3(min_x, min_y, min_z);
    catMax = make_double3(max_x, max_y, max_z);
}


void survey::shift_catalog(std::vector<double4> &catalog, double3 catMin) {
    #pragma omp parallel for
    for (size_t i = 0; i < catalog.size(); ++i) {
        catalog[i].x -= catMin.x;
        catalog[i].y -= catMin.y;
        catalog[i].z -= catMin.z;
    }
}

void survey::prep_cat_survey(std::vector<double4> &catalog, double3 &box, double3 &ori) {
    double3 catMin, catMax;
    find_min_max_elements(catalog, catMin, catMax);
    box.x=catMax.x-catMin.x+1.0;
    box.y=catMax.y-catMin.y+1.0;
    box.z=catMax.z-catMin.z+1.0;
    ori.x=-catMin.x;
    ori.y=-catMin.y;
    ori.z=-catMin.z;
    shift_catalog(catalog, catMin);
}

void survey::prep_cat_survey(std::vector<double4> &catalog1, std::vector<double4> &catalog2, double3 &box, double3 &ori) {
    double3 cat1Min, cat1Max;
    double3 cat2Min, cat2Max;
    double3 catMin, catMax;
    find_min_max_elements(catalog1, cat1Min, cat1Max);
    find_min_max_elements(catalog2, cat2Min, cat2Max);

    catMin.x=std::min(cat1Min.x,cat2Min.x);
    catMin.y=std::min(cat1Min.y,cat2Min.y);
    catMin.z=std::min(cat1Min.z,cat2Min.z);
    catMax.x=std::max(cat1Max.x,cat2Max.x);
    catMax.y=std::max(cat1Max.y,cat2Max.y);
    catMax.z=std::max(cat1Max.z,cat2Max.z);

    box.x=catMax.x-catMin.x+1.0;
    box.y=catMax.y-catMin.y+1.0;
    box.z=catMax.z-catMin.z+1.0;
    ori.x=-catMin.x;
    ori.y=-catMin.y;
    ori.z=-catMin.z;
    shift_catalog(catalog1, catMin);
    shift_catalog(catalog2, catMin);
}




