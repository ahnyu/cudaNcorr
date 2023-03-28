// Survey.cpp

#include "../include/survey.hpp"
#include <vector>
#include <vector_types.h>
#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

constexpr double SPEED_OF_LIGHT = 299792.458; // Speed of light in km/s
constexpr double DEG_TO_RAD = M_PI / 180.0; // degr

Survey::Survey(const Cosmology &cosmo) : cosmo_(cosmo) {}

double Survey::integrand(double a, void *params) {
    Cosmology *cosmo = static_cast<Cosmology *>(params);
//    double H0 = cosmo->H0;
    double OmegaM = cosmo->OmegaM;
    double OmegaLambda = cosmo->OmegaLambda;

    return 1 / sqrt(OmegaM * pow(a, -3) + OmegaLambda);
}

double Survey::comoving_distance(double z) {
    gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(1000);
    gsl_function F;
    F.function = &Survey::integrand;
    F.params = &cosmo_;

    double result, error;
    gsl_integration_qags(&F, 1 / (1 + z), 1, 0, 1e-7, 1000, workspace, &result, &error);
    gsl_integration_workspace_free(workspace);

    double h = cosmo_.H0 / 100.0;
    return (SPEED_OF_LIGHT * result / cosmo_.H0) / h; // Divide by h to get the output in Mpc/h
}

std::vector<double4> Survey::convert_coordinates(const std::vector<double4> &catalog) {
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


