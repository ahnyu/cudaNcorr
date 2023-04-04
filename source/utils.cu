#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <fitsio.h>
#include <vector>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <random>
#include <omp.h>
#include <../include/utils.hpp>


std::vector<double> logspace(double start, double end, int Nedge) {
    double start_log = std::log10(start);
    double end_log = std::log10(end);
    double step = (end_log - start_log) / (Nedge - 1);

    std::vector<double> logspace_vector(Nedge);
    for (int i = 0; i < Nedge; ++i) {
        logspace_vector[i] = std::pow(10, start_log + i * step);
    }

    return logspace_vector;
}

double4 generateRandomPosition(std::mt19937& gen, const double3 &box) {
    std::uniform_real_distribution<double> distx(0, box.x);
    std::uniform_real_distribution<double> disty(0, box.y);
    std::uniform_real_distribution<double> distz(0, box.z);
    double4 position;
    position.x = distx(gen);
    position.y = disty(gen);
    position.z = distz(gen);
    position.w = 1.0f;
    return position;
}

std::vector<double4> generateRandomCatalog(int Np, const double3 &box) {
    std::vector<double4> p(Np);
    #pragma omp parallel
    {
        std::mt19937 gen(omp_get_thread_num() + static_cast<unsigned>(time(0)));
        #pragma omp for
        for (int i = 0; i < Np; i++) {
            p[i] = generateRandomPosition(gen, box);
        }
    }
    return p;
}

std::vector<double4> readCatalogFits(
    const std::string& filename, 
    const std::vector<std::string>& column_names, 
    int hdu_index,
    double zmin,
    double zmax
) {
    std::vector<double4> result;

    fitsfile* fptr;
    int status = 0;

    if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
        throw std::runtime_error("Unable to open FITS file: " + filename);
    }   

    int ncols, hdutype;
    long nrows;

    if (fits_movabs_hdu(fptr, hdu_index, &hdutype, &status)) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Error moving to the specified HDU.");
    }   

    if (hdutype != BINARY_TBL) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("The specified HDU is not a Binary Table.");
    }   

    if (fits_get_num_rows(fptr, &nrows, &status) || fits_get_num_cols(fptr, &ncols, &status)) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Error reading FITS file structure.");
    }   

    for (long i = 1; i <= nrows; ++i) {
        double4 temp;

        for (size_t j = 0; j < column_names.size(); ++j) {
            int col;
            double value;
            if (fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(column_names[j].c_str()), &col, &status)) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("Error finding column named " + column_names[j]);
            }

            if (fits_read_col(fptr, TDOUBLE, col, i, 1, 1, NULL, &value, NULL, &status)) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("Error reading FITS file data.");
            }
            if (column_names[j] == "RA") temp.x = value;
            else if (column_names[j] == "DEC") temp.y = value;
            else if (column_names[j] == "Z") temp.z = value;
            else if (column_names[j] == "WEIGHT") temp.w = value;
        }
        if (temp.z > zmin && temp.z < zmax) {
            result.push_back(temp);
        }
    }

    fits_close_file(fptr, &status);
    return result;
}



std::vector<double4> readCatalogTxt(
    const std::string& filename,
    const double3& box,
    const std::vector<int>& pos_columns
) {
    std::vector<double4> result;
    std::ifstream input_file(filename);

    if (!input_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::string line;
    size_t line_number = 0;

    while (std::getline(input_file, line)) {
        ++line_number;
        std::istringstream line_stream(line);
        double4 temp;
        double value;
        int column = 0;

        while (line_stream >> value) {
            if (column == pos_columns[0]) temp.x = value;
            else if (column == pos_columns[1]) temp.y = value;
            else if (column == pos_columns[2]) temp.z = value;
            else if (column == pos_columns[3]) temp.w = value;
            ++column;
        }

        if (temp.x < 0 || temp.y < 0 || temp.z < 0 ||
            temp.x > box.x || temp.y > box.y || temp.z > box.z) {
            input_file.close();
            throw std::runtime_error("Error at line " + std::to_string(line_number) + ": double3 value is out of bounds.");
        }
        result.push_back(temp);
    }

    input_file.close();
    return result;
}

void writeToFileThreeD(
    const std::string& filename, 
    const std::vector<double3>& double3_vec, 
    const std::vector<double>& double_vec
) {
    if (double3_vec.size() != double_vec.size()) {
        throw std::runtime_error("Vectors have different sizes.");
    }           
                
    std::ofstream output_file(filename);
        
    if (!output_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }               

    output_file << std::fixed << std::setprecision(6);
                        
    for (size_t i = 0; i < double3_vec.size(); ++i) {
        output_file << double3_vec[i].x << " " << double3_vec[i].y << " " << double3_vec[i].z << " " << double_vec[i] << std::endl;
    }                   
     
    output_file.close();
}

void writeToFileRppi(
    const std::string& filename, 
    const std::vector<double3>& double3_vec, 
    const std::vector<double2>& double2_vec, 
    const std::vector<double>& double_vec
) {
    if (double3_vec.size() != double_vec.size() || double3_vec.size() != double2_vec.size() ||double_vec.size() != double2_vec.size()) {
        throw std::runtime_error("Vectors have different sizes.");
    }           
                
    std::ofstream output_file(filename);
        
    if (!output_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }               

    output_file << std::fixed << std::setprecision(6);
                        
    for (size_t i = 0; i < double3_vec.size(); ++i) {
        output_file << double3_vec[i].x << " " << double3_vec[i].y << " " << double3_vec[i].z << " " << double2_vec[i].x << " " << double2_vec[i].y << " " << double_vec[i] << std::endl;
    }                   
     
    output_file.close();
}

