#ifndef TSCHUMPERLE_DERICHE_FILTER_HPP
#define TSCHUMPERLE_DERICHE_FILTER_HPP

/**
 * @file TDF.hpp
 * @brief Этот файл хранит объявления функций.
 */

#include <vector>

namespace cv {
    class Mat;
}

namespace TDF {
    std::vector<cv::Mat> CalculateGradient(cv::Mat image, float sigma_d);
    cv::Mat BuildStructureMatrix(std::vector<cv::Mat> gradient_components, float sigma_m);
    cv::Mat CalculateEigenvalues(cv::Mat structure_matrix);
    cv::Mat CalculateGeometryMatrix(cv::Mat eigenvalues);
    void CalculateHessian(cv::Mat image, float sigma_d, cv::Mat Hxx, cv::Mat Hxy, cv::Mat Hyy);
    cv::Mat UpdateImage(cv::Mat image, float dt, int T, float sigma_d, float sigma_m, float a0, float a1);
    cv::Mat Tschumperle_Deriche_Filter(cv::Mat I, int T, float Dt, float sigma_d, float sigma_m, float a0, float a1, float alpha0);
}

#endif // TSCHUMPERLE_DERICHE_FILTER_HPP
