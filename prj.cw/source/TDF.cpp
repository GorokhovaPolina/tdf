/**
 * @file TDF.cpp
 * @brief Этот файл хранит реализацию.
 */

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "TDF.hpp"

namespace TDF {

/**
 * @brief Вычисление градиентов изображения.
 *
 * Эта функция используется для определения краев, углов и границ объектов на изображении.
 *
 * @param[in] image Входное изображение.
 * @param[in] sigma_d Радиус гауссового ядра, используемого для сглаживания градиента.
 * 
 * @return Возвращает вектор матриц, каждая из которых представляет компоненту градиента изображения (горизонтальную и вертикальную).
 */
    //Step 1: calculate image gradients
    std::vector<cv::Mat> CalculateGradient(cv::Mat image, float sigma_d) {
        // Define gradient kernels
        float a = (2 - sqrt(2)) / 4;
        float b = (sqrt(2) - 1) / 2;
        float c = 0;
        cv::Mat kernel_x = (cv::Mat_<float>(3, 3) << -a, c, a, -b, c, b, -a, c, a);
        cv::Mat kernel_y = (cv::Mat_<float>(3, 3) << -a, -b, -a, c, c, c, a, b, a);
        
        // Calculate image gradients
        std::vector<cv::Mat> gradient_components(image.channels());
        for (int i = 0; i < image.channels(); i++) {
            cv::Mat dx;
            cv::Mat dy;
            cv::filter2D(image, dx, -1, kernel_x);
            cv::filter2D(image, dy, -1, kernel_y);
            gradient_components[i] = cv::Mat::zeros(image.size(), CV_64F);
            
            // Smooth gradient components with Gaussian filter
            cv::GaussianBlur(dx.mul(dx), dx, cv::Size(0, 0), sigma_d);
            cv::GaussianBlur(dy.mul(dy), dy, cv::Size(0, 0), sigma_d);
            cv::multiply(dx, dy, gradient_components[i], 1);
        }
        return gradient_components;
    }

/**
 * @brief Построение локальной матрицы структуры.
 *
 * Эта функция используется для анализа текстуры изображения.
 *
 * @param[in] gradient_components Входной вектор матриц, каждая из которых представляет компоненту градиента изображения.
 * @param[in] sigma_m Радиус гауссового ядра, используемого для сглаживания матрицы структуры.
 * 
 * @return Возвращает матрицу структуры, которая представляет собой комбинацию компонент градиента с использованием оператора сдвига.
 */
    //Step 2: build the local structure matrix
    cv::Mat BuildStructureMatrix(std::vector<cv::Mat> gradient_components, float sigma_m) {
        // Define shift operator kernels
        float a = (2 - sqrt(2)) / 4;
        float b = (sqrt(2) - 1) / 2;
        cv::Mat kernel_x = (cv::Mat_<float>(3, 3) << -a, 0, a, -b, 0, b, -a, 0, a);
        cv::Mat kernel_y = (cv::Mat_<float>(3, 3) << -a, -b, -a, 0, 0, 0, a, b, a);
        
        // Calculate structure matrix
        cv::Mat structure_matrix = cv::Mat::zeros(gradient_components[0].size(), CV_64F);
        for (int i = 0; i < gradient_components.size(); i++) {
            cv::Mat dx, dy;
            cv::filter2D(gradient_components[i], dx, -1, kernel_x);
            cv::filter2D(gradient_components[i], dy, -1, kernel_y);

            cv::Scalar dx_mul_dx_scalar = cv::sum(dx.mul(dx));
            cv::Vec3d dx_mul_dx(dx_mul_dx_scalar[0], dx_mul_dx_scalar[1], dx_mul_dx_scalar[2]);
            cv::Scalar dx_mul_dy_scalar = cv::sum(dx.mul(dy));
            cv::Vec3d dx_mul_dy(dx_mul_dy_scalar[0], dx_mul_dy_scalar[1], dx_mul_dy_scalar[2]);
            cv::Scalar dy_mul_dy_scalar = cv::sum(dy.mul(dy));
            cv::Vec3d dy_mul_dy(dy_mul_dy_scalar[0], dy_mul_dy_scalar[1], dy_mul_dy_scalar[2]);
            // Build structure matrix
            structure_matrix.at<cv::Vec3d>(cv::Point(0, 0))[0] += dx_mul_dx[0];
            structure_matrix.at<cv::Vec3d>(cv::Point(0, 0))[1] += dx_mul_dy[1];
            structure_matrix.at<cv::Vec3d>(cv::Point(0, 0))[2] += dy_mul_dy[2];
        }

        // Smooth structure matrix elements with Gaussian filter
        cv::GaussianBlur(structure_matrix, structure_matrix, cv::Size(0, 0), sigma_m);

        return structure_matrix;
    }

/**
 * @brief Подсчет собственных значений локальной структуры матриц.
 *
 * Эта функция вычисляет собственные значения локальной структурной матрицы изображения для анализа геометрических свойств локальных участков изображения.
 *
 * @param[in] structure_matrix Входная матрица.
 * 
 * @return Возвращает матрицу, элементы которой содержат два собственных значения для каждого пикселя на изображении. 
 */
    //Calculate Eigenvalues of Matrix
    cv::Mat CalculateEigenvalues(cv::Mat structure_matrix) {
        cv::Mat eigenvalues = cv::Mat::zeros(structure_matrix.size(), CV_64F);
        for (int i = 0; i < structure_matrix.rows; i++) {
            for (int j = 0; j < structure_matrix.cols; j++) {
                cv::Mat eigenvalues_ij, eigenvectors_ij;
                cv::eigen(structure_matrix.row(i).col(j), eigenvalues_ij, eigenvectors_ij);
                eigenvalues.at<cv::Vec3d>(i, j) = cv::Vec3d(eigenvalues_ij.at<double>(0), eigenvalues_ij.at<double>(1), 0);
            }
        }
        
        return eigenvalues;
    }

/**
 * @brief Вычисление матрицы геометрической зависимости для каждого пикселя на изображении.
 *
 * Эта функция используется для оценки локальной геометрии изображения.
 *
 * @param[in] eigenvalues Входная матрица собственных значений.
 * 
 * @return Возвращает матрицу геометрической зависимости, каждый элемент которой представляет собой вектор с двумя компонентами.
 */
    cv::Mat CalculateGeometryMatrix(cv::Mat eigenvalues) {
        cv::Mat A = cv::Mat::zeros(eigenvalues.size(), CV_64FC3);
        for (int i = 0; i < eigenvalues.rows; i++) {
            for (int j = 0; j < eigenvalues.cols; j++) {
                double l1 = eigenvalues.at<cv::Vec3d>(i, j)[0];
                double l2 = eigenvalues.at<cv::Vec3d>(i, j)[1];
                double v1 = 1.0 / sqrt(1 + (l1 - l2) * (l1 - l2));
                double v2 = 1.0 / sqrt(1 + (2 * l2 - l1 - l2) * (2 * l2 - l1 - l2));
                A.at<cv::Vec3d>(i, j) = cv::Vec3d(v1, v2, 0);
            }
        }
        return A;
    }

/**
 * @brief Вычисление элементов Гессиана.
 *
 * Эта функция используется для сглаживания краев.
 *
 * @param[in] image Входное изображение.
 * @param[in] sigma_d Радиус гауссового ядра, используемого для сглаживания градиента.
 * @param[in, out] Hxx, Hxy, Hyy Входные матрицы для сохранения элементов гессиана для каждого пикселя. Выход будет нормализован согласно определению.
 */
    void CalculateHessian(cv::Mat image, float sigma_d, cv::Mat Hxx, cv::Mat Hxy, cv::Mat Hyy){
        cv::Mat dx, dy;
        cv::Sobel(image, dx, CV_64F, 2, 0);
        cv::Sobel(image, dy, CV_64F, 0, 2);
        
        cv::GaussianBlur(dx.mul(dx), Hxx, cv::Size(0, 0), sigma_d);
        cv::GaussianBlur(dx.mul(dy), Hxy, cv::Size(0, 0), sigma_d);
        cv::GaussianBlur(dy.mul(dy), Hyy, cv::Size(0, 0), sigma_d);
    }

/**
* @brief Преобразование изображения, используя матрицы геометрической и структурной зависимости.
*
* TDF - рекурсивный Гауссовский производный фильтр.
* 1) Вычисление градиентов изображения
* 2) Создание структурной матрицы
* 3) Преобразование изображения
*
* @param[in] image Входное изображение.
* @param[in] dt Шаг времению
* @param[in] T Количество итераций.
* @param[in] sigma_d Радиус гауссового ядра, используемого для сглаживания градиента.
* @param[in] sigma_m Радиус гауссового ядра, используемого для сглаживания матрицы структуры.
* @param[in] a0 Параметры диффузии вдоль углов.
* @param[in] a1 Параметры диффузии поперек краев.
*
* @return Возвращает изображение с более высоким качеством и без шумов.
*
* @throws std::invalid_argument Если изображение пусто.
*/ 
    //Step 3: Update Image
    cv::Mat UpdateImage(cv::Mat image, float dt, int T, float sigma_d, float sigma_m, float a0, float a1) {
        // // Check if image is empty
        // if (image.empty()) {
        //     throw std::invalid_argument("invalid type of image");
        // }
        cv::Mat I = image.clone();
        std::vector<cv::Mat> gradient_components = TDF::CalculateGradient(image, sigma_d);
        cv::Mat M = TDF::BuildStructureMatrix(gradient_components, sigma_m);
        cv::Mat eigenvalues = TDF::CalculateEigenvalues(M);
        cv::Mat A = TDF::CalculateGeometryMatrix(eigenvalues);
        
        for (int t(0); t < T; t++) {
            cv::Mat Hxx, Hxy, Hyy;
            TDF::CalculateHessian(I, sigma_d, Hxx, Hxy, Hyy);
            for (int i = 0; i < I.rows; i++) {
                for (int j = 0; j < I.cols; j++) {
                    float hxx = Hxx.at<float>(i, j);
                    float hxy = Hxy.at<float>(i, j);
                    float hyy = Hyy.at<float>(i, j);
                    float tr = A.at<cv::Vec3d>(i, j)[0] * hxx + A.at<cv::Vec3d>(i, j)[1] * hyy + 2 * A.at<cv::Vec3d>(i, j)[2] * hxy;
                    float beta = tr;
                    float alpha = dt / std::abs(beta);
                    cv::Vec3d pixel = I.at<cv::Vec3d>(i, j) + alpha * beta * A.at<cv::Vec3d>(i, j);
                    I.at<cv::Vec3d>(i, j) = pixel;
                }
            }
        }
        return I;
    }
/**
* @brief Сглаживание углов с промощью DTF.
*
* TDF - рекурсивный Гауссовский производный фильтр.
* 1) Вычисление градиентов изображения
* 2) Создание структурной матрицы
* 3) Преобразование изображения
*
* @param[in] I Входное изображение.
* @param[in] T Количество итераций.
* @param[in] Dt Шаг времению
* @param[in] sigma_d Радиус гауссового ядра, используемого для сглаживания градиента.
* @param[in] sigma_m Радиус гауссового ядра, используемого для сглаживания матрицы структуры.
* @param[in] a0 Параметры диффузии вдоль углов.
* @param[in] a1 Параметры диффузии поперек краев.
* @param[in] alpha0 Начальный шаг времени.
*
* @return Возвращает изображение, преобразованное с помощью алгоритма TDF.
*
* @throws std::invalid_argument Если изображение пусто.
*/    
    //Step 4: Update Image
    cv::Mat Tschumperle_Deriche_Filter(cv::Mat I, int T, float Dt, float sigma_d, float sigma_m, float a0, float a1, float alpha0) {

        // Вычисление структурной матрицы
        std::vector<cv::Mat> gradient_components = TDF::CalculateGradient(I, sigma_d);
        cv::Mat structure_matrix = TDF::BuildStructureMatrix(gradient_components, sigma_m);

        // Вычисление собственных значений структурной матрицы и геометрической матрицы
        cv::Mat eigenvalues = TDF::CalculateEigenvalues(structure_matrix);
        cv::Mat geom_matrix = TDF::CalculateGeometryMatrix(eigenvalues);

        // Вычисление гессиана изображения
        cv::Mat Hxx, Hxy, Hyy;
        TDF::CalculateHessian(I, sigma_d, Hxx, Hxy, Hyy);

        // Обновление изображения многократным применением матрицы Tschumperle-Deriche
        cv::Mat filtered_image = I.clone();
        for (int i = 0; i < T; i++) {
            cv::Mat updated_image = TDF::UpdateImage(filtered_image, Dt, T, sigma_d, sigma_m, a0, a1);
            filtered_image = Hxx.mul(geom_matrix.row(0)) + 2 * Hxy.mul(geom_matrix.row(1)) + Hyy.mul(geom_matrix.row(2)) + alpha0 * updated_image;
        }

        return filtered_image;
    }
}
