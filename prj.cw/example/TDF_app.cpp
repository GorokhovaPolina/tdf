#include "TDF.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

int main() {
    //Load the image
    std::string image_path = "";
    std::cout << "Image path: ";
    std::cin >> image_path;
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cout << "Could not read the image." << "\n";
        return 1;
    }

    //Number of iterations
    int T1(0);
    std::cout << "Number of iterations(5 - 20): ";
    std::cin >> T1;

    //Time increament
    float Dt1 = 20;

    //Width of the Gaussian kernel for smoothing the gradient
    float sigma_d1 = 0;
    std::cout << "Width of the Gaussian kernel for smoothing the gradient(0.0 - 0.5): ";
    std::cin >> sigma_d1;

    //Width of the Gaussian kernel for smoothing the structure matrix
    float sigma_m1 = 0;
    std::cout << "Width of the Gaussian kernel for smoothing the structure matrix(0.0 - 0.5): ";
    std::cin >> sigma_m1;

    //Difusion parameters along edges
    float a01 = 0;
    std::cout << "Difusion parameters along edges(0.0 - 0.5): ";
    std::cin >> a01;

    //Difusion parameters across edges
    float a11 = 0.9;

    // initial time step
    float alpha01 = 0.5;

    //Save the result or not 
    int answer_save(0);
    std::cout << "Do you want to save the answer? (1 - yes; 0 - no): ";
    std::cin >> answer_save;

    int answer_show(0);
    std::cout << "Do want to see the result? (1 - yes; 0 - no): ";
    std::cin >> answer_show;

    cv::Mat result = TDF::Tschumperle_Deriche_Filter(image, T1, Dt1, sigma_d1, sigma_m1, a01, a11, alpha01);


    if (answer_save == 1) {
        cv::Mat result = TDF::Tschumperle_Deriche_Filter(image, T1, Dt1, sigma_d1, sigma_m1, a01, a11, alpha01);
        cv::imwrite(image_path + "0.png", result);
    }
    if (answer_show == 1) {
        // Display the image.
        cv::namedWindow("TDF result", cv::WINDOW_AUTOSIZE);
        cv::imshow("TDF result", result);
        // Wait for a keystroke in the window.
        int k = cv::waitKey(0);
        // Close the window.
        cv::destroyAllWindows();
    }

    return 0;
}
