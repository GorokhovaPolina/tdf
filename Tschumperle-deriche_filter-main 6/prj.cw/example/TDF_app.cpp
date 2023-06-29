#include "TDF.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cerr << "Error: Not enough arguments. Usage: " << argv[0] << " <image_path> <T> <sigma_d> <sigma_m> <a0> <show_result(yes/no)> <output_image_path>" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];
    int T1 = std::stod(argv[2]);
    double Dt1 = 20;
    double sigma_d1 = std::stod(argv[3]);
    double sigma_m1 = std::stod(argv[4]);
    double a01 = std::stod(argv[5]);
    double a11 = 0.9;
    double alpha01 = 0.5;
    std::string answer_show = argv[6];
    std::string output_path = argv[7];

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cout << "Image is empty." << std::endl;
        return 1;
    }

    cv::Mat result = TDF::Tschumperle_Deriche_Filter(image, T1, Dt1, sigma_d1, sigma_m1, a01, a11, alpha01);

    cv::imwrite(output_path + "0.png", (image, result));

    if (answer_show == 'yes') {
        // Display the image.
        namedWindow("TDF result", cv::WINDOW_AUTOSIZE);
        cv::imshow("TDF result", result);

        // Wait for a keystroke in the window.
        int k = cv::waitKey(0);
        // Close the window.
        cv::destroyAllWindows();
    }

    return 0;
}