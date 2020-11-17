//
// Created by biba_bo on 2020-11-17.
//

#include "pose_detector_src/pose_detector_wrapper.h"


int main(int argv, char *argc[]) {
    std::shared_ptr<PoseDetectorWrapper> poseDetector = std::make_shared<PoseDetectorWrapper>();
    cv::Mat image = cv::imread("src/images/6.jpg");
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(poseDetector->_widthInputLayerPoseNetModel, poseDetector->_heightInputLayerPoseNetModel));
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGRA2RGB);
    cv::Mat flat = resizedImage.reshape(1, resizedImage.total() * resizedImage.channels());
    std::vector<uint8_t> inputDataVector = resizedImage.isContinuous() ? flat : flat.clone();
    std::vector<float> outputData;
    auto *input = poseDetector->_modelInterpreter->typed_input_tensor<uint8_t>(0);
    std::memcpy(input, inputDataVector.data(), inputDataVector.size());
    std::cout << "Ok 1: \n";
    poseDetector->_modelInterpreter->Invoke();
    std::cout << "Ok 2: \n";
    return 0;
}