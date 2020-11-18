//
// Created by biba_bo on 2020-11-17.
//

#ifndef CORAL_POSE_DETECTION_CPP_POSE_ANGLE_ENGINE_H
#define CORAL_POSE_DETECTION_CPP_POSE_ANGLE_ENGINE_H


#include <cmath>
#include <vector>

const int MAX_INIT_STEPS = 30;

class PoseAngleEngine {
private:
    bool _isModelHasFaceBasePoints;
    int _widthImage;
    int _heightImage;
    float _eyesDistanceOnImage;
    float _shoulderDistanceOnImage;
    float _lastDetectedAngle;

    int _countInitSteps;

    void init_angle_detector(const std::vector<int> &xCoords, const std::vector<int> &yCoords);
public:
    PoseAngleEngine(bool isModelHasFaceBasePoints, int widthImage, int heightImage);
    ~PoseAngleEngine();

    float get_angle(const std::vector<int> &xCoords, const std::vector<int> &yCoords);
    float get_eyes_distance();
    float get_shoulder_distance();
};


#endif //CORAL_POSE_DETECTION_CPP_POSE_ANGLE_ENGINE_H
