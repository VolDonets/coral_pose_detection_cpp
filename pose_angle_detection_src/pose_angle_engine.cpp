//
// Created by biba_bo on 2020-11-17.
//

#include "pose_angle_engine.h"

PoseAngleEngine::PoseAngleEngine(bool isModelHasFaceBasePoints, int widthImage, int heightImage) {
    _isModelHasFaceBasePoints = isModelHasFaceBasePoints;
    _widthImage = widthImage;
    _heightImage = heightImage;
    _countInitSteps = 0;
    _eyesDistanceOnImage = 0.0;
    _shoulderDistanceOnImage = 0.0;
    _lastDetectedAngle = 0.0;
}

PoseAngleEngine::~PoseAngleEngine() {
    // nothing to do here
}

float PoseAngleEngine::get_angle(const std::vector<int> &xCoords, const std::vector<int> &yCoords) {
    if (_countInitSteps != MAX_INIT_STEPS) {
        init_angle_detector(xCoords, yCoords);
        return 0.0;
    }
    // check is image has face elements line eyes or nose
    // it helps to detect a human body direction on camera
    if (xCoords[0] != -1) {
        if (xCoords[5] != -1 && xCoords[6] != -1) {
            float currentShoulderDistance = sqrt(pow((xCoords[5] - xCoords[6]), 2) +
                                                 pow((yCoords[5] - yCoords[6]), 2));
            if (currentShoulderDistance >= _shoulderDistanceOnImage) {
                _lastDetectedAngle = 0;
            } else {
                float middleFaceX = 0.0;
                float middleShoulderX = 0.0;
                int countMiddleFaceX = 0;
                for (int inx = 0; inx < 5; inx++) {
                    if (xCoords[inx] != -1) {
                        middleFaceX += xCoords[inx];
                        countMiddleFaceX++;
                    }
                }
                middleFaceX = middleFaceX / countMiddleFaceX;
                middleShoulderX = (xCoords[5] + xCoords[6]) / 2;
                _lastDetectedAngle = (middleShoulderX > middleFaceX)
                                     ? (acos(currentShoulderDistance / _shoulderDistanceOnImage))
                                     : -(acos(currentShoulderDistance / _shoulderDistanceOnImage));
            }
        }
    } else {
        if (xCoords[5] != -1 && xCoords[6] != -1) {
            float currentShoulderDistance = sqrt(pow((xCoords[5] - xCoords[6]), 2) +
                                                 pow((yCoords[5] - yCoords[6]), 2));
            if (currentShoulderDistance >= _shoulderDistanceOnImage) {
                _lastDetectedAngle = 0;
            } else {
                float middleFaceX = 0.0;
                float middleShoulderX = 0.0;
                int countMiddleFaceX = 0;
                for (int inx = 0; inx < 5; inx++) {
                    if (xCoords[inx] != -1) {
                        middleFaceX += xCoords[inx];
                        countMiddleFaceX++;
                    }
                }
                middleFaceX = middleFaceX / countMiddleFaceX;
                middleShoulderX = (xCoords[5] + xCoords[6]) / 2;
                _lastDetectedAngle = (middleShoulderX > middleFaceX)
                                     ? (3.14 - (acos(currentShoulderDistance / _shoulderDistanceOnImage)))
                                     : -(3.13 - (acos(currentShoulderDistance / _shoulderDistanceOnImage)));
            }
        }
    }
    return (180 * _lastDetectedAngle) / 3.14;
}

void PoseAngleEngine::init_angle_detector(const std::vector<int> &xCoords, const std::vector<int> &yCoords) {
    if ((xCoords[1] != -1 && xCoords[2] != -1)
        && (xCoords[5] != -1 && xCoords[6] != -1)) {
        _eyesDistanceOnImage += sqrt(pow((xCoords[1] - xCoords[2]), 2) +
                                     pow((yCoords[1] - yCoords[2]), 2));
        _shoulderDistanceOnImage += sqrt(pow((xCoords[5] - xCoords[6]), 2) +
                                         pow((yCoords[5] - yCoords[6]), 2));
        _countInitSteps++;
        if (_countInitSteps == MAX_INIT_STEPS) {
            _eyesDistanceOnImage /= MAX_INIT_STEPS;
            _shoulderDistanceOnImage /= MAX_INIT_STEPS;
        }
    }
}

float PoseAngleEngine::get_eyes_distance() {
    return _eyesDistanceOnImage;
}

float PoseAngleEngine::get_shoulder_distance() {
    return _shoulderDistanceOnImage;
}

