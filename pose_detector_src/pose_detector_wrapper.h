//
// Created by biba_bo on 2020-11-13.
//

#ifndef CORAL_POSE_DETECTION_CPP_POSE_DETECTOR_WRAPPER_H
#define CORAL_POSE_DETECTION_CPP_POSE_DETECTOR_WRAPPER_H

#include <list>
#include <thread>
#include <mutex>
#include <atomic>

#include <opencv2/opencv.hpp>

#include <edgetpu.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "kernels/register.h"

#include "../posenet/posenet_decoder_op.h"

/** @brief This constant contains a code of successful operation. */
const int CODE_STATUS_OK = 0;
/** @brief This code contains a code of unsuccessful operation loading a poseNet model from file
 *         Usually problems are 1st wrong path to file, 2nd wrong type of model. */
const int CODE_FAILED_BUILD_MODEL_FROM_FILE = -1;
/** @brief This is a path to file with the poseNet model,
 *         which contained pretrained model and weights for detecting human poses key points. */
const std::string PATH_TO_POSENET_MODEL = "src/models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite";
/** @brief This constant contains a maximum count of frames in the frames queue. */
const int MAX_COUNT_FRAMES_IN_QUEUE = 2;
/** @brief This constant contains a maximum count of the vectors with poses in the
 *         queue of detected frames. */
const int MAX_COUNT_POSE_VECTORS_IN_QUEUE = 2;
/** @brief This constant contains a threshold for removing wrong detected points. */
const float POSE_THRESHOLD = 0.1;

/** This is a data structure which contains a result with the last detected poses*/
struct DetectedPose {
    /// vector of keypoint scopes
    std::vector<float> keypointScores;
    /// vector of keypoint coordinates
    std::vector<float> keypointCoordinates;
};

/** @brief This a class PoseDetectorWrapper - is a wrapper for detection a pose via edgetpulib and tensorflow-line C++ API
 *         on the google coral dev board, and possibly on a USB stick.
 *         Here are an initialization and model loading into tensorflow-lite and for calculations are using edgetpu.
 *         It from input image gets a key points. This key points can be used as you wish*/
class PoseDetectorWrapper {
private:
    /** @brief This is a queue of frames which are waiting for processing
     *         (means finding a key points of the human pose)*/
    std::list<cv::Mat> _queueFrames;

    /** @brief This is a queue of detected poses for the lastFrames. */
    std::list<std::vector<DetectedPose>> _queueDetectedPoses;

    /** @brief This is a thread for detecting a key point, it used for separating
     *         other processes from the keypoints detection. */
    std::thread _keyPointDetectionThread;

    /** @brief This mutex synchronize a block code where we process a resources
     *         (here is - std::list<cv::Mat> queueFrames)*/
    std::mutex _mutexRes;

    /** @brief This mutex used for blocking processing, it blocks code when we trying to process image, but actually we don't have a Mat object for this
    *          (it means std::list<cv::Mat> queueFrames is empty)
    *          Firstly it blocks when we are creating a PoseDetectorWrapper
    *         class (calls in constructor)*/
    std::mutex _mutexProc;

    /** @brief this variable shows should server process new position for the iron turtle
     *         also it means should works a rotation thread (is it works now?)*/
    std::atomic<bool> _isProcessThread;

    /** @brief this variable contains a status of poseNet model initialization.
     *         if model are not initialized, you cannot to start a thread. */
    int _statusModelInterpreterActivation;

    std::shared_ptr<edgetpu::EdgeTpuContext> _edgetpuContext;

    /** @brief this is a model loaded from file */
    std::unique_ptr<tflite::FlatBufferModel> _model;

    /** @brief this variable contains an initialized model with poseNet and possibility to use edgetpu*/
    std::unique_ptr<tflite::Interpreter> _modelInterpreter;

    /** @brief this variable contains a description of output neuron network description.
     *         it used for converting output data in usable format.     */
    std::vector<size_t> _outputModelShape;

    /** @brief this is a width of an input image in the poseNet model
     *         (also it can be a width of the input neuron network layer).*/
    int _widthInputLayerPoseNetModel;

    /** @brief this is a height of an input image in the poseNet model
     *         (also it can be a height of the input neuron network layer).*/
    int _heightInputLayerPoseNetModel;

    std::unique_ptr<tflite::Interpreter>
    build_edge_tpu_interpreter(const tflite::FlatBufferModel &model, edgetpu::EdgeTpuContext *edgetpu_context);

    /** @brief This function converts a cv::Mat object into the special format for the poseNet model
     *  @return converted data from cv::Mat in the vector.*/
    std::vector<uint8_t> get_input_data_from_frame(cv::Mat &inputFrame);

    /** @brief This function set an input data (from frame) into the input of the neural network
     *  @param inputData - a vector of the converted data from the cv::Mat
     *  @return a vector of float values (here is raw output of the neural network) */
    std::vector<float> get_raw_output_data_from_model(const std::vector<uint8_t> &inputData);

    /** @brief This function converts raw data into usable format for using in image
     *  @param outputRawDataVector - this is vector<float>, which contain output neural network after work
     *  @param threshold - this is threshold value for removing wrong results
     *  @return - vector of DectedPose-s - in usable format*/
    std::vector<DetectedPose>
    get_pose_estimate_from_output_raw_data(const std::vector<float> &outputRawDataVector, const float &threshold);

    /** @brief This function is used for loading a model from file into a tensorflow and connects
     *         it with an edgetpu context for processing an calculations.
     *  @param pathToModelFile */
    void init_pose_detector(const std::string &pathToModelFile);

    /** @brief This function starts new thread for the detecting pose key points from an each frame
     *         from the frames queue.*/
    void process_pose_detection();


public:
    /** @brief this a default constructor, here inits all needed fields*/
    PoseDetectorWrapper();

    /** @brief this a constructor with params, here inits all needed fields
     *  @param pathToPoseNetModel - is a path to poseNet model. */
    PoseDetectorWrapper(const std::string &pathToPoseNetModel);

    /** @brief this is a default destructor.
     *  @warning this on doing nothing */
    ~PoseDetectorWrapper();

    /** @brief this function adds a frame into frames queue, which used for detecting frames from the image
     *  @param frame - this a new frame for recognition.*/
    void add_frame(cv::Mat frame);

    /** @brief this function for getting last detected humans poses
     *  @return a vector with detected poses coordinates,
     *          here can be a poses for multiple persons.  */
    std::vector<DetectedPose> getLastDetectedPose();

    /** @brief This function starts a pose detection thread if it possible.
     *  @return result of operation. */
    int start_pose_detection();

    /** @brief This function stops a pose detection thread if it possible.
     *  @return result of operation. */
    int stop_pose_detection();

    /** @brief This function draw pose on the any frame (it can draw, or no, dependently from detection status)
     *  @param frame - this is a pointer to image, on which can be drawn pose pointers*/
    void draw_last_pose_on_image(cv::Mat &frame);
};


#endif //CORAL_POSE_DETECTION_CPP_POSE_DETECTOR_WRAPPER_H
