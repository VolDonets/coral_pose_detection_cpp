//
// Created by biba_bo on 2020-11-13.
//

#include "pose_detector_wrapper.h"

PoseDetectorWrapper::PoseDetectorWrapper() {
    // set status of thread working as disabled
    _isProcessThread = false;
    // lock mutexProc - it means, no frames to process
    _mutexProc.lock();
    // load model from file and prepare space for pose detection processing
    init_pose_detector(PATH_TO_POSENET_MODEL);
}

PoseDetectorWrapper::PoseDetectorWrapper(const std::string &pathToPoseNetModel) {
    // set status of thread working as disabled
    _isProcessThread = false;
    // lock mutexProc - it means, no frames to process
    _mutexProc.lock();
    // load model from file and prepare space for pose detection processing
    init_pose_detector(pathToPoseNetModel);
}

PoseDetectorWrapper::~PoseDetectorWrapper() {
    //Now here is nothing to do.
}

void PoseDetectorWrapper::init_pose_detector(const std::string &pathToModelFile) {
    _statusModelInterpreterActivation = CODE_STATUS_OK;
    std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(pathToModelFile.c_str());

    if (model == nullptr) {
#ifdef MY_DEBUG_DEF
        std::cerr << "FAIL to build FlatBufferModel from file!\n";
#endif //MY_DEBUG_DEF
        _statusModelInterpreterActivation = CODE_FAILED_BUILD_MODEL_FROM_FILE;
        return;
    }
#ifdef MY_DEBUG_DEF
    else {
        std::cout << "Model from file SUCCESSFULLY built\n";
    }
#endif //MY_DEBUG_DEF

    const auto &available_tpus = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpuContext =
            edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
                    available_tpus[0].type, available_tpus[0].path);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(coral::kPosenetDecoderOp, coral::RegisterPosenetDecoderOp());
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

    if (tflite::InterpreterBuilder(*model, resolver)(&_modelInterpreter) != kTfLiteOk) {
#ifdef MY_DEBUG_DEF
        std::cerr << "Failed to build interpreter." << std::endl;
#endif //MY_DEBUG_DEFF
        _statusModelInterpreterActivation = CODE_FAILED_BUILD_MODEL_FROM_FILE;
        return;
    }
    // Bind given context with interpreter.
    _modelInterpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpuContext.get());
    _modelInterpreter->SetNumThreads(1);
    if (_modelInterpreter->AllocateTensors() != kTfLiteOk) {
#ifdef MY_DEBUG_DEF
        std::cerr << "Failed to allocate tensors." << std::endl;
#endif //MY_DEBUG_DEF
        _statusModelInterpreterActivation = CODE_FAILED_BUILD_MODEL_FROM_FILE;
        return;
    }

    const auto *dims = _modelInterpreter->tensor(_modelInterpreter->inputs()[0])->dims;
    _widthInputLayerPoseNetModel = dims->data[2];
    _heightInputLayerPoseNetModel = dims->data[1];

    // ////// init _outputModelShape
    const auto &out_tensor_indices = _modelInterpreter->outputs();
    _outputModelShape.resize(out_tensor_indices.size());
#ifdef MY_DEBUG_DEF
    //for debugging
    std::cout << "out_tensor_indices.size() : " << out_tensor_indices.size() << std::endl;
    std::cout << "tensor_dims: " << _widthInputLayerPoseNetModel << ", " << _heightInputLayerPoseNetModel << "\n";
#endif //MY_DEBUG_DEF
    for (size_t i = 0; i < out_tensor_indices.size(); i++) {
        const auto *tensor = _modelInterpreter->tensor(out_tensor_indices[i]);
        // We are assuming that outputs tensor are only of type float.
        _outputModelShape[i] = tensor->bytes / sizeof(float);
    }
    // ////// END of the initing outputModelShape
}

void PoseDetectorWrapper::add_frame(cv::Mat frame) {
    _mutexRes.lock();
    if (_queueFrames.size() > MAX_COUNT_FRAMES_IN_QUEUE)
        _queueFrames.pop_front();

    _queueFrames.push_back(frame);
    _mutexProc.unlock();
    _mutexRes.unlock();
}

int PoseDetectorWrapper::start_pose_detection() {
    if (_isProcessThread)
        return CODE_STATUS_OK;
    else {
        if (_statusModelInterpreterActivation == CODE_STATUS_OK) {
            _isProcessThread = true;
            process_pose_detection();
            return CODE_STATUS_OK;
        } else
            return _statusModelInterpreterActivation;
    }
}

int PoseDetectorWrapper::stop_pose_detection() {
    _isProcessThread = false;
    _queueDetectedPoses.clear();
    return CODE_STATUS_OK;
}

std::vector<uint8_t> PoseDetectorWrapper::get_input_data_from_frame(cv::Mat &inputFrame) {
    cv::Mat resizedFrame;
    cv::resize(inputFrame, resizedFrame, cv::Size(_widthInputLayerPoseNetModel, _heightInputLayerPoseNetModel));
    std::cout << "Ok1\n";
    cv::cvtColor(resizedFrame, resizedFrame, cv::COLOR_BGRA2RGB);
    std::cout << "OK2\n";
    cv::Mat flat = resizedFrame.reshape(1, resizedFrame.total() * resizedFrame.channels());
    std::cout << "Flat: " << flat.rows << " " << flat.cols << "\n";
    std::vector<uint8_t> inputDataVector = resizedFrame.isContinuous() ? flat : flat.clone();
    std::cout << "Vector: " << inputDataVector.size() << "\n";
    return inputDataVector;
}

std::vector<float> PoseDetectorWrapper::get_raw_output_data_from_model(const std::vector<uint8_t> &inputData) {
    std::vector<float> outputData;
    std::cout << "Step OK OK\n";
    auto *input = _modelInterpreter->typed_input_tensor<uint8_t>(0);
    std::cout << "Step OK OK2\n";
    std::memcpy(input, inputData.data(), inputData.size());
    std::cout << "Step OK OK3\n";
    _modelInterpreter->Invoke();
    // TODO when I run program as su _modelInterpreter is stops thread and just wait,
    // TODO when, I run without su it crash with 'Segmentation fault'
    std::cout << "Step OK OK4\n";


    const auto &output_indices = _modelInterpreter->outputs();
    const int num_outputs = output_indices.size();
    int out_idx = 0;
    for (int i = 0; i < num_outputs; ++i) {
        const auto *outTensor = _modelInterpreter->tensor(output_indices[i]);
        assert(outTensor != nullptr);
        if (outTensor->type == kTfLiteUInt8) {
            const int num_values = outTensor->bytes;
            outputData.resize(out_idx + num_values);
            const uint8_t *output = _modelInterpreter->typed_output_tensor<uint8_t>(i);
            for (int j = 0; j < num_values; ++j) {
                outputData[out_idx++] =
                        (output[j] - outTensor->params.zero_point) * outTensor->params.scale;
            }
        } else if (outTensor->type == kTfLiteFloat32) {
            const int num_values = outTensor->bytes / sizeof(float);
            outputData.resize(out_idx + num_values);
            const float *output = _modelInterpreter->typed_output_tensor<float>(i);
            for (int j = 0; j < num_values; ++j) {
                outputData[out_idx++] = output[j];
            }
        }
#ifdef MY_DEBUG_DEF
        else {
            std::cerr << "Tensor " << outTensor->name
                      << " has unsupported output type: " << outTensor->type << std::endl;
        }
#endif //MY_DEBUG_DEF
    }
    return outputData;
}

std::vector<DetectedPose>
PoseDetectorWrapper::get_pose_estimate_from_output_raw_data(const std::vector<float> &outputRawDataVector,
                                                            const float &threshold) {
    const auto *result_raw = outputRawDataVector.data();
    std::vector<std::vector<float>> results(_outputModelShape.size());
    int offset = 0;
    for (size_t i = 0; i < _outputModelShape.size(); ++i) {
        const size_t size_of_output_tensor_i = _outputModelShape[i];
        results[i].resize(size_of_output_tensor_i);
        std::memcpy(results[i].data(), result_raw + offset, sizeof(float) * size_of_output_tensor_i);
        offset += size_of_output_tensor_i;
    }
    std::vector<DetectedPose> inf_results;
    int n = lround(results[3][0]);
    for (int i = 0; i < n; i++) {
        float overall_score = results[2][i];
        if (overall_score > threshold) {
            DetectedPose result;
            std::copy(results[1].begin() + (17 * i), results[1].begin() + (17 * i) + 16,
                      std::back_inserter(result.keypointScores));
            std::copy(results[0].begin() + (17 * 2 * i), results[0].begin() + (17 * 2 * i) + 33,
                      std::back_inserter((result.keypointCoordinates)));
            inf_results.push_back(result);
        }
    }
    return inf_results;
}

void PoseDetectorWrapper::process_pose_detection() {
    _keyPointDetectionThread = std::thread([this]() {
        cv::Mat currentFrame;
        while (_isProcessThread) {
            _mutexProc.lock();
            _mutexRes.lock();
            currentFrame = _queueFrames.front();
            _queueFrames.pop_front();
            _mutexRes.unlock();
            std::cout << "Thread works\n";

            std::vector<uint8_t> inputData = get_input_data_from_frame(currentFrame);
            std::cout << "Step OK!\n";
            std::vector<float> rawOutputData = get_raw_output_data_from_model(inputData);
            std::cout << "Step OK2!\n";
            std::vector<DetectedPose> detectedPoses = get_pose_estimate_from_output_raw_data(rawOutputData,
                                                                                             POSE_THRESHOLD);
            std::cout << "Step OK3!\n";
            if (_queueDetectedPoses.size() > MAX_COUNT_POSE_VECTORS_IN_QUEUE)
                _queueDetectedPoses.pop_front();
            _queueDetectedPoses.push_back(detectedPoses);
            if (!_queueFrames.empty()) {
                _mutexProc.unlock();
            }
            std::cout << "DONE THREAD LOOP \n";
        }
    });
}

void PoseDetectorWrapper::draw_last_pose_on_image(cv::Mat &frame) {
    float camera_width = frame.cols;
    float camera_height = frame.rows;
    if (!_queueDetectedPoses.empty()) {
        std::vector<int> k_x(17), k_y(17);
        const auto &green = cv::Scalar(0, 255, 0);
        for (auto &candidate : _queueDetectedPoses.front()) {
            for (int i = 0; i < 17; i++) {
                if (candidate.keypointScores[i] > POSE_THRESHOLD) {
                    float x_coordinate =
                            candidate.keypointCoordinates[(2 * i) + 1] * (camera_width / _widthInputLayerPoseNetModel);
                    float y_coordinate =
                            candidate.keypointCoordinates[2 * i] * (camera_height / _heightInputLayerPoseNetModel);
                    k_x[i] = static_cast<int>(x_coordinate);
                    k_y[i] = static_cast<int>(y_coordinate);
                    cv::circle(frame, cv::Point(k_x[i], k_y[i]), 0, green, 6, 1, 0);
                }
            }
        }
    }
}

