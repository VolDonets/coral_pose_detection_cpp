//
// Created by biba_bo on 2020-11-06.
//

#include <iostream>

#include <opencv2/opencv.hpp>

#include <edgetpu.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "kernels/register.h"

#include "posenet/posenet_decoder_op.h"


std::unique_ptr<tflite::Interpreter>
BuildEdgeTpuInterpreter(const tflite::FlatBufferModel &model, edgetpu::EdgeTpuContext *edgetpu_context) {
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(coral::kPosenetDecoderOp, coral::RegisterPosenetDecoderOp());
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter." << std::endl;
    }
    // Bind given context with interpreter.
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
    interpreter->SetNumThreads(1);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
    }
    return interpreter;
}

struct PoseCandidate {
    std::vector<float> keypoint_scores;
    std::vector<float> keypoint_coordinates;

};

void img_overlay(cv::Mat &frame, const std::vector<PoseCandidate> &ret, const float &keypoint_threshold,
                 const float &inp_width, const float &inp_height, const float &camera_width,
                 const float &camera_height,
                 const std::string &image_path, const std::string &image_name, const std::string &image_type) {
    std::vector<int> k_x(17), k_y(17);
    const auto &green = cv::Scalar(0, 255, 0);
    for (auto &candidate : ret) {
        for (int i = 0; i < 17; i++) {
            if (candidate.keypoint_scores[i] > keypoint_threshold) {
                float x_coordinate = candidate.keypoint_coordinates[(2 * i) + 1] * (camera_width / inp_width);
                float y_coordinate = candidate.keypoint_coordinates[2 * i] * (camera_height / inp_height);
                k_x[i] = static_cast<int>(x_coordinate);
                k_y[i] = static_cast<int>(y_coordinate);
                cv::circle(frame, cv::Point(k_x[i], k_y[i]), 0, green, 6, 1, 0);
            }
        }
    }
    cv::imwrite((image_path + "detected_" + image_name + image_type), frame);
}

int main() {
    const std::string image_path = "src/images/";
    const std::string image_name = "6";
    const std::string image_type = ".jpg";
    const std::string model_path =
            //        "src/models/resnet/posenet_resnet_50_640_480_16_quant_edgetpu_decoder.tflite";
            "src/models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite";

    std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

    if (model == nullptr) {
        std::cerr << "FAIL to build FlatBufferModel from file!\n";
        std::abort();
    } else {
        std::cout << "Model from file SUCCESSFULLY built\n";
    }

    const auto &available_tpus = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    std::cout << "Available tpus here is " << available_tpus.size() << "\n"; // hopefully we'll see 1 here
    std::cout << "Tpu type: " << available_tpus[0].type << "\n";
    std::cout << "Tpu path: " << available_tpus[0].path << "\n";

    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
            edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
                    available_tpus[0].type, available_tpus[0].path);;

    std::unique_ptr<tflite::Interpreter> model_interpreter =
            BuildEdgeTpuInterpreter(*model, edgetpu_context.get());

    const auto *dims = model_interpreter->tensor(model_interpreter->inputs()[0])->dims;
    std::cout << "Dims info: " << dims->data[0] << " " << dims->data[1] << " " << dims->data[2] << " " << dims->data[3]
              << "\n";

    // ////// init m_output_shape
    std::vector<size_t> m_output_shape;
    const auto &out_tensor_indices = model_interpreter->outputs();
    m_output_shape.resize(out_tensor_indices.size());
    //for debugging
    std::cout << "out_tensor_indices.size() : " << out_tensor_indices.size() << std::endl;
    for (size_t i = 0; i < out_tensor_indices.size(); i++) {
        const auto *tensor = model_interpreter->tensor(out_tensor_indices[i]);
        // We are assuming that outputs tensor are only of type float.
        m_output_shape[i] = tensor->bytes / sizeof(float);
    }
    // ////// END of the initing m_output_shape

    cv::Mat image_file_normal = cv::imread((image_path + image_name + image_type));
    cv::Mat image_file;
    cv::resize(image_file_normal, image_file, cv::Size(dims->data[2], dims->data[1]));
    cv::Mat flat = image_file.reshape(1, image_file.total() * image_file.channels());
    std::vector<uint8_t> in_vec = image_file.isContinuous() ? flat : flat.clone();
    std::cout << "Vector: " << in_vec.size();

    std::vector<float> output_data;
    auto *input = model_interpreter->typed_input_tensor<uint8_t>(0);
    std::memcpy(input, in_vec.data(), in_vec.size());
    model_interpreter->Invoke();

    const auto &output_indices = model_interpreter->outputs();
    const int num_outputs = output_indices.size();
    int out_idx = 0;

    for (int i = 0; i < num_outputs; ++i) {
        const auto *out_tensor = model_interpreter->tensor(output_indices[i]);
        assert(out_tensor != nullptr);
        if (out_tensor->type == kTfLiteUInt8) {
            const int num_values = out_tensor->bytes;
            output_data.resize(out_idx + num_values);
            const uint8_t *output = model_interpreter->typed_output_tensor<uint8_t>(i);
            for (int j = 0; j < num_values; ++j) {
                output_data[out_idx++] =
                        (output[j] - out_tensor->params.zero_point) * out_tensor->params.scale;
            }
        } else if (out_tensor->type == kTfLiteFloat32) {
            const int num_values = out_tensor->bytes / sizeof(float);
            output_data.resize(out_idx + num_values);
            const float *output = model_interpreter->typed_output_tensor<float>(i);
            for (int j = 0; j < num_values; ++j) {
                output_data[out_idx++] = output[j];
            }
        } else {
            std::cerr << "Tensor " << out_tensor->name
                      << " has unsupported output type: " << out_tensor->type << std::endl;
        }
    }

    // output_data == inf_vec
    float threshold = 0.2;

    const auto *result_raw = output_data.data();
    std::vector<std::vector<float>> results(output_data.size());
    int offset = 0;
    for (size_t i = 0; i < m_output_shape.size(); ++i) {
        const size_t size_of_output_tensor_i = m_output_shape[i];
        results[i].resize(size_of_output_tensor_i);
        std::memcpy(results[i].data(), result_raw + offset, sizeof(float) * size_of_output_tensor_i);
        offset += size_of_output_tensor_i;
    }
    std::vector<PoseCandidate> inf_results;
    int n = lround(results[3][0]);
    for (int i = 0; i < n; i++) {
        float overall_score = results[2][i];
        if (overall_score > threshold) {
            PoseCandidate result;
            std::copy(results[1].begin() + (17 * i), results[1].begin() + (17 * i) + 16,
                      std::back_inserter(result.keypoint_scores));
            std::copy(results[0].begin() + (17 * 2 * i), results[0].begin() + (17 * 2 * i) + 33,
                      std::back_inserter((result.keypoint_coordinates)));
            inf_results.push_back(result);
        }
    }
    img_overlay(image_file_normal, inf_results, threshold,
                dims->data[2], dims->data[1],
                image_file_normal.cols, image_file_normal.rows,
                image_path, image_name, image_type);
    return 0;
}