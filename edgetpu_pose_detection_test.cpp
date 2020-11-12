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

int main() {
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

    const auto* dims = model_interpreter->tensor(model_interpreter->inputs()[0])->dims;
    std::cout << "Dims info: " << dims->data[0] << " " << dims->data[1] << " " << dims->data[2] << " " << dims->data[3] << "\n";

    cv::Mat image_file = cv::imread("src/images/4.jpg");
    cv::resize(image_file, image_file, cv::Size(dims->data[2], dims->data[1]));
    cv::Mat flat = image_file.reshape(1, image_file.total() * image_file.channels());
    std::vector<uint8_t> in_vec= image_file.isContinuous()? flat : flat.clone();

    std::vector<float> output_data;
    auto* input = model_interpreter->typed_input_tensor<uint8_t>(0);
    std::memcpy(input, in_vec.data(), in_vec.size());
    model_interpreter->Invoke();

    const auto& output_indices = model_interpreter->outputs();
    const int num_outputs = output_indices.size();
    int out_idx = 0;

    for (int i = 0; i < num_outputs; ++i) {
        const auto* out_tensor = model_interpreter->tensor(output_indices[i]);
        assert(out_tensor != nullptr);
        if (out_tensor->type == kTfLiteUInt8) {
            const int num_values = out_tensor->bytes;
            output_data.resize(out_idx + num_values);
            const uint8_t* output = model_interpreter->typed_output_tensor<uint8_t>(i);
            for (int j = 0; j < num_values; ++j) {
                output_data[out_idx++] =
                        (output[j] - out_tensor->params.zero_point) * out_tensor->params.scale;
            }
        } else if (out_tensor->type == kTfLiteFloat32) {
            const int num_values = out_tensor->bytes / sizeof(float);
            output_data.resize(out_idx + num_values);
            const float* output = model_interpreter->typed_output_tensor<float>(i);
            for (int j = 0; j < num_values; ++j) {
                output_data[out_idx++] = output[j];
            }
        } else {
            std::cerr << "Tensor " << out_tensor->name
                      << " has unsupported output type: " << out_tensor->type << std::endl;
        }
    }

    std::cout << "Result: " << output_data.size() + "\n\n";
    for (int i = 0; i < output_data.size(); i++) {
        std::cout << output_data[i] << "\n";
    }
    return 0;
}