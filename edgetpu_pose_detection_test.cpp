//
// Created by biba_bo on 2020-11-06.
//

#include <iostream>
#include <edgetpu.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/stderr_reporter.h>
#include <kernels/register.h>


std::unique_ptr<tflite::Interpreter>
BuildEdgeTpuInterpreter(const tflite::FlatBufferModel &model, edgetpu::EdgeTpuContext *edgetpu_context) {
    tflite::ops::builtin::BuiltinOpResolver resolver;
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

int main() {
    const std::string model_path =
            "src/models/resnet/posenet_resnet_50_640_480_16_quant_edgetpu_decoder.tflite";
    //        "src/models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite";

    std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

    if (model == nullptr) {
        std::cerr << "Fail to build FlatBufferModel from wile: " << model_path << "\n";
        std::abort();
    } else {
        std::cout << "Model from " << model_path << " file successfully built\n";
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
    return 0;
}