//
// Created by biba_bo on 2020-11-06.
//

#include <iostream>
#include <edgetpu.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/stderr_reporter.h>
#include <kernels/register.h>

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(const tflite::FlatBufferModel &model,
                                        edgetpu::EdgeTpuContext *edgetpu_context);

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(const tflite::FlatBufferModel &model, edgetpu::EdgeTpuContext *edgetpu_context) {
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
    std::cout << "Hello World :D" << "\n";
    const std::string model_path =
            "src/models/resnet/posenet_resnet_50_640_480_16_quant_edgetpu_decoder.tflite";
//            "src/models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite";
    std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
            edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    std::unique_ptr<tflite::Interpreter> model_interpreter =
            BuildEdgeTpuInterpreter(*model, edgetpu_context.get());
    return 0;
}