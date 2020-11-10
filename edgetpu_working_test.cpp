//
// Created by biba_bo on 2020-11-10.
//

#include <iostream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "edgetpu.h"

int main(int argc, char** argv){

    // check for correct CLI input
//    if (argc != 3) {
//        std::cout << "Invalid number of arguments.\nUsage: " << argv[0] << " <tflite model> <input value>\n";
//        exit(0);
//    }
    const std::string model_path =
            "src/models/resnet/posenet_resnet_50_640_480_16_quant_edgetpu_decoder.tflite";

    auto tpu_context = edgetpu::EdgeTpuManager::GetSingleton()->NewEdgeTpuContext();
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    // load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

    if(!model){
        std::cout << "Failed to mmap model\n";
        exit(0);
    }

    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

    // Build the interpreter
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk){
        std::cout << "Failed to build interpreter\n";
        exit(0);
    }
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, tpu_context.get());

    // Allocate memory
    if(interpreter->AllocateTensors() != kTfLiteOk){
        std::cout << "Failed to allocate tensors\n";
        exit(0);
    }

    // Get the input from the CL
    float real_input = strtof(argv[2], NULL);

    // Quantize the inputs
    const std::vector<int>& inputs = interpreter->inputs();
    TfLiteTensor* input_tensor = interpreter->tensor(inputs[0]);
    const TfLiteQuantizationParams& input_params = input_tensor->params;
    uint8_t quant_input = (real_input / input_params.scale) + input_params.zero_point;
    interpreter->typed_tensor<uint8_t>(4)[0] = quant_input;

    // Invoke the interpreter
    interpreter->Invoke();

    // Get the output from the model
    uint8_t quant_output = interpreter->typed_tensor<uint8_t>(5)[0];

    // Dequantize the outputs
    const std::vector<int>& outputs = interpreter->outputs();
    TfLiteTensor* output_tensor = interpreter->tensor(outputs[0]);
    const TfLiteQuantizationParams& output_params = output_tensor->params;
    float real_output = output_params.scale * (quant_output - output_params.zero_point);

    // Free resources
    interpreter.reset();
    tpu_context.reset();

    // Do something with the output
    std::cout << "Result: " << real_output << "\n";

    return 0;
}