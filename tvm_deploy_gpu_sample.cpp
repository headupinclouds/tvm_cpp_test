#define TCT_SAVE_LAYERS 1
#define TCT_TEST_DEBUG_GET_OUTPUT 0

#include <string>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iomanip>

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#if TCT_SAVE_LAYERS

#include "GraphRuntime.h"

#endif // TCT_SAVE_LAYERS

// Iterator for N dimensional array
inline void iterate(int dimensions, int64_t* ordinates, int64_t* maximums)
{
    for (int d = dimensions - 1; d >= 0; d--)
    {
        if ((ordinates[d]+1) < maximums[d])
        {
            ordinates[d]++;
            break;
        }

        ordinates[d] = 0;
    }
}

// Print DLTensor in flat diff friendly 'value[i,j,k...] = value' format:
void print(DLTensor *layer_output, const std::vector<float> &values, std::ofstream &ofs)
{
    std::vector<int64_t> ord( layer_output->ndim , 0);
    for (int k = 0; k < static_cast<int>(values.size()); k++)
    {
        std::stringstream index, shape;
        for(int d = 0; d < ord.size(); d++)
        {
            index << ord[d];
            shape << layer_output->shape[d];
            if(d < ord.size()-1)
            {
                index << ", ";
                shape << ", ";
            }
        }

        std::stringstream ss;
        ss  << "value"
            << "["
            << index.str()
            << "] = "
            << "{"
            << shape.str()
            << "} = "
            << values[k]
#if TUI_TEST_DEBUG_GET_OUTPUT
            << " "
            << values2[k]
#endif
          ;

        ofs << ss.str() << std::endl;
    }
}

int main(int argc, char** argv) try
{
    using Clock = std::chrono::high_resolution_clock;
    using Timepoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;

    if (argc < 2)
    {
        std::cerr << "usage: tvm_deploy_gpu_sample /full/path/to/from_mxnet.so " << std::endl;
        return 1;
    }

    std::string lib = argv[1];

    const std::string json_file("from_mxnet.json");
    const std::string param_file("from_mxnet.params");
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(lib);

#if TCT_SAVE_LAYERS
    GraphRuntimePrivateStuff info;
#endif

    std::string json_data;
    {
        std::ifstream json_in(json_file.c_str(), std::ios::in);
        if (!json_in)
        {
            std::cerr << "Failed to read json file " << json_file << std::endl;
            return 1;
        }

#if TCT_SAVE_LAYERS
        dmlc::JSONReader json(&json_in);
        info.Load(&json);
        json_in.clear();                 // clear fail and eof bits
        json_in.seekg(0, std::ios::beg); // back to the start
#endif

        json_data.assign((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    }

    std::string params_data;
    {
        std::ifstream params_in(param_file.c_str(), std::ios::binary);
        if (!params_in)
        {
            std::cerr << "Failed to read param file " << param_file << std::endl;
            return 1;
        }

        params_data.assign((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    }

    TVMByteArray params_arr;
    params_arr.data = params_data.data();
    params_arr.size = params_data.length();

    constexpr int dtype_code = kDLFloat;
    constexpr int dtype_bits = 32;
    constexpr int dtype_lanes = 1;

    // If you receive an error stating "device_type
#if defined(TVM_OPENCL_RUNTIME)
    constexpr int device_type = kDLOpenCL;
#elif defined(TVM_OPENGL_RUNTIME)
    constexpr int device_type = 11; // kDLOpenGL;
#elif defined(TVM_VULKAN_RUNTIME)
    constexpr int device_type = kDLVulkan;
#elif defined(TVM_METAL_RUNTIME)
    constexpr int device_type = kDLMetal;
#elif defined(TVM_CUDA_RUNTIME)
    constexpr int device_type = kDLGPU;
#elif defined(TVM_CPU_RUNTIME)
    constexpr int device_type = kDLCPU;
#else
#  error Must define a valid TVM_<KIND>_RUNTIME flag, see CMakeLists.txt
#endif

    std::cout << "device_type " << int(device_type) << std::endl;

    constexpr int device_id = 0;

    //const char * runtime = "tvm.graph_runtime.create";
    const char* runtime = "tvm.graph_runtime_debug.create";

    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get(runtime))(json_data, mod_syslib, device_type, device_id);

    std::cout << "load_params" << std::endl;
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    DLTensor* x = nullptr;
    DLTensor* y = nullptr;

    const int n_samples = 1;

    // Configure input tensor for single 1x3x224x224 RGB image (floating point)
    const int in_ndim = 4;
    const int64_t in_shape[] = { 1, 3, 224, 224 };
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
    const size_t in_size = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3];
    std::vector<float> tvm_input(1 * in_size, 0);

    // load image data saved in binary to tvm_input array
    std::ifstream data_fin("cat.bin", std::ios::binary);
    if (!data_fin)
    {
        std::cerr << "Failed to read input file cat.bin" << std::endl;
        return 1;
    }

    data_fin.read(reinterpret_cast<char*>(tvm_input.data()), in_size * sizeof(float));

    // Configure output tensor for 1x100 softmax class "probability" vector
    const int out_ndim = 2;
    const int64_t out_shape[] = { 1, 1000 };
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);
    const size_t out_size = out_shape[0] * out_shape[1];
    std::vector<float> tvm_output(1 * out_size, 0);

    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    tvm::runtime::PackedFunc get_output_by_layer = mod.GetFunction("get_output_by_layer");
#if TCT_TEST_DEBUG_GET_OUTPUT
    tvm::runtime::PackedFunc debug_get_output = mod.GetFunction("debug_get_output");
#endif

    const int warmup = 10;

    int count = 0;
    double total = 0.0;
    for (int i = 0; i < n_samples; ++i)
    {
        std::cout << "iteration " << i << std::endl;

        std::cout << "set_input(data, x)" << std::endl;
        TVMArrayCopyFromBytes(x, tvm_input.data(), in_size * sizeof(float));

        set_input("data", x);

        std::cout << "run()" << std::endl;
        auto tic = Clock::now();
        run();
        auto toc = Clock::now();
        auto elapsed = Duration(toc - tic).count();
        if (i > warmup)
        {
            count++;
            total += elapsed;
            std::cout << "elapsed: " << elapsed << std::endl;
        }

        std::cout << "get_output(0, y)" << std::endl;
        get_output(0, y);

#if TCT_SAVE_LAYERS
        for (int j = 0, k = 0; j < info.attrs_.shape.size(); j += 1, k++)
        {
            DLTensor* layer_output = nullptr;
            DLTensor* layer_output2 = nullptr;

            auto& shape = info.attrs_.shape[j];
            auto code = info.attrs_.dltype;

            std::size_t total = shape.front();
            for (auto iter = shape.begin() + 1; iter != shape.end(); iter++)
            {
                total *= (*iter);
            }

            std::cout << "N=" << k << " total = " << total << std::endl;

            std::cout << "get_output_by_layer(" << j << ", layer_output);" << std::endl;
            std::vector<float> values(total);
            layer_output = get_output_by_layer(j, 0);
            TVMArrayCopyToBytes(layer_output, values.data(), values.size() * sizeof(float));

#if TCT_TEST_DEBUG_GET_OUTPUT
            // debug_get_output require pre-allocation:
            TVMArrayAlloc(shape.data(), shape.size(), dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &layer_output2);

            std::cout << "debug_get_output(" << j << ", layer_output);" << std::endl;
            auto result = debug_get_output(j, layer_output2);
            std::vector<float> values2(total);
            TVMArrayCopyToBytes(layer_output2, values2.data(), values2.size() * sizeof(float));
            TVMArrayFree(layer_output2);
#endif // TCT_TEST_DEBUG_GET_OUTPUT

            std::stringstream ss;
            ss << "tvm_" << std::setw(4) << std::setfill('0') << j << '_' << info.nodes_[j].name << ".txt";
            std::ofstream ofs(ss.str());
            if (ofs)
            {
    	        print(layer_output, values, ofs);
            }
        }
#endif // TCT_SAVE_LAYERS

        std::cout << "TVMArrayCopyToBytes(y, y_iter, out_size * sizeof(float));" << std::endl;
        float* y_iter = tvm_output.data();
        TVMArrayCopyToBytes(y, y_iter, out_size * sizeof(float));

        for (std::size_t i = 0; i < tvm_output.size(); i++)
        {
            std::cout << "score[" << i << "] = " << tvm_output[i] << std::endl;
        }

        // get the maximum position in output vector
        auto max_iter = std::max_element(y_iter, y_iter + 1000);
        auto max_index = std::distance(y_iter, max_iter);
        std::cout << "The maximum position in output vector is: " << max_index << std::endl;

        if (max_index != 282) // 282: 'tiger cat' (see synset.txt)
        {
            std::cerr << "Expected 282 but got: " << max_index << std::endl;
            exit(1);
        }
    }

    TVMArrayFree(x);
    TVMArrayFree(y);

    std::cout << "average: " << total / static_cast<double>(count) << std::endl;

    exit(0);
}
catch (const dmlc::Error& e)
{
    std::cerr << "error: " << e.what() << std::endl;
}
catch (const std::exception& e)
{
    std::cerr << "exception: " << e.what() << std::endl;
}
