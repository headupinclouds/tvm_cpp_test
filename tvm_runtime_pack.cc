/*!
 * \brief This is an all in one TVM runtime file.
 *
 *   You only have to use this file to compile libtvm_runtime to
 *   include in your project.
 *
 *  - Copy this file into your project which depends on tvm runtime.
 *  - Compile with -std=c++11
 *  - Add the following include path
 *     - /path/to/tvm/include/
 *     - /path/to/tvm/3rdparty/dmlc-core/include/
 *     - /path/to/tvm/3rdparty/dlpack/include/
 *   - Add -lpthread -ldl to the linked library.
 *   - You are good to go.
 *   - See the Makefile in the same folder for example.
 *
 *  The include files here are presented with relative path
 *  You need to remember to change it to point to the right file.
 *
 */
#include "../../src/runtime/c_runtime_api.cc"
#include "../../src/runtime/cpu_device_api.cc"
#include "../../src/runtime/workspace_pool.cc"
#include "../../src/runtime/module_util.cc"
#include "../../src/runtime/module.cc"
#include "../../src/runtime/registry.cc"
#include "../../src/runtime/file_util.cc"
#include "../../src/runtime/threading_backend.cc"
#include "../../src/runtime/thread_pool.cc"
#include "../../src/runtime/ndarray.cc"

// NOTE: all the files after this are optional modules
// that you can include remove, depending on how much feature you use.

// Likely we only need to enable one of the following
// If you use Module::Load, use dso_module
// For system packed library, use system_lib_module
#include "../../src/runtime/dso_module.cc"
#include "../../src/runtime/system_lib_module.cc"

// Graph runtime
#include "../../src/runtime/graph/graph_runtime.cc"

#if defined(TVM_USE_GRAPH_RUNTIME_DEBUG)
#  include "../../src/runtime/graph/debug/graph_runtime_debug.cc"
#endif

#if defined(TVM_USE_RPC)
#  include "../../src/runtime/rpc/rpc_session.cc"
#  include "../../src/runtime/rpc/rpc_event_impl.cc"
#  include "../../src/runtime/rpc/rpc_server_env.cc"
#endif

#if defined(TVM_CUDA_RUNTIME)
#  include "../../src/runtime/cuda/cuda_device_api.cc"
#  include "../../src/runtime/cuda/cuda_module.cc"
#endif

#if defined(TVM_METAL_RUNTIME)
#  include "../../src/runtime/metal/metal_device_api.mm"
#  include "../../src/runtime/metal/metal_module.mm"
#endif

#if defined(TVM_OPENCL_RUNTIME)
#  include "../../src/runtime/opencl/opencl_device_api.cc"
#  include "../../src/runtime/opencl/opencl_module.cc"
#endif

#if defined(TVM_OPENGL_RUNTIME)
#  include "../../src/runtime/opengl/opengl_device_api.cc"
#  include "../../src/runtime/opengl/opengl_module.cc"
#endif

#if defined(TVM_VULKAN_RUNTIME)
#  include "../../src/runtime/vulkan/vulkan_device_api.cc"
#  include "../../src/runtime/vulkan/vulkan_module.cc"
#endif
