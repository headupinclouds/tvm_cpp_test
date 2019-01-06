cmake_minimum_required(VERSION 3.2)
project(tvm_cpp_test)

include_directories(/dl/mxnet/3rdparty/tvm/3rdparty/dlpack/include)
include_directories(/dl/mxnet/3rdparty/tvm/3rdparty/dmlc-core/include)
include_directories(/dl/mxnet/3rdparty/tvm/include)

function(print_cmake_vars)
  get_cmake_property(_variableNames VARIABLES)
  list (SORT _variableNames)
  foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endfunction()

# add include so that relative paths in the upstream tvm_runtime_pack.cc will work
include_directories(/dl/mxnet/3rdparty/tvm/apps/howto_deploy)

add_executable(
  tvm_deploy_gpu_sample
  tvm_deploy_gpu_sample.cpp

  # Adopt tvm_runtime_pack.cc for local modifications
  tvm_runtime_pack.cc
  # This could eventually go back here:
  #/dl/mxnet/3rdparty/tvm/apps/howto_deploy/tvm_runtime_pack.cc
  )

find_package(Threads REQUIRED)
target_link_libraries(tvm_deploy_gpu_sample PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

option(TCT_USE_CPU "Use cpu runtime" OFF)
if(TCT_USE_CPU)
  # currently nothing is required
  target_compile_definitions(tvm_deploy_gpu_sample PUBLIC TVM_CPU_RUNTIME=1)
endif()

option(TCT_USE_OPENGL "Use opengl runtime" OFF)
if(TCT_USE_OPENGL)
  find_package(OpenGL REQUIRED)
  target_link_libraries(tvm_deploy_gpu_sample PUBLIC OpenGL::OpenGL)

  find_package(glfw3)
  target_link_libraries(tvm_deploy_gpu_sample PUBLIC glfw)

  target_compile_definitions(tvm_deploy_gpu_sample PUBLIC TVM_OPENGL_RUNTIME=1)
endif()

option(TCT_USE_OPENCL "Use opencl runtime" OFF)
if(TCT_USE_OPENCL)
  find_package(OpenCL REQUIRED)
  target_link_libraries(tvm_deploy_gpu_sample PUBLIC OpenCL::OpenCL)
  target_compile_definitions(tvm_deploy_gpu_sample PUBLIC TVM_OPENCL_RUNTIME=1)
endif()

option(TCT_USE_VULKAN "Use vulkan runtime" OFF)
if(TCT_USE_VULKAN)
  find_package(Vulkan REQUIRED)
  target_link_libraries(tvm_deploy_gpu_sample PUBLIC Vulkan::Vulkan)
  target_compile_definitions(tvm_deploy_gpu_sample PUBLIC TVM_VULKAN_RUNTIME=1)
endif()

option(TCT_USE_CUDA "Use cuda runtime" OFF)
if(TCT_USE_CUDA)
  find_package(CUDA REQUIRED)
  #print_cmake_vars()
  target_compile_definitions(tvm_deploy_gpu_sample PUBLIC TVM_CUDA_RUNTIME=1)
  target_include_directories(tvm_deploy_gpu_sample PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(tvm_deploy_gpu_sample PUBLIC
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CURAND_LIBRARY})
endif()

option(TCT_USE_METAL "Use metal runtime" OFF)
if(TCT_USE_METAL)
  find_package(Metal REQUIRED)
  target_link_libraries(tvm_deploy_gpu_sample PUBLIC Metal::Metal)
  target_compile_definitions(tvm_deploy_gpu_sample PUBLIC TVM_METAL_RUNTIME=1)
endif()

option(TCT_USE_GRAPH_RUNTIME_DEBUG "use debug runtime" ON)
target_compile_definitions(tvm_deploy_gpu_sample PUBLIC TVM_USE_GRAPH_RUNTIME_DEBUG=1)

enable_testing()
add_test(NAME tvm_deploy_gpu_sample COMMAND tvm_deploy_gpu_sample)
