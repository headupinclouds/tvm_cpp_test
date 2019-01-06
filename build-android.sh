#!/bin/bash
#
# https://discuss.tvm.ai/t/tvm-ndk-arm64-compile-without-rpc/65/3?u=headupinclouds
#
# if exec_gpu:
#     # Mobile GPU
#     target = 'opencl'
#     target_host = "llvm -target=%s-linux-android" % arch
# else:
#     # Mobile CPU
#     target = "llvm -target=%s-linux-android" % arch
#     target_host = None

set -e

# See "Cross Compiling for Android with the NDK".
# https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html#id20  
TOOLCHAIN=${PWD}/android.toolchain.cmake

if [ -z "${TVM_NDK_CC}" ]; then
    echo "Must set TVM_NDK_CC to point to vaid android compiler" 2>&1
    exit 1
fi

cmake_args=(
    -H.
    -DCMAKE_VERBOSE_MAKEFILE=ON
    -DCMAKE_BUILD_TYPE=Debug
    -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN}
)

platforms=(
    # This reports the wrong result
    "vulkan;vulkan;llvm -target=aarch64-linux-android"
    # The Android CPU path works fine
    "cpu;llvm -target=aarch64-linux-android;None"
)

exe_dir=/data/local/tmp

for ((i=0; i < ${#platforms[@]}; i++)); do

    IFS=';'; fields=(${platforms[i]}); unset IFS

    flag="${fields[0]}"
    target="${fields[1]}"
    target_host="${fields[2]}"

    echo "Test flag: ${flag}"
    echo "Test target: ${target}"
    echo "Test target host: ${target_host}"

    # "detoxify" special characters for a clean dir name
    toolchain_name=$(sed -E 's/( |-|=)/_/g' <<< "${target}")
    target_dir=_builds/ndk/${toolchain_name}
    target_upper=$(echo $flag | tr '[:lower:]' '[:upper:]')

    # adb pull /system/vendor/lib64

    mkdir -p ${target_dir}
    [ -f ${target_dir}/CMakeCache.txt ] && rm ${target_dir}/CMakeCache.txt
    cmake ${cmake_args[@]} -B${target_dir} -DTCT_USE_${target_upper}=ON && cmake --build ${target_dir}

    pycmd="${PWD}/from_mxnet.py"

    (
	cd $target_dir
	python ${pycmd} --target="${target}" --target-host="${target_host}"
    )

    exe_toolchain_dir=${exe_dir}/${toolchain_name}
    adb push ${target_dir} ${exe_dir}
    adb shell "cd ${exe_toolchain_dir} && ./tvm_deploy_gpu_sample \${PWD}/from_mxnet.so"

    echo "adb pull output..."

    result_dir=_results/ndk/
    mkdir -p ${result_dir}
    (
	cd ${result_dir}
	adb pull ${exe_toolchain_dir}
    )

    exit

done
