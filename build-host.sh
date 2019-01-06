#!/bin/bash

set -e

cmake_args=(
    -H.
    -DCMAKE_VERBOSE_MAKEFILE=ON
    -DCMAKE_BUILD_TYPE=Debug
)

platforms=(
    "vulkan;vulkan;None"
    "cpu;llvm;None"
    "cuda;cuda;None"
    "opencl;opencl;None"
    "opengl;opengl;None"
    #"metal:metal"
)

#cmake ${cmake_args[@]} -DTCT_USE_OPENCL=1 && cmake --build _builds/

for info in ${platforms[@]}; do

    fields=(${info//;/ })
    flag=${fields[0]}
    target=${fields[1]}
    target_host=${fields[2]}

    target_dir=_builds/${target}
    target_upper=$(echo $flag | tr '[:lower:]' '[:upper:]')

    # Build the C++ executable:
    mkdir -p ${target_dir}
    #[ -f ${target_dir}/CMakeCache.txt ] && rm ${target_dir}/CMakeCache.txt
    cmake ${cmake_args[@]} -B${target_dir} -DTCT_USE_${target_upper}=ON && cmake --build ${target_dir}

    # Create the TVM shared library module:
    pycmd="${PWD}/from_mxnet.py"
    (
	cd $target_dir
	python ${pycmd} --target=${target} --target-host=${target_host}
    	./tvm_deploy_gpu_sample ${PWD}/from_mxnet.so
    )

    exit
done
