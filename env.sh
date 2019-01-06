#!/bin/bash

mxnet_dir=/dl/mxnet
tvm_dir=${mxnet_dir}/3rdparty/tvm

export PYTHONPATH=${mxnet_dir}/python:${tvm_dir}/python:${tvm_dir}/topi/python:${tvm_dir}/nnvm/python
export TVM_NDK_CC=/dl/toolchains/android-toolchain-arm64/bin/aarch64-linux-android-clang++
