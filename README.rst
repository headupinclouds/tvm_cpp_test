Ubuntu
------

.. code-block:: none

  > lsb_release -a
  No LSB modules are available.
  Distributor ID:  Ubuntu
  Description:     Ubuntu 18.04.1 LTS
  Release:         18.04
  Codename:        bionic

.. code-block:: none

  > g++ --version
  g++ (Ubuntu 7.3.0-27ubuntu1~18.04) 7.3.0
  Copyright (C) 2017 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Packages
--------

Install the required system packages:

.. code-block:: none

    > sudo apt-get install -y \
    apt-transport-https \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libatlas-base-dev \
    libcurl4-openssl-dev \
    libjemalloc-dev \
    liblapack-dev \
    libopenblas-dev \
    libopencv-dev \
    libzmq3-dev \
    ninja-build \
    software-properties-common \
    sudo \
    unzip \
    wget \
    libtinfo-dev \
    zlib1g-dev

Conda
-----

Install ``conda``:

.. code-block:: none

  [/dl]> wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
  [/dl]> chmod +x Anaconda3-5.3.1-Linux-x86_64.sh

Run the ``*.sh`` script and set ``/dl/anaconda`` as an install destination:

.. code-block:: none

  [/dl]> ./Anaconda3-5.3.1-Linux-x86_64.sh

Activate conda:

.. code-block:: none

  > source /dl/anaconda/etc/profile.d/conda.sh

Test the installation (activate it first):

.. code-block:: none

  > conda list

.. notes::

  * https://conda.io/docs/user-guide/install/linux.html#installing-on-linux
  * https://conda.io/docs/user-guide/install/test-installation.html

Conda environment
-----------------

Load conda:

.. code-block:: none

  > source /dl/anaconda/etc/profile.d/conda.sh

Check for existing environments:

.. code-block:: none

  > conda info --envs

If ``dl`` environment is present and you want to remove it:

.. code-block:: none

  > conda remove --name dl --all

Create a fresh dl environment for a reproducible test sandbox:

.. code-block:: none

  > conda create --name dl python=3.6

Activate the environment: 

.. code-block:: none

  > conda activate dl
  (dl)>

.. note::

  * https://conda.io/docs/user-guide/tasks/manage-environments.html

Pip dependencies
----------------

Install the python dependencies using``pip``:

.. code-block:: none

  (dl)> pip install \
      cpplint==1.3.0 \
      h5py==2.8.0rc1 \
      nose \
      nose-timer \
      'numpy<=1.15.2,>=1.8.2' \
      pylint==1.8.3 \
      'requests<2.19.0,>=2.18.4' \
      scipy==1.0.1 \
      boto3 \
      decorator \
      Pillow \
      matplotlib

There should be no ``mxnet`` or ``nnvm`` package installed on the system. Run a sanity check:

.. code-block:: none

  (dl)> python -c 'import mxnet'
  Traceback (most recent call last):
    File "<string>", line 1, in <module>
  ModuleNotFoundError: No module named 'mxnet'

.. code-block:: none

  (dl)> python -c 'import nnvm'
  Traceback (most recent call last):
    File "<string>", line 1, in <module>
  ModuleNotFoundError: No module named 'nnvm'

MXNet
-----

Get the ``mxnet`` sources:

.. code-block:: none

  (dl)> cd /dl
  (dl) [/dl]> git clone https://github.com/apache/incubator-mxnet mxnet
  (dl) [/dl]> cd mxnet
  (dl) [/dl/mxnet]>

Lock the version:

.. code-block:: none

  (dl) [/dl/mxnet]> git branch test-1eb3344 1eb3344
  (dl) [/dl/mxnet]> git checkout test-1eb3344
  (dl) [/dl/mxnet]> git submodule update --init --recursive

Run the ``mxnet`` build:

.. code-block:: none

  (dl) [/dl/mxnet]> cmake -H. -B_builds -DUSE_LAPACK=OFF -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=ON -DUSE_F16C=OFF
  (dl) [/dl/mxnet]> cmake --build _builds -j $(nproc)

Check the created ``libmxnet.so`` library:

.. code-block:: none

  (dl) [/dl/mxnet]> ls _builds/libmxnet.so

We have to create a symbolic link, because that's where the Python code expects the library:

.. code-block:: none

  (dl) [/dl/mxnet]> mkdir -p lib
  (dl) [/dl/mxnet]> ln -s /dl/mxnet/_builds/libmxnet.so /dl/mxnet/lib/libmxnet.so

Test the import command:

.. code-block:: none

  (dl) [/dl/mxnet] > (PYTHONPATH=/dl/mxnet/python python -c 'import mxnet')

.. note::

  * https://mxnet.apache.org/install/ubuntu_setup.html#build-mxnet-from-source

LLVM
----

LLVM is needed for the TVM build:

.. code-block:: none

  (dl) > cd /dl
  (dl) [/dl]> wget http://releases.llvm.org/6.0.1/llvm-6.0.1.src.tar.xz
  (dl) [/dl]> tar xf llvm-6.0.1.src.tar.xz
  (dl) [/dl]> cmake -H/dl/llvm-6.0.1.src -B/dl/llvm-6.0.1.src/_builds -DCMAKE_INSTALL_PREFIX=/dl/llvm -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_BUILD_TYPE=Debug
  (dl) [/dl]> cmake --build /dl/llvm-6.0.1.src/_builds --target install -- -j $(nproc)

TVM
---

Build TVM:

.. code-block:: none

  (dl) > cd /dl/mxnet/3rdparty/tvm
  (dl) [/dl/mxnet/3rdparty/tvm]> cmake -H. -B_builds -DUSE_GRAPH_RUNTIME_DEBUG=ON -DUSE_CUDA=ON -DUSE_OPENCL=ON -DUSE_VULKAN=ON -DUSE_OPENGL=ON -DUSE_LLVM=/dl/llvm/bin/llvm-config -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_BUILD_TYPE=Debug
  (dl) [/dl/mxnet/3rdparty/tvm]> cmake --build _builds -- -j $(nproc)

Check the library ``libtvm.so``:

.. code-block:: none

  (dl) [/dl/mxnet/3rdparty/tvm]> ls ./_builds/libtvm.so

We have to create a symbolic link, because that's where Python code expects the libraries:

.. code-block:: none

  (dl) [/dl/mxnet/3rdparty/tvm]> mkdir -p lib nnvm/lib
  (dl) [/dl/mxnet/3rdparty/tvm]> ln -s /dl/mxnet/3rdparty/tvm/_builds/libtvm.so /dl/mxnet/3rdparty/tvm/lib/libtvm.so
  (dl) [/dl/mxnet/3rdparty/tvm]> ln -s /dl/mxnet/3rdparty/tvm/_builds/libnnvm_compiler.so /dl/mxnet/3rdparty/tvm/nnvm/lib/libnnvm_compiler.so

Test the import command:

.. code-block:: none

  (dl) [/dl/mxnet/3rdparty/tvm]> (PYTHONPATH=/dl/mxnet/3rdparty/tvm/python:/dl/mxnet/3rdparty/tvm/topi/python:/dl/mxnet/3rdparty/tvm/nnvm/python python -c 'import nnvm')

.. notes::

  * https://docs.tvm.ai/install/from_source.html#build-the-shared-library
  * https://docs.tvm.ai/install/from_source.html#python-package-installation

OpenCL
------

The OpenCL setup described here can be skipped for the end-to-end Vulkan test.

Install the OpenCL headers. For debian systems we can install OpenCL with ``apt-get``.

If we are building for an Android device, we can download the lib directly from
the device and grab the appropriate headers from the KhroosGroup site.

.. code-block:: none

  (dl) [/dl]> cd /dl
  (dl) [/dl]> git clone https://github.com/KhronosGroup/OpenCL-Headers.git

We must set ``CL_TARGET_OPENCL_VERSION`` appropriately to specify a particular version.

* https://github.com/KhronosGroup/OpenCL-Headers#compiling-for-a-specific-opencl-version

For OpenCL version 1.20 we would do the following:

.. code-block:: none

    #define CL_TARGET_OPENCL_VERSION 120
    #include <CL/opencl.h>


Android NDK
-----------

Here we install a recent Android NDK.
First, we will use the NDK to build a standalone toolchain that provides a single
compiler command for the TVM python build step
(i.e., ``export TVM_NDK_CC=/dl/toolchains/android-toolchain-arm64/bin/aarch64-linux-android-clang++``)
to produce our ``from_mxnet.so`` Android device inference library.
Then we will use the NDK directly through our minimal CMake toolchain to cross compile
the TMV inference application (``tvm_deploy_gpu_sample.cpp``) -- the thing that will
open and run the generated ``from_mxnet.so`` library on the Android device using the
``cat.bin`` file that was created from the ``cat.png`` in the python ``from_mxnet.py``
scrip.

.. code-block:: none

    (dl) [/dl] > cd /dl
    (dl) [/dl] > wget https://dl.google.com/android/repository/android-ndk-r18-linux-x86_64.zip
    (dl) [/dl] > unzip android-ndk-r18-linux-x86_64.zip
    (dl) [/dl] > cd /dl/android-ndk-r18/build/tools
    (dl) [/dl/android-ndk-r18/build/tools] > ./make-standalone-toolchain.sh --platform=android-24 --use-llvm --arch=arm64 --install-dir=/dl/toolchains/android-toolchain-arm64	 


The minimal NDK toolchain (``tvm_cpp_test/android.toolchain.cmake``) looks like this:

.. code-block:: none

    set(CMAKE_SYSTEM_NAME Android)
    set(CMAKE_SYSTEM_VERSION 24) # API level
    set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
    set(CMAKE_ANDROID_NDK "${ndk_home}")
    set(CMAKE_ANDROID_STL_TYPE c++_static)

Details can be found in the official CMake documentation here
`Cross Compiling for Android with the NDK <https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html#cross-compiling-for-android-with-the-ndk>`__


from_mxnet.py
-------------

.. code-block:: none

  (dl) > export OMP_NUM_THREADS=1

We reduce the OMP threads with the above environment variable as a workaround
for the following errors. This happens frequently enough in various tested
configurations using TVM to warrant making this a default setup step, although
it may not be necessary at some point in the future.

.. code-block:: none

  Assertion failure at kmp_runtime.cpp(6481): __kmp_team_pool == __null.
  OMP: Error #13: Assertion failure at kmp_runtime.cpp(6481).

.. code-block:: none

  RuntimeError: Compilation error:
  gcc: error trying to exec 'cc1plus': execvp: No such file or directory

Clone the C++ test repository (``tvm_cpp_test``), compile with NNVM/TVM and run:

.. code-block:: none

  (dl) > cd /dl
  (dl) [/dl]> git clone https://github.com/headupinclouds/tvm_cpp_test.git
  (dl) [/dl]> cd tvm_cpp_test
  (dl) [/dl/tvm_cpp_test]> source env.sh # set TVM_NDK_CC and PYTHONPATH
  (dl) [/dl/tvm_cpp_test]> ./build-host.sh # reproduce host=target builds for all back-ends
  (dl) [/dl/tvm_cpp_test]> ./build-android.sh # reproduce android vulkan failure


When we run the final build-android.sh script above, an arm64 development device
should be tethered to the host machine and accessible through ``adb``.  This build
script will cross compile the C++ executable, install it on the device, and run
inference on the centered cat image data.  It should report a max response at 282,
but the Vulkan back-end on Android reports an inccorect result:

.. code-block:: none

    The maximum position in output vector is: 669
    Expected 282 but got: 669

Note that the curret example will hang at this point.  Presumably some tear down
step is blocking on a thread or something similar.  To retrieve the results for
comparison, it is necessary to kill the application and run the remainder
of the script by hand manually to fetch the logged results.

.. code-block:: none

  (dl) > cd /dl/tvm_cpp_test
  (dl) [/dl/tvm_cpp_test]> mkdir -p _results/ndk && cd _results/ndk && adb pull /data/local/tmp/vulkan
  (dl) [/dl/tvm_cpp_test]> bash -fx ./cmp.sh # compare the android output with the ubuntu outputf

The final ``cmp.sh`` script will ``diff`` ubuntu vs android vulkan output line by line.  By default, it output normalized differences > 0.1.
