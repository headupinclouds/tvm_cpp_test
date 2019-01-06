# See "Cross Compiling for Android with the NDK".
# https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html#id20  

set(ndk_home "/dl/android-ndk-r18")
if(NOT EXISTS ${ndk_home})
  message(FATAL_ERROR "Please install NDK r18 in ${ndk_home}, see README.rst for details")
endif()

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 24) # API level
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
set(CMAKE_ANDROID_NDK "${ndk_home}")
set(CMAKE_ANDROID_STL_TYPE c++_static)
