# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(matx_stf CUDA CXX)

# This example uses the CMake Package Manager (CPM) to simplify fetching CCCL from GitHub
# For more information, see https://github.com/cpm-cmake/CPM.cmake
include(cmake/CPM.cmake)

# This will automatically clone MatX from GitHub and make the exported cmake targets available
CPMAddPackage(
  NAME MatX
  GITHUB_REPOSITORY "sidelnik/MatX"
  GIT_TAG cudastf
)

# Default to building for the GPU on the current system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(matx_stf matx_stf.cu)
target_link_libraries(matx_stf PRIVATE matx::matx)
