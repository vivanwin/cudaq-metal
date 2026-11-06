# Out-of-tree build of the Metal NVQIR backend for CUDA-Q (Apple Silicon only).
#
# Default mode builds against a pip-installed cudaq (the `cuda-quantum-cu13`
# macOS wheel) plus the CUDA-Q headers fetched from the matching source tag —
# no 600 MB toolkit needed. scripts/build_backend.sh wires this up:
#
#   cmake -B build -G Ninja \
#     -DCUDAQ_HEADERS_DIR=toolchain/cudaq-headers-<ver> \
#     -DCUDAQ_LIB_DIR=<venv>/lib/python3.X/site-packages/lib
#
# Alternatively, pass -DNVQIR_DIR=<toolkit>/lib/cmake/nvqir to build against an
# official CUDA-Q C++ toolkit install (needed only for nvq++ C++ workflows).
cmake_minimum_required(VERSION 3.22)

# Match the cudaq macOS wheels (tagged macosx_13_0_arm64), so the plugin loads
# on every macOS that cudaq itself supports. Must be set before project().
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Apple Silicon only")

# OBJCXX so the Objective-C++ Metal engine (metal_sv.mm) compiles in-tree.
project(cudaq-metal-backend LANGUAGES CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "cudaq-metal targets Apple Silicon macOS only.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)

set(METAL_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/MetalCircuitSimulator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metal_sv.mm)

if(NVQIR_DIR)
  # --- Official toolkit mode -------------------------------------------------
  find_package(NVQIR REQUIRED)
  nvqir_add_backend(metal false ${METAL_SOURCES})
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/targets/metal.yml
          DESTINATION ${NVQIR_SIMULATOR_CONFIG_PATH})
else()
  # --- pip-wheel mode (default) ----------------------------------------------
  set(CUDAQ_HEADERS_DIR "" CACHE PATH "CUDA-Q headers (scripts/fetch_cudaq_headers.py)")
  set(CUDAQ_LIB_DIR "" CACHE PATH "site-packages/lib of the installed cudaq wheel")
  if(NOT EXISTS "${CUDAQ_HEADERS_DIR}/nvqir/CircuitSimulator.h")
    message(FATAL_ERROR "Set CUDAQ_HEADERS_DIR (see scripts/fetch_cudaq_headers.py).")
  endif()
  if(NOT EXISTS "${CUDAQ_LIB_DIR}/libnvqir.dylib")
    message(FATAL_ERROR "Set CUDAQ_LIB_DIR to <site-packages>/lib of a cudaq install.")
  endif()

  add_library(nvqir-metal SHARED ${METAL_SOURCES})
  target_include_directories(nvqir-metal PRIVATE
    ${CUDAQ_HEADERS_DIR} ${CUDAQ_HEADERS_DIR}/nvqir)
  foreach(lib nvqir cudaq-common cudaq-operator cudaq-logger)
    target_link_libraries(nvqir-metal PRIVATE ${CUDAQ_LIB_DIR}/lib${lib}.dylib)
  endforeach()
  # The plugin lives next to cudaq's own dylibs in <site-packages>/lib, so
  # resolve them relative to ourselves, exactly like NVIDIA's own plugins do.
  set_target_properties(nvqir-metal PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH ON)
endif()

# The engine needs the Apple GPU frameworks.
target_link_libraries(nvqir-metal PRIVATE
  "-framework Metal" "-framework Foundation")

# The Metal engine relies on ARC for its retained Objective-C members (command
# buffer / encoder held across gate calls). Without -fobjc-arc those objects are
# autoreleased and freed mid-flight ("encoder released without endEncoding").
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/metal_sv.mm
  PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
