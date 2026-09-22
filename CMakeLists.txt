cmake_minimum_required(VERSION 3.20)
project(SAGE LANGUAGES CXX ASM)

add_library(SMEPlanCore STATIC lib/Core/CompilerGemmPolicy.cpp)
target_include_directories(SMEPlanCore PUBLIC include)
target_compile_features(SMEPlanCore PUBLIC cxx_std_17)
set_target_properties(SMEPlanCore PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(SME_ENABLE_SANITIZERS "Check the standalone C++ policy with ASan/UBSan" OFF)
if(SME_ENABLE_SANITIZERS)
  target_compile_options(SMEPlanCore PUBLIC -fsanitize=address,undefined)
  target_link_options(SMEPlanCore PUBLIC -fsanitize=address,undefined)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(SMEPlanPolicyTests tests/cpp/CompilerGemmPolicyTest.cpp)
  target_link_libraries(SMEPlanPolicyTests PRIVATE SMEPlanCore)
  add_test(NAME SMEPlanPolicyTests COMMAND SMEPlanPolicyTests)
endif()

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec('torch').origin).parent/'lib')"
    OUTPUT_VARIABLE TORCH_LIBRARY_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  find_program(BREW brew REQUIRED)
  execute_process(COMMAND "${BREW}" --prefix libomp
    OUTPUT_VARIABLE OMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  if(NOT EXISTS "${TORCH_LIBRARY_DIR}/libomp.dylib")
    message(FATAL_ERROR "The selected Torch installation must provide libomp.dylib")
  endif()
  add_executable(compiler-device-calibration adapters/runtime/CompilerDeviceCalibration.cpp)
  target_compile_features(compiler-device-calibration PRIVATE cxx_std_17)
  target_compile_options(compiler-device-calibration PRIVATE -O3 -Wall -Wextra -Werror
    -Xpreprocessor -fopenmp)
  target_include_directories(compiler-device-calibration PRIVATE "${OMP_PREFIX}/include")
  target_link_libraries(compiler-device-calibration PRIVATE "${TORCH_LIBRARY_DIR}/libomp.dylib")
  set_target_properties(compiler-device-calibration PROPERTIES BUILD_RPATH "${TORCH_LIBRARY_DIR}")
  # Some Torch wheels retain a build-machine install name in libomp. Resolve
  # that same library through our rpath without modifying the wheel itself.
  execute_process(COMMAND otool -D "${TORCH_LIBRARY_DIR}/libomp.dylib"
    OUTPUT_VARIABLE OMP_INSTALL_NAMES OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  string(REPLACE "\n" ";" OMP_INSTALL_NAMES "${OMP_INSTALL_NAMES}")
  list(GET OMP_INSTALL_NAMES 1 OMP_INSTALL_NAME)
  add_custom_command(TARGET compiler-device-calibration POST_BUILD
    COMMAND install_name_tool -change "${OMP_INSTALL_NAME}" "@rpath/libomp.dylib"
      "$<TARGET_FILE:compiler-device-calibration>")

  add_executable(sme-probe adapters/runtime/SmeProbe.cpp adapters/runtime/SmeProbe.S)
  target_compile_features(sme-probe PRIVATE cxx_std_17)
  target_compile_options(sme-probe PRIVATE -O2 -march=armv9.2-a+sme)
  set_target_properties(compiler-device-calibration sme-probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.build/native")
endif()
