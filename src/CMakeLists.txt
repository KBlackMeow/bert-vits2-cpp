cmake_minimum_required(VERSION 3.16)

get_filename_component(BERT_VITS2_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
list(APPEND CMAKE_MODULE_PATH "${BERT_VITS2_PROJECT_ROOT}/cmake")

project(bert_vits2_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BERT_VITS2_WITH_ONNXRUNTIME "Build ONNX Runtime based Bert-VITS2 inference" ON)

set(BERT_VITS2_EXE_NAME "bert-vits2-project" CACHE STRING "Output executable name")

if(EXISTS "${BERT_VITS2_PROJECT_ROOT}/bin")
    set(BERT_VITS2_OUTPUT_DIR "${BERT_VITS2_PROJECT_ROOT}/bin")
else()
    set(BERT_VITS2_OUTPUT_DIR "${BERT_VITS2_PROJECT_ROOT}/build-cl")
endif()

include(Sentencepiece)
include(OpenJtalk)

add_executable(${BERT_VITS2_EXE_NAME}
    main.cpp
    bv2_tts.cpp
    bv2_openjtalk.cpp
    bv2_sentencepiece.cpp
)
set_target_properties(${BERT_VITS2_EXE_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${BERT_VITS2_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${BERT_VITS2_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${BERT_VITS2_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${BERT_VITS2_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${BERT_VITS2_OUTPUT_DIR}"
)

target_include_directories(${BERT_VITS2_EXE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${sentencepiece_SOURCE_DIR}/src
)
target_link_libraries(${BERT_VITS2_EXE_NAME} PRIVATE
    ${BERT_VITS2_SENTENCEPIECE_TARGET}
    ${BERT_VITS2_OPENJTALK_TARGET}
)
target_compile_definitions(${BERT_VITS2_EXE_NAME} PRIVATE BV2_WITH_OPENJTALK=1)
target_include_directories(${BERT_VITS2_EXE_NAME} PRIVATE ${BERT_VITS2_OPENJTALK_INCLUDE_DIRS})
if(WIN32)
    target_link_libraries(${BERT_VITS2_EXE_NAME} PRIVATE winmm ws2_32)
endif()

if(BERT_VITS2_WITH_ONNXRUNTIME)
    include(OnnxRuntime)
    bert_vits2_find_onnxruntime("${BERT_VITS2_PROJECT_ROOT}")

    if(NOT ONNXRUNTIME_FOUND)
        message(FATAL_ERROR
            "ONNX Runtime was not found. Place a NuGet package under "
            "third_party/onnxruntime-nuget or third_party/onnxruntime-gpu-windows-nuget, "
            "or configure with -DONNXRUNTIME_ROOT=/path/to/onnxruntime.")
    endif()

    message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

    target_compile_definitions(${BERT_VITS2_EXE_NAME} PRIVATE BV2_WITH_ONNXRUNTIME=1)
    target_include_directories(${BERT_VITS2_EXE_NAME} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(${BERT_VITS2_EXE_NAME} PRIVATE ${ONNXRUNTIME_LIBRARY})

    if(WIN32 AND ONNXRUNTIME_DLL_DIR)
        file(GLOB _ort_dlls "${ONNXRUNTIME_DLL_DIR}/*.dll")
        if(_ort_dlls)
            add_custom_command(TARGET ${BERT_VITS2_EXE_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_ort_dlls}
                    "$<TARGET_FILE_DIR:${BERT_VITS2_EXE_NAME}>"
                COMMENT "Copying ONNX Runtime DLLs to ${BERT_VITS2_OUTPUT_DIR}"
            )
        endif()
    endif()
endif()

if(MSVC)
    target_compile_options(${BERT_VITS2_EXE_NAME} PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(${BERT_VITS2_EXE_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()
