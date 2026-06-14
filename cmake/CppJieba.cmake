include(FetchContent)

# Header-only library; avoid add_subdirectory so cppjieba's own install()
# rules (which omit DESTINATION on newer CMake) don't fail our configure step.
if(NOT DEFINED cppjieba_SOURCE_DIR OR NOT EXISTS "${cppjieba_SOURCE_DIR}/include/cppjieba/Jieba.hpp")
    FetchContent_Declare(
        cppjieba
        GIT_REPOSITORY https://github.com/yanyiwu/cppjieba.git
        GIT_TAG master
    )
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_GetProperties(cppjieba)
    if(NOT cppjieba_POPULATED)
        FetchContent_Populate(cppjieba)
    endif()
endif()

set(BERT_VITS2_CPPJIEBA_INCLUDE "${cppjieba_SOURCE_DIR}/include")
set(BERT_VITS2_CPPJIEBA_DICT "${cppjieba_SOURCE_DIR}/dict")

if(NOT EXISTS "${BERT_VITS2_CPPJIEBA_DICT}/jieba.dict.utf8")
    message(FATAL_ERROR "cppjieba dict was not found at ${BERT_VITS2_CPPJIEBA_DICT}")
endif()

file(MAKE_DIRECTORY "${BERT_VITS2_PROJECT_ROOT}/text/jieba")
file(GLOB _jieba_dict_files "${BERT_VITS2_CPPJIEBA_DICT}/*")
foreach(_dict_file ${_jieba_dict_files})
    if(IS_DIRECTORY "${_dict_file}")
        file(COPY "${_dict_file}" DESTINATION "${BERT_VITS2_PROJECT_ROOT}/text/jieba")
    else()
        get_filename_component(_dict_name "${_dict_file}" NAME)
        configure_file("${_dict_file}" "${BERT_VITS2_PROJECT_ROOT}/text/jieba/${_dict_name}" COPYONLY)
    endif()
endforeach()
