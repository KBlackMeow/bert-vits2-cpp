include(FetchContent)

if(NOT TARGET openjtalk)
    FetchContent_Declare(
        openjtalk
        GIT_REPOSITORY https://github.com/r9y9/open_jtalk.git
        GIT_TAG 1.11
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(openjtalk)
    if(NOT openjtalk_POPULATED)
        FetchContent_Populate(openjtalk)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
        set(CHARSET "utf8" CACHE STRING "" FORCE)
        add_subdirectory(${openjtalk_SOURCE_DIR}/src ${openjtalk_BINARY_DIR})
    endif()
endif()

set(BERT_VITS2_OPENJTALK_TARGET openjtalk)
set(BERT_VITS2_OPENJTALK_INCLUDE_DIRS
    ${openjtalk_SOURCE_DIR}/src/jpcommon
    ${openjtalk_SOURCE_DIR}/src/mecab/src
    ${openjtalk_SOURCE_DIR}/src/mecab2njd
    ${openjtalk_SOURCE_DIR}/src/njd
    ${openjtalk_SOURCE_DIR}/src/njd2jpcommon
    ${openjtalk_SOURCE_DIR}/src/njd_set_accent_phrase
    ${openjtalk_SOURCE_DIR}/src/njd_set_accent_type
    ${openjtalk_SOURCE_DIR}/src/njd_set_digit
    ${openjtalk_SOURCE_DIR}/src/njd_set_long_vowel
    ${openjtalk_SOURCE_DIR}/src/njd_set_pronunciation
    ${openjtalk_SOURCE_DIR}/src/njd_set_unvoiced_vowel
    ${openjtalk_SOURCE_DIR}/src/text2mecab
)
