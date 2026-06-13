include(FetchContent)

if(NOT TARGET sentencepiece-static AND NOT TARGET sentencepiece)
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_TRAIN OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_TRAIN_EXECUTABLE OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_ENCODE_SERVER OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_DECODE_SERVER OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_NMT_SERVER OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_PROFILING OFF CACHE BOOL "" FORCE)
    set(SPM_COVERAGE OFF CACHE BOOL "" FORCE)
    set(SPM_USE_BUILTIN_PROTOBUF ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
        sentencepiece
        GIT_REPOSITORY https://github.com/google/sentencepiece.git
        GIT_TAG v0.2.1
    )
    FetchContent_MakeAvailable(sentencepiece)
endif()

if(TARGET sentencepiece-static)
    set(BERT_VITS2_SENTENCEPIECE_TARGET sentencepiece-static)
elseif(TARGET sentencepiece)
    set(BERT_VITS2_SENTENCEPIECE_TARGET sentencepiece)
else()
    message(FATAL_ERROR "sentencepiece target was not created")
endif()
