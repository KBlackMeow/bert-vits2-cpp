function(bert_vits2_copy_onnxruntime_libs target dll_dir)
    if(NOT dll_dir OR NOT TARGET ${target})
        return()
    endif()

    if(WIN32)
        file(GLOB _ort_libs "${dll_dir}/*.dll")
        if(_ort_libs)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_ort_libs}
                    "$<TARGET_FILE_DIR:${target}>"
                COMMENT "Copying ONNX Runtime DLLs next to ${target}"
            )
        endif()
        return()
    endif()

    if(APPLE)
        set(_lib_ext "dylib")
    else()
        set(_lib_ext "so")
    endif()

    file(GLOB _ort_real_libs
        "${dll_dir}/libonnxruntime*.${_lib_ext}.*"
        "${dll_dir}/libonnxruntime*.${_lib_ext}"
    )

    foreach(_src IN LISTS _ort_real_libs)
        if(IS_SYMLINK "${_src}")
            continue()
        endif()
        get_filename_component(_name "${_src}" NAME)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_src}" "$<TARGET_FILE_DIR:${target}>/${_name}"
            COMMENT "Copy ${_name} next to ${target}"
        )
    endforeach()

    foreach(_src IN LISTS _ort_real_libs)
        if(NOT IS_SYMLINK "${_src}")
            continue()
        endif()
        get_filename_component(_name "${_src}" NAME)
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E true
            OUTPUT_QUIET
        )
        file(READ_SYMLINK "${_src}" _target_name)
        get_filename_component(_target_name "${_target_name}" NAME)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rm -f "$<TARGET_FILE_DIR:${target}>/${_name}"
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                "${_target_name}" "$<TARGET_FILE_DIR:${target}>/${_name}"
            COMMENT "Symlink ${_name} -> ${_target_name}"
        )
    endforeach()
endfunction()

function(bert_vits2_find_onnxruntime project_root)
    unset(_include_dir)
    unset(_library)
    unset(_dll_dir)

    if(DEFINED ONNXRUNTIME_ROOT AND ONNXRUNTIME_ROOT)
        find_path(_include_dir
            NAMES onnxruntime_cxx_api.h
            HINTS
                "${ONNXRUNTIME_ROOT}/include"
                "${ONNXRUNTIME_ROOT}/include/onnxruntime"
            NO_DEFAULT_PATH
        )
        find_library(_library
            NAMES onnxruntime
            HINTS
                "${ONNXRUNTIME_ROOT}/lib"
                "${ONNXRUNTIME_ROOT}/lib64"
            NO_DEFAULT_PATH
        )
        if(_include_dir AND _library)
            get_filename_component(_dll_dir "${_library}" DIRECTORY)
        endif()
    endif()

    if(NOT _include_dir OR NOT _library)
        set(_local_roots
            "${project_root}/third_party/onnxruntime-linux"
            "${project_root}/third_party/onnxruntime-gpu-linux"
        )
        foreach(_root IN LISTS _local_roots)
            unset(_include_dir)
            unset(_library)
            unset(_dll_dir)
            if(NOT EXISTS "${_root}/include/onnxruntime_cxx_api.h")
                continue()
            endif()
            find_path(_include_dir
                NAMES onnxruntime_cxx_api.h
                HINTS "${_root}/include"
                NO_DEFAULT_PATH
            )
            find_library(_library
                NAMES onnxruntime
                HINTS "${_root}/lib" "${_root}/lib64"
                NO_DEFAULT_PATH
            )
            if(_include_dir AND _library)
                get_filename_component(_dll_dir "${_library}" DIRECTORY)
                break()
            endif()
        endforeach()
    endif()

    if(NOT _include_dir OR NOT _library)
        set(_nuget_candidates
            "${project_root}/third_party/onnxruntime-gpu-windows-nuget"
            "${project_root}/third_party/onnxruntime-nuget"
            "${project_root}/third_party/onnxruntime-linux-nuget"
            "${project_root}/third_party/onnxruntime-gpu-linux-nuget"
        )
        foreach(_package IN LISTS _nuget_candidates)
            unset(_include_dir)
            unset(_library)
            unset(_dll_dir)
            if(EXISTS "${_package}/buildTransitive/native/include/onnxruntime_cxx_api.h")
                set(_include_dir "${_package}/buildTransitive/native/include")
            elseif(EXISTS "${_package}/build/native/include/onnxruntime_cxx_api.h")
                set(_include_dir "${_package}/build/native/include")
            else()
                continue()
            endif()

            if(WIN32)
                set(_native_dir "${_package}/runtimes/win-x64/native")
            elseif(APPLE)
                if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
                    set(_native_dir "${_package}/runtimes/osx-arm64/native")
                else()
                    set(_native_dir "${_package}/runtimes/osx-x64/native")
                endif()
            else()
                set(_native_dir "${_package}/runtimes/linux-x64/native")
            endif()

            if(WIN32)
                if(EXISTS "${_native_dir}/onnxruntime.lib")
                    set(_library "${_native_dir}/onnxruntime.lib")
                    set(_dll_dir "${_native_dir}")
                endif()
            else()
                find_library(_library
                    NAMES onnxruntime
                    HINTS "${_native_dir}"
                    NO_DEFAULT_PATH
                )
                if(_library)
                    get_filename_component(_dll_dir "${_library}" DIRECTORY)
                endif()
            endif()

            if(_include_dir AND _library)
                break()
            endif()
        endforeach()
    endif()

    if(NOT _include_dir OR NOT _library)
        set(ONNXRUNTIME_FOUND FALSE PARENT_SCOPE)
        return()
    endif()

    set(ONNXRUNTIME_FOUND TRUE PARENT_SCOPE)
    set(ONNXRUNTIME_INCLUDE_DIR "${_include_dir}" PARENT_SCOPE)
    set(ONNXRUNTIME_LIBRARY "${_library}" PARENT_SCOPE)
    set(ONNXRUNTIME_DLL_DIR "${_dll_dir}" PARENT_SCOPE)
endfunction()
