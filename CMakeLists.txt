# パス: CMakeLists.txt

cmake_minimum_required(VERSION 3.28)
project(ShioRIS3 VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt6設定
find_package(Qt6 REQUIRED COMPONENTS Core Widgets OpenGLWidgets Concurrent Network Multimedia)
# Core5Compatを条件付きで検索
find_package(Qt6Core5Compat QUIET)

qt_standard_project_setup()

# OpenCV と OpenGL
find_package(OpenCV REQUIRED)

if(APPLE)
    # CMake の OpenGL モジュールは古い AGL フレームワークへ依存してしまうため、
    # macOS では直接 OpenGL.framework を解決してリンクする。
    find_library(MAC_OPENGL_FRAMEWORK OpenGL)
    if(NOT MAC_OPENGL_FRAMEWORK)
        message(FATAL_ERROR "OpenGL framework not found. Please install the Command Line Tools.")
    endif()

    # AVFoundation framework for microphone permissions check
    find_library(AVFOUNDATION_FRAMEWORK AVFoundation)
    if(NOT AVFOUNDATION_FRAMEWORK)
        message(WARNING "AVFoundation framework not found. Microphone permission checks will be disabled.")
    endif()

    # Qt のインポートターゲットから不要な AGL 依存を取り除く。
    function(strip_agl_from_target qt_target)
        if(NOT TARGET "${qt_target}")
            return()
        endif()

        get_target_property(_target_type "${qt_target}" TYPE)
        if(NOT _target_type)
            set(_target_type "")
        endif()

        set(_base_props
            INTERFACE_LINK_LIBRARIES
            INTERFACE_LINK_OPTIONS
            IMPORTED_LINK_INTERFACE_LIBRARIES
            IMPORTED_LINK_OPTIONS)
        if(NOT _target_type STREQUAL "INTERFACE_LIBRARY")
            list(APPEND _base_props LINK_LIBRARIES LINK_OPTIONS)
        endif()
        set(_configs "" RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)

        foreach(_prop IN LISTS _base_props)
            foreach(_cfg IN LISTS _configs)
                if(_cfg STREQUAL "")
                    set(_prop_name "${_prop}")
                else()
                    set(_prop_name "${_prop}_${_cfg}")
                endif()

                get_target_property(_values "${qt_target}" "${_prop_name}")
                if(NOT _values OR _values STREQUAL "NOTFOUND")
                    continue()
                endif()

                set(_list ${_values})
                list(LENGTH _list _len)
                if(_len EQUAL 0)
                    continue()
                endif()

                set(_filtered "")
                set(_index 0)
                while(_index LESS _len)
                    list(GET _list ${_index} _item)

                    set(_skip FALSE)
                    if(_item STREQUAL "-framework")
                        math(EXPR _next "${_index} + 1")
                        if(_next < _len)
                            list(GET _list ${_next} _next_item)
                            if(_next_item STREQUAL "AGL")
                                set(_skip TRUE)
                                math(EXPR _index "${_index} + 2")
                                continue()
                            endif()
                        endif()
                    endif()

                    if(NOT _skip)
                        if(_item MATCHES "^\\$<[^>]*-framework[ ]+AGL[^>]*>$")
                            set(_skip TRUE)
                        elseif(_item MATCHES "^SHELL:-framework[ ]+AGL$")
                            set(_skip TRUE)
                        elseif(_item MATCHES "(^|[\\s,;-])AGL($|[\\s,;-])" OR _item MATCHES "AGL\\.framework")
                            set(_skip TRUE)
                        elseif(_item MATCHES "^-framework[ ]+AGL$")
                            set(_skip TRUE)
                        endif()
                    endif()

                    if(NOT _skip)
                        list(APPEND _filtered "${_item}")
                    endif()

                    math(EXPR _index "${_index} + 1")
                endwhile()

                list(JOIN _filtered ";" _filtered_joined)
                list(JOIN _list ";" _original_joined)
                if(NOT _filtered_joined STREQUAL _original_joined)
                    set_target_properties("${qt_target}" PROPERTIES "${_prop_name}" "${_filtered}")
                endif()
            endforeach()
        endforeach()
    endfunction()

    strip_agl_from_target(Qt6::Gui)
    strip_agl_from_target(Qt6::OpenGL)
    strip_agl_from_target(Qt6::OpenGLWidgets)
else()
    find_package(OpenGL REQUIRED)
endif()
# SQLite3 はプラットフォームによって提供方法が異なるため、
# 最初に通常の `find_package` を行い、得られた情報からターゲットを決定する。
set(SHIO_SQLITE_TARGET "")
find_package(SQLite3 QUIET)

foreach(_sqlite_candidate IN ITEMS SQLite::SQLite3 SQLite3::SQLite3 sqlite3 sqlite3::sqlite3)
    if(TARGET "${_sqlite_candidate}")
        set(SHIO_SQLITE_TARGET "${_sqlite_candidate}")
        break()
    endif()
endforeach()

if(NOT SHIO_SQLITE_TARGET AND SQLite3_FOUND)
    # モジュールモードで見つかった場合はターゲットが提供されないため、
    # INTERFACE IMPORTED ターゲットを自前で定義する。
    set(_sqlite_include "")
    if(DEFINED SQLite3_INCLUDE_DIRS)
        set(_sqlite_include "${SQLite3_INCLUDE_DIRS}")
    elseif(DEFINED SQLite3_INCLUDE_DIR)
        set(_sqlite_include "${SQLite3_INCLUDE_DIR}")
    endif()

    set(_sqlite_libs "")
    if(DEFINED SQLite3_LIBRARIES)
        set(_sqlite_libs "${SQLite3_LIBRARIES}")
    elseif(DEFINED SQLite3_LIBRARY)
        set(_sqlite_libs "${SQLite3_LIBRARY}")
    endif()

    if(_sqlite_libs)
        add_library(SQLite::SQLite3 INTERFACE IMPORTED)
        set_property(TARGET SQLite::SQLite3 PROPERTY INTERFACE_LINK_LIBRARIES ${_sqlite_libs})
        if(_sqlite_include)
            set_property(TARGET SQLite::SQLite3 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${_sqlite_include})
        endif()
        set(SHIO_SQLITE_TARGET SQLite::SQLite3)
    endif()
endif()

if(NOT SHIO_SQLITE_TARGET)
    find_package(unofficial-sqlite3 CONFIG QUIET)
    if(TARGET unofficial::sqlite3::sqlite3)
        set(SHIO_SQLITE_TARGET unofficial::sqlite3::sqlite3)
    endif()
endif()

if(NOT SHIO_SQLITE_TARGET)
    message(FATAL_ERROR "SQLite3 が見つかりません。以下のいずれかを実行してください:\n  - vcpkg で `vcpkg install sqlite3` を実行する\n  - 開発用の SQLite3 ライブラリとヘッダーへのパスを CMAKE_PREFIX_PATH に追加する")
endif()

# コンパイラ固有の設定（DCMTKの問題対策）
if(APPLE)
    # macOS環境でのDCMTK C++17互換性問題を解決
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDCMTK_CXX17_STD=1")
    
    # Apple Clangでのテンプレート問題対策
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDCMTK_HAVE_CXX11=1")
    
    # DCMTK設定の強制
    add_definitions(-DHAVE_CONFIG_H)
    add_definitions(-DDCMTK_HAVE_STD_NAMESPACE)
endif()

# ===== GPU Dose Calculation 設定 =====
option(ENABLE_GPU_DOSE_CALCULATION "Enable GPU-accelerated dose calculation" ON)

if(ENABLE_GPU_DOSE_CALCULATION)
    message(STATUS "=== GPU Dose Calculation Configuration ===")

    set(GPU_BACKEND_FOUND FALSE)

    # Metal (macOS only - recommended for Apple Silicon)
    if(APPLE)
        find_library(METAL_FRAMEWORK Metal)
        find_library(FOUNDATION_FRAMEWORK Foundation)
        if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
            message(STATUS "✓ Metal framework found")
            set(GPU_BACKEND_METAL TRUE)
            set(GPU_BACKEND_FOUND TRUE)
            message(STATUS "  Metal Framework: ${METAL_FRAMEWORK}")
        else()
            message(STATUS "✗ Metal framework not found")
            set(GPU_BACKEND_METAL FALSE)
        endif()
    else()
        set(GPU_BACKEND_METAL FALSE)
    endif()

    # CUDA (NVIDIA GPUs - best performance for RTX series)
    if(NOT APPLE)
        include(CheckLanguage)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            find_package(CUDAToolkit)
            if(CUDAToolkit_FOUND)
                message(STATUS "✓ CUDA Toolkit found: ${CUDAToolkit_VERSION}")
                set(GPU_BACKEND_CUDA TRUE)
                set(GPU_BACKEND_FOUND TRUE)
                message(STATUS "  CUDA Toolkit Root: ${CUDAToolkit_INCLUDE_DIRS}")
                message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

                # Set CUDA architecture for RTX 3090 (Compute Capability 8.6)
                # Also support other common architectures
                if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures")
                    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
                endif()
            else()
                message(STATUS "✗ CUDA Toolkit not found")
                set(GPU_BACKEND_CUDA FALSE)
            endif()
        else()
            message(STATUS "✗ CUDA compiler not found")
            set(GPU_BACKEND_CUDA FALSE)
        endif()
    else()
        set(GPU_BACKEND_CUDA FALSE)
    endif()

    # OpenCL (Windows/Linux/macOS fallback)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        message(STATUS "✓ OpenCL found: ${OpenCL_VERSION_STRING}")
        set(GPU_BACKEND_OPENCL TRUE)
        set(GPU_BACKEND_FOUND TRUE)
        message(STATUS "  OpenCL Include: ${OpenCL_INCLUDE_DIRS}")
        message(STATUS "  OpenCL Library: ${OpenCL_LIBRARIES}")
    else()
        message(STATUS "✗ OpenCL not found")
        set(GPU_BACKEND_OPENCL FALSE)
    endif()

    if(GPU_BACKEND_FOUND)
        if(GPU_BACKEND_CUDA)
            message(STATUS "✓ GPU dose calculation enabled with CUDA backend (primary - NVIDIA)")
        endif()
        if(GPU_BACKEND_METAL)
            message(STATUS "✓ GPU dose calculation enabled with Metal backend (primary - Apple)")
        endif()
        if(GPU_BACKEND_OPENCL)
            if(APPLE)
                message(STATUS "  OpenCL backend available (fallback - deprecated on macOS)")
            elseif(GPU_BACKEND_CUDA)
                message(STATUS "  OpenCL backend available (fallback - CUDA preferred for NVIDIA)")
            else()
                message(STATUS "✓ GPU dose calculation enabled with OpenCL backend")
            endif()
        endif()
    else()
        message(WARNING "No GPU backend available. GPU dose calculation will be disabled.")
        set(ENABLE_GPU_DOSE_CALCULATION OFF)
    endif()
endif()

# ===== ONNX Runtime 設定 =====
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime for AI functionality" ON)

if(ENABLE_ONNXRUNTIME)
    message(STATUS "=== ONNX Runtime Detection ===")
    
    if(APPLE)
        # macOS設定（既存の設定を保持）
        if(DEFINED ENV{ONNXRUNTIME_ROOT})
            set(ONNXRUNTIME_ROOT_PATH $ENV{ONNXRUNTIME_ROOT})
            message(STATUS "Using ONNXRUNTIME_ROOT: ${ONNXRUNTIME_ROOT_PATH}")
        else()
            set(ONNXRUNTIME_ROOT_PATH "/opt/homebrew/onnxruntime")
            message(STATUS "Using default path: ${ONNXRUNTIME_ROOT_PATH}")
        endif()
        
        if(EXISTS "${ONNXRUNTIME_ROOT_PATH}")
            set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_PATH}/include")
            if(EXISTS "${ONNXRUNTIME_INCLUDE_DIRS}/onnxruntime_cxx_api.h")
                message(STATUS "Headers found: ${ONNXRUNTIME_INCLUDE_DIRS}")
                
                set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.dylib")
                if(EXISTS "${ONNXRUNTIME_LIB}")
                    message(STATUS "Library found: ${ONNXRUNTIME_LIB}")
                    set(ONNXRUNTIME_FOUND TRUE)
                else()
                    message(WARNING "ONNX Runtime library not found: ${ONNXRUNTIME_LIB}")
                    set(ONNXRUNTIME_FOUND FALSE)
                endif()
            else()
                message(WARNING "ONNX Runtime headers not found")
                set(ONNXRUNTIME_FOUND FALSE)
            endif()
        else()
            message(WARNING "ONNX Runtime directory not found: ${ONNXRUNTIME_ROOT_PATH}")
            set(ONNXRUNTIME_FOUND FALSE)
        endif()
        
    elseif(UNIX AND NOT APPLE) # Linux設定
        message(STATUS "Configuring ONNX Runtime for Linux...")
        
        # 複数のパスを試行
        set(POTENTIAL_ONNX_PATHS
            "/usr/local/onnxruntime"
            "/opt/onnxruntime"
            "/usr/share/onnxruntime"
            "$ENV{HOME}/onnxruntime"
        )
        
        # 環境変数が設定されている場合は最優先
        if(DEFINED ENV{ONNXRUNTIME_ROOT})
            list(INSERT POTENTIAL_ONNX_PATHS 0 $ENV{ONNXRUNTIME_ROOT})
        endif()
        
        set(ONNXRUNTIME_FOUND FALSE)
        foreach(ONNX_PATH ${POTENTIAL_ONNX_PATHS})
            message(STATUS "Checking ONNX Runtime path: ${ONNX_PATH}")
            
            if(EXISTS "${ONNX_PATH}")
                # ヘッダーファイル確認
                set(ONNX_INCLUDE_PATH "${ONNX_PATH}/include")
                if(EXISTS "${ONNX_INCLUDE_PATH}/onnxruntime_cxx_api.h")
                    message(STATUS "✓ Headers found: ${ONNX_INCLUDE_PATH}")
                    
                    # ライブラリファイル確認（複数の可能性）
                    set(POTENTIAL_LIBS
                        "${ONNX_PATH}/lib/libonnxruntime.so"
                        "${ONNX_PATH}/lib64/libonnxruntime.so"
                        "${ONNX_PATH}/lib/x86_64-linux-gnu/libonnxruntime.so"
                    )
                    
                    foreach(LIB_PATH ${POTENTIAL_LIBS})
                        if(EXISTS "${LIB_PATH}")
                            message(STATUS "✓ Library found: ${LIB_PATH}")
                            set(ONNXRUNTIME_ROOT_PATH ${ONNX_PATH})
                            set(ONNXRUNTIME_INCLUDE_DIRS ${ONNX_INCLUDE_PATH})
                            set(ONNXRUNTIME_LIB ${LIB_PATH})
                            set(ONNXRUNTIME_FOUND TRUE)
                            break()
                        endif()
                    endforeach()
                    
                    if(ONNXRUNTIME_FOUND)
                        break()
                    endif()
                endif()
            endif()
        endforeach()
        
        # pkg-config を使用した検索
        if(NOT ONNXRUNTIME_FOUND)
            message(STATUS "Trying pkg-config for ONNX Runtime...")
            find_package(PkgConfig QUIET)
            if(PkgConfig_FOUND)
                pkg_check_modules(ONNXRUNTIME QUIET libonnxruntime)
                if(ONNXRUNTIME_FOUND)
                    message(STATUS "✓ Found ONNX Runtime via pkg-config")
                    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIRS})
                    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_LIBRARIES})
                endif()
            endif()
        endif()
        
        # CUDA対応の確認
        if(ONNXRUNTIME_FOUND)
            message(STATUS "Checking CUDA availability...")

            # CUDA 12.0対応の検出方法
            find_package(CUDAToolkit QUIET)
            if(CUDAToolkit_FOUND)
                message(STATUS "✓ CUDA Toolkit found: Version ${CUDAToolkit_VERSION}")

                # まず初期値をFALSEに設定
                set(ONNXRUNTIME_USE_CUDA FALSE)

                # 既存のCUDA EPライブラリの探索ロジック
                set(POTENTIAL_CUDA_LIBS
                    "/usr/local/onnxruntime/lib/libonnxruntime_providers_cuda.so"
                    "${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime_providers_cuda.so"
                    "${ONNXRUNTIME_ROOT_PATH}/lib64/libonnxruntime_providers_cuda.so"
                    "${ONNXRUNTIME_ROOT_PATH}/lib/x86_64-linux-gnu/libonnxruntime_providers_cuda.so"
                    "/opt/onnxruntime/lib/libonnxruntime_providers_cuda.so"
                )

                foreach(CUDA_LIB ${POTENTIAL_CUDA_LIBS})
                    if(EXISTS "${CUDA_LIB}")
                        message(STATUS "✓ CUDA provider library found: ${CUDA_LIB}")
                        set(ONNXRUNTIME_CUDA_LIB ${CUDA_LIB})
                        set(ONNXRUNTIME_USE_CUDA TRUE)

                        # Shared providerライブラリの確認
                        get_filename_component(CUDA_LIB_DIR "${CUDA_LIB}" DIRECTORY)
                        set(POTENTIAL_SHARED_LIBS
                            "${CUDA_LIB_DIR}/libonnxruntime_providers_shared.so"
                        )
                        foreach(SHARED_LIB ${POTENTIAL_SHARED_LIBS})
                            if(EXISTS "${SHARED_LIB}")
                                message(STATUS "✓ Shared provider library found: ${SHARED_LIB}")
                                set(ONNXRUNTIME_SHARED_LIB ${SHARED_LIB})
                            endif()
                        endforeach()

                        # TensorRTライブラリの確認（オプション）
                        set(POTENTIAL_TRT_LIBS "${CUDA_LIB_DIR}/libonnxruntime_providers_tensorrt.so")
                        foreach(TRT_LIB ${POTENTIAL_TRT_LIBS})
                            if(EXISTS "${TRT_LIB}")
                                message(STATUS "✓ TensorRT provider library found: ${TRT_LIB}")
                                set(ONNXRUNTIME_TRT_LIB ${TRT_LIB})
                            endif()
                        endforeach()

                        break()
                    endif()
                endforeach()

                if(NOT ONNXRUNTIME_USE_CUDA)
                    message(WARNING "⚠ CUDA Toolkit found but onnxruntime_providers_cuda library not found")
                    message(WARNING "  Searched paths:")
                    foreach(CUDA_LIB ${POTENTIAL_CUDA_LIBS})
                        message(WARNING "    - ${CUDA_LIB}")
                    endforeach()
                    message(WARNING "  Using CPU execution provider")
                endif()
            else()
                message(STATUS "CUDA Toolkit not found, using CPU execution provider")
                set(ONNXRUNTIME_USE_CUDA FALSE)
            endif()
        endif()

        # 最終確認とログ出力
        if(ONNXRUNTIME_USE_CUDA)
            message(STATUS "")
            message(STATUS "🚀 ===== CUDA SUPPORT ENABLED =====")
            message(STATUS "   CUDA Provider: ${ONNXRUNTIME_CUDA_LIB}")
            if(ONNXRUNTIME_SHARED_LIB)
                message(STATUS "   Shared Provider: ${ONNXRUNTIME_SHARED_LIB}")
            endif()
            if(ONNXRUNTIME_TRT_LIB)
                message(STATUS "   TensorRT Provider: ${ONNXRUNTIME_TRT_LIB}")
            endif()
            message(STATUS "   Macro ONNXRUNTIME_USE_CUDA will be defined")
            message(STATUS "===================================")
            message(STATUS "")
        else()
            message(STATUS "")
            message(STATUS "ℹ CUDA support disabled - using CPU execution provider")
            message(STATUS "")
        endif()


        
    endif() # Linux設定終了
    
    # 最終確認
    if(ONNXRUNTIME_FOUND)
        message(STATUS "✓ ONNX Runtime enabled")
        message(STATUS "  Include dirs: ${ONNXRUNTIME_INCLUDE_DIRS}")
        message(STATUS "  Library: ${ONNXRUNTIME_LIB}")
        if(UNIX AND NOT APPLE AND ONNXRUNTIME_USE_CUDA)
            message(STATUS "  CUDA support: Enabled")
        endif()
    else()
        message(WARNING "ONNX Runtime not found. AI functionality will be disabled.")
        set(ENABLE_ONNXRUNTIME OFF)
    endif()
endif()

# DCMTK設定（既存の設定を保持）
set(DCMTK_INCLUDE_DIRS_PRIVATE "")

if(APPLE)
    # HomebrewのDCMTKを使用
    execute_process(
        COMMAND brew --prefix dcmtk
        OUTPUT_VARIABLE DCMTK_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(DCMTK_PREFIX)
        message(STATUS "Found DCMTK via Homebrew: ${DCMTK_PREFIX}")
        set(DCMTK_DIR "${DCMTK_PREFIX}/lib/cmake/dcmtk")
        find_package(DCMTK REQUIRED NO_MODULE)
        set(DCMTK_LIBS ${DCMTK_LIBRARIES})
        set(DCMTK_INCLUDE_DIRS_PRIVATE ${DCMTK_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "DCMTK not found. Please install with: brew install dcmtk")
    endif()

elseif(WIN32)
    # vcpkg経由で提供されるCMake configを利用
    find_package(DCMTK CONFIG REQUIRED COMPONENTS
        ofstd
        oflog
        dcmdata
        dcmimgle
        dcmimage
    )
    set(DCMTK_LIBS
        DCMTK::ofstd
        DCMTK::oflog
        DCMTK::dcmdata
        DCMTK::dcmimgle
        DCMTK::dcmimage
    )

else()
    # Linux環境ではpkg-configを使用
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DCMTK REQUIRED dcmtk)
    set(DCMTK_LIBS ${DCMTK_LIBRARIES})
    set(DCMTK_INCLUDE_DIRS_PRIVATE ${DCMTK_INCLUDE_DIRS})
endif()

# ソースファイル定義
set(HEADERS
    include/mainwindow.h
    include/theme_manager.h
    include/dicom/dicom_reader.h
    include/dicom/dicom_volume.h
    include/dicom/rtdose_volume.h
    include/dicom/rtstruct.h
    include/dicom/dose_resampled_volume.h
    include/dicom/dvh_calculator.h
    include/dicom/dpsd_calculator.h
    include/dicom/brachy_plan.h
    include/dicom/dose_evaluation_point.h
    include/brachy/ir_source_data.h
    include/brachy/brachy_dose_calculator.h
    include/brachy/dwell_time_optimizer.h
    include/qcustomplot.h
    include/visualization/dicom_viewer.h
    include/visualization/collapsible_group_box.h
    include/visualization/opengl_image_widget.h
    include/visualization/opengl_3d_widget.h
    include/visualization/quad_viewer.h
    include/visualization/volume_viewer.h
    include/visualization/volume_viewer_window.h
    include/visualization/dvh_window.h
    include/visualization/dose_profile_window.h
    include/visualization/dpsd_window.h
    include/visualization/gamma_analysis_window.h
    include/visualization/random_study_dialog.h
    include/visualization/data_window.h
    include/visualization/auto_segmentation_dialog.h
    include/visualization/fusion_dialog.h
    include/visualization/license_dialog.h
    include/visualization/translator_window.h
    include/platform/macos_audio_permissions.h
    include/ai/lmstudio_client.h
    include/database/database_manager.h
    include/database/smart_scanner.h
    include/database/database_sync_manager.h
    include/data/file_structure_manager.h
    include/data/multi_modality_manager.h
    include/data/metadata_generator.h
    include/cyberknife/beam_data_parser.h
    include/cyberknife/beam_data_locator.h
    include/cyberknife/beam_data_manager.h
    include/cyberknife/geometry_calculator.h
    include/cyberknife/dose_calculator.h
    include/cyberknife/gpu_dose_backend.h
    include/cyberknife/opencl_dose_backend.h
    include/web/web_server.h
    include/export/usdz_exporter.h
)

# Add Metal backend header on macOS
if(APPLE AND GPU_BACKEND_METAL)
    list(APPEND HEADERS include/cyberknife/metal_dose_backend.h)
endif()

set(SOURCES
    src/main.cpp
    src/theme_manager.cpp
    src/mainwindow.cpp
    src/dicom/dicom_reader.cpp
    src/dicom/dicom_volume.cpp
    src/dicom/rtdose_volume.cpp
    src/dicom/dose_isosurface.cpp
    src/dicom/rtstruct.cpp
    src/dicom/structure_surface.cpp
    src/dicom/dose_resampled_volume.cpp
    src/dicom/dvh_calculator.cpp
    src/dicom/dpsd_calculator.cpp
    src/dicom/brachy_plan.cpp
    src/brachy/ir_source_data.cpp
    src/brachy/brachy_dose_calculator.cpp
    src/brachy/dwell_time_optimizer.cpp
    src/visualization/dicom_viewer.cpp
    src/visualization/collapsible_group_box.cpp
    src/visualization/opengl_image_widget.cpp
    src/visualization/opengl_3d_widget.cpp
    src/visualization/quad_viewer.cpp
    src/visualization/volume_viewer.cpp
    src/visualization/volume_viewer_window.cpp
    src/visualization/dvh_window.cpp
    src/visualization/dose_profile_window.cpp
    src/visualization/dpsd_window.cpp
    src/visualization/gamma_analysis_window.cpp
    src/visualization/random_study_dialog.cpp
    src/visualization/data_window.cpp
    src/visualization/auto_segmentation_dialog.cpp
    src/visualization/fusion_dialog.cpp
    src/visualization/license_dialog.cpp
    src/visualization/translator_window.cpp
    src/ai/lmstudio_client.cpp
    src/qrcodegen.cpp
    src/database/database_manager.cpp
    src/database/smart_scanner.cpp
    src/database/database_sync_manager.cpp
    src/data/file_structure_manager.cpp
    src/data/multi_modality_manager.cpp
    src/data/metadata_generator.cpp
    src/cyberknife/beam_data_parser.cpp
    src/cyberknife/beam_data_locator.cpp
    src/cyberknife/beam_data_manager.cpp
    src/cyberknife/geometry_calculator.cpp
    src/cyberknife/dose_calculator.cpp
    src/cyberknife/gpu_dose_backend.cpp
    src/cyberknife/opencl_dose_backend.cpp
    src/web/web_server.cpp
    src/export/usdz_exporter.cpp
)

# Add Metal backend implementation on macOS
if(APPLE AND GPU_BACKEND_METAL)
    list(APPEND SOURCES src/cyberknife/metal_dose_backend.mm)
endif()

# Add macOS audio permissions implementation
if(APPLE)
    list(APPEND SOURCES src/platform/macos_audio_permissions.mm)
endif()

# Add CUDA backend implementation on Linux/Windows with NVIDIA GPUs
if(GPU_BACKEND_CUDA)
    list(APPEND HEADERS include/cyberknife/cuda_dose_backend.h)
    list(APPEND HEADERS include/cyberknife/multi_gpu_cuda_manager.h)
    list(APPEND SOURCES
        src/cyberknife/cuda_dose_backend.cu
        src/cyberknife/multi_gpu_cuda_manager.cpp
    )
    # Note: cuda_kernels.cu is included directly in cuda_dose_backend.cu
endif()

if(ENABLE_ONNXRUNTIME)
    list(APPEND HEADERS include/ai/onnx_segmenter.h)
    list(APPEND SOURCES src/ai/onnx_segmenter.cpp)
    list(APPEND HEADERS include/ai/linux_auto_segmenter.h)
    list(APPEND HEADERS include/ai/segmentation_pipeline.h)
    list(APPEND SOURCES src/ai/linux_auto_segmenter.cpp)
    list(APPEND SOURCES src/ai/segmentation_pipeline.cpp)
endif()

# Whisper音声認識機能の追加
list(APPEND HEADERS include/ai/whisper_client.h)
list(APPEND SOURCES src/ai/whisper_client.cpp)
list(APPEND HEADERS include/ai/audio_recorder.h)
list(APPEND SOURCES src/ai/audio_recorder.cpp)

# whisper.cpp設定
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "Build whisper.cpp tests")
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "Build whisper.cpp examples")
add_subdirectory(external/whisper.cpp)

# リソースファイル
set(RESOURCES
    resources/resources.qrc
)

# 実行ファイル作成
qt_add_executable(ShioRIS3 ${SOURCES} ${HEADERS} ${RESOURCES})

# インクルードディレクトリ
target_include_directories(ShioRIS3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/whisper.cpp
    ${OpenCV_INCLUDE_DIRS}
    ${DCMTK_INCLUDE_DIRS_PRIVATE}
)

# ライブラリリンク
target_link_libraries(ShioRIS3 PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::OpenGLWidgets
    Qt6::Concurrent
    Qt6::Network
    Qt6::Multimedia
    ${OpenCV_LIBS}
    ${DCMTK_LIBS}
    ${SHIO_SQLITE_TARGET}
    whisper
)

if(APPLE)
  function(drop_wrapopengl tgt)
    if(NOT TARGET "${tgt}")
      return()
    endif()
    foreach(prop IN ITEMS INTERFACE_LINK_LIBRARIES LINK_LIBRARIES)
      get_target_property(v "${tgt}" "${prop}")
      if(v)
        list(REMOVE_ITEM v WrapOpenGL::WrapOpenGL)
        # 念のため AGL 文字列も掃除
        list(FILTER v EXCLUDE REGEX "AGL(\\.framework)?")
        set_target_properties("${tgt}" PROPERTIES "${prop}" "${v}")
      endif()
    endforeach()
  endfunction()

  drop_wrapopengl(Qt6::Gui)
  drop_wrapopengl(Qt6::OpenGL)
  drop_wrapopengl(Qt6::OpenGLWidgets)
  drop_wrapopengl(ShioRIS3)
endif()

if(APPLE)
    target_link_libraries(ShioRIS3 PRIVATE ${MAC_OPENGL_FRAMEWORK})
    if(AVFOUNDATION_FRAMEWORK)
        target_link_libraries(ShioRIS3 PRIVATE ${AVFOUNDATION_FRAMEWORK})
        message(STATUS "✓ AVFoundation framework linked for microphone permissions")
    endif()
elseif(TARGET OpenGL::GL)
    target_link_libraries(ShioRIS3 PRIVATE OpenGL::GL)
elseif(DEFINED OPENGL_LIBRARIES AND OPENGL_LIBRARIES)
    target_link_libraries(ShioRIS3 PRIVATE ${OPENGL_LIBRARIES})
endif()

if(APPLE)
    strip_agl_from_target(ShioRIS3)
endif()

if(ENABLE_ONNXRUNTIME)
    target_include_directories(ShioRIS3 PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
    target_link_libraries(ShioRIS3 PRIVATE ${ONNXRUNTIME_LIB})
    target_compile_definitions(ShioRIS3 PRIVATE USE_ONNXRUNTIME)

    message(STATUS "Configuring ONNX Runtime linking...")

    if(UNIX AND NOT APPLE AND ONNXRUNTIME_USE_CUDA)
        # CUDA サポート有効化
        target_compile_definitions(ShioRIS3 PRIVATE ONNXRUNTIME_USE_CUDA)
        message(STATUS "  ✓ ONNXRUNTIME_USE_CUDA macro defined")

        # CUDA providerライブラリをリンク
        if(ONNXRUNTIME_CUDA_LIB)
            target_link_libraries(ShioRIS3 PRIVATE ${ONNXRUNTIME_CUDA_LIB})
            message(STATUS "  ✓ Linked CUDA provider: ${ONNXRUNTIME_CUDA_LIB}")
        endif()

        # Shared providerライブラリをリンク（必要な場合）
        if(ONNXRUNTIME_SHARED_LIB)
            target_link_libraries(ShioRIS3 PRIVATE ${ONNXRUNTIME_SHARED_LIB})
            message(STATUS "  ✓ Linked Shared provider: ${ONNXRUNTIME_SHARED_LIB}")
        endif()

        # TensorRT providerライブラリをリンク（オプション）
        if(ONNXRUNTIME_TRT_LIB)
            target_link_libraries(ShioRIS3 PRIVATE ${ONNXRUNTIME_TRT_LIB})
            message(STATUS "  ✓ Linked TensorRT provider: ${ONNXRUNTIME_TRT_LIB}")
        endif()
    else()
        message(STATUS "  ℹ Using CPU execution provider")
    endif()
endif()

# GPU Dose Calculation用のリンク設定
if(ENABLE_GPU_DOSE_CALCULATION)
    target_compile_definitions(ShioRIS3 PRIVATE
        ENABLE_GPU_DOSE_CALCULATION
        SHIORIS_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    # Metal backend (macOS)
    if(GPU_BACKEND_METAL)
        target_link_libraries(ShioRIS3 PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
        target_compile_definitions(ShioRIS3 PRIVATE USE_METAL_BACKEND)
        message(STATUS "✓ GPU dose calculation: Metal backend linked")
        message(STATUS "  Metal Framework: ${METAL_FRAMEWORK}")
        message(STATUS "  Foundation Framework: ${FOUNDATION_FRAMEWORK}")

        # Copy Metal shader file to bundle Resources on macOS
        if(APPLE)
            add_custom_command(TARGET ShioRIS3 POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Resources"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_SOURCE_DIR}/src/cyberknife/metal_kernels.metal"
                    "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Resources/metal_kernels.metal"
                COMMENT "Copying Metal shader to app bundle"
            )
        endif()
    endif()

    # OpenCL backend (cross-platform)
    if(GPU_BACKEND_OPENCL)
        target_include_directories(ShioRIS3 PRIVATE ${OpenCL_INCLUDE_DIRS})
        target_link_libraries(ShioRIS3 PRIVATE ${OpenCL_LIBRARIES})
        target_compile_definitions(ShioRIS3 PRIVATE USE_OPENCL_BACKEND)

        if(APPLE AND GPU_BACKEND_METAL)
            message(STATUS "✓ GPU dose calculation: OpenCL backend linked (fallback)")
        else()
            message(STATUS "✓ GPU dose calculation: OpenCL backend linked")
        endif()
        message(STATUS "  OpenCL Libraries: ${OpenCL_LIBRARIES}")

        # Copy OpenCL kernel file to build directory
        if(APPLE)
            add_custom_command(TARGET ShioRIS3 POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_SOURCE_DIR}/src/cyberknife/opencl_kernels.cl"
                    "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Resources/opencl_kernels.cl"
                COMMENT "Copying OpenCL kernel to app bundle"
            )
        else()
            add_custom_command(TARGET ShioRIS3 POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_SOURCE_DIR}/src/cyberknife/opencl_kernels.cl"
                    "$<TARGET_FILE_DIR:ShioRIS3>/opencl_kernels.cl"
                COMMENT "Copying OpenCL kernel to build directory"
            )
        endif()
    endif()

    # CUDA backend (NVIDIA GPUs - Linux/Windows)
    if(GPU_BACKEND_CUDA)
        target_compile_definitions(ShioRIS3 PRIVATE USE_CUDA_BACKEND)
        target_link_libraries(ShioRIS3 PRIVATE CUDA::cudart CUDA::cuda_driver)

        # Set CUDA properties for target
        set_target_properties(ShioRIS3 PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )

        message(STATUS "✓ GPU dose calculation: CUDA backend linked")
        message(STATUS "  CUDA Runtime: CUDA::cudart")
        message(STATUS "  CUDA Driver: CUDA::cuda_driver")
        message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# Core5Compatが利用可能な場合のみリンク
if(TARGET Qt6::Core5Compat)
    target_link_libraries(ShioRIS3 PRIVATE Qt6::Core5Compat)
    target_compile_definitions(ShioRIS3 PRIVATE QT_CORE5_COMPAT_LIB)
    message(STATUS "Qt6Core5Compat found and linked")
endif()

# Windows固有の設定
if(WIN32)
    set_target_properties(ShioRIS3 PROPERTIES
        WIN32_EXECUTABLE TRUE
    )
    
    target_link_libraries(ShioRIS3 PRIVATE
        ws2_32
        netapi32
        wsock32
    )
endif()

# Linux環境の設定
if(UNIX AND NOT APPLE)
    if(DCMTK_LIBRARY_DIRS)
        target_link_directories(ShioRIS3 PRIVATE ${DCMTK_LIBRARY_DIRS})
    endif()

    find_package(Threads REQUIRED)
    target_link_libraries(ShioRIS3 PRIVATE Threads::Threads)

    if(DCMTK_CFLAGS_OTHER)
        target_compile_options(ShioRIS3 PRIVATE ${DCMTK_CFLAGS_OTHER})
    endif()

    # RPATH設定（Whisper + ONNX Runtime）
    set(RPATH_DIRS "")

    # Whisper library用のRPATH設定
    # Whisperライブラリはビルドディレクトリ内に生成されるため、$ORIGINベースのパスを追加
    list(APPEND RPATH_DIRS "$ORIGIN")
    list(APPEND RPATH_DIRS "${CMAKE_BINARY_DIR}/external/whisper.cpp/src")

    # ONNX Runtime用のRPATH設定
    if(ENABLE_ONNXRUNTIME AND ONNXRUNTIME_FOUND)
        get_filename_component(ONNX_LIB_DIR ${ONNXRUNTIME_LIB} DIRECTORY)
        list(APPEND RPATH_DIRS ${ONNX_LIB_DIR})
        if(ONNXRUNTIME_CUDA_LIB)
            get_filename_component(ONNX_CUDA_LIB_DIR ${ONNXRUNTIME_CUDA_LIB} DIRECTORY)
            list(APPEND RPATH_DIRS ${ONNX_CUDA_LIB_DIR})
        endif()
    endif()

    if(RPATH_DIRS)
        list(REMOVE_DUPLICATES RPATH_DIRS)
        set_target_properties(ShioRIS3 PROPERTIES
            INSTALL_RPATH "${RPATH_DIRS}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
        message(STATUS "Linux RPATH configured: ${RPATH_DIRS}")
    endif()
endif()

# macOS環境の設定
if(APPLE)
    set(MACOSX_BUNDLE_BUNDLE_NAME "${PROJECT_NAME}")
    set(MACOSX_BUNDLE_GUI_IDENTIFIER "jp.co.radlab.ShioRIS3")
    set(MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_EXECUTABLE_NAME "ShioRIS3")

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MacOSXBundleInfo.plist.in
        ${CMAKE_CURRENT_BINARY_DIR}/MacOSXBundleInfo.plist
        @ONLY
    )

    set_target_properties(ShioRIS3 PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_BINARY_DIR}/MacOSXBundleInfo.plist
    )

    find_package(Threads REQUIRED)
    target_link_libraries(ShioRIS3 PRIVATE Threads::Threads)
    
    # Ensure Homebrew library paths are available to linker and at runtime
    set(MAC_EXTRA_RPATHS "")
    if(DCMTK_PREFIX)
        target_link_directories(ShioRIS3 PRIVATE ${DCMTK_PREFIX}/lib)
        list(APPEND MAC_EXTRA_RPATHS "${DCMTK_PREFIX}/lib")
    endif()

    if(ENABLE_ONNXRUNTIME AND ONNXRUNTIME_FOUND)
        get_filename_component(ONNX_LIB_DIR ${ONNXRUNTIME_LIB} DIRECTORY)
        if(ONNX_LIB_DIR)
            target_link_directories(ShioRIS3 PRIVATE ${ONNX_LIB_DIR})
            list(APPEND MAC_EXTRA_RPATHS "${ONNX_LIB_DIR}")
        endif()
    endif()

    # Whisperライブラリ用にFrameworksディレクトリもRPATHに追加
    list(APPEND MAC_EXTRA_RPATHS "@executable_path/../Frameworks")

    if(MAC_EXTRA_RPATHS)
        list(REMOVE_DUPLICATES MAC_EXTRA_RPATHS)
        set_target_properties(ShioRIS3 PROPERTIES
            INSTALL_RPATH "${MAC_EXTRA_RPATHS}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()

    # macOS特有のフレームワーク
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(ShioRIS3 PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()

    # ビルド後にWhisperとGGMLライブラリをFrameworksディレクトリにコピー
    add_custom_command(TARGET ShioRIS3 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks"
        # Copy whisper library
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:whisper>"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks/"
        # Copy all ggml-related libraries
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:ggml>"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:ggml-base>"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:ggml-cpu>"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:ggml-blas>"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:ggml-metal>"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks/"
        COMMENT "Copying Whisper and GGML libraries to app bundle"
    )

    # Copy logo image to app bundle Resources
    add_custom_command(TARGET ShioRIS3 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Resources/images"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/resources/images/ShioRIS3_Logo.png"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Resources/images/ShioRIS3_Logo.png"
        COMMENT "Copying logo image to app bundle"
    )

    # Copy application icon to app bundle Resources
    add_custom_command(TARGET ShioRIS3 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/resources/images/ShioRIS3.icns"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Resources/ShioRIS3.icns"
        COMMENT "Copying application icon to app bundle"
    )

    # libwhisper.dylibとすべてのlibggml*.dylibのrpathとシンボリックリンクを修正
    add_custom_command(TARGET ShioRIS3 POST_BUILD
        # Fix whisper library install name
        COMMAND install_name_tool -id "@rpath/libwhisper.1.dylib"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks/$<TARGET_FILE_NAME:whisper>"
        # Create symlink for whisper: libwhisper.1.dylib -> actual library file
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            "$<TARGET_FILE_NAME:whisper>"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks/libwhisper.1.dylib"
        # Fix all ggml library install names
        COMMAND install_name_tool -id "@rpath/$<TARGET_FILE_NAME:ggml>"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks/$<TARGET_FILE_NAME:ggml>"
        COMMAND install_name_tool -id "@rpath/$<TARGET_FILE_NAME:ggml-base>"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks/$<TARGET_FILE_NAME:ggml-base>"
        COMMAND install_name_tool -id "@rpath/$<TARGET_FILE_NAME:ggml-cpu>"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks/$<TARGET_FILE_NAME:ggml-cpu>"
        COMMAND install_name_tool -id "@rpath/$<TARGET_FILE_NAME:ggml-blas>"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks/$<TARGET_FILE_NAME:ggml-blas>"
        COMMAND install_name_tool -id "@rpath/$<TARGET_FILE_NAME:ggml-metal>"
            "$<TARGET_BUNDLE_DIR:ShioRIS3>/Contents/Frameworks/$<TARGET_FILE_NAME:ggml-metal>"
        COMMENT "Fixing library install names and creating symlinks"
        VERBATIM
    )
endif()



# デバッグ情報
message(STATUS "=== Build Configuration ===")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "DCMTK_INCLUDE_DIRS: ${DCMTK_INCLUDE_DIRS_PRIVATE}")
message(STATUS "DCMTK_LIBS: ${DCMTK_LIBS}")
if(ENABLE_ONNXRUNTIME)
    message(STATUS "ONNX Runtime: Enabled")
    message(STATUS "ONNX Include: ${ONNXRUNTIME_INCLUDE_DIRS}")
    message(STATUS "ONNX Library: ${ONNXRUNTIME_LIB}")
    if(UNIX AND NOT APPLE AND ONNXRUNTIME_USE_CUDA)
        message(STATUS "CUDA Support: Enabled")
    endif()
else()
    message(STATUS "ONNX Runtime: Disabled")
endif()

# ビルド後の実行スクリプト生成（macOS）
if(APPLE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    configure_file(
        "${CMAKE_SOURCE_DIR}/scripts/run_debug.sh.in"
        "${CMAKE_BINARY_DIR}/run_debug.sh"
        @ONLY
    )
    
    file(CHMOD "${CMAKE_BINARY_DIR}/run_debug.sh"
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
endif()

# Linux: ビルド後にWhisperとGGML共有ライブラリを実行可能ファイルのディレクトリにコピー
if(UNIX AND NOT APPLE)
    add_custom_command(TARGET ShioRIS3 POST_BUILD
        # Copy whisper library and create symlink
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:whisper>"
            "$<TARGET_FILE_DIR:ShioRIS3>"
        # Copy ggml libraries
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:ggml>"
            "$<TARGET_FILE_DIR:ShioRIS3>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:ggml-base>"
            "$<TARGET_FILE_DIR:ShioRIS3>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:ggml-cpu>"
            "$<TARGET_FILE_DIR:ShioRIS3>"
        COMMENT "Copying Whisper and GGML libraries to executable directory"
    )

    # Copy logo image to executable directory
    add_custom_command(TARGET ShioRIS3 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:ShioRIS3>/resources/images"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/resources/images/ShioRIS3_Logo.png"
            "$<TARGET_FILE_DIR:ShioRIS3>/resources/images/ShioRIS3_Logo.png"
        COMMENT "Copying logo image to executable directory"
    )
endif()

# Windows: ビルド後にWhisperとGGML DLLを実行可能ファイルのディレクトリにコピー
if(WIN32)
    add_custom_command(TARGET ShioRIS3 POST_BUILD
        # Copy whisper.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:whisper>"
            "$<TARGET_FILE_DIR:ShioRIS3>"
        # Copy ggml.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:ggml>"
            "$<TARGET_FILE_DIR:ShioRIS3>"
        # Copy ggml-base.dll if it exists
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:ggml-base>"
            "$<TARGET_FILE_DIR:ShioRIS3>"
        # Copy ggml-cpu.dll if it exists
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:ggml-cpu>"
            "$<TARGET_FILE_DIR:ShioRIS3>"
        COMMENT "Copying Whisper and GGML DLLs to executable directory"
    )

    # Copy logo image to executable directory
    add_custom_command(TARGET ShioRIS3 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:ShioRIS3>/resources/images"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/resources/images/ShioRIS3_Logo.png"
            "$<TARGET_FILE_DIR:ShioRIS3>/resources/images/ShioRIS3_Logo.png"
        COMMENT "Copying logo image to executable directory"
    )
endif()

# Copy web_client directory for Vision Pro web interface (all platforms)
add_custom_command(TARGET ShioRIS3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/web_client"
        "$<TARGET_FILE_DIR:ShioRIS3>/../web_client"
    COMMENT "Copying web_client directory for Vision Pro interface"
)
