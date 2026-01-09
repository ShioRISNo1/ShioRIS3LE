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
    include/visualization/fusion_dialog.h
    include/visualization/license_dialog.h
    include/platform/macos_audio_permissions.h
    include/database/database_manager.h
    include/database/smart_scanner.h
    include/database/database_sync_manager.h
    include/data/file_structure_manager.h
    include/data/multi_modality_manager.h
    include/data/metadata_generator.h
    include/web/web_server.h
    include/export/usdz_exporter.h
)

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
    src/visualization/fusion_dialog.cpp
    src/visualization/license_dialog.cpp
    src/database/database_manager.cpp
    src/database/smart_scanner.cpp
    src/database/database_sync_manager.cpp
    src/data/file_structure_manager.cpp
    src/data/multi_modality_manager.cpp
    src/data/metadata_generator.cpp
    src/web/web_server.cpp
    src/export/usdz_exporter.cpp
)

# Add macOS audio permissions implementation
if(APPLE)
    list(APPEND SOURCES src/platform/macos_audio_permissions.mm)
endif()


# リソースファイル
set(RESOURCES
    resources/resources.qrc
)

# 実行ファイル作成
qt_add_executable(ShioRIS3 ${SOURCES} ${HEADERS} ${RESOURCES})

# インクルードディレクトリ
target_include_directories(ShioRIS3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
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

    # RPATH設定
    set(RPATH_DIRS "")

    if(RPATH_DIRS)
        list(REMOVE_DUPLICATES RPATH_DIRS)
        set_target_properties(ShioRIS3 PROPERTIES
            INSTALL_RPATH "${RPATH_DIRS}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
        message(STATUS "Linux RPATH configured: ${RPATH_DIRS}")
    else()
        message(STATUS "No RPATH configured for Linux")
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
endif()



# デバッグ情報
message(STATUS "=== Build Configuration ===")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "DCMTK_INCLUDE_DIRS: ${DCMTK_INCLUDE_DIRS_PRIVATE}")
message(STATUS "DCMTK_LIBS: ${DCMTK_LIBS}")

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

# Linux: Copy logo image to executable directory
if(UNIX AND NOT APPLE)
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

# Windows: Copy logo image to executable directory
if(WIN32)
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
