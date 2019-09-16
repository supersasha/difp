cmake_minimum_required (VERSION 3.5.1)
project (difp)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -g -O3 -std=c++17")
set(SOURCES
    src/main.cc
    src/filebrowser.cc
    src/jpeg.cc
    src/texture.cc
    src/utils.cc
    src/image.cc
    src/gui.cc
    src/cldriver.cc
    src/photo_process_opts.cc
    src/film.cc
    src/frame.cc
)
set(EXTRA_SOURCES
        third-party/imgui/imgui.cpp
        third-party/imgui/imgui_draw.cpp
        third-party/imgui/imgui_widgets.cpp
        third-party/imgui-impl/imgui_impl_glfw.cpp
        third-party/imgui-impl/imgui_impl_opengl3.cpp
        third-party/imgui-impl/libs/gl3w/GL/gl3w.c
)
include_directories(
    third-party/imgui
    third-party/imgui-impl
    third-party/imgui-impl/libs/gl3w
    /usr/local/include/libraw
    ${GLFW_INCLUDE_DIRS}
)

add_executable(difp ${EXTRA_SOURCES} ${SOURCES})
target_link_libraries(difp dl GL turbojpeg raw OpenCL pthread ${GLFW_LIBRARIES})
