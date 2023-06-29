cmake_minimum_required(VERSION 3.21)
project(example)

find_package(OpenCV CONFIG REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})
message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIR})

add_executable(${PROJECT_NAME} TDF_app.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} TDF)

target_compile_definitions(${PROJECT_NAME} PRIVATE INSTALL_PATH="${CMAKE_INSTALL_PREFIX}/bin")
