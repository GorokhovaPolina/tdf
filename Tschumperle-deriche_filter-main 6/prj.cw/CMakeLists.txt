# Specify the minimum version.
cmake_minimum_required(VERSION 3.21)

# Specify the project info.
project(TDF VERSION 1.0.0 DESCRIPTION "TDF PROJECT")

# For access to standard installation directory variables (CMAKE_INSTALL_xDIR).
include(GNUInstallDirs)

# Declare the library target.
add_library(
        ${PROJECT_NAME}
        source/TDF.cpp
        source/TDF.hpp
)

if(DEFINED MYLIB_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${MYLIB_SHARED_LIBS})
endif()

if (PROJECT_IS_TOP_LEVEL)
    set_target_properties(
            ${PROJECT_NAME}
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out
    )
endif ()

# Set the version property.
set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        PUBLIC_HEADER source/TDF.hpp
)

# Configure the directories to search for header files.
target_include_directories(${PROJECT_NAME} PUBLIC source)

find_package(Doxygen)
if (DOXYGEN_FOUND)
    set(TDF_DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxfile.in)
    set(TDF_DOXYGEN_OUT ${CMAKE_BINARY_DIR}/Doxfile)
    configure_file(${TDF_DOXYGEN_IN} ${TDF_DOXYGEN_OUT})
    add_custom_target(TDF_doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${TDF_DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
    install (DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/TDF_doc
            DESTINATION ./bin)
else (DOXYGEN_FOUND)
    message(STATUS "Doxygen was not found!")
endif (DOXYGEN_FOUND)

find_package(OpenCV CONFIG REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

# Set library shared object and API header file to install

add_subdirectory(example)

if(WIN32)
    set(OpenCV_RUNTIME_PATH "${OpenCV_DIR}/../../bin")
else()
    set(OpenCV_RUNTIME_PATH "${OpenCV_DIR}/../../lib")
endif()

install(DIRECTORY "${OpenCV_RUNTIME_PATH}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*opencv_core*"
        )

install(DIRECTORY "${OpenCV_RUNTIME_PATH}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*highgui*"
        )

install(DIRECTORY "${OpenCV_RUNTIME_PATH}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*imgcodecs*"
        )

install(DIRECTORY "${OpenCV_RUNTIME_PATH}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*imgproc*"
        )

install(DIRECTORY "${OpenCV_RUNTIME_PATH}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*videoio*"
        )

install(DIRECTORY "${OpenCV_RUNTIME_PATH}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*zlib*"
        )

install(DIRECTORY "${OpenCV_RUNTIME_PATH}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*jpeg*"
        )

install(DIRECTORY "${OpenCV_RUNTIME_PATH}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*libwebpdecoder*"
        )

install(DIRECTORY "${OpenCV_RUNTIME_PATH}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*libwebp*"
        )

install(DIRECTORY "${OpenCV_RUNTIME_PATH}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*libpng*"
        )

install(DIRECTORY "${OpenCV_RUNTIME_PATH}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*libsharpyuv*"
        )

install(DIRECTORY "${OpenCV_RUNTIME_PATH}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*tiff*"
        )
        
install(DIRECTORY "${OpenCV_RUNTIME_PATH}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*liblzma*"
        )

install(
        TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
