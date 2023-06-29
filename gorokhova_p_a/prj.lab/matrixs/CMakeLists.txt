add_library(matrixs matrixs.cpp include/matrixs/matrixs.hpp)
set property(TARGET matrixs PROPERTY CXX_STANDARD 20)
target_include_directories(matrixs PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}
	$<INSTALL_INTERFACE:include>
)
install(TARGETS matrixs)