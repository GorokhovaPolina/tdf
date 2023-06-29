add_library(orto orto.cpp include/orto/orto.hpp)
set_property(TARGET orto PROPERTY CXX_STANDARD 20)
target_include_directories(orto PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
install(TARGETS orto)