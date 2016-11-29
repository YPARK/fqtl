cmake_minimum_required(VERSION 3.6)
project(FQTL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPPFLAGS} -std=c++14 -DNDEBUG -O2")

include_directories("${PROJECT_SOURCE_DIR}/util")
include_directories("${PROJECT_SOURCE_DIR}/repr")
include_directories("${PROJECT_SOURCE_DIR}/param")
add_subdirectory(util)

file(GLOB param_files param "*.hh")
file(GLOB util_files util "*.hh")

add_executable(fqtl fqtl_main.cc fqtl.hh qtl_model.hh
        repr/gaus_repr.hh dummy.hh
        ${param_files} ${util_files} fqtl_main.hh)
target_link_libraries(fqtl GZIP z)

add_executable(fqtl_zscore fqtl_zscore.cc fqtl_zscore.hh qtl_model.hh
        repr/gaus_repr.hh dummy.hh
        ${param_files} ${util_files})
target_link_libraries(fqtl_zscore GZIP z)

add_executable(fqtl_mf fqtl_mf.cc fqtl_mf.hh qtl_model.hh
        repr/gaus_repr.hh dummy.hh
        ${param_files} ${util_files})
target_link_libraries(fqtl_mf GZIP z)
