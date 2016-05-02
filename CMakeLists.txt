# cmake version used
cmake_minimum_required(VERSION 2.8.12.2)
# project name
project(icp_program)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
#SET(FLANN_ROOT /home/dipendra/Dropbox/icp/flann-1.8.4-src)

# includes header files
include_directories(${PROJECT_SOURCE_DIR}/include /usr/local/include /usr/include/hdf5/serial)
link_directories(/usr/local/lib /usr/lib/x86_64-linux-gnu)

#add_subdirectory(${FLANN_ROOT} sub_project/flann)
add_subdirectory(src)
add_subdirectory(res)
