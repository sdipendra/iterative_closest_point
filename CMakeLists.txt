# cmake version used
cmake_minimum_required(VERSION 2.8.12.2)
# project name
project(icp_program)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP)
#SET(FLANN_ROOT /home/dipendra/Dropbox/icp/flann-1.8.4-src)

# includes header files
include_directories(${PROJECT_SOURCE_DIR}/include /usr/local/include /usr/include/hdf5/serial /usr/local/cuda-7.5/targets/x86_64-linux/include)
link_directories(/usr/local/lib /usr/lib/x86_64-linux-gnu /usr/local/cuda-7.5/targets/x86_64-linux/lib)

#add_subdirectory(${FLANN_ROOT} sub_project/flann)
add_subdirectory(src)
add_subdirectory(res)
