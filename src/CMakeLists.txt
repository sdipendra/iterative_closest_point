#compile options
add_compile_options(-std=c++11)

#SET(HDF5_LIBRARIES /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so /usr/lib/x86_64-linux-gnu/libpthread.so /usr/lib/x86_64-linux-gnu/libsz.so /usr/lib/x86_64-linux-gnu/libz.so /usr/lib/x86_64-linux-gnu/libdl.so /usr/lib/x86_64-linux-gnu/libm.so)

add_subdirectory(cu)

add_executable(icp1 icp1.cpp)
target_link_libraries(icp1 libflann_cuda_s.a libcudart.so libflann_cpp_s.a gsl gslcblas)

#add_executable(flann_example flann_example.cpp)
#target_link_libraries(flann_example flann_cpp)
#target_link_libraries(flann_example ${HDF5_LIBRARIES})
