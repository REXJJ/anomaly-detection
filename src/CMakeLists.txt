cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11 -O3  -lopencv_imgcodecs)
project( Anomaly1 )
find_package(OpenCV REQUIRED )
# find_package(Boost 1.57.0 COMPONENTS serialization)
# find_package(Armadillo REQUIRED) 

# find_library(MLPACK_LIBRARY
#   NAMES mlpack
#   PATHS "$ENV{ProgramFiles}/mlpack/" /usr/lib64/ /usr/lib/ /usr/local/lib64/ /usr/local/
# )

# include_directories(include)
include_directories(/usr/local/include/eigen3)
# include_directories(${ARMADILLO_INCLUDE_DIRS})

# add_executable(main  main.cpp)
# # target_link_libraries( main ${OpenCV_LIBS} ${Boost_LIBRARIES} ${MLPACK_LIBRARY} ${ARMADILLO_LIBRARIES})
# target_link_libraries( main ${OpenCV_LIBS})

# add_executable(contour_method  contour_method.cpp)
# target_link_libraries( contour_method ${OpenCV_LIBS})

# add_executable(gradient_filtering  gradient_filtering.cpp)
# target_link_libraries(gradient_filtering ${OpenCV_LIBS})

# add_executable(clustering_test  clustering_test.cpp)
# target_link_libraries(clustering_test ${OpenCV_LIBS})

add_executable(integrated  integrated.cpp)
target_link_libraries(integrated ${OpenCV_LIBS} ${Boost_LIBRARIES})

# add_executable(reference_image_test  reference_image_test.cpp)
# target_link_libraries(reference_image_test ${OpenCV_LIBS})