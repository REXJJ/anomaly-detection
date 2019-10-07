cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11 -O3)
project( Anomaly1 )
find_package(OpenCV REQUIRED )
find_package(Boost 1.57.0 COMPONENTS serialization)
find_package(Armadillo REQUIRED) 

find_library(MLPACK_LIBRARY
  NAMES mlpack
  PATHS "$ENV{ProgramFiles}/mlpack/" /usr/lib64/ /usr/lib/ /usr/local/lib64/ /usr/local/
)

include_directories(include)
include_directories(/usr/local/include/)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable(main  main.cpp)
target_link_libraries( main ${OpenCV_LIBS} ${Boost_LIBRARIES} ${MLPACK_LIBRARY} ${ARMADILLO_LIBRARIES})

