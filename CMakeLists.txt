cmake_minimum_required(VERSION 3.16)
project(P)
# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native")

# For normal building (optimized, no debug symbols)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)

# Add sanitizers for Debug builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address -fsanitize=undefined")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined")

#cmake_minimum_required(VERSION 3.16)
#project(P)

# Set C++ standard
#set(CMAKE_CXX_STANDARD 23)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native")
# For normal building (optimized, no debug symbols)

#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)

# Add your source files here
add_executable(P src/main.cpp
        src/math_utils.cpp
        src/math_utils.h
        src/plotting_utils.cpp
        src/plotting_utils.h
        src/data_types.cpp
        src/data_types.h
        src/bodies.cpp
        src/bodies.h
        src/system_of_bodies.cpp
        src/system_of_bodies.h
        src/tests_utils.cpp
        src/tests_utils.h
       # src/system_of_bodies_inverse_dynamics.cpp
        src/system_of_bodies_forward_dynamics.cpp
        src/system_structure_utils.cpp
        src/Tree_forward_dynamics.cpp
)

target_link_libraries(P
    armadillo
    openblas
    raylib
    muparser
    Python3::Python
)


# Find Python development files
find_package(Python3 COMPONENTS Development)

# Add Python include directory
# Get NumPy include directory
execute_process(
    COMMAND python -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Add to include directories
target_include_directories(P PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src  # your source directory
)
