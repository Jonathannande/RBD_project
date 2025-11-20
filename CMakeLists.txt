cmake_minimum_required(VERSION 3.16)
project(MyProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

add_definitions(-DARMA_NO_DEBUG)
# Add your source files here
add_executable(MyProject src/main.cpp
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
        src/Tests_utils.cpp
        src/Tests_utils.h
       # src/system_of_bodies_inverse_dynamics.cpp
        src/system_of_bodies_forward_dynamics.cpp
        src/system_structure_utils.cpp
        #src/Tree_forward_dynamics.cpp
)

# Include directories
target_include_directories(MyProject PRIVATE
        "C:/cpp_libraries/armadillo-14.2.2/armadillo-14.2.2/include"
        "C:/msys64/mingw64/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "C:/Python310/include"
        "C:/Python310/Lib/site-packages/numpy/core/include"
)

# Library directories
target_link_directories(MyProject PRIVATE
        "C:/cpp_libraries/OpenBLAS-0.3.29_x64/lib"
        "C:/msys64/mingw64/lib"
        "C:/Python310/libs"
)

# Link libraries
target_link_libraries(MyProject
        python310
        openblas
        raylib
        opengl32
        muparser
)

# Set the compiler (MinGW)
set(CMAKE_CXX_COMPILER "C:/msys64/mingw64/bin/g++.exe")
set(CMAKE_C_COMPILER "C:/msys64/mingw64/bin/gcc.exe")