cmake_minimum_required(VERSION 3.14)
project(cudacompat C CXX)

set(CMAKE_C_STANDARD 11)
option(CUDA_COMPAT_DEBUG "Enable debug mode with ACL error message logging" OFF)
if(CUDA_COMPAT_DEBUG)
    message(STATUS "CUDA_COMPAT_DEBUG enabled - ACL errors will print detailed messages")
    add_definitions(-DCUDA_COMPAT_DEBUG_MODE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
endif()

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CANN paths - can be overridden with -DCANN_PATH
if(NOT DEFINED CANN_PATH)
    if($ENV{HOME} STREQUAL "/root")
        set(CANN_PATH "/usr/local/Ascend/cann")
    else()
        set(CANN_PATH "$ENV{HOME}/Ascend/cann")
    endif()

    if(NOT EXISTS ${CANN_PATH})
        message(FATAL_ERROR
            "CANN_PATH not set and default path ${CANN_PATH} does not exist. "
            "Please set CANN_PATH manually: cmake -DCANN_PATH=/your/path ..")
    endif()
endif()

set(CANN_INCLUDE_DIR "${CANN_PATH}/include")
set(CANN_LIB_DIR "${CANN_PATH}/lib64")
set(CANN_PKG_INC_DIR "")
foreach(CANN_ARCH_DIR aarch64-linux x86_64-linux)
    if(EXISTS "${CANN_PATH}/${CANN_ARCH_DIR}/pkg_inc")
        set(CANN_PKG_INC_DIR "${CANN_PATH}/${CANN_ARCH_DIR}/pkg_inc")
        break()
    endif()
endforeach()

# Export CANN paths to parent scope for tests and other components
set(CANN_INCLUDE_DIR "${CANN_INCLUDE_DIR}" PARENT_SCOPE)
set(CANN_LIB_DIR "${CANN_LIB_DIR}" PARENT_SCOPE)

# Include directories
include_directories(${CANN_INCLUDE_DIR})
if(CANN_PKG_INC_DIR)
    include_directories(${CANN_PKG_INC_DIR})
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Link directories
link_directories(${CANN_LIB_DIR})

# Source files for the compatibility library
set(CUDACOMPAT_SOURCES
    cann_compat.c
    cann_compat_ipc.c
    cann_compat_library.c
)

# Header files for installation
set(CUDACOMPAT_HEADERS
    cann_runtime_compat.h
    cann_compat_error_types.h
    cann_compat_types.h
    cann_compat_device_types.h
    cann_compat_device.h
    cann_compat_memory.h
    cann_compat_mempool.h
    cann_compat_stream.h
    cann_compat_event.h
    cann_compat_ipc.h
    cann_compat_symbol.h
    cann_compat_library.h
    cann_compat_graph.h
    cann_compat_unsupported.h
    cann_compat_cu_types.h
    cann_compat_context.h
    cann_compat_cu_vmm.h
    cann_compat_cu_extra.h
    cann_compat_exec.h
)

# Create static library
add_library(cudacompat_static STATIC ${CUDACOMPAT_SOURCES})
set_target_properties(cudacompat_static PROPERTIES
    OUTPUT_NAME cudacompat
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Create shared library
add_library(cudacompat SHARED ${CUDACOMPAT_SOURCES})
set_target_properties(cudacompat PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)

# Find pthread package
find_package(Threads REQUIRED)

# Link libraries
target_link_libraries(cudacompat acl_rt runtime msprofiler Threads::Threads)
target_link_libraries(cudacompat_static acl_rt runtime msprofiler Threads::Threads)

# Installation
install(TARGETS cudacompat cudacompat_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${CUDACOMPAT_HEADERS}
    DESTINATION include/cudacompat
)

# Print configuration info
message(STATUS "CANN Path: ${CANN_PATH}")
message(STATUS "CANN Include: ${CANN_INCLUDE_DIR}")
message(STATUS "CANN Lib: ${CANN_LIB_DIR}")
if(CUDA_COMPAT_DEBUG)
    message(STATUS "Build Mode: DEBUG (with ACL error logging)")
else()
    message(STATUS "Build Mode: RELEASE")
endif()
