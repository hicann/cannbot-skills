# set the library output dir to the python dir for wheel package build
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/ascend_kernel/ascend_kernel/lib)

# host side files
FILE(GLOB OP_SRCS
    ${PROJECT_OP_SRC_BASE}/register.cpp
    ${PROJECT_OP_SRC_BASE}/ops/helloworld/op_host/helloworld.cpp
    ${PROJECT_OP_SRC_BASE}/ops/abs/op_host/abs.cpp
    ${PROJECT_OP_SRC_BASE}/ops/cumsum/op_host/cumsum.cpp
    ${PROJECT_OP_SRC_BASE}/ops/histc/op_host/histc.cpp
    ${PROJECT_OP_SRC_BASE}/ops/sum/op_host/sum.cpp
    ${PROJECT_OP_SRC_BASE}/ops/avg_pool3_d/op_host/avg_pool3_d.cpp
    )
if(BUILD_CATLASS_MODULE)
    list(APPEND OP_SRCS
        ${PROJECT_OP_SRC_BASE}/ops/catlass/op_host/catlass_matmul_basic.cpp
    )
endif()

# set the so name
set(OP_PLUGIN_NAME ascend_kernel)

# kernel side files without workspace
ascendc_library(no_workspace_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/ops/helloworld/op_kernel/kernel_helloworld.cpp
    ${PROJECT_OP_SRC_BASE}/ops/abs/op_kernel/abs.cpp
    ${PROJECT_OP_SRC_BASE}/ops/cumsum/op_kernel/cumsum.cpp
    ${PROJECT_OP_SRC_BASE}/ops/histc/op_kernel/histc.cpp
    ${PROJECT_OP_SRC_BASE}/ops/sum/op_kernel/sum.cpp
    ${PROJECT_OP_SRC_BASE}/ops/avg_pool3_d/op_kernel/avg_pool3_d_generic.cpp
    ${PROJECT_OP_SRC_BASE}/ops/avg_pool3_d/op_kernel/avg_pool3_d_reduce_d.cpp
    ${PROJECT_OP_SRC_BASE}/ops/avg_pool3_d/op_kernel/avg_pool3_d_split_c.cpp
    ${PROJECT_OP_SRC_BASE}/ops/avg_pool3_d/op_kernel/avg_pool3_d_split_w.cpp
    ${PROJECT_OP_SRC_BASE}/ops/avg_pool3_d/op_kernel/avg_pool3_d_multi_w.cpp
    )

# # kernel side files with workspace
# set(WORKSPACE_KERNEL_SRCS
#     ${PROJECT_OP_SRC_BASE}/ops/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
# )
# if(BUILD_CATLASS_MODULE)
#     list(APPEND WORKSPACE_KERNEL_SRCS
#         ${PROJECT_OP_SRC_BASE}/ops/catlass/op_kernel/catlass_matmul_basic_kernel.cpp
#     )
# endif()
# ascendc_library(workspace_kernel STATIC ${WORKSPACE_KERNEL_SRCS})
# if(BUILD_CATLASS_MODULE)
#     ascendc_include_directories(workspace_kernel PRIVATE
#         ${CATLASS_DIR}/include
#     )
# endif()

# ascendc_compile_definitions(workspace_kernel PRIVATE
#        -DHAVE_WORKSPACE
#        -DHAVE_TILING
# )

# create shared library libascend_kernel.so
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        # workspace_kernel
        no_workspace_kernel
        torch_npu
        ascendcl
        tiling_api
        nnopbase
        opapi
        register
        platform
        ascendalog
        dl
)

target_link_directories(${OP_PLUGIN_NAME} PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
)

target_include_directories(${OP_PLUGIN_NAME} PRIVATE
        ${PROJECT_OP_SRC_BASE}/utils
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_OP_SRC_BASE}/ops/avg_pool3_d/op_kernel
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include/third_party/acl/inc
        ${TORCH_NPU_DIR}/include/third_party/hccl/inc
        ${TORCH_NPU_DIR}/include
        ${PYTHON_INCLUDE_DIR}
        ${ASCEND_INCLUDE_DIR}/external
        ${ASCEND_INCLUDE_DIR}/experiment/platform
        ${ASCEND_INCLUDE_DIR}/experiment/runtime
)
