npu_op_kernel_sources(ascendc_kernels
    OP_TYPE SampleOp
    KERNEL_DIR .
    KERNEL_FILE sample_op_arch22.cpp
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)
