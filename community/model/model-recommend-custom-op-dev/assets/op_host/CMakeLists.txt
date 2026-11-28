set(host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/sample_op.cpp
)

set(host_ops_infershape_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/sample_op_infershape.cpp
)

set(host_ops_tiling_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/arch22/sample_op_tiling.cpp
)

set(host_ops_srcs
    ${host_ops_def_srcs}
    ${host_ops_infershape_srcs}
    ${host_ops_tiling_srcs}
)

if(NOT DEFINED ASCEND_HOST_ARCH)
    set(ASCEND_HOST_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
endif()

npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/include
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/asc/include/tiling
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc/op_common
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc/base
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc/exe_graph
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc/graph
        -I$ENV{ASCEND_HOME_PATH}/opp
        -I${CMAKE_CURRENT_SOURCE_DIR}
)

npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

target_include_directories(cust_optiling PRIVATE
    $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/include
    $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/opp
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(SECURE_COMPILE_OPTIONS -fPIC -fstack-protector-all)
set(SECURE_LINK_OPTIONS -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack)

target_compile_options(cust_optiling PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_optiling PRIVATE ${SECURE_LINK_OPTIONS})

option(ENABLE_OPAPI "Build and package opapi (ACLNN) library" ON)

if(ENABLE_OPAPI)
    set(op_api_dir ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
    set(op_api_srcs
        ${op_api_dir}/aclnn_sample_op.cpp
        ${op_api_dir}/sample_op.cpp
    )
    npu_op_library(cust_opapi ACLNN
        ${op_api_srcs}
    )

    target_include_directories(cust_opapi PRIVATE
        ${op_api_dir}
        $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/include
        $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/include/aclnn
        $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/asc/include
        $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc
        $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc/op_common
        $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc/base
        $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc/aicpu
        $ENV{ASCEND_HOME_PATH}/opp
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(cust_opapi PRIVATE opapi_math)
    target_compile_options(cust_opapi PRIVATE ${SECURE_COMPILE_OPTIONS} -std=c++17)
    target_link_options(cust_opapi PRIVATE ${SECURE_LINK_OPTIONS})
endif()

file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${proto_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_op_proto GRAPH
    ${host_ops_srcs}
    ${proto_src}
)

target_include_directories(cust_op_proto PRIVATE
    $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/include
    $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/${ASCEND_HOST_ARCH}-linux/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/opp
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(cust_op_proto PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_op_proto PRIVATE ${SECURE_LINK_OPTIONS})

set(package_libs
    cust_optiling
    cust_op_proto
)

if(ENABLE_OPAPI)
    list(APPEND package_libs cust_opapi)
endif()

npu_op_package_add(${package_name}
    LIBRARY
        ${package_libs}
)
