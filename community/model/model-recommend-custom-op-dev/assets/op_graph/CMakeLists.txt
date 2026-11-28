file(GLOB proto_headers "${CMAKE_CURRENT_SOURCE_DIR}/*_proto.h")
if(NOT proto_headers)
    message(FATAL_ERROR "op_graph/ directory must contain at least one *_proto.h file with REG_OP definition")
endif()

if(TARGET cust_op_proto)
    target_include_directories(cust_op_proto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
