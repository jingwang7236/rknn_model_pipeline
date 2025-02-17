message("sub CMakeLists.txt")
message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

# 设置生成库文件的保存地址
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/custom_lib_dir)
# message("CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/utils
    ${CMAKE_SOURCE_DIR}/include/outer_model  # 外部模型使用的头文件，提供给端侧使用
    ${CMAKE_SOURCE_DIR}/include/inter_model  # 内部模型使用的头文件
    ${CMAKE_SOURCE_DIR}/include/ppocr        # ppocr使用的头文件
    # ${CMAKE_SOURCE_DIR}/third_party/rknpu2/include  # 第三方库头文件
    ${CMAKE_SOURCE_DIR}/third_party/librga/include
    ${CMAKE_SOURCE_DIR}/third_party/jpeg_turbo/include
    ${OpenCV_INCLUDE_DIRS}  # 使用rk3588系统安装动态库地址
    ${RKNN_RT_LIB_INCLUDE_DIRS}
)

# 获取所有子目录的头文件到HEADER变量中
file(GLOB_RECURSE HEADER 
    ${CMAKE_SOURCE_DIR}/include/*.h*
    ${CMAKE_SOURCE_DIR}/third_party/*.h*
    )
file(GLOB_RECURSE SOURCE 
    ${CMAKE_SOURCE_DIR}/src/ssd_det/*.c*
    ${CMAKE_SOURCE_DIR}/src/yolo_det/*.c*
    ${CMAKE_SOURCE_DIR}/src/yolo_pose/*.c*
    ${CMAKE_SOURCE_DIR}/src/classify_model/*.c*
    ${CMAKE_SOURCE_DIR}/src/business_pipeline/*.c*  # 业务逻辑
    ${CMAKE_SOURCE_DIR}/src/utils/*.c*
    ${CMAKE_SOURCE_DIR}/src/ppocr/*.c*
    ${CMAKE_SOURCE_DIR}/src/resnet18/*.c*
    ${CMAKE_SOURCE_DIR}/src/mobilenet/*.c*
    ${CMAKE_SOURCE_DIR}/src/tests/*.c*  # 测试模型指标,包含分类模型的准确率和检测模型的map/pr
    
    )
message("HEADER: ${HEADER}")
message("SOURCE: ${SOURCE}")

add_library(
    rknn_model_pipeline
    STATIC  # 静态库
    # SHARED #动态库
    ${HEADER}  # 自己调用的头文件全部打包在静态库中，提供给外部使用的头文件放在model_params.hpp和model_func.cpp中
    ${SOURCE}
)

# 第三方动态库查找和链接
find_package(OpenCV REQUIRED)
message("OpenCV_LIBS:${OpenCV_LIBS}")
find_library(RKNN_RT_LIB NAMES rknnrt PATHS /usr/lib)
message("RKNN_RT_LIB:${RKNN_RT_LIB}")
# find_library(RGA_LIB NAMES rga PATHS /usr/lib)
# message("RGA_LIB:${RGA_LIB}")

# 手动指定静态库路径
# set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/third_party/rknpu2/Linux/aarch64/librknnrt.so)
set(RGA_LIB ${CMAKE_SOURCE_DIR}/third_party/librga/Linux/aarch64/librga.so)
set(jpeg_turbo ${CMAKE_SOURCE_DIR}/third_party/jpeg_turbo/Linux/aarch64/libturbojpeg.a)


# 添加其他依赖库
set(EXTRA_LIBS
    pthread
    m
    dl
)
message("EXTRA_LIBS:${EXTRA_LIBS}")

# find_package(Threads REQUIRED)
# if(Threads_FOUND)
#     message(STATUS "Threads library found")
# else()
#     message(FATAL_ERROR "Threads library not found")
# endif()

target_link_libraries(
    rknn_model_pipeline
    ${RKNN_RT_LIB}
    ${RGA_LIB}
    ${jpeg_turbo}
    ${OpenCV_LIBS}
    ${EXTRA_LIBS}
)

# 设置调试标志，增加编译器-g选项以包含调试信息
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -o0")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -o0")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -o0")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g -o0")
# set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -g -o0")
# set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Os")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Os")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Os")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -Os")
set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} -Os")
set(CMAKE_BUILD_TYPE Release)