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
    ${CMAKE_SOURCE_DIR}/third_party/rknpu2/include
    ${CMAKE_SOURCE_DIR}/third_party/librga/include
    ${CMAKE_SOURCE_DIR}/third_party/jpeg_turbo/include
    # ${CMAKE_SOURCE_DIR}/third_party/opencv/opencv-linux-aarch64/include
    ${OpenCV_INCLUDE_DIRS}  # 使用rk3588系统安装动态库地址
)

# 获取所有子目录的头文件到HEADER变量中
file(GLOB_RECURSE HEADER 
    ${CMAKE_SOURCE_DIR}/include/*.h*
    ${CMAKE_SOURCE_DIR}/third_party/*.h*
    )
# file(GLOB_RECURSE SOURCE ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE SOURCE 
    ${CMAKE_SOURCE_DIR}/src/ssd_det/*.c*
    ${CMAKE_SOURCE_DIR}/src/yolo_det/*.c*
    ${CMAKE_SOURCE_DIR}/src/classify_model/*.c*
    ${CMAKE_SOURCE_DIR}/src/business_pipeline/*.c*
    ${CMAKE_SOURCE_DIR}/src/utils/*.c
    )
message("HEADER: ${HEADER}")
message("SOURCE: ${SOURCE}")

add_library(
    rknn_model_pipeline
    STATIC
    ${HEADER}  # 自己调用的头文件全部打包在静态库中，提供给外部使用的头文件放在model_params.hpp和model_func.cpp中
    ${SOURCE}
)
# set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/third_party/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)
# message("OpenCV_DIR: ${OpenCV_DIR}")
message("OpenCV_LIBS:${OpenCV_LIBS}")

# 手动指定静态库路径
# set(OpenCV_STATIC_LIB_PATH ${CMAKE_SOURCE_DIR}/third_party/opencv/opencv-linux-aarch64/lib)
# set(OpenCV_STATIC_LIBS
#     ${OpenCV_STATIC_LIB_PATH}/libopencv_core.a
#     ${OpenCV_STATIC_LIB_PATH}/libopencv_imgcodecs.a
#     ${OpenCV_STATIC_LIB_PATH}/libopencv_imgproc.a
#     ${OpenCV_STATIC_LIB_PATH}/libopencv_calib3d.a
#     ${OpenCV_STATIC_LIB_PATH}/libopencv_features2d.a
#     ${OpenCV_STATIC_LIB_PATH}/libopencv_video.a
#     ${OpenCV_STATIC_LIB_PATH}/libopencv_dnn.a
# )
# message("OpenCV_STATIC_LIBS:${OpenCV_STATIC_LIBS}")

# 添加其他依赖库
set(EXTRA_LIBS
    pthread
    m
    dl
)
message("EXTRA_LIBS:${EXTRA_LIBS}")

set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/third_party/rknpu2/Linux/aarch64/librknnrt.so)
set(RGA_LIB ${CMAKE_SOURCE_DIR}/third_party/librga/Linux/aarch64/librga.so)
set(jpeg_turbo ${CMAKE_SOURCE_DIR}/third_party/jpeg_turbo/Linux/aarch64/libturbojpeg.a)

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
    # ${OpenCV_STATIC_LIBS}
    ${EXTRA_LIBS}
)

