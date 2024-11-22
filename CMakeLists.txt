cmake_minimum_required(VERSION 3.15)

project(rknn_model_pipeline)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 如果启用了地址 sanitizer(ASAN)，则设置相应的编译器和链接器标志
if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rk3588_model_pipeline_${CMAKE_SYSTEM_NAME})


# 设置CMake安装时的RPATH（运行时库搜索路径）
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/lib")

if(CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/yaml-cpp)
include_directories(${CMAKE_SOURCE_DIR}/include/stb_image)
include_directories(${CMAKE_SOURCE_DIR}/include/yolov10)
include_directories(${CMAKE_SOURCE_DIR}/include/model)
include_directories(${CMAKE_SOURCE_DIR}/include/utils)
include_directories(${CMAKE_SOURCE_DIR}/third_party/opencv/opencv-linux-aarch64/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/rknpu2/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/librga/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/jpeg_turbo/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/opencv/opencv-linux-aarch64/include)
# 获取当前源目录下所有的C++源文件
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.c*)

# 添加一个可执行文件目标，包括main.cpp和选定的retinaface实现文件
add_executable(${PROJECT_NAME}
    src/main.cpp
    
    src/person/yolov10_det_person.cc
    src/person/yolov10.cc
    src/person/postprocess.cc
    src/person/person_attr.cpp
    src/person/person_det_attr.cpp
    
    src/face/face_attr.cpp
    src/face/face_det.cpp
    src/face/face_det_attr.cpp
    src/face/retinaface.cc
    
    src/object/phone_det.cpp
    src/object/retinanet.cc
    src/object/header_det.cpp
    src/face/header_det_attr.cpp
    src/face/header_attr.cpp

    src/utils/file_utils.c
    src/utils/image_utils.c
    src/utils/image_drawing.c
)

# 链接必要的库文件，包括fileutils、imageutils、imagedrawing、librknnrt和dl

set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/third_party/rknpu2/Linux/aarch64/librknnrt.so)
set(YAML_LIB ${CMAKE_SOURCE_DIR}/third_party/libyaml-cpp.a)
# rga, comes from https://github.com/airockchip/librga
set(RGA_LIB ${CMAKE_SOURCE_DIR}/third_party/librga/Linux/aarch64/librga.so)
set(jpeg_turbo ${CMAKE_SOURCE_DIR}/third_party/jpeg_turbo/Linux/aarch64/libturbojpeg.a)
set(OPENCV_LIB ${CMAKE_SOURCE_DIR}/third_party/opencv/opencv-linux-aarch64/lib/libopencv_core.a)
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/third_party/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)

target_link_libraries(${PROJECT_NAME}
    ${RKNN_RT_LIB}
    ${YAML_LIB}
    ${RGA_LIB}
    ${jpeg_turbo}
    ${OpenCV_LIBS}
    ${OPENCV_LIB}
)

# 如果系统是Linux，设置使用pthread，链接Threads库
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

# 为项目目标设置私有包含目录，包括当前源目录和librknnrt的头文件目录
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
)

# 安装目标可执行文件到当前目录下
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(FILES ${YAML_LIB} DESTINATION lib)
install(FILES ${RKNN_RT_LIB} DESTINATION lib)
install(FILES ${jpeg_turbo} DESTINATION lib)
install(TARGETS ${PROJECT_NAME} DESTINATION ./)
# 安装测试图片到model目录下
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/test.png DESTINATION ./data)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/phone.png DESTINATION ./data)
# 模型文件拷贝到model目录下
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/RetinaFace.rknn DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/FaceAttr.rknn DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/PersonAttr.rknn DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/yolov10s.rknn DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/coco_80_labels_list.txt DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/PhoneDet.rknn DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/HeaderDet.rknn DESTINATION ./model)
# 设置调试标志，增加编译器-g选项以包含调试信息
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -o0")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -o0")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -o0")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g -o0")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -g -o0")
set(CMAKE_BUILD_TYPE Debug)
