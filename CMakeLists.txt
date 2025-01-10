cmake_minimum_required(VERSION 3.15)

project(main)

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
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rk3588_test_pipeline)
 

# 设置CMake安装时的RPATH（运行时库搜索路径）
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/lib")

if(CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include/utils)
include_directories(${CMAKE_SOURCE_DIR}/include/stb_image)
include_directories(${CMAKE_SOURCE_DIR}/include/outer_model)
include_directories(${CMAKE_SOURCE_DIR}/include/ppocr)



# 添加静态库目标
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 添加一个可执行文件目标，包括main.cpp和选定的retinaface实现文件
add_executable(${PROJECT_NAME}
    main.cpp
    )

target_link_libraries(${PROJECT_NAME}
    rknn_model_pipeline
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
install(TARGETS rknn_model_pipeline ARCHIVE DESTINATION lib)
install(TARGETS ${PROJECT_NAME} DESTINATION ./)
# 安装测试图片到model目录下
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/test.png DESTINATION ./data)
# 模型文件拷贝到model目录下
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/RetinaFace.rknn DESTINATION ./model)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/FaceAttr.rknn DESTINATION ./model)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/yolov10s.rknn DESTINATION ./model)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/coco_80_labels_list.txt DESTINATION ./model)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/PhoneDet.rknn DESTINATION ./model)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/HeaderDet.rknn DESTINATION ./model)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/rec_ren_1225_resnet18.rknn DESTINATION ./model)
# 头文件拷贝到include目录下
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/outer_model/model_params.hpp DESTINATION ./include/outer_model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/outer_model/model_func.hpp DESTINATION ./include/outer_model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/inter_model/resnet18.h DESTINATION ./include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/inter_model/yolo_image_preprocess.h DESTINATION ./include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/inter_model/yolo_postprocess.h DESTINATION ./include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/inter_model/yolov8.h DESTINATION ./include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/inter_model/yolov10.h DESTINATION ./include)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/common.h DESTINATION ./include)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/stb_image/stb_image.h DESTINATION ./include)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/stb_image/stb_image_resize2.h DESTINATION ./include)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknpu2/include/rknn_api.h DESTINATION ./include)

# 设置调试标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -o0")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -o0")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -o0")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g -o0")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -g -o0")
set(CMAKE_BUILD_TYPE Debug)
