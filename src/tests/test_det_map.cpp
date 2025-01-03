/* 适用于检测模型计算map指标的测试文件，该文件最终会编译成库，
 * 输入：测试集列表文件(包含图片路径和标注xml文件路径) 模型地址
 * 输出：map指标
 * 
 * 该功能包含三个部分
 * 1. 每张图片使用模型推理得到结果---保存结果，后续需要使用这个结果计算PR等其他指标
 * 2. 每张图片读取标注结果---保存结果，后续需要使用这个结果计算PR等其他指标
 * 3. 同一张图像的预测结果和标注结果进行对比，计算map指标
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <string.h>

#include "model_func.hpp"
#include "model_params.hpp"
// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


int cal_det_map(const char *test_list_file, const char *model_file){
    // 模型初始化
    int ret=0;
    const char* model_path = "model/HeaderDet.rknn";
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    ret = init_model(model_path, &rknn_app_ctx);  // 初始化
    if (ret != 0) {
        printf("init_retinanet_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    // 读取测试集列表
    std::ifstream test_list_file_stream(test_list_file);
    std::string test_list_file_line;
    std::vector<std::string> test_list_file_lines;
    while (std::getline(test_list_file_stream, test_list_file_line)) {
        test_list_file_lines.push_back(test_list_file_line);
    }
    for (int i = 0; i < test_list_file_lines.size(); i++) {
        std::cout << test_list_file_lines[i] << std::endl;
        // Load image
        const char *image_path = test_list_file_lines[i].c_str();
        int width, height, channel;
        unsigned char* data = stbi_load(image_path, &width, &height, &channel, 3);
        if (data == NULL) {
            printf("Failed to load image from path: %s\n", image_path);
            return -1;
        }

        // init input data
        det_model_input input_data;
        input_data.data = data;
        input_data.width = width;
        input_data.height = height;
        input_data.channel = channel;

        ssd_det_result result = inference_header_det_model(&rknn_app_ctx, input_data, true); //推理

        for (int i = 0; i < result.count; i++) {
            printf("box: %d %d %d %d, score: %f, cls: %d\n", result.object[i].box.left, result.object[i].box.top, result.object[i].box.right, result.object[i].box.bottom, result.object[i].score, result.object[i].cls);
        }

    }
    ret = release_model(&rknn_app_ctx);  //释放
    if (ret != 0) {
        printf("release_retinanet_model fail! ret=%d\n", ret);
        return -1;
    }
    return 0;
}
    