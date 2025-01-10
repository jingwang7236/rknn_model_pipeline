/*
分类模型计算准确率；
以人脸属性分类模型为例，计算准确率
输入：测试数据集文件（img_path label1_id label2_id label3_id）、模型名、模型路径、推理函数
输出：准确率
*/
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string.h>
#include <vector>
#include <map>

#include "stb_image.h"
#include "model_func.hpp"


static det_model_input read_image(const char *image_path){
    // Load image
    int width, height, channel;
    unsigned char* data = stbi_load(image_path, &width, &height, &channel, 3);
    if (data == NULL) {
        printf("Failed to load image from path: %s\n", image_path);
        exit(1);
    }
    // init input data
    det_model_input input_data;
    input_data.data = data;
    input_data.width = width;
    input_data.height = height;
    input_data.channel = channel;
    return input_data;
}

int ClsModelAccuracyCalculator(ClsModelManager& modelManager, const std::string& modelName, const char *testset_file){

    // 获取模型信息
    ClsModelInfo modelInfo;
    try {
        modelInfo = modelManager.getModel(modelName);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    // 模型初始化
    int ret = 0;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    const char* cls_model_path =  modelInfo.modelPath.c_str();
    ret = init_model(cls_model_path, &rknn_app_ctx);

    // 读取文件中的每一行的图片路径和属性值：image label_id
    std::ifstream infile(testset_file);
    if (!infile) {
        std::cout << "Failed to open file: " << testset_file << std::endl;
        return -1;
    }
    std::string line;
    int total_cnt = 0;
    int num_class = rknn_app_ctx.io_num.n_output;
    int attr_right_cnt[num_class];
    for (int i = 0; i < num_class; i++) {
        attr_right_cnt[i] = 0;
    }
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string image_path;
        int attr_label[num_class];
        if (!(iss >> image_path)) {
            std::cerr << "Error reading image path from line: " << line << std::endl;
            continue;
        }
        for (int i = 0; i < num_class; i++) {
            if (!(iss >> attr_label[i])) {
                std::cerr << "Error reading attribute label " << i + 1 << " from line: " << line << std::endl;
                break;
            }
        }
        total_cnt++;

        // 输出读取的结果
        std::cout << "Image Path: " << image_path << std::endl;
        for (int i = 0; i < num_class; i++) {
            std::cout << "Attribute " << i + 1 << ": " << attr_label[i] << std::endl;
        }

        // 模型推理图片获取结果
        det_model_input input_data = read_image(image_path.c_str());
        box_rect header_box;
        header_box.left = 0;
        header_box.top = 0;
        header_box.right = input_data.width;
        header_box.bottom = input_data.height;
        cls_model_result result = modelInfo.inferenceFunc(&rknn_app_ctx, input_data, header_box, false);
        for (int i = 0; i < num_class; i++) {
            int pred = result.cls_output[i];
            if (attr_label[i] == pred && attr_label[i] != -1) {
                attr_right_cnt[i]++;
            }
        }
    }
    ret = release_model(&rknn_app_ctx);
    infile.close();
    // 分别计算三个属性的准确率
    printf("Total number of %s is : %d\n", testset_file, total_cnt);
    for (int i = 0; i < num_class; i++) {
        printf("Accuracy of attribute %d: %.2f%%\n", i + 1, (float)attr_right_cnt[i] / total_cnt * 100);
    }
    return 0;
}
