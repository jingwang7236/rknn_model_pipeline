#include <stdio.h>
#include <iostream>
#include <string.h>

#include "model_func.hpp"
#include "model_params.hpp"
// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <chrono>  // 计算耗时
using namespace std::chrono; 


/*-------------------------------------------
                Main  Functions
-------------------------------------------*/
int main(int argc, char **argv) {
    if (argc != 3) {
        printf("%s <model_name> <image_path>\n", argv[0]);
        return -1;
    }
    int ret;

    const char *model_name = argv[1];
    const char *image_path = argv[2];  // single image path or testset file path

    // 计算模型准确率
    if (std::string(model_name) == "test_face_attr"){ 
        // 创建模型管理器并添加模型
        ClsModelManager clsmodelManager;
        clsmodelManager.addModel("FaceAttr", "model/FaceAttr.rknn", inference_face_attr_model); // 模型名、模型路径、推理函数
        ret = ClsModelAccuracyCalculator(clsmodelManager, "FaceAttr", image_path);
        return ret;
    }else if (std::string(model_name) == "test_coco_det"){
        // 定义label_name和label_id的映射关系，传入函数
        std::map<std::string, int> label_name_map = {
            {"ren", 0},
            {"person", 0},  // labelme标注工具：label_name和模型输出的label_id对应
        };
        float CONF_THRESHOLD = 0.3; // 计算某个阈值区间的PR
        float NMS_THRESHOLD = 0.45; // 计算MAP
        DetModelManager modelManager;
        printf("inference_coco_person_model start\n");
        auto start = std::chrono::high_resolution_clock::now();       
        modelManager.addModel("CocoPersonDet", "model/yolov10s.rknn", inference_coco_person_det_model);
        ret = DetModelMapCalculator(modelManager, "CocoPersonDet", image_path, label_name_map, CONF_THRESHOLD, NMS_THRESHOLD);
        auto end = std::chrono::high_resolution_clock::now();
        printf("inference_coco_person_model time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
        printf("inference_coco_person_model end\n");
        return ret;
    }
    else if (std::string(model_name) == "test_person_det"){
        // 定义label_name和label_id的映射关系，传入函数
        std::map<std::string, int> label_name_map = {
            {"ren", 0},
            {"person", 0},  // labelme标注工具：label_name和模型输出的label_id对应
        };
        float CONF_THRESHOLD = 0.3; // 计算某个阈值区间的PR
        float NMS_THRESHOLD = 0.45; // 计算MAP
        printf("inference_person_det_model start\n");
        auto start = std::chrono::high_resolution_clock::now();  
        DetModelManager modelManager;
        modelManager.addModel("PersonDet", "model/PersonDet.rknn", inference_person_det_model);
        ret = DetModelMapCalculator(modelManager, "PersonDet", image_path, label_name_map, CONF_THRESHOLD, NMS_THRESHOLD);
        auto end = std::chrono::high_resolution_clock::now();
        printf("inference_person_det_model time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
        printf("inference_person_det_model end\n");
        return ret;
    }
        else if (std::string(model_name) == "test_header_det"){
        // 定义label_name和label_id的映射关系，传入函数
        std::map<std::string, int> label_name_map = {
            {"header", 0},// labelme标注工具：label_name和模型输出的label_id对应
            {"head", 0},  
        };
        float CONF_THRESHOLD = 0.3; // 计算某个阈值区间的PR
        float NMS_THRESHOLD = 0.45; // 计算MAP
        printf("inference_header_det_model start\n");
        auto start = std::chrono::high_resolution_clock::now();  
        DetModelManager modelManager;
        modelManager.addModel("HeaderDet", "model/HeaderDet.rknn", inference_header_det_model);
        ret = DetModelMapCalculator(modelManager, "HeaderDet", image_path, label_name_map, CONF_THRESHOLD, NMS_THRESHOLD);
        auto end = std::chrono::high_resolution_clock::now();
        printf("inference_header_det_model time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
        printf("inference_header_det_model end\n");
        return ret;
    }
    return 0;
}
