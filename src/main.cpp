#include <stdio.h>
#include <iostream>

#include "opencv2/opencv.hpp"

#include "model_func.hpp"
#include "model_params.hpp"
int main(int argc, char **argv) {
    if (argc != 3) {
        printf("%s <model_name> <image_path>\n", argv[0]);
        return -1;
    }
    const char *model_name = argv[1];
    const char *image_path = argv[2];
    
    //  Load image
    cv::Mat orig_img = cv::imread(image_path, cv::IMREAD_COLOR);
    // 检查图像是否成功读取
    if (orig_img.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;

        // 尝试获取更详细的错误信息
        try {
            // 重新尝试读取图像，捕获可能的异常
            orig_img = cv::imread(image_path, cv::IMREAD_COLOR);
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV Exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Standard Exception: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown Exception" << std::endl;
        }
    }
    int ret;
    if (std::string(model_name) == "face_det_attr"){
        det_cls_result result;
        ret = infer_face_det_attr_model(orig_img, &result);
        if (ret != 0) {
            printf("inference_face_det_attr_model fail! ret=%d \n", ret);
            return -1;
        }
        printf("face result count: %d\n", result.count);
        for (int i = 0; i < result.count; i++) {
            printf("face det %d: left: %d, top: %d, right: %d, bottom: %d, score: %f\n", i, result.object[i].box.left, result.object[i].box.top, result.object[i].box.right, result.object[i].box.bottom, result.object[i].score);
            printf("face attr %d: hat %d, glass %d, mask %d\n", i, result.object[i].cls[0], result.object[i].cls[1], result.object[i].cls[2]);
        }
    }else if (std::string(model_name) == "person_det"){ // person_det
        object_detect_result_list result;
        ret = inference_person_det_model(orig_img, &result);
        if (ret != 0) {
            printf("inference_person_det_model fail! ret=%d \n", ret);
            return -1;
        }
        printf("person result count: %d\n", result.count);
        for (int i = 0; i < result.count; i++) {
            printf("person det %d: left: %d, top: %d, right: %d, bottom: %d, score: %f\n", i, result.results[i].box.left, result.results[i].box.top, result.results[i].box.right, result.results[i].box.bottom, result.results[i].prop);
        }
    }else if (std::string(model_name) == "person_det_attr"){ // person_det_attr
        det_cls_result result;
        ret = infer_person_det_attr_model(orig_img, &result);
        if (ret != 0) {
            printf("infer_person_det_attr_model fail! ret=%d \n", ret);
            return -1;
        }
        printf("person result count: %d\n", result.count);
        for (int i = 0; i < result.count; i++) {
            printf("person det %d: left: %d, top: %d, right: %d, bottom: %d, score: %f\n", i, result.object[i].box.left, result.object[i].box.top, result.object[i].box.right, result.object[i].box.bottom, result.object[i].score);
            printf("person attr %d: hat %d, carry_stuff %d\n", i, result.object[i].cls[0], result.object[i].cls[1]);
        }
    }else if(std::string(model_name) == "phone_det"){
        retinaface_result result;
        ret = inference_phone_det_model(orig_img, &result);
        if (ret != 0) {
            printf("inference_phone_det_model fail! ret=%d \n", ret);
            return -1;
        }
        printf("phone result count: %d\n", result.count);
        for (int i = 0; i < result.count; i++) {
            printf("phone %d: left: %d, top: %d, right: %d, bottom: %d, score: %f\n", i, result.object[i].box.left, result.object[i].box.top, result.object[i].box.right, result.object[i].box.bottom, result.object[i].score);
        }
    }else if(std::string(model_name) == "header_det"){
        retinaface_result result;
        ret = inference_header_det_model(orig_img, &result);
        if (ret != 0) {
            printf("inference_header_det_model fail! ret=%d \n", ret);
            return -1;
        }
        printf("header result count: %d\n", result.count);
        for (int i = 0; i < result.count; i++) {
            printf("header %d: left: %d, top: %d, right: %d, bottom: %d, score: %f\n", i, result.object[i].box.left, result.object[i].box.top, result.object[i].box.right, result.object[i].box.bottom, result.object[i].score);
        }
    }else if (std::string(model_name) == "header_det_attr"){ // 
        det_cls_result result;
        ret = infer_header_det_attr_model(orig_img, &result);
        if (ret != 0) {
            printf("infer_header_det_attr_model fail! ret=%d \n", ret);
            return -1;
        }
        printf("header result count: %d\n", result.count);
        for (int i = 0; i < result.count; i++) {
            printf("header det %d: left: %d, top: %d, right: %d, bottom: %d, score: %f\n", i, result.object[i].box.left, result.object[i].box.top, result.object[i].box.right, result.object[i].box.bottom, result.object[i].score);
            printf("header attr %d: hat %d, glass %d, mask %d\n", i, result.object[i].cls[0], result.object[i].cls[1], result.object[i].cls[2]);
        }
    }
    else {
        std::cerr << "Unknown model_name: " << model_name << std::endl;
        throw std::invalid_argument("Unknown model_name: " + std::string(model_name));
    }
}
