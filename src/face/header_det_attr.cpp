#include <stdio.h>
#include <iostream>

#include "opencv2/opencv.hpp"

#include "model_func.hpp"
#include "model_params.hpp"
#include "retinaface.h"

int infer_header_det_attr_model(cv::Mat orig_img, det_cls_result* result) {
    bool save_image = false;
    // face_det_model
    int ret;
    retinaface_result det_result;
    ret = inference_header_det_model(orig_img, &det_result);
    if (ret != 0) {
        printf("inference_face_det_model fail! ret=%d \n", ret);
        return -1;
    }
    
    result->count = det_result.count;

    for (int i = 0; i < det_result.count; ++i) {
        image_rect_t header_box;  // header的box
        // box越界检查
        header_box.left = std::max(det_result.object[i].box.left, 0);
        header_box.top = std::max(det_result.object[i].box.top, 0);
        header_box.right = std::min(det_result.object[i].box.right, orig_img.cols);
        header_box.bottom = std::min(det_result.object[i].box.bottom, orig_img.rows);

        // orig_img 根据large_box截图
        cv::Mat crop_img = orig_img(cv::Rect(header_box.left, header_box.top, header_box.right - header_box.left, header_box.bottom - header_box.top));
        
        // face_attr_model
        int num = 3;
        int cls_result[num];
        ret = inference_header_attr_model(crop_img, cls_result);
        if (ret != 0) {
            printf("inference_face_attr_model fail! ret=%d \n", ret);
            return -1;
        }
        result->object[i].cls_num = num;
        for (int j = 0; j < num; j++) {
            // std::cout << "属性:" << i << "的结果为:" << cls_result[i] << std::endl;
            result->object[i].cls[j] = cls_result[j];
        }

        // save result to det_cls_result
        result->object[i].box.left = header_box.left;
        result->object[i].box.top = header_box.top;
        result->object[i].box.right = header_box.right;
        result->object[i].box.bottom = header_box.bottom;
        result->object[i].score = det_result.object[i].score;

        if (save_image) {
            // 保存截图,目前只支持Png格式，好像opencv库不支持jpg格式
            std::string image_path = "face_" + std::to_string(i) + ".png";
            bool success = cv::imwrite(image_path.c_str(), crop_img);
            std::cout << "保存截图" << image_path.c_str() << "成功: " << success << std::endl;
        }
        
    }
    return ret;
}