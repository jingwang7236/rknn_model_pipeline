#include <stdio.h>
#include <iostream>

#include "opencv2/opencv.hpp"

#include "model_func.hpp"
#include "model_params.hpp"
#include "retinaface.h"

int infer_face_det_attr_model(cv::Mat orig_img, det_cls_result* result) {
    bool save_image = false;
    // face_det_model
    int ret;
    retinaface_result det_result;
    ret = inference_face_det_model(orig_img, &det_result);
    if (ret != 0) {
        printf("inference_face_det_model fail! ret=%d \n", ret);
        return -1;
    }
    
    result->count = det_result.count;

    for (int i = 0; i < det_result.count; ++i) {
        image_rect_t large_box;  // 扩边后的box
        int box_width = det_result.object[i].box.right - det_result.object[i].box.left;
        int box_height = det_result.object[i].box.bottom - det_result.object[i].box.top;
        large_box.left = (int)(det_result.object[i].box.left - box_width / 2);
        large_box.top = (int)(det_result.object[i].box.top - box_height / 2);
        large_box.right = (int)(det_result.object[i].box.right + box_width / 2);
        large_box.bottom = (int)(det_result.object[i].box.bottom + box_height / 2);
        // box越界检查
        large_box.left = std::max(large_box.left, 0);
        large_box.top = std::max(large_box.top, 0);
        large_box.right = std::min(large_box.right, orig_img.cols);
        large_box.bottom = std::min(large_box.bottom, orig_img.rows);

        // orig_img 根据large_box截图
        cv::Mat crop_img = orig_img(cv::Rect(large_box.left, large_box.top, large_box.right - large_box.left, large_box.bottom - large_box.top));
        
        // face_attr_model
        int num = 3;
        int cls_result[num];
        ret = inference_face_attr_model(crop_img, cls_result);
        if (ret != 0) {
            printf("inference_face_attr_model fail! ret=%d \n", ret);
            return -1;
        }
        result->object[i].cls_num = num;
        for (int j = 0; j < num; j++) {
            // std::cout << "属性:" << i << "的结果为:" << cls_result[i] << std::endl;
            result->object[i].cls[i] = cls_result[j];
        }

        // save result to det_cls_result
        result->object[i].box.left = det_result.object[i].box.left;
        result->object[i].box.top = det_result.object[i].box.top;
        result->object[i].box.right = det_result.object[i].box.right;
        result->object[i].box.bottom = det_result.object[i].box.bottom;
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