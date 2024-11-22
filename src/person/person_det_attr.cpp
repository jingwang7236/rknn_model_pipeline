#include <stdio.h>
#include <iostream>

#include "opencv2/opencv.hpp"

#include "model_func.hpp"
#include "model_params.hpp"

int infer_person_det_attr_model(cv::Mat orig_img, det_cls_result* result) {
    bool save_image = false;
    // person_det_model
    int ret;
    object_detect_result_list det_result_list;
    ret = inference_person_det_model(orig_img, &det_result_list);
    printf("inference_person_det_model success! ret=%d \n", ret);
    if (ret != 0) {
        printf("inference_person_det_model fail! ret=%d \n", ret);
        return -1;
    }
    
    result->count = det_result_list.count;
    // printf("det_result_list.count=%d\n", det_result_list.count);
    for (int i = 0; i < result->count; i++)
    {
        object_detect_result *det_result = &(det_result_list.results[i]);
        // printf("person @ (%d %d %d %d) %.3f\n",
        //        det_result->box.left, det_result->box.top,
        //        det_result->box.right, det_result->box.bottom,
        //        det_result->prop);
        // box越界检查
        int left = (int)std::max(det_result->box.left, 0);
        int top = (int)std::max(det_result->box.top, 0);
        int right = (int)std::min(det_result->box.right, orig_img.cols);
        int bottom = (int)std::min(det_result->box.bottom, orig_img.rows);
        // printf("left:%d top:%d right:%d bottom:%d\n", left, top, right, bottom);
        // orig_img 根据box截图
        cv::Mat crop_img = orig_img(cv::Rect(left, top, right - left, bottom - top));
        // person_attr_model
        int num = 2;
        int cls_result[num];
        ret = inference_person_attr_model(crop_img, cls_result);
        if (ret != 0) {
            printf("inference_person_attr_model fail! ret=%d \n", ret);
            return -1;
        }
        result->object[i].cls_num = num;
        for (int i = 0; i < num; i++) {
            // std::cout << "属性:" << i << "的结果为:" << cls_result[i] << std::endl;
            result->object[i].cls[i] = cls_result[i];
        }
        // save result to det_cls_result
        result->object[i].box.left = left;
        result->object[i].box.top = top;
        result->object[i].box.right = right;
        result->object[i].box.bottom = bottom;
        result->object[i].score = det_result->prop;
        
        if (save_image) {
            // 保存截图,目前只支持Png格式，好像opencv库不支持jpg格式
            std::string image_path = "person_" + std::to_string(i) + ".png";
            bool success = cv::imwrite(image_path.c_str(), crop_img);
            std::cout << "保存截图" << image_path.c_str() << "成功: " << success << std::endl;
        }

    }
    return ret;
}