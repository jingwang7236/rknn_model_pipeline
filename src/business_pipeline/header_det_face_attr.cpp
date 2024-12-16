// 头肩检测+人脸属性,包含帽子、头盔、墨镜、口罩；

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <iostream>
#include "opencv2/opencv.hpp"

#include "outer_model/model_func.hpp"

/*-------------------------------------------
                  Functions
-------------------------------------------*/
face_det_attr_result inference_face_det_attr_model(rknn_app_context_t *det_app_ctx, rknn_app_context_t *cls_app_ctx, det_model_input input_data, bool enable_logger){
    face_det_attr_result result;
    unsigned char* data = input_data.data;
    int width = input_data.width;
    int height = input_data.height;
    int channel = input_data.channel;

    if (enable_logger){
        printf("Image size: %d x %d x %d\n", width, height, channel);
    }

    cv::Mat cv_img(height, width, CV_8UC3, data);
    // cv::Mat orig_img(height, width, CV_MAKETYPE(CV_8U, channels), input_data);
    if (cv_img.empty()) {
        std::cerr << "Image is empty or invalid." << std::endl;
    }
    // orig_img的通道顺序为cv图片的默认顺序bgr
    cv::Mat orig_img;
    cv::cvtColor(cv_img, orig_img, cv::COLOR_RGB2BGR); 
    
    ssd_det_result det_result = inference_header_det_model(det_app_ctx, input_data, false);

    result.count = det_result.count;
    for (int i = 0; i < det_result.count; ++i) {
        image_rect_t header_box;  // header的box
        // box越界检查
        header_box.left = std::max(det_result.object[i].box.left, 0);
        header_box.top = std::max(det_result.object[i].box.top, 0);
        header_box.right = std::min(det_result.object[i].box.right, width);
        header_box.bottom = std::min(det_result.object[i].box.bottom, height);

        // orig_img 根据large_box截图
        cv::Mat crop_img = orig_img(cv::Rect(header_box.left, header_box.top, header_box.right - header_box.left, header_box.bottom - header_box.top));
        
        // face_attr_model
        det_model_input cls_input_data;
        cls_input_data.data = crop_img.data;
        cls_input_data.width = crop_img.cols;
        cls_input_data.height = crop_img.rows;
        cls_input_data.channel = crop_img.channels();
        face_attr_cls_object cls_result = inference_face_attr_model(cls_app_ctx, cls_input_data, false); //推理
        face_det_attr_object det_cls_res;
        det_cls_res.box = det_result.object[i].box;
        det_cls_res.score = det_result.object[i].score;
        det_cls_res.face_attr_output = cls_result;
        result.object[i] = det_cls_res;
        
        // if (save_image) {
        //     // 保存截图,目前只支持Png格式，好像opencv库不支持jpg格式
        //     std::string image_path = "face_" + std::to_string(i) + ".png";
        //     bool success = cv::imwrite(image_path.c_str(), crop_img);
        //     std::cout << "保存截图" << image_path.c_str() << "成功: " << success << std::endl;
        // }

        
    }
    bool enable_draw_image = false; //画图,本地测试
    if (enable_draw_image) {
        cv::Mat draw_img = orig_img.clone();
        for (int i = 0; i < result.count; ++i) {
            int rx = result.object[i].box.left;
            int ry = result.object[i].box.top;
            int rw = result.object[i].box.right - result.object[i].box.left;
            int rh = result.object[i].box.bottom - result.object[i].box.top;
            cv::Rect box(rx, ry, rw, rh);
            std::string text = "header";
            std::string score_str = std::to_string(result.object[i].score);
            text += " " + score_str;
            int hat = result.object[i].face_attr_output.cls_output[0];
            int glass = result.object[i].face_attr_output.cls_output[1];
            int mask = result.object[i].face_attr_output.cls_output[2];
            text += " " + std::to_string(hat);
            text += " " + std::to_string(glass);
            text += " " + std::to_string(mask);
            cv::Scalar color(0, 255, 0);  // 绿色
            cv::rectangle(draw_img, box, color, 2);
            cv::Point textOrg(box.x, box.y - 10);
            cv::putText(draw_img, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2);
        }
        const char* image_path = "header_det_face_attr_result.png";
        cv::imwrite(image_path, draw_img);
        std::cout << "Draw result on " << image_path << " is finished." << std::endl;
    }

    return result;
}

